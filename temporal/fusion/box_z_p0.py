"""Box-based Z0-Z3 fusion at the existing post-selection / pre-range-attention input.

Raw descriptors are interpolated on the actual R/A/E bins, with a separate
observation mask. No KNN, Top-64/8, GT input, or P4 approximation is used.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def normalize_descriptor(raw):
    continuous = torch.sign(raw)*torch.log1p(raw.abs())/80.
    return torch.cat((continuous[..., :3], raw[..., 3:6]/64., continuous[..., 6:]), -1)


def scatter_field(features, coords, shape):
    """Return [1,8,R,A,E] descriptor means and [1,1,R,A,E] observation mask.

    coords index the original physical bins. Duplicate returns in one cell
    are averaged, independent of token ordering. Missing cells remain masked.
    """
    r, a, e = shape
    ids = (coords[:, 0]*a+coords[:, 1])*e+coords[:, 2]
    count = features.new_zeros(r*a*e).scatter_add(0, ids, features.new_ones(len(ids)))
    sums = features.new_zeros(r*a*e, features.shape[-1]).index_add(0, ids, features)
    values = sums/count.clamp_min(1)[:, None]
    return values.T.reshape(1, features.shape[-1], r, a, e), (count > 0).to(features).reshape(1, 1, r, a, e)


def physical_grid(xyz, bins):
    """Differentiable fractional indices on actual monotonic bins; grid tuple E,A,R."""
    radius = xyz.square().sum(-1).clamp_min(1e-12).sqrt()
    horizontal = xyz[..., :2].square().sum(-1).clamp_min(1e-12).sqrt()
    spherical = (radius, torch.atan2(xyz[..., 1], xyz[..., 0]+1e-12),
                 torch.atan2(xyz[..., 2], horizontal))
    indices, valid = [], torch.isfinite(xyz).all(-1) & (radius > 1e-6)
    for value, table in zip(spherical, bins):
        hi = torch.searchsorted(table, value.contiguous()).clamp(1, len(table)-1)
        lo = hi-1
        fraction = (value-table[lo])/(table[hi]-table[lo])
        indices.append(2*(lo+fraction+.5)/len(table)-1)
        valid = valid & (value >= table[0]) & (value <= table[-1])
    return torch.stack(indices[::-1], -1), valid


def masked_interpolate(field, observation, xyz, bins, min_weight=1e-4):
    grid, inside = physical_grid(xyz, bins)
    shape = xyz.shape[:-1]
    grid = grid.reshape(1, -1, 1, 1, 3)
    numerator = F.grid_sample(field, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    weight = F.grid_sample(observation, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    values = (numerator/weight.clamp_min(min_weight))[0, :, :, 0, 0].T.reshape(*shape, field.shape[1])
    valid = inside & (weight.reshape(shape) >= min_weight)
    return values, valid


def map_coordinates(points, current, history, mode, alpha=None, beta=None, samples=8):
    st, sh = current[..., 3:].clamp_min(.1), history[..., 3:].clamp_min(.1)
    mapped = history[..., :3] + (points-current[..., :3])*sh/st
    if mode == 'Z0':
        levels = (torch.arange(samples, dtype=points.dtype, device=points.device)+.5)/samples-.5
        mapped = mapped[..., None, :].expand(*mapped.shape[:-1], samples, 3).clone()
        mapped[..., 2] = history[..., 2, None]+levels*sh[..., 2, None]
    elif mode == 'Z1':
        mapped = torch.cat((mapped[..., :2], points[..., 2:3]), -1)
    elif mode == 'Z3':
        z = history[..., 2]+sh[..., 2]/st[..., 2]*alpha.exp()*(points[..., 2]-current[..., 2])+beta
        mapped = torch.cat((mapped[..., :2], z[..., None]), -1)
    elif mode != 'Z2':
        raise ValueError(mode)
    return mapped


class BoxZP0Fusion(nn.Module):
    def __init__(self, bins, mode='Z1', dim=64, heads=4, samples=8, chunk_size=2048,
                 lambda_scale=.001, lambda_shift=.001, min_interp_weight=1e-4, enabled=True):
        super().__init__()
        if mode not in ('Z0', 'Z1', 'Z2', 'Z3') or dim % heads or min(samples, chunk_size) < 1:
            raise ValueError('Invalid Z ablation settings')
        self.mode, self.dim, self.heads = mode, dim, heads
        self.samples, self.chunk_size, self.enabled = samples, chunk_size, enabled
        self.lambda_scale, self.lambda_shift = lambda_scale, lambda_shift
        self.min_interp_weight = min_interp_weight
        for name, table in zip(('range_bins', 'azimuth_bins', 'elevation_bins'), bins):
            table = torch.as_tensor(table, dtype=torch.float32)
            if len(table) < 2 or not torch.isfinite(table).all() or not (table[1:] > table[:-1]).all():
                raise ValueError('Physical bins must be finite, strictly increasing')
            self.register_buffer(name, table)
        self.q, self.k, self.v = (nn.Linear(8, dim) for _ in range(3))
        self.time_bias = nn.Embedding(3, heads)
        self.output = nn.Linear(dim, 5)
        self.gate = nn.Sequential(nn.Linear(8+dim, dim), nn.ReLU(), nn.Linear(dim, 5))
        # Construct Z3-only layer last: all shared parameters get identical seeded initialization.
        with torch.random.fork_rng(devices=[]):
            self.z_linear = nn.Linear(7, 2) if mode == 'Z3' else None
        nn.init.zeros_(self.time_bias.weight)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        nn.init.constant_(self.gate[-1].bias, -2.)
        if self.z_linear is not None:
            nn.init.zeros_(self.z_linear.weight); nn.init.zeros_(self.z_linear.bias)
        self.losses, self.diagnostics, self.pair_values = {}, {}, {}

    @property
    def bins(self):
        return (self.range_bins, self.azimuth_bins, self.elevation_bins)

    def xyz(self, indices):
        radius, az, el = (table[indices[:, i]] for i, table in enumerate(self.bins))
        return torch.stack((radius*el.cos()*az.cos(), radius*el.cos()*az.sin(), radius*el.sin()), -1)

    def pair_parameters(self, current, history):
        ct = current[:, None].expand_as(history)
        st, sh = ct[..., 3:].clamp_min(.1), history[..., 3:].clamp_min(.1)
        gaps = current.new_tensor([1., 2., 3.]).view(1, 3, 1).expand(len(current), -1, -1)/3
        geometry = torch.cat(((sh/st).log(), (history[..., :3]-ct[..., :3])/st, gaps), -1)
        if self.z_linear is None:
            return geometry[..., 0]*0, geometry[..., 0]*0
        ab = self.z_linear(geometry)
        return math.log(2)*ab[..., 0].tanh(), .5*ab[..., 1].tanh()

    def forward(self, features, coords, pairs):
        """features [4,N,8], coords [4,N,3] R/A/E; temporal order oldest to current."""
        if features.shape[0] != 4 or features.shape[-1] != 8 or coords.shape != (*features.shape[:2], 3):
            raise ValueError('Expected T=4 raw descriptor sequence')
        if not torch.isfinite(features).all():
            raise ValueError('Nonfinite raw descriptors')
        ft = features[-1]
        current, history, matched, inverse = (pairs[k] for k in ('current', 'history', 'matched', 'inverse'))
        matched = matched.bool()
        alpha, beta = self.pair_parameters(current, history)
        # Summing raw powers can overflow FP32 even though every input is finite.
        anchor = sum((p.sum()*0 for p in self.parameters()), ft.new_zeros(()))
        self.losses = {}
        if self.mode == 'Z3':
            denominator = matched.sum().clamp_min(1)
            self.losses = dict(loss_z_scale=self.lambda_scale*(alpha.square()*matched).sum()/denominator,
                               loss_z_shift=self.lambda_shift*(beta.square()*matched).sum()/denominator)
        xyz = self.xyz(coords[-1])
        owner = torch.full((len(xyz),), -1, device=xyz.device, dtype=torch.long)
        if self.enabled:
            for i in torch.argsort(current[:, 3:].prod(-1), stable=True):
                inside = ((xyz-current[i, :3]).abs() <= current[i, 3:]/2+1e-6).all(-1)
                owner[(owner < 0) & inside & matched[i].any()] = i
        selected = torch.where(owner >= 0)[0]
        normalized = normalize_descriptor(features)
        grids = []
        if len(selected):
            for j in range(3):
                grids.append(scatter_field(normalized[2-j], coords[2-j], tuple(len(b) for b in self.bins)))
        result = ft.clone()+anchor
        valid_samples = attempted = valid_tokens = used_queries = 0
        gate_sum = correction_sum = 0.
        for start in range(0, len(selected), self.chunk_size):
            ids = selected[start:start+self.chunk_size]; box_ids = owner[ids]
            points, fquery = xyz[ids], normalized[-1, ids]
            tokens, masks = [], []
            for j in range(3):
                hb = history[box_ids, j]
                mapped = map_coordinates(points, current[box_ids], hb, self.mode,
                    alpha[box_ids, j], beta[box_ids, j], self.samples)
                mapped = mapped if self.mode == 'Z0' else mapped[:, None]
                inside = ((mapped-hb[:, None, :3]).abs() <= hb[:, None, 3:]/2+1e-6).all(-1)
                local = mapped @ inverse[j, :3, :3].T+inverse[j, :3, 3]
                values, valid = masked_interpolate(*grids[j], local, self.bins, self.min_interp_weight)
                exists = matched[box_ids, j, None]
                valid = valid & inside & exists
                attempted += int(exists.expand_as(valid).sum()); valid_samples += int(valid.sum())
                tokens.append((values*valid[..., None]).sum(1)/valid.sum(1).clamp_min(1)[:, None])
                masks.append(valid.any(1))
            tokens, mask = torch.stack(tokens, 1), torch.stack(masks, 1)
            q = self.q(fquery).reshape(-1, self.heads, self.dim//self.heads)
            k = self.k(tokens).reshape(-1, 3, self.heads, self.dim//self.heads)
            v = self.v(tokens).reshape_as(k)
            scores = (q[:, None]*k).sum(-1)/math.sqrt(self.dim//self.heads)+self.time_bias.weight
            weights = scores.masked_fill(~mask[..., None], -1e9).softmax(1)*mask[..., None]
            weights = weights/weights.sum(1, keepdim=True).clamp_min(1e-9)
            mixed = (weights[..., None]*v).sum(1).flatten(1)
            gate = self.gate(torch.cat((fquery, mixed), -1)).sigmoid()
            good = mask.any(1)
            # Shared raw adapter: bounded relative residual on five continuous channels.
            # Preserve the current Doppler bin IDs, zeros and signs exactly.
            relative = gate*torch.expm1(.5*self.output(mixed).tanh())*good[:, None]
            source = ft[ids][:, [0, 1, 2, 6, 7]]
            power = source+source*relative
            result[ids] = torch.cat((power[:, :3], ft[ids, 3:6], power[:, 3:]), -1)
            valid_tokens += int(mask.sum()); used_queries += int(good.sum())
            gate_sum += float((gate.mean(-1)*good).detach().sum())
            correction_sum += float(relative.detach().abs().mean(-1).sum())
        with torch.no_grad():
            scale = history[..., 5]/current[:, None, 5].clamp_min(.1)*alpha.exp()
            self.pair_values = {k: v[matched].detach().cpu().tolist() for k, v in
                                [('alpha', alpha), ('shift', beta), ('scale', scale)]}
            self.diagnostics = dict(sample_attempted=attempted, sample_valid=valid_samples,
                valid_sampling_rate=valid_samples/max(attempted, 1),
                invalid_sampling_rate=1-valid_samples/attempted if attempted else 0.,
                valid_tokens=valid_tokens, fused_queries=used_queries, selected_queries=len(selected),
                gate_sum=gate_sum, gate_mean=gate_sum/max(used_queries, 1),
                correction_sum=correction_sum, relative_contribution=correction_sum/max(used_queries, 1),
                pairs=int(matched.sum()))
        return result
