"""Fixed-descriptor pair ranking; never predicts offsets or edits coordinates."""
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .box_candidates import temporal_box_candidates


FORMAT = 'radarocc_learned_top8_v1'
DEFAULT_POLICY = dict(candidate_k=64, top_k=8, x_half_by_history=[4., 6., 8.],
                      y_half=1.5, z_half=1., descriptor_version='signed_log_power_ids_v1',
                      time_unit='seconds')


def normalize_descriptors(raw):
    """Fixed transforms only: preserve all 8 channels including bin IDs.

    Power/statistic channels use signed log1p / 80 for the large raw dynamic
    range; bin IDs use /64 as a fixed numerical scale, not a velocity table.
    No per-batch normalization or learned encoder is involved.
    """
    raw = raw.float()
    scale = raw.new_tensor([80., 80., 80., 64., 64., 64., 80., 80.])
    powers = torch.sign(raw) * torch.log1p(raw.abs())
    ids = torch.tensor([False, False, False, True, True, True, False, False],
                       device=raw.device)
    return torch.where(ids, raw, powers) / scale


class PairRanker(nn.Module):
    def __init__(self, hidden_dims=64):
        super().__init__()
        if hidden_dims <= 0:
            raise ValueError('hidden_dims must be positive')
        self.hidden_dims = hidden_dims
        # Q(8), H(8), absolute descriptor difference(8), XYZ(3), distance(1), dt(1).
        self.network = nn.Sequential(nn.Linear(29, hidden_dims), nn.ReLU(),
                                     nn.Linear(hidden_dims, hidden_dims), nn.ReLU(),
                                     nn.Linear(hidden_dims, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, query, history, relative_xyz, delta_t):
        q = normalize_descriptors(query).unsqueeze(-2).expand_as(history)
        h = normalize_descriptors(history)
        # Retain geometry precision for the distance prior: casting XYZ before
        # subtraction can reorder near ties relative to CPU cuboid search.
        distance_prior = torch.linalg.vector_norm(relative_xyz.double(), dim=-1) / 8.
        relative = relative_xyz.float()
        distance = distance_prior.float().unsqueeze(-1)
        dt = delta_t.float().reshape(-1, 1, 1).expand(*history.shape[:-1], 1)
        position = relative / relative.new_tensor([8., 1.5, 1.])
        pairs = torch.cat((q, h, (q-h).abs(), position, distance, dt), dim=-1)
        # Larger score means a smaller learned matching distance. At initialization
        # the residual is zero, reproducing geometric-distance ranking exactly.
        return -distance_prior + self.network(pairs).squeeze(-1)


def masked_topk(scores, valid, k=8):
    if scores.shape != valid.shape or scores.shape[-1] < k:
        raise ValueError('Matching score/mask shapes and candidate count >= K required')
    order = torch.argsort(scores.masked_fill(~valid, -torch.inf), dim=-1,
                          descending=True, stable=True)[..., :k]
    return order, valid.gather(-1, order)


def ranking_loss(scores, positive, valid):
    """Balanced per-query positive/negative loss on ALL candidates.

    Queries with no positive are not given invented targets. Loss-valid masks
    can exclude ambiguous bbox memberships without removing them from search.
    """
    pos, neg = positive & valid, ~positive & valid
    rows = pos.any(-1)
    if not rows.any():
        return scores.sum() * 0
    pos_loss = (F.softplus(-scores) * pos).sum(-1) / pos.sum(-1).clamp_min(1)
    neg_loss = (F.softplus(scores) * neg).sum(-1) / neg.sum(-1).clamp_min(1)
    return (pos_loss[rows] + neg_loss[rows]).mean()


def save_ranker(path, ranker, policy=None, **metadata):
    torch.save(dict(format=FORMAT, model_config=dict(hidden_dims=ranker.hidden_dims),
                    policy=dict(DEFAULT_POLICY if policy is None else policy),
                    state_dict=ranker.state_dict(), metadata=metadata), path)


def load_ranker(path, expected_policy=None):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Pretrain the Top-8 ranker first; checkpoint missing: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != FORMAT:
        raise ValueError('Not a standalone learned Top-8 checkpoint')
    if expected_policy is not None and payload['policy'] != expected_policy:
        raise ValueError('Ranker checkpoint search/descriptor/time policy disagrees with config')
    ranker = PairRanker(**payload['model_config'])
    ranker.load_state_dict(payload['state_dict'], strict=True)
    return ranker, payload


class BoxTopKSelector(nn.Module):
    """64 -> 8 per frame. Ranks detached fixed descriptors, returns indices only."""
    def __init__(self, method='learned', checkpoint=None, frozen=True, candidate_k=64,
                 top_k=8, x_half_by_history=(4., 6., 8.), y_half=1.5, z_half=1.,
                 query_chunk_size=512, cache_dir=None, cache_mode='readwrite'):
        super().__init__()
        if method not in ('learned', 'distance') or candidate_k != 64 or top_k != 8:
            raise ValueError('This experiment uses learned/distance 64-candidate Top-8')
        if not frozen:
            raise ValueError('Pretrain independently; full-model selector must be frozen')
        if query_chunk_size <= 0 or not x_half_by_history or not np.isfinite(
                [*x_half_by_history, y_half, z_half]).all() or min(
                *x_half_by_history, y_half, z_half) <= 0:
            raise ValueError('Positive finite box dimensions and query chunk required')
        self.method, self.query_chunk_size = method, query_chunk_size
        self.policy = dict(DEFAULT_POLICY, candidate_k=candidate_k, top_k=top_k,
                           x_half_by_history=list(x_half_by_history), y_half=y_half, z_half=z_half)
        self.ranker = None
        if method == 'learned':
            if checkpoint is None:
                raise ValueError('learned selection requires a pretrained checkpoint')
            self.ranker, _ = load_ranker(checkpoint, self.policy)
            self.ranker.requires_grad_(False)
            self.ranker.eval()
        from .selection_cache import SelectionCache
        self.cache = SelectionCache(cache_dir, cache_mode) if cache_dir else None
        self.cache_metadata = {}
        self._cache_weights = None
        self.register_load_state_dict_post_hook(self._reset_cache_weights)

    def _reset_cache_weights(self, module, incompatible_keys):
        self._cache_weights = None

    def _cache_namespace(self, features):
        from .selection_cache import update_array
        if self._cache_weights is None:
            digest = hashlib.sha256()
            if self.ranker is not None:
                for name, value in self.ranker.state_dict().items():
                    digest.update(name.encode())
                    update_array(digest, value.detach().cpu().numpy())
            self._cache_weights = digest.hexdigest()
        source = hashlib.sha256()
        for name in ('learned_topk.py', 'box_candidates.py', 'selection_cache.py'):
            source.update(Path(__file__).with_name(name).read_bytes())
        device = features.device
        return json.dumps(dict(format='selection_indices_v1', policy=self.policy,
            method=self.method, weights=self._cache_weights, source=source.hexdigest(),
            torch=str(torch.__version__), device=str(device),
            hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
            matmul_precision=torch.get_float32_matmul_precision(),
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            chunk=self.query_chunk_size), sort_keys=True)

    def train(self, mode=True):
        super().train(mode)
        if self.ranker is not None:
            self.ranker.eval()
        return self

    @torch.no_grad()
    def forward(self, features, xyz, valid_mask, timestamps=None, point_valid_mask=None):
        if self.cache is None:
            return self._select_online(features, xyz, valid_mask, timestamps, point_valid_mask)
        b, t, n, c = features.shape
        if c != 8 or np.shape(xyz) != (b, t, n, 3) or t < 2:
            raise ValueError('Selection cache expects P1 [B,T,N,8] and T>=2')
        raw = features.detach().float().cpu().numpy()
        namespace = self._cache_namespace(features)
        indices, masks = [], []
        for batch in range(b):
            xyz_one = np.asarray(xyz)[batch:batch+1]
            frames = np.asarray(valid_mask)[batch:batch+1]
            times = None if timestamps is None else np.asarray(timestamps)[batch:batch+1]
            points = None if point_valid_mask is None else np.asarray(point_valid_mask)[batch:batch+1]
            key = self.cache.key(namespace, raw[batch:batch+1], xyz_one, frames, times, points)
            selected = self.cache.load(key, (1, n, (t-1)*8), (t-1)*n)
            if selected is None:
                selected = self._select_online(features[batch:batch+1], xyz_one, frames, times, points)
                self.cache.save(key, *selected, metadata=dict(namespace=json.loads(namespace),
                    sample=self.cache_metadata, input_sha256=key))
            indices.append(selected[0])
            masks.append(selected[1])
        return np.concatenate(indices), np.concatenate(masks)

    @torch.no_grad()
    def _select_online(self, features, xyz, valid_mask, timestamps=None, point_valid_mask=None):
        b, t, n, _ = features.shape
        if features.shape[-1] != 8 or np.shape(xyz) != (b, t, n, 3):
            raise ValueError('Expected matching [B,T,N,8] descriptors and [B,T,N,3] XYZ')
        policy = self.policy
        pool, good = temporal_box_candidates(xyz, valid_mask,
            policy['x_half_by_history'], policy['y_half'], policy['z_half'],
            policy['candidate_k'], point_valid_mask)
        chosen = np.zeros((b, n, (t-1)*8), dtype=np.int64)
        chosen_valid = np.zeros_like(chosen, dtype=bool)
        if self.method == 'distance':
            for frame in range(t-1):
                chosen[..., frame*8:(frame+1)*8] = pool[..., frame*64:frame*64+8]
                chosen_valid[..., frame*8:(frame+1)*8] = good[..., frame*64:frame*64+8]
            return chosen, chosen_valid
        if timestamps is None:
            raise ValueError('Learned selection requires real temporal timestamps in seconds')
        times = np.asarray(timestamps, dtype=np.float64)
        frames = np.asarray(valid_mask, dtype=bool)
        if times.shape != (b, t) or not np.isfinite(times[frames]).all():
            raise ValueError('Finite real timestamps required for all valid temporal frames')
        device = features.device
        # Disable caller autocast: the small frozen scorer uses fixed FP32 inputs.
        context = torch.autocast(device_type=device.type, enabled=False) if device.type in ('cuda', 'cpu') else nullcontext()
        with context:
            raw = features.detach().float()
            coordinates = torch.as_tensor(xyz, dtype=torch.float64, device=device)
            for batch in range(b):
                history = raw[batch, :-1].reshape(-1, 8)
                history_xyz = coordinates[batch, :-1].reshape(-1, 3)
                for frame in range(t-1):
                    if not frames[batch, frame] or not good[batch, :, frame*64:(frame+1)*64].any():
                        continue
                    delta_t = float(times[batch, frame]-times[batch, -1])
                    if delta_t >= 0:
                        raise ValueError('Historical timestamp must precede current timestamp')
                    for start in range(0, n, self.query_chunk_size):
                        end = min(start+self.query_chunk_size, n)
                        ix = torch.as_tensor(pool[batch, start:end, frame*64:(frame+1)*64], device=device)
                        mask = torch.as_tensor(good[batch, start:end, frame*64:(frame+1)*64], device=device)
                        relative = history_xyz[ix] - coordinates[batch, -1, start:end, None]
                        scores = self.ranker(raw[batch, -1, start:end], history[ix], relative,
                                             raw.new_full((end-start,), delta_t))
                        order, valid = masked_topk(scores, mask, 8)
                        chosen[batch, start:end, frame*8:(frame+1)*8] = ix.gather(-1, order).cpu().numpy()
                        chosen_valid[batch, start:end, frame*8:(frame+1)*8] = valid.cpu().numpy()
        return chosen, chosen_valid
