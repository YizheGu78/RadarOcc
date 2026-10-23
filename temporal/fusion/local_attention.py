"""P1 local temporal attention. Geometry search is CPU; learned ops are torch."""
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn


def local_neighbors(xyz, valid_mask, k=8, radius_m=2.0, point_valid_mask=None):
    """Return [B,N,(T-1)*K] flattened history IDs and validity.

    A separate tree per frame prevents one dense frame taking every slot.
    Optional point mask excludes both historical candidates and current queries.
    """
    b, t, n, _ = xyz.shape
    point_mask = (np.ones((b,t,n), dtype=bool) if point_valid_mask is None
                  else np.asarray(point_valid_mask, dtype=bool))
    if point_mask.shape != (b,t,n):
        raise ValueError('point_valid_mask must be [B,T,N]')
    ids = np.zeros((b, n, (t-1)*k), dtype=np.int64)
    mask = np.zeros_like(ids, dtype=bool)
    for batch in range(b):
        for frame in range(t-1):
            if not valid_mask[batch, frame]:
                continue
            candidates = np.flatnonzero(point_mask[batch, frame])
            queries = np.flatnonzero(point_mask[batch, -1])
            if candidates.size == 0 or queries.size == 0:
                continue
            dist, ix = cKDTree(xyz[batch, frame, candidates]).query(
                xyz[batch, -1, queries], k=list(range(1, k+1)),
                distance_upper_bound=radius_m, workers=1)
            good = np.isfinite(dist) & (ix < len(candidates))
            original_ix = candidates[np.minimum(ix, len(candidates)-1)]
            slots = slice(frame*k, (frame+1)*k)
            ids[batch, queries, slots] = np.where(good, original_ix + frame*n, 0)
            mask[batch, queries, slots] = good
    return ids, mask


class LocalTemporalAttention(nn.Module):
    def __init__(self, channels=8, embed_dims=32, num_heads=4,
                 neighbors_per_frame=8, radius_m=2.0, chunk_size=1024,
                 time_scale_frames=3.0, max_log_correction=0.5, selection=None):
        super().__init__()
        if channels != 8 or embed_dims % num_heads or min(
                embed_dims, num_heads, neighbors_per_frame, chunk_size) <= 0:
            raise ValueError('P1 requires 8 channels and positive compatible attention dimensions')
        if min(radius_m, time_scale_frames, max_log_correction) <= 0:
            raise ValueError('Attention scales must be positive')
        self.k, self.radius_m, self.chunk_size = neighbors_per_frame, radius_m, chunk_size
        self.heads, self.dim = num_heads, embed_dims // num_heads
        self.time_scale = time_scale_frames
        self.max_correction = max_log_correction
        self.selector = None
        if selection is not None:
            from ..core.learned_topk import BoxTopKSelector
            self.selector = BoxTopKSelector(**dict(selection))
            if neighbors_per_frame != 8:
                raise ValueError('Box Top-8 selection requires neighbors_per_frame=8')
        self.input = nn.Sequential(nn.Linear(channels, embed_dims), nn.LayerNorm(embed_dims))
        self.q = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v = nn.Linear(embed_dims, embed_dims)
        self.position = nn.Sequential(nn.Linear(4, embed_dims), nn.ReLU(),
                                      nn.Linear(embed_dims, embed_dims))
        self.output = nn.Linear(embed_dims, 5)
        # Residual starts at exact identity; output projection learns first.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, aligned_xyz, frame_offsets, valid_mask, point_valid_mask=None,
                timestamps=None):
        current = features[:, -1]
        if features.shape[-1] != 8 or not torch.isfinite(features).all():
            raise ValueError('P1 descriptors must be finite with 8 channels')
        b, t, n, _ = features.shape
        if t == 1:
            return current, dict(used_history=False, queries_with_history=0)
        if self.selector is None:
            ids_np, mask_np = local_neighbors(aligned_xyz, valid_mask, self.k, self.radius_m,
                                            point_valid_mask=point_valid_mask)
        else:
            ids_np, mask_np = self.selector(features, aligned_xyz, valid_mask,
                timestamps=timestamps, point_valid_mask=point_valid_mask)
        if not mask_np.any():
            return current, dict(used_history=False, queries_with_history=0)
        device = features.device
        ids = torch.as_tensor(ids_np, device=device)
        mask = torch.as_tensor(mask_np, device=device)
        xyz = torch.as_tensor(aligned_xyz, device=device, dtype=torch.float32)
        offsets = torch.as_tensor(frame_offsets, device=device, dtype=torch.float32)
        if offsets.shape != (b, t) or not torch.isfinite(offsets).all():
            raise ValueError('frame_offsets must be finite [B,T]')
        # Log-scale only the attention branch; baseline input stays untouched.
        x = features.float()
        encoded = self.input(torch.sign(x) * torch.log1p(x.abs()))
        history = encoded[:, :-1].reshape(b, -1, encoded.shape[-1])
        hist_xyz = xyz[:, :-1].reshape(b, -1, 3)
        hist_time = (offsets[:, :-1] - offsets[:, -1:]).repeat_interleave(n, dim=1)
        result = []
        for batch in range(b):
            chunks = []
            for start in range(0, n, self.chunk_size):
                end = min(start+self.chunk_size, n)
                ix, good = ids[batch, start:end], mask[batch, start:end]
                rel = (hist_xyz[batch][ix] - xyz[batch, -1, start:end, None]) / self.radius_m
                dt = hist_time[batch][ix, None] / self.time_scale
                pos = self.position(torch.cat((rel, dt), dim=-1))
                kv = history[batch][ix]
                q = self.q(encoded[batch, -1, start:end]).view(-1, self.heads, self.dim)
                key = (self.k_proj(kv) + pos).view(end-start, -1, self.heads, self.dim)
                value = (self.v(kv) + pos).view_as(key)
                logits = (q[:, None] * key).sum(-1) / self.dim**0.5
                logits = logits.masked_fill(~good[..., None], -1e9)
                weights = logits.softmax(dim=1) * good[..., None]
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
                mixed = (weights[..., None] * value).sum(dim=1).flatten(1)
                correction = self.max_correction * torch.tanh(self.output(mixed))
                correction = correction * good.any(dim=1, keepdim=True)
                # Relative residual f + f*expm1(delta): positive power stays
                # positive, zeros stay zero, Doppler IDs 3:6 remain exact.
                src = current[batch, start:end]
                power = src[:, [0, 1, 2, 6, 7]]
                fused = power + power * torch.expm1(correction).to(power.dtype)
                chunks.append(torch.cat((fused[:, :3], src[:, 3:6], fused[:, 3:]), dim=-1))
            result.append(torch.cat(chunks, dim=0))
        return torch.stack(result), dict(used_history=True,
            queries_with_history=int(mask_np.any(axis=-1).sum()))

