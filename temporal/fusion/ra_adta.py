"""Range-Aware Anisotropic Deformable Temporal Attention at P1."""
import math

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..core.range_anisotropic_geometry import RangeAnisotropicGeometry
from .sparse_deformable_sampler import (anisotropic_candidates, anisotropic_knn_candidates,
                                        masked_softmax, sparse_sample)


class RangeAwareDeformableTemporalAttention(nn.Module):
    def __init__(self, geometry, channels=8, embed_dims=32, num_heads=4,
                 sampling_points_per_head=2, candidate_pool_size=24,
                 alpha_range=3.0, alpha_azimuth=3.0, alpha_elevation=3.0,
                 min_angular_scale_m=0.25, offset_bound=0.5,
                 temperature=0.5, interpolation_cutoff=2.0, chunk_size=256,
                 search_chunk_size=256, time_scale_frames=3.0,
                 max_log_correction=0.5, checkpoint_chunks=True,
                 search_mode='radius_anchor'):
        super().__init__()
        sizes = (embed_dims, num_heads, sampling_points_per_head, candidate_pool_size,
                 chunk_size, search_chunk_size)
        if (channels != 8 or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
                                 for v in sizes) or embed_dims % num_heads):
            raise ValueError('P1 requires 8 channels and positive compatible integer dimensions')
        scales = (offset_bound, temperature, interpolation_cutoff,
                  time_scale_frames, max_log_correction)
        if any(not math.isfinite(v) or v <= 0 for v in scales):
            raise ValueError('RA-ADTA scales must be finite and positive')
        self.geometry = RangeAnisotropicGeometry(geometry, alpha_range, alpha_azimuth,
                                                alpha_elevation, min_angular_scale_m)
        self.heads, self.dim = num_heads, embed_dims // num_heads
        self.points, self.pool = sampling_points_per_head, candidate_pool_size
        self.chunk_size, self.search_chunk_size = chunk_size, search_chunk_size
        self.bound, self.temperature, self.cutoff = offset_bound, temperature, interpolation_cutoff
        self.time_scale, self.max_correction = time_scale_frames, max_log_correction
        if search_mode not in ('radius_anchor', 'anisotropic_knn'):
            raise ValueError('Unsupported RA-ADTA search_mode')
        self.search_mode = search_mode
        self.checkpoint_chunks = bool(checkpoint_chunks)
        self.input = nn.Sequential(nn.Linear(channels, embed_dims), nn.LayerNorm(embed_dims))
        self.q = nn.Linear(embed_dims, embed_dims)
        self.k_proj, self.v = nn.Linear(embed_dims, embed_dims), nn.Linear(embed_dims, embed_dims)
        self.offset = nn.Sequential(nn.Linear(2 * embed_dims + 1, embed_dims), nn.ReLU(),
                                    nn.Linear(embed_dims, num_heads * self.points * 3))
        self.position = nn.Sequential(nn.Linear(4, embed_dims), nn.ReLU(),
                                      nn.Linear(embed_dims, self.dim))
        self.temporal_output = nn.Linear(embed_dims, embed_dims)
        self.output = nn.Linear(embed_dims, 5)
        # Only the final residual is zero-initialized. A second zero projection
        # here would unnecessarily delay gradients to the offset MLP's first layer.
        nn.init.normal_(self.offset[-1].weight, std=1e-3)
        # Small distinct initial samples break symmetry between heads/points.
        angles = torch.arange(num_heads * self.points, dtype=torch.float32) * 2.39996323
        height = torch.linspace(-0.75, 0.75, num_heads * self.points)
        initial = 0.25 * torch.stack((angles.cos(), angles.sin(), height), dim=-1)
        with torch.no_grad():
            self.offset[-1].bias.copy_(torch.atanh(initial).flatten())
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _fuse_chunk(self, current, history, history_key, history_value,
                    ids, good, local, relative_time):
        queries, frames, _ = ids.shape
        candidates = history[ids]
        context_weights = masked_softmax(-local.square().sum(-1) / 2, good, dim=-1)
        context = (candidates * context_weights[..., None]).sum(dim=2)
        dt = relative_time[None, :, None].expand(queries, frames, 1) / self.time_scale
        offset_input = torch.cat((current[:, None].expand(-1, frames, -1), context, dt), dim=-1)
        offsets = self.bound * torch.tanh(self.offset(offset_input)).view(
            queries, frames, self.heads, self.points, 3)
        key, value, sample_valid = sparse_sample(history_key[ids], history_value[ids],
            local, offsets, good, self.temperature, self.cutoff)
        sample_dt = dt[:, :, None, None].expand(-1, -1, self.heads, self.points, -1)
        position = self.position(torch.cat((offsets, sample_dt), dim=-1))
        key, value = key + position, value + position
        query = self.q(current).view(queries, self.heads, self.dim)
        logits = (query[:, None, :, None] * key).sum(-1) / self.dim ** 0.5
        logits = logits.permute(0, 2, 1, 3).reshape(queries, self.heads, -1)
        valid = sample_valid.permute(0, 2, 1, 3).reshape_as(logits)
        weights = masked_softmax(logits, valid, dim=-1).view(
            queries, self.heads, frames, self.points).permute(0, 2, 1, 3)
        mixed = (weights[..., None] * value).sum(dim=(1, 3)).flatten(1)
        correction = self.max_correction * torch.tanh(self.output(self.temporal_output(mixed)))
        has_history = sample_valid.any(dim=(1, 2, 3))
        return correction * has_history[:, None], has_history

    def forward(self, features, aligned_xyz, frame_offsets, valid_mask, point_valid_mask=None):
        if features.ndim != 4 or features.shape[-1] != 8 or not torch.isfinite(features).all():
            raise ValueError('P1 descriptors must be finite [B,T,N,8]')
        b, t, n, _ = features.shape
        if min(b, t, n) <= 0:
            raise ValueError('P1 batch, frame and point dimensions must be positive')
        if np.shape(aligned_xyz) != (b, t, n, 3):
            raise ValueError('aligned_xyz must match P1 [B,T,N,3]')
        frame_offsets = np.asarray(frame_offsets, dtype=np.float64)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if frame_offsets.shape != (b, t) or not np.isfinite(frame_offsets).all():
            raise ValueError('frame_offsets must be finite [B,T]')
        if valid_mask.shape != (b, t) or not valid_mask[:, -1].all():
            raise ValueError('valid_mask must be [B,T] with valid current slots')
        current = features[:, -1]
        if t == 1 or not valid_mask[:, :-1].any():
            return current, dict(used_history=False, queries_with_history=0)
        candidate_search = (anisotropic_knn_candidates if self.search_mode == 'anisotropic_knn'
                            else anisotropic_candidates)
        ids_np, good_np, local_np = candidate_search(aligned_xyz, valid_mask,
            self.geometry, self.pool, self.bound, self.cutoff, self.search_chunk_size,
            point_valid_mask)
        if not good_np.any():
            return current, dict(used_history=False, queries_with_history=0)
        device = features.device
        # Project each historical token once, before gathering query candidates.
        x = features.float()
        encoded = self.input(torch.sign(x) * torch.log1p(x.abs()))
        history = encoded[:, :-1].reshape(b, -1, encoded.shape[-1])
        history_key = self.k_proj(history).view(b, -1, self.heads, self.dim)
        history_value = self.v(history).view(b, -1, self.heads, self.dim)
        relative_time = torch.as_tensor(frame_offsets[:, :-1] - frame_offsets[:, -1:],
                                        device=device, dtype=torch.float32)
        result, counts = [], []
        for batch in range(b):
            chunks = []
            for start in range(0, n, self.chunk_size):
                end = min(start + self.chunk_size, n)
                args = (encoded[batch, -1, start:end], history[batch], history_key[batch],
                        history_value[batch],
                        torch.as_tensor(ids_np[batch, start:end], device=device),
                        torch.as_tensor(good_np[batch, start:end], device=device),
                        torch.as_tensor(local_np[batch, start:end], device=device),
                        relative_time[batch])
                if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                    correction, has_history = checkpoint(self._fuse_chunk, *args, use_reentrant=False)
                else:
                    correction, has_history = self._fuse_chunk(*args)
                src = current[batch, start:end]
                power = src[:, [0, 1, 2, 6, 7]]
                fused = power + power * torch.expm1(correction).to(power.dtype)
                chunks.append(torch.cat((fused[:, :3], src[:, 3:6], fused[:, 3:]), dim=-1))
                counts.append(has_history.sum())
            result.append(torch.cat(chunks, dim=0))
        count = int(torch.stack(counts).sum().detach().cpu())
        return torch.stack(result), dict(used_history=count > 0, queries_with_history=count,
            candidate_pool_size=self.pool, sampling_points_per_head=self.points,
            offset_bound=self.bound)
