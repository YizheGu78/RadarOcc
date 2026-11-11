# CenterFormer, Copyright (c) 2022 Zixiang Zhou, MIT; see LICENSE.MIT.
# MODIFIED: temporal-only P0 adaptation, current point queries, no box heatmap.
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from temporal.p0_common import P0TemporalBase, sample_volume, masked_xy_pool
from .attention import TemporalMSDeformAttn


class CenterFormerP0(P0TemporalBase):
    def __init__(self, heads=4, points=4, checkpoint_chunks=True, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_chunks = checkpoint_chunks
        c = self.channels
        # Current three scales + one high-resolution level per historical frame.
        self.attention = TemporalMSDeformAttn(c, 3+self.frame_nums-1, heads, points)
        self.query = nn.Linear(5, c)
        self.position = nn.Linear(3, c)
        self.time_embedding = nn.Linear(1, c)
        # Independent XY processing at every physical height, with shared weights.
        self.spatial_attention = nn.Conv3d(2, 1, (1, 7, 7), padding=(0, 3, 3), bias=False)
        self.spatial_fusion = nn.Conv3d(c*self.frame_nums, c, (1, 3, 3), padding=(0, 1, 1), bias=False)
        self.norm1 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, 2*c), nn.GELU(), nn.Linear(2*c, c))
        self.norm2 = nn.LayerNorm(c)

    def fuse(self, maps, masks, query, positions, times, valid):
        b, t, c, z, h, w = maps.shape
        current = maps[:, -1]
        spatial = torch.cat((current.mean(1, keepdim=True), current.amax(1, keepdim=True)), 1)
        gate = self.spatial_attention(spatial).sigmoid()
        # Upstream spatial-aware fusion used for the query context. No GT enters.
        history = maps[:, :-1] * gate[:, None]
        context = torch.cat((current[:, None], history), 1)
        ordered_times = torch.cat((times[:, -1:], times[:, :-1]), 1)
        ordered_valid = torch.cat((valid[:, -1:], valid[:, :-1]), 1)
        te = self.time_embedding(ordered_times[..., None]) * ordered_valid[..., None]
        context_mask = masks.amax(1)
        context = context + te[..., None, None, None]*masks
        context = self.spatial_fusion(context.reshape(b, t*c, z, h, w))*context_mask
        # Only encoded feature maps are pooled; sparse masks follow the same levels.
        source_maps = [current]
        source_masks = [masks[:, -1]]
        for scale in (2, 4):
            pooled, mask = masked_xy_pool(current, masks[:, -1], scale)
            source_maps.append(pooled)
            source_masks.append(mask)
        source_maps.extend(maps[:, j] for j in range(t-1))
        source_masks.extend(masks[:, j] for j in range(t-1))
        shapes = [x.shape[-3:] for x in source_maps]
        values = self.attention.project_values(source_maps, source_masks)
        level_valid = torch.cat((valid[:, -1:].expand(-1, 3), valid[:, :-1]), 1)
        results, supports, offsets = [], [], []
        def run(raw, pos, ctx, val):
            q = self.query(raw) + ctx + self.position(pos)
            mixed, support, offset = self.attention(q, pos, val, shapes, source_masks, level_valid)
            mixed = self.norm1(q + mixed)
            return self.norm2(mixed + self.ffn(mixed)), support, offset
        for start in range(0, query.shape[1], self.chunk_size):
            pos = positions[:, start:start+self.chunk_size]
            context_values, _ = sample_volume(context, pos, context_mask)
            args = (query[:, start:start+self.chunk_size], pos, context_values, values)
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                mixed, support, offset = checkpoint(run, *args, use_reentrant=False)
            else:
                mixed, support, offset = run(*args)
            results.append(mixed); supports.append(support); offsets.append(offset)
        return torch.cat(results, 1), torch.cat(supports, 1), dict(
            mean_abs_offset_cells=float(torch.stack(offsets).mean()),
            source_levels=len(source_maps), height_slices=z)
