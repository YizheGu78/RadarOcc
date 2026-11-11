# Deformable DETR, Copyright (c) 2020 SenseTime. All Rights Reserved.
# Apache-2.0; see LICENSE.Apache-2.0.
# Adapted from TuSimple/centerformer at 5a949b88ed7bb15aafb39bf78c95f1452063ebea.
# MODIFIED: pure PyTorch backend, optional 3D fields, observation-normalized
# sampling, invalid-candidate softmax masking and chunk support.
import math
import torch
from torch import nn
from torch.nn import functional as F


def sample_deformable_values(value, spatial_shapes, locations, observations=None):
    """Sample XY or XYZ maps; normalize each sample by observed corner weight.

    Returns [B,heads,dim,Q,levels,points] and [B,Q,heads,levels,points]
    validity. Omitting observations preserves the upstream dense operator.
    """
    b, _, heads, dim = value.shape
    _, queries, _, levels, points, _ = locations.shape
    values = value.split([math.prod(shape) for shape in spatial_shapes], dim=1)
    sampled, validity = [], []
    for level, shape in enumerate(spatial_shapes):
        field = values[level].flatten(2).transpose(1, 2).reshape(b*heads, dim, *shape)
        grid = (2*locations[:, :, :, level]-1).transpose(1, 2).flatten(0, 1)
        if len(shape) == 3:
            grid = grid.unsqueeze(3)
        sample = F.grid_sample(field, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        sample = sample.reshape(b, heads, dim, queries, points)
        good = torch.ones(b, heads, queries, points, dtype=torch.bool, device=value.device)
        if observations is not None:
            mask = observations[level][:, None].expand(-1, heads, -1, *shape).reshape(b*heads, 1, *shape)
            weight = F.grid_sample(mask, grid, align_corners=False).reshape(b, heads, queries, points)
            loc = locations[:, :, :, level].transpose(1, 2)
            good = (weight >= 1e-4) & ((loc >= 0) & (loc < 1)).all(-1)
            # Projected values have already been multiplied by observation masks.
            sample = sample/weight.clamp_min(1e-4)[:, :, None]*good[:, :, None]
        sampled.append(sample)
        validity.append(good.permute(0, 2, 1, 3))
    return torch.stack(sampled, -2), torch.stack(validity, -2)


def ms_deform_attn_core_pytorch(value, spatial_shapes, locations, weights):
    """Upstream dense reference core, extended to accept 3D source levels."""
    b, _, heads, dim = value.shape
    queries = locations.shape[1]
    sampled, _ = sample_deformable_values(value, spatial_shapes, locations)
    weights = weights.permute(0, 2, 1, 3, 4)[:, :, None]
    output = (sampled*weights).sum((-1, -2))
    return output.view(b, heads*dim, queries).transpose(1, 2).contiguous()


class TemporalMSDeformAttn(nn.Module):
    """CenterFormer MSDeformAttn with CUDA extension replaced by reference core."""
    def __init__(self, channels=32, levels=6, heads=4, points=4, spatial_dims=3):
        super().__init__()
        if heads < 1 or points < 1 or levels < 1 or channels % heads or spatial_dims not in (2, 3):
            raise ValueError('Invalid attention dimensions')
        self.heads, self.levels, self.points = heads, levels, points
        self.dim = channels // heads
        self.spatial_dims = spatial_dims
        self.sampling_offsets = nn.Linear(channels, heads*levels*points*spatial_dims)
        self.attention_weights = nn.Linear(channels, heads*levels*points)
        self.value_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.sampling_offsets.weight)
        theta = torch.arange(heads)*2*math.pi/heads
        grid = torch.stack((theta.cos(), theta.sin()), -1)
        if spatial_dims == 3:
            # Start in the query's physical height; Z offsets are learnable.
            grid = torch.cat((grid, torch.zeros(heads, 1)), -1)
        grid = grid / grid.abs().amax(-1, keepdim=True)
        grid = grid[:, None, None].repeat(1, levels, points, 1)
        grid = grid * torch.arange(1, points+1)[None, None, :, None]
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.flatten())
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        for layer in (self.value_proj, self.output_proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def project_values(self, maps, masks):
        values = []
        for field, mask in zip(maps, masks):
            x = self.value_proj(field.flatten(2).transpose(1, 2))
            values.append(x * mask.flatten(2).transpose(1, 2))
        return torch.cat(values, 1).view(maps[0].shape[0], -1, self.heads, self.dim)

    def forward(self, query, positions, values, shapes, masks, level_valid, history_start=3):
        b, q, _ = query.shape
        offsets = self.sampling_offsets(query).view(b, q, self.heads, self.levels, self.points, self.spatial_dims)
        normalizer = query.new_tensor([tuple(reversed(shape)) for shape in shapes])
        locations = positions[:, :, None, None, None] + offsets/normalizer[None, None, None, :, None]
        logits = self.attention_weights(query).view(b, q, self.heads, self.levels, self.points)
        sampled, observed = sample_deformable_values(values, shapes, locations, masks)
        good = observed & level_valid[:, None, None, :, None]
        logits = logits.masked_fill(~good, -1e9)
        weights = logits.flatten(-2).softmax(-1).view_as(logits)*good
        weights = weights/weights.sum((-1, -2), keepdim=True).clamp_min(1e-9)
        output = (sampled*weights.permute(0, 2, 1, 3, 4)[:, :, None]).sum((-1, -2))
        output = output.reshape(b, self.heads*self.dim, q).transpose(1, 2)
        # Do not mistake a projected bias or empty cell for historical evidence.
        support = good[:, :, :, history_start:].any(dim=(2, 3, 4))
        return self.output_proj(output), support, offsets.detach().abs().mean()
