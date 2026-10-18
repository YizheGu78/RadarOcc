"""Unaligned P1 temporal memory: no poses, geometry warp or nearest neighbours.

Learned latent queries compress each historical frame. Two independent cross-
attention/FFN blocks then update current queries using the same history memory.
Position is expressed in each frame's ORIGINAL spherical index coordinates.
"""
import torch
from torch import nn


class CrossBlock(nn.Module):
    def __init__(self, dim, heads, ffn_ratio):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ffn_ratio), nn.GELU(),
                                 nn.Linear(dim * ffn_ratio, dim))

    def forward(self, query, memory, padding_mask=None):
        normalized = self.memory_norm(memory)
        update, _ = self.attention(self.query_norm(query), normalized, normalized,
                                   key_padding_mask=padding_mask, need_weights=False)
        query = query + update
        return query + self.ffn(self.ffn_norm(query))


class UnalignedMemoryTemporal(nn.Module):
    def __init__(self, channels=8, embed_dims=64, num_heads=4, num_layers=2,
                 memory_tokens=256, chunk_size=1024, ffn_ratio=2,
                 coord_scales=(256., 107., 37.), time_scale_frames=3.,
                 max_log_correction=0.5):
        super().__init__()
        if channels != 8:
            raise ValueError('P1 currently requires the 8-channel Doppler descriptor')
        sizes = (embed_dims, num_heads, num_layers, memory_tokens, chunk_size, ffn_ratio)
        if any(not isinstance(v, int) or v <= 0 for v in sizes) or embed_dims % num_heads:
            raise ValueError('Positive integer sizes and embed_dims divisible by num_heads required')
        if len(coord_scales) != 3 or any(v <= 0 for v in coord_scales):
            raise ValueError('coord_scales must contain three positive scales')
        if time_scale_frames <= 0 or max_log_correction <= 0:
            raise ValueError('Time and correction scales must be positive')
        self.chunk_size, self.memory_tokens = chunk_size, memory_tokens
        self.time_scale, self.max_correction = time_scale_frames, max_log_correction
        self.register_buffer('coord_scales', torch.tensor(coord_scales), persistent=False)
        self.input = nn.Sequential(nn.Linear(channels, embed_dims), nn.LayerNorm(embed_dims))
        self.position = nn.Sequential(nn.Linear(3, embed_dims), nn.GELU(), nn.Linear(embed_dims, embed_dims))
        # Continuous frame-gap encoding also accepts larger gaps in subset PKLs.
        self.time = nn.Sequential(nn.Linear(1, embed_dims), nn.GELU(), nn.Linear(embed_dims, embed_dims))
        self.latents = nn.Parameter(torch.randn(1, memory_tokens, embed_dims) * 0.02)
        self.compressor = CrossBlock(embed_dims, num_heads, ffn_ratio)
        self.layers = nn.ModuleList([CrossBlock(embed_dims, num_heads, ffn_ratio)
                                     for _ in range(num_layers)])
        self.gate = nn.Linear(embed_dims * 2, embed_dims)
        self.output = nn.Linear(embed_dims, 5)
        self.reset_output()

    def reset_output(self):
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, coords, poses=None, frame_offsets=None,
                doppler=None, valid_mask=None, timestamps=None):
        # poses/doppler/timestamps are deliberately unused interface arguments.
        if features.ndim != 4 or features.shape[-1] != 8 or features.shape[2] == 0:
            raise ValueError('features must be nonempty [B,T,N,8]')
        b, t, n, _ = features.shape
        if t < 1 or tuple(coords.shape) != (b, t, n, 3):
            raise ValueError('coords must be [B,T,N,3] with T >= 1')
        device = features.device
        mask = (torch.ones(b, t, dtype=torch.bool, device=device) if valid_mask is None
                else torch.as_tensor(valid_mask, device=device, dtype=torch.bool))
        if tuple(mask.shape) != (b, t) or not mask[:, -1].all():
            raise ValueError('valid_mask must be [B,T], with a valid current frame')
        current = features[:, -1]
        debug = dict(mode='unaligned_memory', used_history=False, aligned=False,
                     num_layers=len(self.layers), memory_tokens_per_frame=self.memory_tokens)
        if t == 1 or not mask[:, :-1].any():
            return current, debug
        if frame_offsets is None:
            raise ValueError('Relative frame_offsets are required')
        offsets = torch.as_tensor(frame_offsets, device=device, dtype=torch.float32)
        if tuple(offsets.shape) != (b, t) or not torch.isfinite(offsets[mask]).all():
            raise ValueError('frame_offsets must be finite [B,T] on valid frames')
        gaps = offsets - offsets[:, -1:]
        if (gaps[:, :-1][mask[:, :-1]] >= 0).any():
            raise ValueError('Valid history must precede the current frame')
        xyz = torch.as_tensor(coords, device=device, dtype=torch.float32)
        # Invalid placeholders cannot contaminate attention, including NaNs.
        x = torch.where(mask[..., None, None], features.float(), 0.)
        xyz = torch.where(mask[..., None, None], xyz, 0.)
        gaps = torch.where(mask, gaps, 0.)
        if not torch.isfinite(x).all() or not torch.isfinite(xyz).all():
            raise ValueError('Valid features and coordinates must be finite')
        encoded = (self.input(torch.sign(x) * torch.log1p(x.abs()))
                   + self.position(xyz / self.coord_scales)
                   + self.time((gaps / self.time_scale)[..., None])[:, :, None])
        memories = []
        for frame in range(t - 1):
            active = mask[:, frame].nonzero(as_tuple=True)[0]
            memory = encoded.new_zeros(b, self.memory_tokens, encoded.shape[-1])
            if active.numel():
                source = encoded[:, frame].index_select(0, active)
                latent = self.latents.expand(active.numel(), -1, -1)
                compressed = self.compressor(latent, source)
                memory = memory.index_copy(0, active, compressed)
            memories.append(memory)
        memory = torch.cat(memories, dim=1)
        padding = (~mask[:, :-1]).repeat_interleave(self.memory_tokens, dim=1)
        # A zero null token prevents all-masked attention for scene starts in
        # mixed batches. It is available only to samples with no history.
        has_history = mask[:, :-1].any(dim=1)
        memory = torch.cat((memory, memory.new_zeros(b, 1, memory.shape[-1])), dim=1)
        padding = torch.cat((padding, has_history[:, None]), dim=1)
        chunks = []
        for start in range(0, n, self.chunk_size):
            src = current[:, start:start + self.chunk_size]
            query0 = encoded[:, -1, start:start + self.chunk_size]
            query = query0
            for layer in self.layers:
                query = layer(query, memory, padding)
            gate = torch.sigmoid(self.gate(torch.cat((query0, query), dim=-1)))
            correction = self.max_correction * torch.tanh(self.output(gate * query))
            correction = correction * has_history[:, None, None]
            power = src[..., [0, 1, 2, 6, 7]]
            fused = power + power * torch.expm1(correction).to(power.dtype)
            # Preserve current sparse anchors and Doppler bin IDs exactly.
            chunks.append(torch.cat((fused[..., :3], src[..., 3:6], fused[..., 3:]), dim=-1))
        debug['used_history'] = True
        return torch.cat(chunks, dim=1), debug
