# RODNet, Copyright (c) 2021 Yizhou Wang, MIT; see LICENSE.
# MODIFIED from DeformConvPack3D: native grid_sample backend, causal current-only
# output, groups=1, stride=1, odd spatial kernel. Offset layout is explicitly
# (dy,dx) per (time,ky,kx) tap. Optional observation normalization excludes
# missing samples; no temporal offsets or learned modulation masks.
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class CausalTDC(nn.Module):
    def __init__(self, in_channels, out_channels, frames=4, kernel_size=3,
                 chunk_size=1024, checkpoint_chunks=True):
        super().__init__()
        if frames < 1 or kernel_size < 1 or kernel_size % 2 == 0 or chunk_size < 1:
            raise ValueError('Invalid TDC kernel/frames/chunk size')
        self.frames, self.kernel_size = frames, kernel_size
        self.chunk_size, self.checkpoint_chunks = chunk_size, checkpoint_chunks
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, frames, kernel_size, kernel_size))
        self.conv_offset = nn.Conv3d(in_channels, 2*frames*kernel_size**2,
                                     (frames, kernel_size, kernel_size),
                                     padding=(0, kernel_size//2, kernel_size//2))
        nn.init.uniform_(self.weight, -1/math.sqrt(in_channels*frames*kernel_size**2),
                         1/math.sqrt(in_channels*frames*kernel_size**2))
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)

    def _chunk(self, x, offsets, base, observation=None):
        b, c, t, h, w = x.shape
        k = self.kernel_size
        delta = torch.arange(k, device=x.device, dtype=x.dtype)-k//2
        yy, xx = torch.meshgrid(delta, delta, indexing='ij')
        regular = torch.stack((xx.flatten(), yy.flatten()), -1)
        result = x.new_zeros(b, self.weight.shape[0], base.shape[1])
        history_support = x.new_zeros(b, 1, base.shape[1])
        any_support = torch.zeros_like(history_support)
        for frame in range(t):
            # offsets supplied as [B,N,T,K*K,2] in dy,dx order.
            loc = base[:, :, None] + regular[None, None] + offsets[:, :, frame].flip(-1)
            grid = 2*(loc+.5)/x.new_tensor([w, h])-1
            field = x[:, :, frame]
            if observation is not None:
                field = field*observation[:, :, frame]
            values = F.grid_sample(field, grid, align_corners=False, padding_mode='zeros')
            if observation is not None:
                count = F.grid_sample(observation[:, :, frame], grid, align_corners=False)
                inside = ((grid >= -1) & (grid < 1)).all(-1)[:, None]
                good = (count >= 1e-4) & inside
                values = values/count.clamp_min(1e-4)*good
                seen = good.any(-1).to(x)
                any_support = torch.maximum(any_support, seen)
                if frame < t-1:
                    history_support = torch.maximum(history_support, seen)
            result = result + torch.einsum('bcnk,ock->bon', values, self.weight[:, :, frame].flatten(2))
        return result, history_support, any_support

    def forward(self, x, offsets=None, observation=None, return_support=False):
        if x.ndim != 5 or x.shape[2] != self.frames:
            raise ValueError('TDC input must be [B,C,configured_frames,H,W]')
        b, _, t, h, w = x.shape
        if observation is not None and observation.shape != (b, 1, t, h, w):
            raise ValueError('TDC observations must be [B,1,T,H,W]')
        if return_support and observation is None:
            raise ValueError('Support output requires observation masks')
        if offsets is None:
            offsets = self.conv_offset(x)
        expected = (b, 2*t*self.kernel_size**2, 1, h, w)
        if offsets.shape != expected:
            raise ValueError('Invalid TDC offsets shape')
        offsets = offsets[:, :, 0].flatten(2).transpose(1, 2).reshape(
            b, h*w, t, self.kernel_size**2, 2)
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                                torch.arange(w, device=x.device, dtype=x.dtype), indexing='ij')
        base = torch.stack((xx.flatten(), yy.flatten()), -1)[None].expand(b, -1, -1)
        results, history, observed = [], [], []
        for start in range(0, h*w, self.chunk_size):
            args = (x, offsets[:, start:start+self.chunk_size], base[:, start:start+self.chunk_size], observation)
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                result, historical, good = checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                result, historical, good = self._chunk(*args)
            results.append(result); history.append(historical); observed.append(good)
        result = torch.cat(results, -1).reshape(b, self.weight.shape[0], h, w)
        if return_support:
            return result, torch.cat(history, -1).reshape(b, 1, h, w), torch.cat(observed, -1).reshape(b, 1, h, w)
        return result

