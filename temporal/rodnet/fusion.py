"""Temporal-only RODNet TDC at P0; no M-Net/chirp, detection head or velocity."""
import torch
from torch import nn
from temporal.p0_common import P0TemporalBase, sample_volume
from .tdc import CausalTDC


class RODNetP0(P0TemporalBase):
    def __init__(self, kernel_size=3, checkpoint_chunks=True, **kwargs):
        super().__init__(**kwargs)
        self.tdc = CausalTDC(self.channels, self.channels, self.frame_nums, kernel_size,
                             self.chunk_size, checkpoint_chunks)
        self.activation = nn.GELU()

    def fuse(self, maps, masks, query, positions, times, valid):
        # Treat physical height slices as independent batches of the original
        # 2D TDC. All slices share weights; no height pooling or temporal future.
        b, t, c, z, h, w = maps.shape
        x = maps.permute(0, 3, 2, 1, 4, 5).reshape(b*z, c, t, h, w)
        observation = masks.permute(0, 3, 2, 1, 4, 5).reshape(b*z, 1, t, h, w)
        offsets = self.tdc.conv_offset(x)
        mixed, history, observed = self.tdc(x, offsets, observation, return_support=True)
        mixed = self.activation(mixed).reshape(b, z, c, h, w).permute(0, 2, 1, 3, 4)
        observed = observed.reshape(b, z, 1, h, w).permute(0, 2, 1, 3, 4)
        history = history.reshape(b, z, 1, h, w).permute(0, 2, 1, 3, 4)
        values, _ = sample_volume(mixed, positions, observed)
        historical, _ = sample_volume(history, positions)
        return values, historical[..., 0] > 1e-4, dict(
            mean_abs_offset_cells=float(offsets.detach().abs().mean()),
            temporal_taps=self.frame_nums, height_slices=z)
