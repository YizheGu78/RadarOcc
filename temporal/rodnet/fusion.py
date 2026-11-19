"""Temporal-only RODNet TDC at P0; no M-Net/chirp, detection head or velocity."""
import math
import torch
from torch import nn
from temporal.p0_common import P0TemporalBase, sample_volume
from .tdc import CausalTDC


class RODNetP0(P0TemporalBase):
    def __init__(self, kernel_size=3, checkpoint_chunks=True,
                 correction_mode='multiplicative', max_additive_correction=.5, **kwargs):
        super().__init__(**kwargs)
        if correction_mode not in ('multiplicative', 'additive', 'hybrid'):
            raise ValueError('Unknown RODNet correction mode: '+str(correction_mode))
        if not math.isfinite(max_additive_correction) or max_additive_correction <= 0:
            raise ValueError('max_additive_correction must be finite and positive')
        self.correction_mode = correction_mode
        self.add_bound = max_additive_correction
        self.tdc = CausalTDC(self.channels, self.channels, self.frame_nums, kernel_size,
                             self.chunk_size, checkpoint_chunks)
        self.activation = nn.GELU()

        # Keep the default state_dict and seeded TDC initialization unchanged.
        # New heads do not consume randomness used by subsequent model layers.
        if correction_mode != 'multiplicative':
            with torch.random.fork_rng(devices=[]):
                self.add_gate = nn.Linear(5+self.channels, 5)
                nn.init.zeros_(self.add_gate.weight)
                nn.init.zeros_(self.add_gate.bias)
                if correction_mode == 'hybrid':
                    self.add_output = nn.Linear(self.channels, 5)
                    nn.init.zeros_(self.add_output.weight)
                    nn.init.zeros_(self.add_output.bias)

    def correct_continuous(self, raw, mixed, support):
        if self.correction_mode == 'multiplicative':
            return super().correct_continuous(raw, mixed, support)
        # These five channels are nonnegative power/statistics. Never alter the
        # three discrete Doppler IDs, or add an O(1) number to raw 1e34 values.
        if (raw < 0).any():
            raise ValueError('RODNet additive correction requires nonnegative power')
        base = (super().correct_continuous(raw, mixed, support)
                if self.correction_mode == 'hybrid' else raw)
        normalized = torch.log1p(raw)/80.
        gate = self.add_gate(torch.cat((normalized, mixed), -1)).sigmoid()
        head = self.add_output if self.correction_mode == 'hybrid' else self.output
        delta = self.add_bound * head(mixed).tanh() * gate
        # The bound is in natural-log units (normalized delta is delta/80).
        # Clamp before expm1 and preserve an exact identity at initialization.
        log_base = torch.log1p(base)
        limit = math.log(torch.finfo(base.dtype).max)-1.
        shifted = (log_base+delta).clamp(min=0., max=limit)
        change = torch.expm1(shifted)-torch.expm1(log_base.clamp(max=limit))
        corrected = base+change
        # Positive inputs must remain positive for the downstream log10 encoder.
        corrected = torch.where(base > 0, corrected.clamp_min(torch.finfo(base.dtype).tiny), corrected)
        return torch.where(support[..., None], corrected, raw)

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
