import torch
import torch.nn as nn
from .core.ego_alignment import RadarBinGeometry, align_to_current
from .debug.temporal_recorder import TemporalRecorder
from .fusion.local_attention import LocalTemporalAttention


class TemporalPlugin(nn.Module):
    """
    DG-STF temporal plugin.

    Identity mode verifies that
    real T-frame loading preserves the RadarOcc baseline. ego_attention adds
    local history fusion on current P1 anchors.

    Expected canonical interface:
        output, debug = plugin(
            features=[B,T,N,C],
            coords=[B,T,N,3],
            poses=[B,T,4,4],
            frame_offsets=[B,T],
            doppler=[B,T,N,Cd],
            valid_mask=[B,T],
            timestamps=[B,T],
        )
    """

    def __init__(
        self,
        mode="identity",
        position="P1",
        alignment=None,
        debug=None,
        attention=None,
    ):
        super().__init__()
        self.mode = str(mode).lower()
        self.position = str(position).upper()
        self.alignment_cfg = dict(alignment or {})
        self.geometry = None
        if self.alignment_cfg:
            self.geometry = RadarBinGeometry.from_file(
                self.alignment_cfg['bins_path'],
                azimuth_sign=self.alignment_cfg.get('azimuth_sign', 1),
                elevation_sign=self.alignment_cfg.get('elevation_sign', 1),
            )
        debug = dict(debug or {})
        self.recorder = (TemporalRecorder(debug['output_dir'], debug.get('save_every', 50))
                         if debug.get('enabled', False) else None)
        if self.recorder is not None and self.geometry is None:
            raise ValueError('Alignment recording requires alignment.bins_path')

        if self.mode not in ('identity', 'ego_attention'):
            raise ValueError(f'Unsupported temporal mode: {self.mode}')
        self.fusion = None
        if self.mode == 'ego_attention':
            if self.geometry is None:
                raise ValueError('ego_attention requires alignment.bins_path')
            self.fusion = LocalTemporalAttention(**dict(attention or {}))

        if self.position != "P1":
            raise NotImplementedError(
                "The current regression stage supports P1 only."
            )

    def forward(
        self,
        features,
        coords,
        poses=None,
        frame_offsets=None,
        doppler=None,
        valid_mask=None,
        timestamps=None,
    ):
        if features.ndim != 4:
            raise ValueError(
                "TemporalPlugin expects features [B,T,N,C], got "
                f"{tuple(features.shape)}"
            )

        if coords.ndim != 4:
            raise ValueError(
                "TemporalPlugin expects coords [B,T,N,3], got "
                f"{tuple(coords.shape)}"
            )

        batch_size, frame_nums, num_points, channels = features.shape

        if coords.shape[:3] != features.shape[:3]:
            raise ValueError(
                "Temporal feature/coordinate shapes disagree: "
                f"features={tuple(features.shape)}, "
                f"coords={tuple(coords.shape)}"
            )

        if valid_mask is not None and torch.is_tensor(valid_mask):
            if valid_mask.shape[-1] != frame_nums:
                raise ValueError(
                    "valid_mask frame dimension does not match T: "
                    f"{tuple(valid_mask.shape)} vs T={frame_nums}"
                )
            if not bool(valid_mask[..., -1].all()):
                raise ValueError(
                    "Current temporal slot must always be valid."
                )

        # Identity guarantee: return the current P1 features byte-for-byte
        # (modulo view/stack semantics), and do not use history in any way.
        output = features[:, -1]

        alignment_debug = {}
        fusion_debug = {}
        if self.geometry is not None:
            def cpu_array(value):
                return value.detach().cpu().numpy() if torch.is_tensor(value) else value
            raw_xyz = self.geometry.to_xyz(cpu_array(coords))
            aligned_xyz, transforms = align_to_current(
                raw_xyz, cpu_array(poses),
                radar_to_lidar=self.alignment_cfg.get('radar_to_lidar'),
                valid_mask=cpu_array(valid_mask),
            )
            if self.fusion is not None:
                import numpy as np
                mask = (np.ones(features.shape[:2], dtype=bool) if valid_mask is None
                        else cpu_array(valid_mask))
                if frame_offsets is None:
                    raise ValueError('ego_attention requires relative frame_offsets')
                output, fusion_debug = self.fusion(
                    features, aligned_xyz, cpu_array(frame_offsets), mask)
            alignment_debug = dict(coords_raw_xyz=raw_xyz,
                                   coords_aligned_xyz=aligned_xyz,
                                   transforms=transforms,
                                   valid_mask=cpu_array(valid_mask),
                                   frame_offsets=cpu_array(frame_offsets))
            alignment_debug['power_scores'] = cpu_array(features[..., 2])
            if self.recorder is not None:
                self.recorder.record(**alignment_debug)

        debug = {
            "mode": self.mode,
            "position": self.position,
            "batch_size": int(batch_size),
            "frame_nums": int(frame_nums),
            "num_points": int(num_points),
            "channels": int(channels),
            "used_history": False,
            "coord_type": "spherical_index_range_azimuth_elevation",
        }
        debug.update(fusion_debug)
        # Do not retain large CPU debug arrays on the detector during training.
        if self.fusion is None or self.recorder is not None:
            debug.update(alignment_debug)

        return output, debug

