import torch
import torch.nn as nn
import numpy as np
from .core.ego_alignment import RadarBinGeometry, align_to_current
from .debug.temporal_recorder import TemporalRecorder
from .fusion.local_attention import LocalTemporalAttention
from .fusion.ra_adta import RangeAwareDeformableTemporalAttention


class TemporalPlugin(nn.Module):
    """
    DG-STF temporal plugin.

    Identity mode verifies that
    real T-frame loading preserves the RadarOcc baseline. ego_attention adds
    local history fusion on current P1 anchors. ra_adta adds a separate,
    configurable range-aware deformable fusion path on the same anchors.

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
        power_mask=None,
    ):
        super().__init__()
        self.mode = str(mode).lower()
        self.position = str(position).upper()
        self.alignment_cfg = dict(alignment or {})
        self.geometry = None
        self.power_mask_cfg = dict(power_mask or {})
        self.power_mask_enabled = bool(self.power_mask_cfg.get('enabled', False))
        self.power_mask_radius = float(self.power_mask_cfg.get('radius_m', 3.7))
        if self.power_mask_enabled and (not np.isfinite(self.power_mask_radius)
                                        or self.power_mask_radius <= 0):
            raise ValueError('power_mask.radius_m must be finite and positive')
        if self.alignment_cfg:
            self.geometry = RadarBinGeometry.from_file(
                self.alignment_cfg['bins_path'],
                azimuth_sign=self.alignment_cfg.get('azimuth_sign', 1),
                elevation_sign=self.alignment_cfg.get('elevation_sign', 1),
            )
        debug = dict(debug or {})
        if self.power_mask_enabled and self.geometry is None:
            raise ValueError('power_mask requires alignment.bins_path for physical coordinates')
        self.recorder = (TemporalRecorder(debug['output_dir'], debug.get('save_every', 50))
                         if debug.get('enabled', False) else None)
        if self.recorder is not None and self.geometry is None:
            raise ValueError('Alignment recording requires alignment.bins_path')

        if self.mode not in ('identity', 'ego_attention', 'ra_adta'):
            raise ValueError(f'Unsupported temporal mode: {self.mode}')
        self.fusion = None
        if self.mode == 'ego_attention':
            if self.geometry is None:
                raise ValueError('ego_attention requires alignment.bins_path')
            self.fusion = LocalTemporalAttention(**dict(attention or {}))
        elif self.mode == 'ra_adta':
            if self.geometry is None:
                raise ValueError('ra_adta requires alignment.bins_path')
            self.fusion = RangeAwareDeformableTemporalAttention(
                geometry=self.geometry, **dict(attention or {}))

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
            point_valid = None
            if self.power_mask_enabled:
                if channels != 8:
                    raise ValueError('Power masking expects the 8-channel P1 descriptor')
                # Local XY mask, before any pose transform, in every frame
                # including the current one. Never edit caller/source tensors.
                excluded = np.hypot(raw_xyz[..., 0], raw_xyz[..., 1]) <= self.power_mask_radius
                point_valid = ~excluded
                keep = torch.as_tensor(point_valid, device=features.device)
                features = features.clone()
                for channel in (0, 1, 2, 6, 7):
                    features[..., channel] = features[..., channel].masked_fill(~keep, 0)
                output = features[:, -1]
                fusion_debug.update(power_mask_enabled=True,
                    power_mask_radius_m=self.power_mask_radius,
                    power_masked_per_frame=excluded.sum(axis=-1).tolist())
            aligned_xyz, transforms = align_to_current(
                raw_xyz, cpu_array(poses),
                radar_to_lidar=self.alignment_cfg.get('radar_to_lidar'),
                valid_mask=cpu_array(valid_mask),
            )
            if self.fusion is not None:
                mask = (np.ones(features.shape[:2], dtype=bool) if valid_mask is None
                        else cpu_array(valid_mask))
                if frame_offsets is None:
                    raise ValueError(f'{self.mode} requires relative frame_offsets')
                output, attention_debug = self.fusion(
                    features, aligned_xyz, cpu_array(frame_offsets), mask,
                    point_valid_mask=point_valid)
                fusion_debug.update(attention_debug)
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

