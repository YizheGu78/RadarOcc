from __future__ import annotations

import numpy as np

from tradition.core.config import CFARConfig, KRadarConfig
from tradition.core.geometry import (
    angle_from_index,
    doppler_from_index,
    radar_to_lidar,
    spherical_to_cartesian,
)
from tradition.core.interfaces import CFARBackend, TargetDetector
from tradition.core.types import RadarDetection


_EPS = 1e-12


class ClassicalTargetDetector(TargetDetector):
    """Radar cube -> CFAR -> peak grouping -> angle/parameter estimation."""

    def __init__(
        self,
        cfar_backend: CFARBackend,
        radar_cfg: KRadarConfig | None = None,
        cfar_cfg: CFARConfig | None = None,
    ) -> None:
        self.cfar_backend = cfar_backend
        self.radar_cfg = radar_cfg or KRadarConfig()
        self.cfar_cfg = cfar_cfg or CFARConfig()

    @staticmethod
    def _power_to_db(power: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(np.asarray(power, dtype=np.float64), _EPS))

    def _dual_axis_cfar(self, rd_power_dr: np.ndarray) -> np.ndarray:
        cfg = self.cfar_cfg
        rd_db = self._power_to_db(rd_power_dr)
        d_bins, r_bins = rd_db.shape

        range_pass = np.zeros_like(rd_db, dtype=bool)
        for d in range(d_bins):
            threshold = self.cfar_backend.threshold(
                rd_db[d],
                guard_len=cfg.guard_range,
                noise_len=cfg.noise_range,
            )
            range_pass[d] = rd_db[d] > threshold

        doppler_pass = np.zeros_like(rd_db, dtype=bool)
        for r in range(r_bins):
            threshold = self.cfar_backend.threshold(
                rd_db[:, r],
                guard_len=cfg.guard_doppler,
                noise_len=cfg.noise_doppler,
            )
            doppler_pass[:, r] = rd_db[:, r] > threshold

        mask = range_pass & doppler_pass
        if cfg.min_power_db is not None:
            mask &= rd_db >= cfg.min_power_db
        return mask

    def _is_local_maximum(self, rd_power: np.ndarray, d: int, r: int) -> bool:
        radius = self.cfar_cfg.local_max_radius
        if radius <= 0:
            return True
        d_bins, r_bins = rd_power.shape
        center = rd_power[d, r]
        for dd in range(d - radius, d + radius + 1):
            for rr in range(max(0, r - radius), min(r_bins, r + radius + 1)):
                d_wrap = dd % d_bins
                if d_wrap == d and rr == r:
                    continue
                if rd_power[d_wrap, rr] > center:
                    return False
        return True

    def detect(self, radar_tensor_drea: np.ndarray) -> list[RadarDetection]:
        cube = np.asarray(radar_tensor_drea)
        if cube.ndim != 4:
            raise ValueError(f"Expected [D,R,E,A], got {cube.shape}")

        rd_power = np.max(cube, axis=(2, 3))
        cfar_mask = self._dual_axis_cfar(rd_power)
        candidates = np.argwhere(cfar_mask)
        candidates = [
            (int(d), int(r))
            for d, r in candidates
            if self._is_local_maximum(rd_power, int(d), int(r))
        ]

        if len(candidates) > self.cfar_cfg.max_detections:
            candidates.sort(
                key=lambda pair: float(rd_power[pair[0], pair[1]]),
                reverse=True,
            )
            candidates = candidates[: self.cfar_cfg.max_detections]

        radar_cfg = self.radar_cfg
        detections: list[RadarDetection] = []
        for d, r in candidates:
            angular_slice = cube[d, r]
            e, a = np.unravel_index(int(np.argmax(angular_slice)), angular_slice.shape)
            power = float(angular_slice[e, a])
            range_m = r * radar_cfg.range_resolution_m
            azimuth_rad = angle_from_index(
                int(a), radar_cfg.azimuth_min_deg, radar_cfg.azimuth_step_deg
            )
            elevation_rad = angle_from_index(
                int(e), radar_cfg.elevation_min_deg, radar_cfg.elevation_step_deg
            )
            radial_velocity = doppler_from_index(d, radar_cfg)
            xyz_radar = spherical_to_cartesian(range_m, azimuth_rad, elevation_rad)
            xyz_lidar = radar_to_lidar(xyz_radar, radar_cfg)
            detections.append(
                RadarDetection(
                    range_index=r,
                    doppler_index=d,
                    elevation_index=int(e),
                    azimuth_index=int(a),
                    power=power,
                    range_m=range_m,
                    radial_velocity_mps=radial_velocity,
                    azimuth_rad=azimuth_rad,
                    elevation_rad=elevation_rad,
                    xyz_radar_m=xyz_radar,
                    xyz_lidar_m=xyz_lidar,
                )
            )
        return detections
