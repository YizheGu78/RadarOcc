from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from tradition.core.config import GridConfig, KRadarConfig, MappingConfig
from tradition.core.geometry import ray_voxels_dda, xyz_to_voxel
from tradition.core.interfaces import OccupancyMapper
from tradition.core.types import MotionLabel, RadarDetection


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("Probability must be strictly between 0 and 1.")
    return math.log(probability / (1.0 - probability))


class LogOddsOccupancyGrid3D(OccupancyMapper):
    """Classical 3D inverse sensor model with free-ray carving."""

    def __init__(
        self,
        grid_cfg: GridConfig | None = None,
        radar_cfg: KRadarConfig | None = None,
        mapping_cfg: MappingConfig | None = None,
    ) -> None:
        self.grid_cfg = grid_cfg or GridConfig()
        self.radar_cfg = radar_cfg or KRadarConfig()
        self.mapping_cfg = mapping_cfg or MappingConfig()
        self.sensor_origin_lidar_m = np.asarray(
            self.radar_cfg.radar_to_lidar_translation_xyz_m, dtype=np.float64
        )
        self._hit_update = _logit(self.mapping_cfg.p_hit)
        self._free_update = _logit(self.mapping_cfg.p_free)
        self.reset()

    def reset(self) -> None:
        shape = self.grid_cfg.shape_xyz
        self.occupancy_log_odds = np.zeros(shape, dtype=np.float32)
        self.static_evidence = np.zeros(shape, dtype=np.float32)
        self.dynamic_evidence = np.zeros(shape, dtype=np.float32)

    def _update_hit_neighborhood(
        self,
        endpoint: tuple[int, int, int],
        motion: MotionLabel,
    ) -> None:
        cfg = self.mapping_cfg
        ex, ey, ez = endpoint
        sx, sy, sz = self.grid_cfg.shape_xyz
        for dx in range(-cfg.hit_radius_xy_voxels, cfg.hit_radius_xy_voxels + 1):
            for dy in range(-cfg.hit_radius_xy_voxels, cfg.hit_radius_xy_voxels + 1):
                for dz in range(-cfg.hit_radius_z_voxels, cfg.hit_radius_z_voxels + 1):
                    x, y, z = ex + dx, ey + dy, ez + dz
                    if not (0 <= x < sx and 0 <= y < sy and 0 <= z < sz):
                        continue
                    weight = math.exp(-0.5 * float(dx * dx + dy * dy + dz * dz))
                    self.occupancy_log_odds[x, y, z] += self._hit_update * weight
                    if motion == MotionLabel.DYNAMIC:
                        self.dynamic_evidence[x, y, z] += weight
                    else:
                        self.static_evidence[x, y, z] += weight

    def update(
        self,
        detections: Sequence[RadarDetection],
        motion_labels: Sequence[MotionLabel],
    ) -> None:
        if len(detections) != len(motion_labels):
            raise ValueError("detections and motion_labels must have equal length.")

        for det, motion in zip(detections, motion_labels):
            endpoint = xyz_to_voxel(det.xyz_lidar_m, self.grid_cfg)
            if endpoint is None:
                continue
            ray = ray_voxels_dda(
                self.sensor_origin_lidar_m, det.xyz_lidar_m, self.grid_cfg
            )
            for voxel in ray[:-1]:
                self.occupancy_log_odds[voxel] += self._free_update
            self._update_hit_neighborhood(endpoint, motion)

        cfg = self.mapping_cfg
        np.clip(
            self.occupancy_log_odds,
            cfg.log_odds_min,
            cfg.log_odds_max,
            out=self.occupancy_log_odds,
        )

    def labels(self) -> np.ndarray:
        probability = 1.0 / (1.0 + np.exp(-self.occupancy_log_odds))
        occupied = probability >= self.mapping_cfg.occupancy_probability_threshold
        labels = np.zeros(self.grid_cfg.shape_xyz, dtype=np.uint8)
        static = occupied & (
            self.dynamic_evidence
            < self.static_evidence * self.mapping_cfg.dynamic_override_ratio
        )
        dynamic = occupied & ~static
        labels[static] = int(MotionLabel.STATIC)
        labels[dynamic] = int(MotionLabel.DYNAMIC)
        return labels
