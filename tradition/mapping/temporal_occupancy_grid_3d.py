from __future__ import annotations

import numpy as np

from tradition.core.config import GridConfig, KRadarConfig, MappingConfig
from tradition.core.interfaces import TemporalOccupancyMapper
from tradition.core.types import SemanticLabel
from tradition.core.geometry import xyz_to_voxel
from tradition.mapping.occupancy_grid_3d import LogOddsOccupancyGrid3D


class TemporalLogOddsOccupancyGrid3D(
    LogOddsOccupancyGrid3D, TemporalOccupancyMapper
):
    """Current-frame inverse sensor model plus aligned static history hits."""

    def __init__(
        self,
        grid_cfg: GridConfig | None = None,
        radar_cfg: KRadarConfig | None = None,
        mapping_cfg: MappingConfig | None = None,
    ) -> None:
        super().__init__(grid_cfg, radar_cfg, mapping_cfg)

    def update_historic_background(self, xyz_lidar_m: np.ndarray) -> None:
        points = np.asarray(xyz_lidar_m, dtype=np.float64).reshape(-1, 3)
        for point in points:
            endpoint = xyz_to_voxel(point, self.grid_cfg)
            if endpoint is not None:
                self._update_hit_neighborhood(
                    endpoint, SemanticLabel.BACKGROUND
                )
        np.clip(
            self.occupancy_log_odds,
            self.mapping_cfg.log_odds_min,
            self.mapping_cfg.log_odds_max,
            out=self.occupancy_log_odds,
        )
