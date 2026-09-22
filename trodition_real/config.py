"""Project parameters; none of these are claimed as GM2019 published values."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class Config:
    min_xyz: tuple = (0., -25.6, -2.6)
    shape_xyz: tuple = (128, 128, 14)
    resolution: float = .4
    temporal_window: int = 5
    p_hit: float = .7
    p_free: float = .35
    prior: float = .5
    occupied_threshold: float = .55
    free_threshold: float = .45
    fusion: str = "gm2019"
    dbscan_eps_xy: float = 1.2
    dbscan_eps_z: float = .8
    dbscan_min_samples: int = 2
    foreground_threshold: float = .5
    unknown_export: str = "free"

    def __post_init__(self):
        if len(self.min_xyz) != 3 or len(self.shape_xyz) != 3:
            raise ValueError("Expected XYZ grid specification")
        if not all(math.isfinite(x) for x in self.min_xyz):
            raise ValueError("Grid bounds must be finite")
        if any(int(x) != x or x < 1 for x in self.shape_xyz):
            raise ValueError("Grid shape must contain positive integers")
        if not all(math.isfinite(x) and x > 0 for x in
                   [self.resolution, self.dbscan_eps_xy, self.dbscan_eps_z]):
            raise ValueError("Resolution and clustering radii must be positive")
        if self.temporal_window < 1 or self.dbscan_min_samples < 1:
            raise ValueError("Window and minimum sample count must be positive")
        if not 0 < self.p_free < self.prior < self.p_hit < 1:
            raise ValueError("Expected 0 < p_free < prior < p_hit < 1")
        if not 0 < self.free_threshold < self.prior < self.occupied_threshold < 1:
            raise ValueError("Thresholds must enclose the prior")
        if not 0 <= self.foreground_threshold <= 1:
            raise ValueError("Invalid foreground threshold")
        if self.fusion not in ("gm2019", "autoware_bbf"):
            raise ValueError("Unknown fusion backend")
        if self.unknown_export not in ("free", "background"):
            raise ValueError("Unknown export policy must be free or background")

    def signature(self):
        # Canonical JSON-compatible representation persisted with RF model.
        result = asdict(self)
        result['min_xyz'] = list(self.min_xyz)
        result['shape_xyz'] = list(self.shape_xyz)
        return result
