"""OctoMap defaults plus explicit RadarOcc window, grid and RF adapters."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class Config:
    min_xyz: tuple = (0., -25.6, -2.6)
    shape_xyz: tuple = (128, 128, 14)
    resolution: float = .4
    temporal_window: int = 5
    fusion: str = "octomap"
    p_hit: float = .7
    p_miss: float = .4
    occupancy_threshold: float = .5
    clamping_min: float = .1192
    clamping_max: float = .971
    max_range: float = -1.0
    lazy_eval: bool = False
    discretize: bool = False
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
        if not 0 < self.p_miss <= .5 <= self.p_hit < 1:
            raise ValueError("Expected 0 < p_miss <= .5 <= p_hit < 1")
        if not 0 < self.clamping_min < self.occupancy_threshold < self.clamping_max < 1:
            raise ValueError("Clamping limits must enclose the occupancy threshold")
        if not math.isfinite(self.max_range):
            raise ValueError("max_range must be finite; negative means unlimited")
        if not isinstance(self.lazy_eval, bool) or not isinstance(self.discretize, bool):
            raise ValueError("lazy_eval and discretize must be boolean")
        if not 0 <= self.foreground_threshold <= 1:
            raise ValueError("Invalid foreground threshold")
        if self.fusion != "octomap":
            raise ValueError("This pipeline requires OctoMap; rebuild old Autoware/GM2019 caches")
        if self.unknown_export not in ("free", "background"):
            raise ValueError("Unknown export policy must be free or background")

    def signature(self):
        # Canonical JSON-compatible representation persisted with RF model.
        result = asdict(self)
        result['min_xyz'] = list(self.min_xyz)
        result['shape_xyz'] = list(self.shape_xyz)
        return result
