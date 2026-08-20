from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class GridConfig:
    """RadarOcc evaluation grid in the LiDAR/vehicle Cartesian frame."""

    min_xyz: Tuple[float, float, float] = (0.0, -25.6, -2.6)
    max_xyz: Tuple[float, float, float] = (51.2, 25.6, 3.0)
    voxel_size_m: float = 0.4
    shape_xyz: Tuple[int, int, int] = (128, 128, 14)

    def __post_init__(self) -> None:
        derived = tuple(
            int(round((hi - lo) / self.voxel_size_m))
            for lo, hi in zip(self.min_xyz, self.max_xyz)
        )
        if derived != self.shape_xyz:
            raise ValueError(
                f"Grid bounds/voxel size imply {derived}, not {self.shape_xyz}."
            )


@dataclass(frozen=True)
class KRadarConfig:
    """K-Radar tensor geometry used by RadarOcc.

    Raw arrDREA layout is [Doppler, Range, Elevation, Azimuth].
    Angle defaults follow the public K-Radar/RadarOcc tensor dimensions.
    Change them if a different tensor calibration is used.
    """

    range_resolution_m: float = 0.4
    azimuth_min_deg: float = -53.0
    azimuth_step_deg: float = 1.0
    elevation_min_deg: float = -18.0
    elevation_step_deg: float = 1.0
    doppler_min_mps: float = -1.92
    doppler_resolution_mps: float = 0.06
    doppler_period_mps: float = 3.84
    radar_to_lidar_translation_xyz_m: Tuple[float, float, float] = (
        2.54,
        -0.30,
        -0.70,
    )
    expected_doppler_bins: int = 64
    expected_azimuth_bins: int = 107
    expected_elevation_bins: int = 37


@dataclass(frozen=True)
class CFARConfig:
    """Tunable classical CFAR/peak-extraction settings."""

    guard_range: int = 2
    noise_range: int = 8
    guard_doppler: int = 1
    noise_doppler: int = 4
    threshold_offset_db: float = 6.0
    min_power_db: float | None = None
    local_max_radius: int = 1
    max_detections: int = 4096


@dataclass(frozen=True)
class MotionConfig:
    """Doppler-based static/dynamic split."""

    static_residual_threshold_mps: float = 0.30
    stationary_velocity_sign: float = 1.0


@dataclass(frozen=True)
class MappingConfig:
    """Inverse sensor model for one-frame 3D occupancy."""

    p_hit: float = 0.70
    p_free: float = 0.35
    log_odds_min: float = -4.0
    log_odds_max: float = 4.0
    occupancy_probability_threshold: float = 0.55
    hit_radius_xy_voxels: int = 1
    hit_radius_z_voxels: int = 1
    dynamic_override_ratio: float = 0.80
