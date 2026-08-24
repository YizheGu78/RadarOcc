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
    """Tunable classical CFAR/peak-extraction settings for raw 4DRT."""

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
class SemanticConfig:
    """Geometry rules for the classical background/foreground classifier.

    Doppler identifies moving foreground directly.  These settings let compact,
    object-like clusters promote stationary returns (for example parked cars) to
    foreground without using annotation boxes at inference time.
    """

    neighbor_radius_xy_m: float = 1.2
    neighbor_radius_z_m: float = 0.8
    min_cluster_points: int = 3
    max_object_length_m: float = 12.0
    max_object_width_m: float = 4.5
    max_object_height_m: float = 4.5
    min_object_height_m: float = 0.20
    min_object_top_z_m: float = -0.40
    dynamic_cluster_min_fraction: float = 0.15

    def __post_init__(self) -> None:
        if self.neighbor_radius_xy_m <= 0.0 or self.neighbor_radius_z_m <= 0.0:
            raise ValueError("Semantic neighbor radii must be positive.")
        if self.min_cluster_points < 2:
            raise ValueError("min_cluster_points must be at least 2.")
        if not 0.0 <= self.dynamic_cluster_min_fraction <= 1.0:
            raise ValueError("dynamic_cluster_min_fraction must be in [0, 1].")


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
    foreground_override_ratio: float = 0.80
