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

    static_residual_threshold_mps: float = 0.50
    dynamic_residual_threshold_mps: float = 0.80
    # K-Radar RPC convention in the supplied scene: positive Doppler for a
    # stationary return projected along positive ego translation.
    stationary_velocity_sign: float = 1.0

    def __post_init__(self) -> None:
        if self.static_residual_threshold_mps < 0.0:
            raise ValueError("Static residual threshold must be non-negative.")
        if (
            self.dynamic_residual_threshold_mps
            < self.static_residual_threshold_mps
        ):
            raise ValueError(
                "Dynamic residual threshold must be greater than or equal "
                "to the static residual threshold."
            )
        if self.stationary_velocity_sign not in (-1.0, 1.0):
            raise ValueError("Stationary velocity sign must be -1 or +1.")


@dataclass(frozen=True)
class PoseConfig:
    """LiDAR-pose sampling and rigid radar extrinsics."""

    frame_dt_s: float = 0.10
    radar_to_lidar_rotation: Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
    ] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    def __post_init__(self) -> None:
        if self.frame_dt_s <= 0.0:
            raise ValueError("Pose frame interval must be positive.")


@dataclass(frozen=True)
class ReliabilityConfig:
    """Local polar-neighbourhood RPC sidelobe/reliability filter."""

    range_radius_bins: int = 1
    azimuth_radius_bins: int = 2
    elevation_radius_bins: int = 1
    min_local_power_ratio: float = 0.25
    min_local_neighbors: int = 1

    def __post_init__(self) -> None:
        if min(
            self.range_radius_bins,
            self.azimuth_radius_bins,
            self.elevation_radius_bins,
        ) < 0:
            raise ValueError("Reliability neighbourhood radii must be non-negative.")
        if not 0.0 <= self.min_local_power_ratio <= 1.0:
            raise ValueError("Local power ratio must lie in [0, 1].")
        if self.min_local_neighbors < 0:
            raise ValueError("Minimum local neighbour count must be non-negative.")


@dataclass(frozen=True)
class TemporalConfig:
    """Causal temporal consistency settings for pose-aligned RPC frames."""

    window_size: int = 3
    min_static_support: int = 2
    min_dynamic_support: int = 2
    static_match_radius_m: float = 0.60
    dynamic_match_radius_m: float = 2.00

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("Temporal window must contain at least one frame.")
        if not 1 <= self.min_static_support <= self.window_size:
            raise ValueError("Invalid static temporal support.")
        if not 1 <= self.min_dynamic_support <= self.window_size:
            raise ValueError("Invalid dynamic temporal support.")
        if self.static_match_radius_m <= 0.0:
            raise ValueError("Static match radius must be positive.")
        if self.dynamic_match_radius_m < self.static_match_radius_m:
            raise ValueError(
                "Dynamic match radius must not be smaller than static radius."
            )


@dataclass(frozen=True)
class EgoSpeedConfig:
    """Robust per-frame ego-speed search in aliased Doppler space."""

    min_speed_mps: float = 0.0
    max_speed_mps: float = 40.0
    coarse_step_mps: float = 0.10
    fine_step_mps: float = 0.01
    min_abs_projection: float = 0.25
    min_detections: int = 16
    robust_quantile: float = 0.50

    def __post_init__(self) -> None:
        if self.min_speed_mps < 0.0 or self.max_speed_mps <= self.min_speed_mps:
            raise ValueError("Invalid ego-speed search interval.")
        if self.coarse_step_mps <= 0.0 or self.fine_step_mps <= 0.0:
            raise ValueError("Ego-speed search steps must be positive.")
        if not 0.0 < self.min_abs_projection <= 1.0:
            raise ValueError("min_abs_projection must be in (0, 1].")
        if self.min_detections < 1:
            raise ValueError("min_detections must be positive.")
        if not 0.0 < self.robust_quantile <= 1.0:
            raise ValueError("robust_quantile must be in (0, 1].")


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
