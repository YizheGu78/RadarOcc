from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .config import GridConfig, KRadarConfig


def angle_from_index(index: int, minimum_deg: float, step_deg: float) -> float:
    return math.radians(minimum_deg + index * step_deg)


def doppler_from_index(index: int, cfg: KRadarConfig) -> float:
    return cfg.doppler_min_mps + index * cfg.doppler_resolution_mps


def wrapped_velocity_residual(value_mps: float, period_mps: float) -> float:
    """Map a Doppler residual to [-period/2, period/2)."""
    half = period_mps / 2.0
    return (value_mps + half) % period_mps - half


def transform_points(
    xyz_m: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """Apply a 4x4 rigid transform to an [N,3] Cartesian point array."""
    xyz = np.asarray(xyz_m, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N,3], got {xyz.shape}.")
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {matrix.shape}.")
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def relative_lidar_transform(
    source_lidar_to_world: np.ndarray,
    target_lidar_to_world: np.ndarray,
) -> np.ndarray:
    """Transform points from a source LiDAR frame into a target frame."""
    source = np.asarray(source_lidar_to_world, dtype=np.float64)
    target = np.asarray(target_lidar_to_world, dtype=np.float64)
    if source.shape != (4, 4) or target.shape != (4, 4):
        raise ValueError("Pose matrices must both be 4x4.")
    return np.linalg.inv(target) @ source


def spherical_to_cartesian(
    range_m: float,
    azimuth_rad: float,
    elevation_rad: float,
) -> np.ndarray:
    cos_el = math.cos(elevation_rad)
    return np.array(
        [
            range_m * cos_el * math.cos(azimuth_rad),
            range_m * cos_el * math.sin(azimuth_rad),
            range_m * math.sin(elevation_rad),
        ],
        dtype=np.float64,
    )


def radar_to_lidar(xyz_radar_m: np.ndarray, cfg: KRadarConfig) -> np.ndarray:
    return np.asarray(xyz_radar_m, dtype=np.float64) + np.asarray(
        cfg.radar_to_lidar_translation_xyz_m,
        dtype=np.float64,
    )


def xyz_to_voxel(
    xyz_m: Iterable[float],
    grid: GridConfig,
) -> tuple[int, int, int] | None:
    xyz = np.asarray(tuple(xyz_m), dtype=np.float64)
    lo = np.asarray(grid.min_xyz, dtype=np.float64)
    hi = np.asarray(grid.max_xyz, dtype=np.float64)
    if np.any(xyz < lo) or np.any(xyz >= hi):
        return None
    voxel = np.floor((xyz - lo) / grid.voxel_size_m).astype(np.int64)
    if np.any(voxel < 0) or np.any(voxel >= np.asarray(grid.shape_xyz)):
        return None
    return int(voxel[0]), int(voxel[1]), int(voxel[2])


def voxel_center(
    voxel_xyz: tuple[int, int, int],
    grid: GridConfig,
) -> np.ndarray:
    return (
        np.asarray(grid.min_xyz, dtype=np.float64)
        + (np.asarray(voxel_xyz, dtype=np.float64) + 0.5) * grid.voxel_size_m
    )


def clip_segment_to_grid(
    start_xyz_m: np.ndarray,
    end_xyz_m: np.ndarray,
    grid: GridConfig,
) -> np.ndarray | None:
    """Clip a finite segment to the grid AABB and return its last in-grid point.

    The returned maximum-bound coordinates are nudged into the half-open grid so
    they can safely be passed to :func:`xyz_to_voxel`.
    """
    start = np.asarray(start_xyz_m, dtype=np.float64)
    end = np.asarray(end_xyz_m, dtype=np.float64)
    lo = np.asarray(grid.min_xyz, dtype=np.float64)
    hi = np.asarray(grid.max_xyz, dtype=np.float64)
    direction = end - start
    t_enter, t_exit = 0.0, 1.0

    for axis in range(3):
        if abs(float(direction[axis])) <= 1e-12:
            if start[axis] < lo[axis] or start[axis] >= hi[axis]:
                return None
            continue
        t0 = float((lo[axis] - start[axis]) / direction[axis])
        t1 = float((hi[axis] - start[axis]) / direction[axis])
        if t0 > t1:
            t0, t1 = t1, t0
        t_enter = max(t_enter, t0)
        t_exit = min(t_exit, t1)
        if t_enter > t_exit:
            return None

    if t_exit < 0.0 or t_enter > 1.0:
        return None
    clipped = start + min(1.0, t_exit) * direction
    interior_hi = np.nextafter(hi, lo)
    return np.minimum(np.maximum(clipped, lo), interior_hi)


def ray_voxels_dda(
    start_xyz_m: np.ndarray,
    end_xyz_m: np.ndarray,
    grid: GridConfig,
) -> list[tuple[int, int, int]]:
    """Sample a 3D ray densely enough to visit each crossed RadarOcc voxel."""
    start = np.asarray(start_xyz_m, dtype=np.float64)
    end = np.asarray(end_xyz_m, dtype=np.float64)
    distance = float(np.linalg.norm(end - start))
    if distance <= 1e-9:
        voxel = xyz_to_voxel(end, grid)
        return [] if voxel is None else [voxel]

    steps = max(1, int(math.ceil(distance / (grid.voxel_size_m * 0.5))))
    points = np.linspace(start, end, steps + 1)
    result: list[tuple[int, int, int]] = []
    previous = None
    for point in points:
        voxel = xyz_to_voxel(point, grid)
        if voxel is not None and voxel != previous:
            result.append(voxel)
            previous = voxel
    return result
