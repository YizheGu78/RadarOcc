"""Calibrated P1 geometry and LiDAR-pose alignment (no feature fusion)."""
from pathlib import Path

import numpy as np


def validate_transform(matrix, name="transform"):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain finite 4x4 matrices")
    if not np.allclose(matrix[..., 3, :], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name}: invalid homogeneous last row")
    rotation = matrix[..., :3, :3]
    if not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"{name}: rotation is not orthonormal")
    if not np.allclose(np.linalg.det(rotation), 1, atol=2e-3):
        raise ValueError(f"{name}: rotation determinant must be +1")
    return matrix


def radar_to_lidar_default():
    # Same convention as tradition/core/config.py and RadarOcc's inverse
    # Cartesian-to-radar reference transform. Override for other calibration.
    matrix = np.eye(4)
    matrix[:3, 3] = [2.54, -0.30, -0.70]
    return matrix


class RadarBinGeometry:
    """Actual bin lookup; angles in radians, XYZ x-forward/y-left/z-up.

    NPZ keys: range_m, azimuth_rad, elevation_rad.
    K-Radar MAT keys: arrRange, arrAzimuth, arrElevation (angles in radians).
    Signs are explicit overrides for source angle-axis conventions.
    """
    def __init__(self, range_m, azimuth_rad, elevation_rad, azimuth_sign=1, elevation_sign=1):
        if azimuth_sign not in (-1, 1) or elevation_sign not in (-1, 1):
            raise ValueError("Angle signs must be -1 or +1")
        self.bins = [np.asarray(x, dtype=np.float64).reshape(-1) for x in
                     (range_m, azimuth_rad, elevation_rad)]
        if any(x.size == 0 or not np.isfinite(x).all() for x in self.bins):
            raise ValueError("Bin tables must be nonempty and finite")
        if (self.bins[0] < 0).any():
            raise ValueError("Negative range bins")
        if any(np.abs(x).max() > np.pi for x in self.bins[1:]):
            raise ValueError("Angle tables must be radians, not degrees")
        self.azimuth_sign, self.elevation_sign = azimuth_sign, elevation_sign

    @classmethod
    def from_file(cls, path, **kwargs):
        path = Path(path)
        if path.suffix.lower() == ".mat":
            from scipy.io import loadmat
            data = loadmat(path)
            keys = ("arrRange", "arrAzimuth", "arrElevation")
            return cls(*(data[k] for k in keys), **kwargs)
        with np.load(path, allow_pickle=False) as data:
            return cls(*(data[k] for k in ("range_m", "azimuth_rad", "elevation_rad")), **kwargs)

    def to_xyz(self, coords):
        coords = np.asarray(coords)
        if coords.shape[-1] != 3 or not np.isfinite(coords).all():
            raise ValueError("P1 coords must be finite [...,3] R/A/E indices")
        indices = coords.astype(np.int64)
        if not np.array_equal(indices, coords):
            raise ValueError("P1 indices must be integers")
        values = []
        for axis, bins in enumerate(self.bins):
            ix = indices[..., axis]
            if (ix < 0).any() or (ix >= len(bins)).any():
                raise ValueError(f"P1 axis {axis} exceeds actual bin table")
            values.append(bins[ix])
        r, a, e = values
        a, e = a * self.azimuth_sign, e * self.elevation_sign
        return np.stack((r * np.cos(e) * np.cos(a), r * np.cos(e) * np.sin(a), r * np.sin(e)), axis=-1)


def align_to_current(coords_xyz, poses_world_from_lidar, radar_to_lidar=None, valid_mask=None):
    """[B,T,N,3] -> current radar frame, returning XYZ and [B,T,4,4].

    T_currentRadar_from_historyRadar = inverse(P_current @ E) @ (P_history @ E).
    Invalid history is retained only as a placeholder; consumers must mask it.
    """
    xyz = np.asarray(coords_xyz, dtype=np.float64)
    if xyz.ndim != 4 or xyz.shape[-1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("coords_xyz must be finite [B,T,N,3]")
    poses = validate_transform(poses_world_from_lidar, "ego poses")
    if poses.shape != xyz.shape[:2] + (4, 4):
        raise ValueError("Pose B/T dimensions disagree with coordinates")
    mask = np.ones(xyz.shape[:2], dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if mask.shape != xyz.shape[:2] or not mask[:, -1].all():
        raise ValueError("valid_mask must be [B,T] with valid current slots")
    extrinsic = validate_transform(radar_to_lidar_default() if radar_to_lidar is None else radar_to_lidar, "radar_to_lidar")
    if extrinsic.shape != (4, 4):
        raise ValueError("radar_to_lidar must be one 4x4 matrix")
    world_from_radar = poses @ extrinsic
    transforms = np.linalg.inv(world_from_radar[:, -1])[:, None] @ world_from_radar
    transforms[:, -1] = np.eye(4)
    transforms[~mask] = np.eye(4)
    aligned = np.einsum("btij,btnj->btni", transforms[..., :3, :3], xyz) + transforms[..., None, :3, 3]
    aligned[:, -1] = xyz[:, -1]
    return aligned, transforms
