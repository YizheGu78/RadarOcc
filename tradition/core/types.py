from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class MotionLabel(IntEnum):
    STATIC = 1
    DYNAMIC = 2


@dataclass(frozen=True)
class RadarDetection:
    range_index: int
    doppler_index: int
    elevation_index: int
    azimuth_index: int
    power: float
    range_m: float
    radial_velocity_mps: float
    azimuth_rad: float
    elevation_rad: float
    xyz_radar_m: np.ndarray
    xyz_lidar_m: np.ndarray


@dataclass(frozen=True)
class RPCPointCloudFrame:
    """Enhanced K-Radar density-reduced Cartesian point cloud.

    Source columns are [x, y, z, power, doppler, range, azimuth, elevation,
    range_index, azimuth_index, elevation_index]. Angles are radians and
    Doppler is already physical radial velocity in m/s.
    """

    xyz_radar_m: np.ndarray
    power: np.ndarray
    radial_velocity_mps: np.ndarray
    range_m: np.ndarray
    azimuth_rad: np.ndarray
    elevation_rad: np.ndarray
    range_ind: np.ndarray
    azimuth_ind: np.ndarray
    elevation_ind: np.ndarray

    @property
    def size(self) -> int:
        return int(self.power.size)


@dataclass
class FramePrediction:
    """One-frame result in RadarOcc dense grid convention [X,Y,Z]."""

    dense_labels_xyz: np.ndarray
    detections: list[RadarDetection]
    motion_labels: list[MotionLabel]
    metadata: dict[str, Any]
