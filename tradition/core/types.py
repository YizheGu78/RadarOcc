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


@dataclass
class FramePrediction:
    """One-frame result in RadarOcc dense grid convention [X,Y,Z]."""

    dense_labels_xyz: np.ndarray
    detections: list[RadarDetection]
    motion_labels: list[MotionLabel]
    metadata: dict[str, Any]
