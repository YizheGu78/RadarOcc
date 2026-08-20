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
class SparseRadarFrame:
    """One RadarOcc EAsparse frame.

    descriptor rows follow the existing generator:
      0:3 top-3 Doppler powers
      3:6 matching Doppler-bin indices
      6   mean Doppler power
      7   Doppler variance
    """

    range_ind: np.ndarray
    elevation_ind: np.ndarray
    azimuth_ind: np.ndarray
    descriptor: np.ndarray

    @property
    def size(self) -> int:
        return int(self.range_ind.size)

    @property
    def top3_power(self) -> np.ndarray:
        return self.descriptor[0:3]

    @property
    def top3_doppler_ind(self) -> np.ndarray:
        return self.descriptor[3:6]

    @property
    def mean_power(self) -> np.ndarray:
        return self.descriptor[6]

    @property
    def variance(self) -> np.ndarray:
        return self.descriptor[7]


@dataclass
class FramePrediction:
    """One-frame result in RadarOcc dense grid convention [X,Y,Z]."""

    dense_labels_xyz: np.ndarray
    detections: list[RadarDetection]
    motion_labels: list[MotionLabel]
    metadata: dict[str, Any]
