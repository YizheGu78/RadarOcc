from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class MotionLabel(IntEnum):
    STATIC = 1
    DYNAMIC = 2


class DopplerEvidence(IntEnum):
    """Internal evidence; UNCERTAIN is never emitted as a RadarOcc class."""

    DYNAMIC = -1
    UNCERTAIN = 0
    STATIC = 1


class SemanticLabel(IntEnum):
    BACKGROUND = 1
    FOREGROUND = 2


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


@dataclass(frozen=True)
class EgoMotion:
    """Per-frame sensor motion derived from synchronized LiDAR poses."""

    pose_lidar_to_world: np.ndarray
    linear_velocity_lidar_mps: np.ndarray
    linear_velocity_radar_mps: np.ndarray
    yaw_rate_rps: float
    source: str = "pose"


@dataclass(frozen=True)
class TemporalDetectionFrame:
    """Reliable detections and motion evidence for one buffered frame."""

    token: str
    pose_lidar_to_world: np.ndarray
    detections: list[RadarDetection]
    doppler_residuals_mps: np.ndarray
    doppler_evidence: np.ndarray


@dataclass(frozen=True)
class TemporalClassification:
    """Accepted current detections plus pose-aligned historic background."""

    current_indices: np.ndarray
    current_motion_labels: list[MotionLabel]
    historic_background_lidar_m: np.ndarray
    static_support: np.ndarray
    dynamic_support: np.ndarray


@dataclass
class FramePrediction:
    """One-frame result in RadarOcc dense grid convention [X,Y,Z]."""

    dense_labels_xyz: np.ndarray
    detections: list[RadarDetection]
    motion_labels: list[MotionLabel]
    semantic_labels: list[SemanticLabel]
    metadata: dict[str, Any]
