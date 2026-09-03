from __future__ import annotations

import math

import numpy as np

from tradition.core.config import KRadarConfig, PoseConfig
from tradition.core.types import EgoMotion


def _yaw(rotation: np.ndarray) -> float:
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def _wrap_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


class PoseEgoMotionEstimator:
    """Derive the current LiDAR/radar twist from LiDAR-to-world poses."""

    def __init__(
        self,
        radar_cfg: KRadarConfig | None = None,
        pose_cfg: PoseConfig | None = None,
    ) -> None:
        self.radar_cfg = radar_cfg or KRadarConfig()
        self.pose_cfg = pose_cfg or PoseConfig()

    def estimate(
        self,
        current_pose: np.ndarray,
        previous_pose: np.ndarray | None = None,
        next_pose: np.ndarray | None = None,
        previous_dt_s: float | None = None,
        next_dt_s: float | None = None,
    ) -> EgoMotion:
        """Use a causal backward difference, with forward fallback at frame 0."""
        current = np.asarray(current_pose, dtype=np.float64)
        if current.shape != (4, 4):
            raise ValueError("Current pose must be 4x4.")

        if previous_pose is not None:
            reference = np.asarray(previous_pose, dtype=np.float64)
            dt_s = previous_dt_s or self.pose_cfg.frame_dt_s
            world_delta = current[:3, 3] - reference[:3, 3]
            yaw_delta = _wrap_angle(
                _yaw(current[:3, :3]) - _yaw(reference[:3, :3])
            )
            source = "pose_backward"
        elif next_pose is not None:
            reference = np.asarray(next_pose, dtype=np.float64)
            dt_s = next_dt_s or self.pose_cfg.frame_dt_s
            world_delta = reference[:3, 3] - current[:3, 3]
            yaw_delta = _wrap_angle(
                _yaw(reference[:3, :3]) - _yaw(current[:3, :3])
            )
            source = "pose_forward"
        else:
            raise ValueError(
                "At least one adjacent pose is required to estimate ego motion."
            )
        if dt_s <= 0.0:
            raise ValueError("Pose time interval must be positive.")

        # Express the world translation derivative in the current LiDAR frame.
        velocity_lidar = current[:3, :3].T @ (world_delta / dt_s)
        yaw_rate = yaw_delta / dt_s

        # The radar origin has an additional lever-arm velocity during yaw.
        omega_lidar = np.array([0.0, 0.0, yaw_rate], dtype=np.float64)
        translation_lidar_radar = np.asarray(
            self.radar_cfg.radar_to_lidar_translation_xyz_m,
            dtype=np.float64,
        )
        radar_origin_velocity_lidar = velocity_lidar + np.cross(
            omega_lidar, translation_lidar_radar
        )
        rotation_lidar_radar = np.asarray(
            self.pose_cfg.radar_to_lidar_rotation,
            dtype=np.float64,
        )
        velocity_radar = rotation_lidar_radar.T @ radar_origin_velocity_lidar

        return EgoMotion(
            pose_lidar_to_world=current.copy(),
            linear_velocity_lidar_mps=velocity_lidar,
            linear_velocity_radar_mps=velocity_radar,
            yaw_rate_rps=float(yaw_rate),
            source=source,
        )
