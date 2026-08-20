from __future__ import annotations

import math
from typing import Sequence

from tradition.core.config import KRadarConfig, MotionConfig
from tradition.core.geometry import wrapped_velocity_residual
from tradition.core.interfaces import MotionClassifier
from tradition.core.types import MotionLabel, RadarDetection


class EgoCompensatedDopplerClassifier(MotionClassifier):
    """Classify detections using ego-motion-compensated radial Doppler."""

    def __init__(
        self,
        radar_cfg: KRadarConfig | None = None,
        motion_cfg: MotionConfig | None = None,
    ) -> None:
        self.radar_cfg = radar_cfg or KRadarConfig()
        self.motion_cfg = motion_cfg or MotionConfig()

    def classify(
        self,
        detections: Sequence[RadarDetection],
        ego_speed_mps: float,
    ) -> list[MotionLabel]:
        labels: list[MotionLabel] = []
        for det in detections:
            expected_static = (
                self.motion_cfg.stationary_velocity_sign
                * ego_speed_mps
                * math.cos(det.elevation_rad)
                * math.cos(det.azimuth_rad)
            )
            residual = wrapped_velocity_residual(
                det.radial_velocity_mps - expected_static,
                self.radar_cfg.doppler_period_mps,
            )
            labels.append(
                MotionLabel.STATIC
                if abs(residual) <= self.motion_cfg.static_residual_threshold_mps
                else MotionLabel.DYNAMIC
            )
        return labels
