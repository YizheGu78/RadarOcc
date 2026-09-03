from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from tradition.core.config import KRadarConfig, MotionConfig
from tradition.core.geometry import wrapped_velocity_residual
from tradition.core.interfaces import PoseAwareDopplerClassifier
from tradition.core.types import DopplerEvidence, MotionLabel, RadarDetection


class EgoCompensatedDopplerClassifier(PoseAwareDopplerClassifier):
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

    def residuals_with_velocity(
        self,
        detections: Sequence[RadarDetection],
        ego_velocity_radar_mps: np.ndarray,
    ) -> np.ndarray:
        """Return wrapped residuals using the full radar velocity vector."""
        velocity = np.asarray(ego_velocity_radar_mps, dtype=np.float64)
        if velocity.shape != (3,):
            raise ValueError(
                f"Radar ego velocity must have shape (3,), got {velocity.shape}."
            )
        residuals = np.empty(len(detections), dtype=np.float64)
        for index, det in enumerate(detections):
            point = np.asarray(det.xyz_radar_m, dtype=np.float64)
            norm = float(np.linalg.norm(point))
            if norm <= 1e-9:
                residuals[index] = np.nan
                continue
            line_of_sight = point / norm
            expected_static = (
                self.motion_cfg.stationary_velocity_sign
                * float(line_of_sight @ velocity)
            )
            residuals[index] = wrapped_velocity_residual(
                det.radial_velocity_mps - expected_static,
                self.radar_cfg.doppler_period_mps,
            )
        return residuals

    def evidence_with_velocity(
        self,
        detections: Sequence[RadarDetection],
        ego_velocity_radar_mps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate static/uncertain/dynamic evidence with a dead band."""
        residuals = self.residuals_with_velocity(
            detections, ego_velocity_radar_mps
        )
        absolute = np.abs(residuals)
        evidence = np.full(
            len(detections), int(DopplerEvidence.UNCERTAIN), dtype=np.int8
        )
        evidence[absolute <= self.motion_cfg.static_residual_threshold_mps] = int(
            DopplerEvidence.STATIC
        )
        evidence[absolute >= self.motion_cfg.dynamic_residual_threshold_mps] = int(
            DopplerEvidence.DYNAMIC
        )
        evidence[~np.isfinite(absolute)] = int(DopplerEvidence.UNCERTAIN)
        return residuals, evidence
