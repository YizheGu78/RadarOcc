from __future__ import annotations

from typing import Sequence

import numpy as np

from tradition.core.config import EgoSpeedConfig, KRadarConfig, MotionConfig
from tradition.core.interfaces import EgoSpeedEstimator
from tradition.core.types import RadarDetection


class RobustDopplerEgoSpeedEstimator(EgoSpeedEstimator):
    """Estimate ego speed from the dominant stationary Doppler consensus.

    K-Radar radial velocity is periodic.  Searching in wrapped-Doppler space
    avoids treating an aliased highway-speed return as a low-speed target.
    The quantile loss is robust to independently moving road users.
    """

    def __init__(
        self,
        radar_cfg: KRadarConfig | None = None,
        motion_cfg: MotionConfig | None = None,
        estimator_cfg: EgoSpeedConfig | None = None,
    ) -> None:
        self.radar_cfg = radar_cfg or KRadarConfig()
        self.motion_cfg = motion_cfg or MotionConfig()
        self.estimator_cfg = estimator_cfg or EgoSpeedConfig()

    def estimate(self, detections: Sequence[RadarDetection]) -> float:
        cfg = self.estimator_cfg
        if len(detections) < cfg.min_detections:
            return 0.0

        radial_velocity = np.asarray(
            [det.radial_velocity_mps for det in detections], dtype=np.float64
        )
        projection = self.motion_cfg.stationary_velocity_sign * np.asarray(
            [
                np.cos(det.elevation_rad) * np.cos(det.azimuth_rad)
                for det in detections
            ],
            dtype=np.float64,
        )
        valid = (
            np.isfinite(radial_velocity)
            & np.isfinite(projection)
            & (np.abs(projection) >= cfg.min_abs_projection)
        )
        radial_velocity = radial_velocity[valid]
        projection = projection[valid]
        if radial_velocity.size < cfg.min_detections:
            return 0.0

        coarse = np.arange(
            cfg.min_speed_mps,
            cfg.max_speed_mps + 0.5 * cfg.coarse_step_mps,
            cfg.coarse_step_mps,
            dtype=np.float64,
        )
        best = self._best_speed(radial_velocity, projection, coarse)

        fine_start = max(cfg.min_speed_mps, best - cfg.coarse_step_mps)
        fine_stop = min(cfg.max_speed_mps, best + cfg.coarse_step_mps)
        fine = np.arange(
            fine_start,
            fine_stop + 0.5 * cfg.fine_step_mps,
            cfg.fine_step_mps,
            dtype=np.float64,
        )
        return self._best_speed(radial_velocity, projection, fine)

    def _best_speed(
        self,
        radial_velocity: np.ndarray,
        projection: np.ndarray,
        candidates: np.ndarray,
    ) -> float:
        residual = radial_velocity[:, None] - projection[:, None] * candidates
        period = self.radar_cfg.doppler_period_mps
        wrapped = (residual + 0.5 * period) % period - 0.5 * period
        score = np.quantile(
            np.abs(wrapped),
            self.estimator_cfg.robust_quantile,
            axis=0,
        )
        return float(candidates[int(np.argmin(score))])
