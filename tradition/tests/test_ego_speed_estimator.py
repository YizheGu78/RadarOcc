import math

import numpy as np

from tradition.core.types import RadarDetection
from tradition.motion.ego_speed_estimator import RobustDopplerEgoSpeedEstimator


def _wrap(value: float, period: float = 3.84) -> float:
    return (value + 0.5 * period) % period - 0.5 * period


def _detection(azimuth: float, radial_velocity: float) -> RadarDetection:
    xyz = np.array([10.0 * math.cos(azimuth), 10.0 * math.sin(azimuth), 0.0])
    return RadarDetection(
        range_index=25,
        doppler_index=-1,
        elevation_index=18,
        azimuth_index=53,
        power=1.0,
        range_m=10.0,
        radial_velocity_mps=radial_velocity,
        azimuth_rad=azimuth,
        elevation_rad=0.0,
        xyz_radar_m=xyz,
        xyz_lidar_m=xyz,
    )


def test_estimates_highway_speed_from_aliased_static_returns():
    true_speed = 12.0
    detections = [
        _detection(azimuth, _wrap(-true_speed * math.cos(azimuth)))
        for azimuth in np.linspace(-0.85, 0.85, 41)
    ]
    detections.extend(
        _detection(azimuth, velocity)
        for azimuth, velocity in [(-0.5, 1.2), (0.0, -0.4), (0.6, 0.9)]
    )

    estimated = RobustDopplerEgoSpeedEstimator().estimate(detections)

    assert abs(estimated - true_speed) <= 0.02


def test_returns_zero_when_target_list_is_too_small():
    detections = [_detection(0.0, 0.0)]
    assert RobustDopplerEgoSpeedEstimator().estimate(detections) == 0.0
