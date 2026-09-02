import numpy as np

from tradition.core.config import MotionConfig
from tradition.core.types import MotionLabel, RadarDetection
from tradition.motion.doppler_classifier import EgoCompensatedDopplerClassifier


def _detection(radial_velocity_mps: float) -> RadarDetection:
    xyz = np.array([10.0, 0.0, 0.0], dtype=np.float64)
    return RadarDetection(
        range_index=0,
        doppler_index=-1,
        elevation_index=0,
        azimuth_index=0,
        power=1.0,
        range_m=10.0,
        radial_velocity_mps=radial_velocity_mps,
        azimuth_rad=0.0,
        elevation_rad=0.0,
        xyz_radar_m=xyz,
        xyz_lidar_m=xyz,
    )


def test_static_residual_threshold_is_configurable():
    detection = _detection(0.40)
    strict = EgoCompensatedDopplerClassifier(
        motion_cfg=MotionConfig(static_residual_threshold_mps=0.30)
    )
    relaxed = EgoCompensatedDopplerClassifier(
        motion_cfg=MotionConfig(static_residual_threshold_mps=0.50)
    )

    assert strict.classify([detection], ego_speed_mps=0.0) == [
        MotionLabel.DYNAMIC
    ]
    assert relaxed.classify([detection], ego_speed_mps=0.0) == [
        MotionLabel.STATIC
    ]
