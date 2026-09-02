import numpy as np

from tradition.core.types import MotionLabel, RadarDetection, SemanticLabel
from tradition.semantics.classical_classifier import DopplerSemanticClassifier


def _detection(x: float, y: float, z: float) -> RadarDetection:
    xyz = np.array([x, y, z], dtype=np.float64)
    return RadarDetection(
        range_index=0,
        doppler_index=-1,
        elevation_index=0,
        azimuth_index=0,
        power=1.0,
        range_m=float(np.linalg.norm(xyz)),
        radial_velocity_mps=0.0,
        azimuth_rad=0.0,
        elevation_rad=0.0,
        xyz_radar_m=xyz,
        xyz_lidar_m=xyz,
    )


def test_dynamic_detection_is_foreground():
    labels = DopplerSemanticClassifier().classify(
        [_detection(10.0, 0.0, -0.7)],
        [MotionLabel.DYNAMIC],
    )
    assert labels == [SemanticLabel.FOREGROUND]


def test_stationary_cluster_remains_background_without_geometry_promotion():
    detections = [
        _detection(10.0, 0.0, -0.6),
        _detection(10.4, 0.2, -0.2),
        _detection(10.8, 0.0, 0.2),
    ]
    labels = DopplerSemanticClassifier().classify(
        detections,
        [MotionLabel.STATIC] * len(detections),
    )
    assert labels == [SemanticLabel.BACKGROUND] * len(detections)


def test_isolated_stationary_return_defaults_to_background():
    labels = DopplerSemanticClassifier().classify(
        [_detection(10.0, 0.0, -1.2)],
        [MotionLabel.STATIC],
    )
    assert labels == [SemanticLabel.BACKGROUND]
