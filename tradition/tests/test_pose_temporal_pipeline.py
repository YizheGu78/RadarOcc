import math

import numpy as np

from tradition.core.config import MotionConfig, ReliabilityConfig, TemporalConfig
from tradition.core.geometry import wrapped_velocity_residual
from tradition.core.types import (
    DopplerEvidence,
    MotionLabel,
    RadarDetection,
    TemporalDetectionFrame,
)
from tradition.detection.rpc_reliability_filter import LocalPowerRPCFilter
from tradition.io.pose_reader import RadarOccPoseReader
from tradition.motion.doppler_classifier import (
    EgoCompensatedDopplerClassifier,
)
from tradition.motion.pose_ego_motion import PoseEgoMotionEstimator
from tradition.motion.temporal_consistency import (
    PoseAlignedTemporalClassifier,
)


def _pose(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    pose[:3, 3] = [x, y, 0.0]
    return pose


def _detection(xyz: tuple[float, float, float], doppler: float = 0.0):
    point = np.asarray(xyz, dtype=np.float64)
    distance = float(np.linalg.norm(point))
    return RadarDetection(
        range_index=int(round(distance / 0.4)),
        doppler_index=-1,
        elevation_index=18,
        azimuth_index=53,
        power=1.0,
        range_m=distance,
        radial_velocity_mps=doppler,
        azimuth_rad=math.atan2(point[1], point[0]),
        elevation_rad=math.atan2(point[2], np.linalg.norm(point[:2])),
        xyz_radar_m=point.copy(),
        xyz_lidar_m=point.copy(),
    )


def _frame(
    token: str,
    pose: np.ndarray,
    point: tuple[float, float, float],
    residual: float,
    evidence: DopplerEvidence,
) -> TemporalDetectionFrame:
    return TemporalDetectionFrame(
        token=token,
        pose_lidar_to_world=pose,
        detections=[_detection(point)],
        doppler_residuals_mps=np.asarray([residual], dtype=np.float64),
        doppler_evidence=np.asarray([int(evidence)], dtype=np.int8),
    )


def test_pose_ego_motion_uses_pose_translation_and_frame_interval():
    estimator = PoseEgoMotionEstimator()
    motion = estimator.estimate(
        current_pose=_pose(x=2.3609),
        previous_pose=_pose(),
        previous_dt_s=0.10,
    )

    np.testing.assert_allclose(
        motion.linear_velocity_lidar_mps, [23.609, 0.0, 0.0], atol=1e-9
    )
    assert motion.source == "pose_backward"


def test_full_vector_doppler_compensation_wraps_aliases():
    classifier = EgoCompensatedDopplerClassifier()
    ego_velocity = np.asarray([23.609, 0.0, 0.0])
    expected = wrapped_velocity_residual(23.609, 3.84)
    detection = _detection((10.0, 0.0, 0.0), doppler=expected)

    residuals, evidence = classifier.evidence_with_velocity(
        [detection], ego_velocity
    )

    np.testing.assert_allclose(residuals, [0.0], atol=1e-9)
    assert evidence.tolist() == [int(DopplerEvidence.STATIC)]


def test_pose_alignment_confirms_static_background_over_two_frames():
    classifier = PoseAlignedTemporalClassifier(
        temporal_cfg=TemporalConfig(window_size=3, min_static_support=2)
    )
    # The world point is at x=10.  After the ego vehicle moves +1 m, its
    # current-frame coordinate becomes x=9; pose alignment restores the match.
    classifier.update(
        _frame("3_00000", _pose(x=0.0), (10.0, 0.0, 0.0), 0.1,
               DopplerEvidence.STATIC)
    )
    result = classifier.update(
        _frame("3_00001", _pose(x=1.0), (9.0, 0.0, 0.0), 0.1,
               DopplerEvidence.STATIC)
    )

    assert result.current_indices.tolist() == [0]
    assert result.current_motion_labels == [MotionLabel.STATIC]
    assert result.static_support.tolist() == [2]
    np.testing.assert_allclose(
        result.historic_background_lidar_m, [[9.0, 0.0, 0.0]], atol=1e-9
    )


def test_persistent_large_residual_becomes_foreground_without_clustering():
    classifier = PoseAlignedTemporalClassifier(
        temporal_cfg=TemporalConfig(
            window_size=3,
            min_static_support=2,
            min_dynamic_support=2,
            static_match_radius_m=0.6,
            dynamic_match_radius_m=2.0,
        ),
        motion_cfg=MotionConfig(
            static_residual_threshold_mps=0.5,
            dynamic_residual_threshold_mps=0.8,
        ),
    )
    classifier.update(
        _frame("3_00000", _pose(), (10.0, 0.0, 0.0), 1.2,
               DopplerEvidence.DYNAMIC)
    )
    result = classifier.update(
        _frame("3_00001", _pose(), (11.0, 0.0, 0.0), 1.1,
               DopplerEvidence.DYNAMIC)
    )

    assert result.current_indices.tolist() == [0]
    assert result.current_motion_labels == [MotionLabel.DYNAMIC]
    assert result.dynamic_support.tolist() == [2]
    assert result.static_support.tolist() == [0]


def test_uncertain_is_discarded_instead_of_forced_to_foreground():
    classifier = PoseAlignedTemporalClassifier()
    result = classifier.update(
        _frame("3_00000", _pose(), (10.0, 0.0, 0.0), 0.65,
               DopplerEvidence.UNCERTAIN)
    )

    assert result.current_indices.size == 0
    assert result.current_motion_labels == []


def test_local_power_filter_rejects_weak_sidelobe_and_isolated_return():
    strong = _detection((10.0, 0.0, 0.0))
    neighbour = RadarDetection(
        **{
            **strong.__dict__,
            "azimuth_index": strong.azimuth_index + 1,
            "power": 0.5,
        }
    )
    weak = RadarDetection(
        **{
            **strong.__dict__,
            "azimuth_index": strong.azimuth_index + 2,
            "power": 0.1,
        }
    )
    isolated = RadarDetection(
        **{
            **strong.__dict__,
            "range_index": strong.range_index + 20,
            "power": 1.0,
        }
    )
    result = LocalPowerRPCFilter(
        ReliabilityConfig(min_local_power_ratio=0.25, min_local_neighbors=1)
    ).filter([strong, neighbour, weak, isolated])

    assert len(result) == 2
    assert result[0] is strong
    assert result[1] is neighbour


def test_pose_reader_indexes_pose_by_lidar_token_not_rpc_number(tmp_path):
    pose_dir = tmp_path / "train" / "3" / "pose"
    pose_dir.mkdir(parents=True)
    expected = _pose(x=12.0)
    np.save(pose_dir / "lidar_ego_pose0.npy", expected)

    pose, path = RadarOccPoseReader(tmp_path).read("3", "3_00000")

    np.testing.assert_allclose(pose, expected)
    assert path.name == "lidar_ego_pose0.npy"
