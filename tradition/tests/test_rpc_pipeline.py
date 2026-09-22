import math

import numpy as np

from tradition.core.config import ReliabilityConfig, TemporalConfig
from tradition.core.types import EgoMotion, MotionLabel, SemanticLabel
from tradition.detection.rpc_target_detector import RPCPointTargetDetector
from tradition.experiment.dataset_runner import _resolve_rpc_radar
from tradition.io.rpc_radar_reader import KRadarRPCReader
from tradition.pipeline.traditional_radar_pipeline import build_rpc_pipeline


def _point(x: float, y: float, doppler: float) -> list[float]:
    z = 0.0
    range_m = math.sqrt(x * x + y * y)
    azimuth = math.atan2(y, x)
    return [
        x,
        y,
        z,
        1.0,
        doppler,
        range_m,
        azimuth,
        0.0,
        round(range_m / 0.4),
        round(math.degrees(azimuth) + 53.0),
        18,
    ]


def test_rpc_reader_detector_and_three_class_ogm(tmp_path):
    path = tmp_path / "rpc_00042.npy"
    points = np.asarray(
        [
            [0.0] * 11,
            _point(10.0, -4.0, 0.0),
            _point(10.0, 4.0, 1.0),
        ],
        dtype=np.float64,
    )
    np.save(path, points, allow_pickle=False)

    frame = KRadarRPCReader().read(path)
    detections = RPCPointTargetDetector().detect(frame)
    assert frame.size == 3
    assert len(detections) == 2
    assert detections[0].doppler_index == -1
    assert detections[1].radial_velocity_mps == 1.0
    np.testing.assert_allclose(
        detections[0].xyz_lidar_m,
        points[1, :3] + np.array([2.54, -0.30, -0.70]),
    )

    prediction = build_rpc_pipeline().predict_file(path, ego_speed_mps=0.0)
    assert prediction.dense_labels_xyz.shape == (128, 128, 14)
    assert set(np.unique(prediction.dense_labels_xyz)).issubset({0, 1, 2})
    assert MotionLabel.STATIC in prediction.motion_labels
    assert MotionLabel.DYNAMIC in prediction.motion_labels
    assert SemanticLabel.BACKGROUND in prediction.semantic_labels
    assert SemanticLabel.FOREGROUND in prediction.semantic_labels
    assert np.any(prediction.dense_labels_xyz == int(SemanticLabel.BACKGROUND))
    assert np.any(prediction.dense_labels_xyz == int(SemanticLabel.FOREGROUND))


def test_rpc_endpoint_outside_grid_still_carves_free_ray(tmp_path):
    path = tmp_path / "rpc_00043.npy"
    np.save(path, np.asarray([_point(60.0, 0.0, 0.0)]), allow_pickle=False)

    pipeline = build_rpc_pipeline()
    prediction = pipeline.predict_file(path, ego_speed_mps=0.0)

    assert np.all(prediction.dense_labels_xyz == 0)
    assert pipeline.mapper.occupancy_log_odds[127, 63, 4] < 0.0
    assert not np.any(pipeline.mapper.occupancy_log_odds > 0.0)


def test_rpc_resolver_pairs_frames_by_scene_order_not_sensor_ids(tmp_path):
    scene_dir = tmp_path / "3"
    scene_dir.mkdir(parents=True)
    first = scene_dir / "rpc_00031.npy"
    second = scene_dir / "rpc_00077.npy"
    np.save(first, np.zeros((1, 11), dtype=np.float32))
    np.save(second, np.zeros((1, 11), dtype=np.float32))
    info = {
        "scene_token": "3",
        "lidar_token": "3_00999",
        "radar_frame_idx": 42,
        "radar_path": "/missing/tesseract_00099.mat",
        "_rpc_sequence_ordinal": 1,
        "_rpc_sequence_count": 2,
    }

    resolved = _resolve_rpc_radar(info, tmp_path, tmp_path)
    assert resolved == second.resolve()


def test_temporal_pipeline_exposes_true_static_and_dynamic_branches(tmp_path):
    path = tmp_path / "rpc_00001.npy"
    np.save(
        path,
        np.asarray(
            [
                _point(10.0, -4.0, 0.1),
                _point(10.0, 4.0, 1.0),
            ],
            dtype=np.float64,
        ),
        allow_pickle=False,
    )
    pipeline = build_rpc_pipeline(
        reliability_cfg=ReliabilityConfig(min_local_neighbors=0),
        temporal_cfg=TemporalConfig(
            window_size=1,
            min_static_support=1,
            min_dynamic_support=1,
        ),
    )
    prediction = pipeline.predict_temporal_file(
        path,
        token="3_00000",
        ego_motion=EgoMotion(
            pose_lidar_to_world=np.eye(4),
            linear_velocity_lidar_mps=np.zeros(3),
            linear_velocity_radar_mps=np.zeros(3),
            yaw_rate_rps=0.0,
        ),
    )

    assert prediction.metadata["doppler_velocity_source"] == "rpc_raw_bin_velocity_mps"
    assert prediction.metadata["doppler_unwrapping"] == "none"
    assert prediction.branch_points_lidar_m is not None
    assert prediction.branch_points_lidar_m["static"].shape == (1, 3)
    assert prediction.branch_points_lidar_m["dynamic"].shape == (1, 3)


def test_rpc_reader_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((8, 5), dtype=np.float32))
    try:
        KRadarRPCReader().read(path)
    except ValueError as error:
        assert "[N,11]" in str(error)
    else:
        raise AssertionError("Expected an invalid RPC schema to be rejected")
