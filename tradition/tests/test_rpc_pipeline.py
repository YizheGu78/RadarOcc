import math

import numpy as np

from tradition.core.types import MotionLabel, SemanticLabel
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


def test_rpc_resolver_prefers_aligned_radar_frame_index(tmp_path):
    rpc_path = tmp_path / "3" / "rpc_00042.npy"
    rpc_path.parent.mkdir(parents=True)
    np.save(rpc_path, np.zeros((1, 11), dtype=np.float32))
    info = {
        "scene_token": "3",
        "lidar_token": "3_00040",
        "radar_frame_idx": 42,
        "radar_path": "/missing/tesseract_00099.mat",
    }

    resolved = _resolve_rpc_radar(info, tmp_path, tmp_path)
    assert resolved == rpc_path.resolve()


def test_rpc_reader_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((8, 5), dtype=np.float32))
    try:
        KRadarRPCReader().read(path)
    except ValueError as error:
        assert "[N,11]" in str(error)
    else:
        raise AssertionError("Expected an invalid RPC schema to be rejected")
