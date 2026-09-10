import math

import numpy as np

from tradition.core.config import ObjectClusteringConfig
from tradition.core.types import MotionLabel, RadarDetection, SemanticLabel
from tradition.mapping.temporal_occupancy_grid_3d import TemporalLogOddsOccupancyGrid3D
from tradition.semantics.object_classifier import (
    FEATURE_NAMES,
    DualBranchCandidateExtractor,
    ObjectAwareSemanticClassifier,
)
from tradition.semantics.training import train_random_forest_objectness


def _detection(x: float, y: float, z: float = 0.0, residual: float = 0.1):
    point = np.asarray([x, y, z], dtype=np.float64)
    return RadarDetection(
        range_index=int(np.linalg.norm(point) / 0.4),
        doppler_index=-1,
        elevation_index=18,
        azimuth_index=53,
        power=1.0,
        range_m=float(np.linalg.norm(point)),
        radial_velocity_mps=residual,
        azimuth_rad=math.atan2(y, x),
        elevation_rad=0.0,
        xyz_radar_m=point.copy(),
        xyz_lidar_m=point.copy(),
    )


class _GeometryEstimator:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, features):
        result = []
        for row in features:
            extent_x = row[FEATURE_NAMES.index("extent_x_m")]
            extent_y = row[FEATURE_NAMES.index("extent_y_m")]
            linearity = row[FEATURE_NAMES.index("linearity")]
            is_object = extent_x < 6.0 and extent_y > 0.5 and linearity < 0.98
            probability = 0.9 if is_object else 0.1
            result.append([1.0 - probability, probability])
        return np.asarray(result)


def test_stationary_object_shape_can_be_foreground_while_guardrail_stays_background():
    vehicle = [
        _detection(10.0 + dx, -1.0 + dy)
        for dx in (0.0, 0.4, 0.8, 1.2)
        for dy in (0.0, 0.4, 0.8)
    ]
    guardrail = [_detection(20.0 + x, -8.0) for x in np.arange(0.0, 8.0, 0.4)]
    detections = vehicle + guardrail
    motion = [MotionLabel.STATIC] * len(detections)
    classifier = ObjectAwareSemanticClassifier(
        _GeometryEstimator(),
        ObjectClusteringConfig(static_dilation_cells=0),
    )

    labels, accepted = classifier.classify_with_acceptance(detections, motion)

    assert all(accepted)
    assert all(label == SemanticLabel.FOREGROUND for label in labels[: len(vehicle)])
    assert all(label == SemanticLabel.BACKGROUND for label in labels[len(vehicle) :])


def test_dynamic_dbscan_is_only_a_proposal_and_model_decides_semantics():
    detections = [_detection(10.0, 0.0, residual=1.1), _detection(10.5, 0.2, residual=1.2)]
    motion = [MotionLabel.DYNAMIC] * 2
    candidates = DualBranchCandidateExtractor(
        ObjectClusteringConfig(dynamic_min_points=2)
    ).extract(detections, motion)
    assert len(candidates) == 1
    assert candidates[0].branch == "dynamic"


def test_isolated_dynamic_return_is_discarded():
    classifier = ObjectAwareSemanticClassifier(
        _GeometryEstimator(), ObjectClusteringConfig(dynamic_min_points=2)
    )
    _, accepted = classifier.classify_with_acceptance(
        [_detection(10.0, 0.0, residual=1.2)], [MotionLabel.DYNAMIC]
    )
    assert accepted.tolist() == [False]


def test_random_forest_bundle_round_trip(tmp_path):
    import joblib

    rng = np.random.default_rng(4)
    features = rng.normal(size=(40, len(FEATURE_NAMES)))
    targets = np.asarray([0] * 20 + [1] * 20, dtype=np.int8)
    path = train_random_forest_objectness(
        features, targets, tmp_path / "objects.joblib", n_estimators=10
    )
    classifier = ObjectAwareSemanticClassifier.from_file(path)
    assert classifier.estimator.n_estimators == 10
    assert classifier.estimator.n_features_in_ == 42
    bundle = joblib.load(path)
    assert bundle["metadata"]["feature_count"] == 42
    assert tuple(bundle["feature_names"]) == FEATURE_NAMES
    predictions = classifier.estimator.predict_proba(features[:3])
    assert predictions.shape == (3, 2)
    np.testing.assert_allclose(predictions.sum(axis=1), np.ones(3))


def test_legacy_28_feature_model_is_rejected_with_retraining_message(tmp_path):
    import joblib

    path = tmp_path / "legacy.joblib"
    joblib.dump({"estimator": _GeometryEstimator(), "feature_names": FEATURE_NAMES[:28]}, path)
    with np.testing.assert_raises_regex(ValueError, "stored 28 features.*expected 42.*retrain"):
        ObjectAwareSemanticClassifier.from_file(path)


def test_inconsistent_estimator_dimension_is_rejected(tmp_path):
    import joblib

    estimator = _GeometryEstimator()
    estimator.n_features_in_ = 28
    path = tmp_path / "inconsistent.joblib"
    joblib.dump({"estimator": estimator, "feature_names": FEATURE_NAMES}, path)
    with np.testing.assert_raises_regex(ValueError, "must accept 42 features"):
        ObjectAwareSemanticClassifier.from_file(path)


def test_training_rejects_legacy_28_feature_arrays(tmp_path):
    with np.testing.assert_raises_regex(ValueError, r"Expected features \[N,42\]"):
        train_random_forest_objectness(
            np.zeros((2, 28)), np.array([0, 1]), tmp_path / "unused.joblib"
        )


def test_historic_static_object_can_be_mapped_as_foreground():
    mapper = TemporalLogOddsOccupancyGrid3D()
    mapper.update_historic_semantics(
        np.asarray([[10.0, 0.0, 0.0]]), [SemanticLabel.FOREGROUND]
    )
    labels = mapper.labels()
    assert int(np.count_nonzero(labels == int(SemanticLabel.FOREGROUND))) > 0
