from dataclasses import replace
import math

import numpy as np

from tradition.core.config import ObjectClusteringConfig
from tradition.core.types import MotionLabel, RadarDetection
from tradition.semantics.object_classifier import (
    DOPPLER_SPREAD_EPS_MPS,
    FEATURE_NAMES,
    DualBranchCandidateExtractor,
    RadarObjectFeatureExtractor,
    _convex_hull,
)
from tradition.semantics.cluster_geometry import (
    mean_diameter_line_distance,
    minimum_area_bbox_perimeter,
)


LEGACY_FEATURE_NAMES = (
    "branch_dynamic", "point_count", "log_point_count", "unique_xy_cells",
    "extent_x_m", "extent_y_m", "extent_z_m", "xy_diagonal_m",
    "bbox_area_m2", "bbox_volume_m3", "hull_area_m2", "hull_perimeter_m",
    "point_density_m2", "cov_major_m2", "cov_minor_m2", "linearity",
    "power_mean", "power_std", "power_max", "power_q90",
    "doppler_residual_mean_mps", "doppler_residual_std_mps",
    "doppler_residual_min_mps", "doppler_residual_max_mps",
    "doppler_residual_abs_mean_mps", "range_mean_m", "range_std_m",
    "dynamic_fraction",
)

ADDED_FEATURE_NAMES = (
    "range_compensated_point_count", "power_span", "oriented_bbox_perimeter_m",
    "max_line_deviation_m", "compactness_m", "major_doppler_spread_ratio",
    "minor_doppler_spread_ratio", "range_doppler_correlation",
    "z_mean_m", "z_std_m", "z_min_m", "z_max_m",
    "elevation_mean_rad", "elevation_std_rad",
)


def _detection(xyz, residual=0.0, power=1.0, range_m=10.0, elevation=0.0):
    point = np.asarray(xyz, dtype=np.float64)
    return RadarDetection(
        range_index=25,
        doppler_index=-1,
        elevation_index=18,
        azimuth_index=53,
        power=power,
        range_m=range_m,
        radial_velocity_mps=residual,
        azimuth_rad=0.0,
        elevation_rad=elevation,
        xyz_radar_m=point.copy(),
        xyz_lidar_m=point.copy(),
    )


def _rectangle():
    xyz = [(1, 0, 0), (5, 0, 1), (1, 2, 2), (5, 2, 3)]
    return [
        _detection(point, residual, power, distance, elevation)
        for point, residual, power, distance, elevation in zip(
            xyz, [-0.3, -0.1, 0.1, 0.3], [1, 3, 8, 10],
            [10, 12, 14, 16], [-0.1, 0.0, 0.2, 0.3]
        )
    ]


def _features(detections, branch="static"):
    motion = MotionLabel.DYNAMIC if branch == "dynamic" else MotionLabel.STATIC
    features = RadarObjectFeatureExtractor().extract(
        detections, [motion] * len(detections), np.arange(len(detections)), branch
    )
    assert features.shape == (42,)
    assert np.all(np.isfinite(features))
    return dict(zip(FEATURE_NAMES, features))


def test_feature_schema_preserves_28_names_and_appends_exactly_14():
    assert FEATURE_NAMES == LEGACY_FEATURE_NAMES + ADDED_FEATURE_NAMES
    assert len(set(FEATURE_NAMES)) == 42


def test_all_14_added_features_have_expected_values():
    features = _features(_rectangle())
    expected = [
        52.0, 9.0, 12.0, 4.0 / math.sqrt(20.0), math.sqrt(5.0),
        4.0 / (0.6 + DOPPLER_SPREAD_EPS_MPS),
        2.0 / (0.6 + DOPPLER_SPREAD_EPS_MPS),
        1.0, 1.5, math.sqrt(1.25), 0.0, 3.0, 0.1, math.sqrt(0.025),
    ]
    np.testing.assert_allclose(
        [features[name] for name in ADDED_FEATURE_NAMES], expected, atol=1e-12
    )


def test_original_28_values_are_preserved():
    features = _features(_rectangle())
    expected = [
        0.0, 4.0, np.log1p(4), 4.0,
        4.0, 2.0, 3.0, math.sqrt(20), 8.0, 24.0, 8.0, 12.0,
        0.5, 16.0 / 3.0, 4.0 / 3.0, 0.75,
        5.5, math.sqrt(13.25), 10.0, 9.4,
        0.0, math.sqrt(0.05), -0.3, 0.3, 0.2, 13.0, math.sqrt(5), 0.0,
    ]
    np.testing.assert_allclose(
        [features[name] for name in LEGACY_FEATURE_NAMES], expected, atol=1e-12
    )


def test_single_point_and_duplicate_points_remain_finite():
    point = _detection((10.0, 2.0, -0.5), elevation=0.25)
    for detections in ([point], [point] * 4):
        features = _features(detections)
        for name in (
            "power_span", "oriented_bbox_perimeter_m", "max_line_deviation_m",
            "compactness_m", "major_doppler_spread_ratio",
            "minor_doppler_spread_ratio", "range_doppler_correlation",
            "z_std_m", "elevation_std_rad",
        ):
            assert features[name] == 0.0
        assert features["z_mean_m"] == -0.5
        assert features["z_min_m"] == features["z_max_m"] == -0.5
        assert features["elevation_mean_rad"] == 0.25


def test_collinear_cluster_has_zero_minor_spread_and_line_deviation():
    detections = [_detection((x, 2 * x, 0), residual=x / 10) for x in (1, 2, 3)]
    features = _features(detections)
    np.testing.assert_allclose(features["oriented_bbox_perimeter_m"], 4 * math.sqrt(5))
    np.testing.assert_allclose(features["minor_doppler_spread_ratio"], 0, atol=1e-12)
    np.testing.assert_allclose(features["max_line_deviation_m"], 0, atol=1e-12)


def test_zero_doppler_spread_is_regularized_and_constant_correlation_is_zero():
    detections = [replace(point, radial_velocity_mps=0.1) for point in _rectangle()]
    features = _features(detections)
    assert features["major_doppler_spread_ratio"] == 4.0 / DOPPLER_SPREAD_EPS_MPS
    assert features["minor_doppler_spread_ratio"] == 2.0 / DOPPLER_SPREAD_EPS_MPS
    assert features["range_doppler_correlation"] == 0.0
    detections = [replace(point, range_m=12.0) for point in _rectangle()]
    assert _features(detections)["range_doppler_correlation"] == 0.0


def test_range_doppler_correlation_keeps_its_sign():
    detections = [
        replace(point, radial_velocity_mps=-point.radial_velocity_mps)
        for point in _rectangle()
    ]
    np.testing.assert_allclose(_features(detections)["range_doppler_correlation"], -1)


def test_planar_geometry_is_rotation_translation_and_order_invariant():
    original = _rectangle()
    expected = _features(original)
    angle = 0.63
    rotation = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    transformed = []
    for point in reversed(original):
        xyz = point.xyz_lidar_m.copy()
        xyz[:2] = rotation @ xyz[:2] + [31.0, -7.0]
        transformed.append(replace(point, xyz_lidar_m=xyz))
    actual = _features(transformed)
    for name in (
        "oriented_bbox_perimeter_m", "max_line_deviation_m", "compactness_m",
        "major_doppler_spread_ratio", "minor_doppler_spread_ratio",
    ):
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-10)
    assert actual["bbox_area_m2"] > expected["bbox_area_m2"]


def test_diameter_line_distance_includes_interior_measurements():
    xy = np.array([[0, 0], [4, 0], [0, 2], [4, 2], [1, 1]], dtype=float)
    hull = _convex_hull(xy)
    # The chosen diagonal runs from (0, 0) to (4, 2). The interior point
    # contributes an additional distance of 2 / sqrt(20).
    expected = (8 + 8 + 2) / (5 * math.sqrt(20))
    np.testing.assert_allclose(mean_diameter_line_distance(xy, hull), expected)


def test_minimum_area_rectangle_not_minimum_perimeter_rectangle():
    # A skew polygon where the two optimizations select different directions.
    hull = _convex_hull(np.array([[0, 0], [6, 0], [7, 2], [1, 4]], dtype=float))
    areas, perimeters = [], []
    for edge in np.roll(hull, -1, axis=0) - hull:
        theta = math.atan2(edge[1], edge[0])
        rotation = np.array([
            [math.cos(theta), math.sin(theta)],
            [-math.sin(theta), math.cos(theta)],
        ])
        sides = np.ptp(hull @ rotation.T, axis=0)
        areas.append(np.prod(sides))
        perimeters.append(2 * np.sum(sides))
    expected = perimeters[int(np.argmin(areas))]
    np.testing.assert_allclose(minimum_area_bbox_perimeter(hull), expected)
    assert expected > min(perimeters) + 1e-6


def test_aligned_height_and_source_measurement_elevation_are_distinct():
    source = _detection((10, 0, 5), range_m=11.2, elevation=0.46)
    aligned = replace(source, xyz_lidar_m=np.array([8, 0, 1], dtype=float))
    features = _features([aligned])
    assert features["z_mean_m"] == 1.0
    assert features["elevation_mean_rad"] == 0.46
    assert features["range_compensated_point_count"] == 11.2


def test_both_candidate_branches_use_the_same_42_feature_extractor():
    base = [_detection((10 + x, y, 0), residual=0.1 * x) for x, y in (
        (0, 0), (0.4, 0), (0, 0.4), (0.4, 0.4)
    )]
    extractor = DualBranchCandidateExtractor(ObjectClusteringConfig())
    for motion, branch in ((MotionLabel.STATIC, "static"), (MotionLabel.DYNAMIC, "dynamic")):
        labels = [motion] * len(base)
        candidates = extractor.extract(base, labels)
        assert len(candidates) == 1
        assert candidates[0].branch == branch
        expected = RadarObjectFeatureExtractor().extract(
            base, labels, candidates[0].indices, branch
        )
        np.testing.assert_allclose(candidates[0].features, expected)
        assert expected.shape == (42,)
