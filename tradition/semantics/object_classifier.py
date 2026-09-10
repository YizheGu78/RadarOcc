from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tradition.core.config import ObjectClusteringConfig
from tradition.core.interfaces import SemanticClassifier
from tradition.core.types import MotionLabel, RadarDetection, SemanticLabel
from tradition.semantics.cluster_geometry import (
    mean_diameter_line_distance,
    minimum_area_bbox_perimeter,
)


FEATURE_NAMES = (
    "branch_dynamic",
    "point_count",
    "log_point_count",
    "unique_xy_cells",
    "extent_x_m",
    "extent_y_m",
    "extent_z_m",
    "xy_diagonal_m",
    "bbox_area_m2",
    "bbox_volume_m3",
    "hull_area_m2",
    "hull_perimeter_m",
    "point_density_m2",
    "cov_major_m2",
    "cov_minor_m2",
    "linearity",
    "power_mean",
    "power_std",
    "power_max",
    "power_q90",
    "doppler_residual_mean_mps",
    "doppler_residual_std_mps",
    "doppler_residual_min_mps",
    "doppler_residual_max_mps",
    "doppler_residual_abs_mean_mps",
    "range_mean_m",
    "range_std_m",
    "dynamic_fraction",
    "range_compensated_point_count",
    "power_span",
    "oriented_bbox_perimeter_m",
    "max_line_deviation_m",
    "compactness_m",
    "major_doppler_spread_ratio",
    "minor_doppler_spread_ratio",
    "range_doppler_correlation",
    "z_mean_m",
    "z_std_m",
    "z_min_m",
    "z_max_m",
    "elevation_mean_rad",
    "elevation_std_rad",
)

# Regularize the denominator in spatial-spread / (residual-spread + epsilon).
# This is a numerical floor scale, not a Doppler classification threshold.
DOPPLER_SPREAD_EPS_MPS = 1e-3


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Pearson correlation, defined as zero for insufficient/constant data."""
    if len(first) < 2 or np.std(first) <= 1e-9 or np.std(second) <= 1e-9:
        return 0.0
    centered_first = first - np.mean(first)
    centered_second = second - np.mean(second)
    denominator = np.linalg.norm(centered_first) * np.linalg.norm(centered_second)
    return float(np.clip(centered_first @ centered_second / denominator, -1.0, 1.0))


@dataclass(frozen=True)
class ObjectCandidate:
    """One stationary-OGM or moving-DBSCAN object proposal."""

    indices: np.ndarray
    branch: str
    features: np.ndarray


def _convex_hull(points_xy: np.ndarray) -> np.ndarray:
    points = np.unique(np.asarray(points_xy, dtype=np.float64), axis=0)
    if len(points) <= 1:
        return points
    ordered = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        first = a - origin
        second = b - origin
        return float(first[0] * second[1] - first[1] * second[0])

    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _hull_area_perimeter(points_xy: np.ndarray) -> tuple[float, float]:
    hull = _convex_hull(points_xy)
    if len(hull) < 2:
        return 0.0, 0.0
    closed = np.vstack([hull, hull[0]])
    perimeter = float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())
    if len(hull) < 3:
        return 0.0, perimeter
    x, y = hull[:, 0], hull[:, 1]
    area = 0.5 * abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1)))
    return area, perimeter


class RadarObjectFeatureExtractor:
    """Shared 42-D training/inference descriptors (original 28 entries first).

    Callers supply ego-compensated wrapped residuals in ``radial_velocity_mps``.
    Geometry/height use pose-aligned ``xyz_lidar_m``. Range and elevation remain
    each point's original radar measurement, also for aligned static history.
    """

    feature_names = FEATURE_NAMES

    def __init__(self, cell_size_m: float = 0.40) -> None:
        self.cell_size_m = float(cell_size_m)

    def extract(
        self,
        detections: Sequence[RadarDetection],
        motion_labels: Sequence[MotionLabel],
        indices: np.ndarray,
        branch: str,
    ) -> np.ndarray:
        selected = [detections[int(index)] for index in indices]
        if not selected:
            raise ValueError("Cannot extract features from an empty candidate.")
        xyz = np.asarray([item.xyz_lidar_m for item in selected], dtype=np.float64)
        xy = xyz[:, :2]
        power = np.asarray([item.power for item in selected], dtype=np.float64)
        residual = np.asarray(
            [item.radial_velocity_mps for item in selected], dtype=np.float64
        )
        ranges = np.asarray([item.range_m for item in selected], dtype=np.float64)
        elevations = np.asarray(
            [item.elevation_rad for item in selected], dtype=np.float64
        )
        labels = [motion_labels[int(index)] for index in indices]

        extent = np.ptp(xyz, axis=0) if len(xyz) > 1 else np.zeros(3)
        bbox_area = float(max(extent[0] * extent[1], 0.0))
        bbox_volume = float(max(bbox_area * extent[2], 0.0))
        hull = _convex_hull(xy)
        hull_area, hull_perimeter = _hull_area_perimeter(hull)
        cell = np.floor(xy / self.cell_size_m).astype(np.int64)
        unique_cells = int(len(np.unique(cell, axis=0)))

        if len(xy) > 1:
            covariance = np.cov(xy, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            major, minor = float(eigenvalues[-1]), float(eigenvalues[0])
            # Projection spans along covariance axes, not ellipse axis lengths.
            principal_coordinates = (xy - np.mean(xy, axis=0)) @ eigenvectors[:, ::-1]
            principal_spreads = np.ptp(principal_coordinates, axis=0)
        else:
            major = minor = 0.0
            principal_spreads = np.zeros(2)
        linearity = 0.0 if major <= 1e-9 else 1.0 - minor / major
        density_area = max(hull_area, self.cell_size_m**2)
        dynamic_fraction = sum(
            label == MotionLabel.DYNAMIC for label in labels
        ) / len(labels)
        residual_spread = float(np.ptp(residual)) + DOPPLER_SPREAD_EPS_MPS
        center = np.mean(xy, axis=0)
        z = xyz[:, 2]

        features = np.asarray(
            [
                float(branch == "dynamic"),
                float(len(selected)),
                float(np.log1p(len(selected))),
                float(unique_cells),
                float(extent[0]),
                float(extent[1]),
                float(extent[2]),
                float(np.linalg.norm(extent[:2])),
                bbox_area,
                bbox_volume,
                hull_area,
                hull_perimeter,
                float(len(selected) / density_area),
                major,
                minor,
                linearity,
                float(np.mean(power)),
                float(np.std(power)),
                float(np.max(power)),
                float(np.quantile(power, 0.90)),
                float(np.mean(residual)),
                float(np.std(residual)),
                float(np.min(residual)),
                float(np.max(residual)),
                float(np.mean(np.abs(residual))),
                float(np.mean(ranges)),
                float(np.std(ranges)),
                float(dynamic_fraction),
                float(len(selected) * np.mean(ranges)),
                float(np.ptp(power)),
                minimum_area_bbox_perimeter(hull),
                mean_diameter_line_distance(xy, hull),
                float(np.mean(np.linalg.norm(xy - center, axis=1))),
                float(principal_spreads[0] / residual_spread),
                float(principal_spreads[1] / residual_spread),
                _safe_correlation(ranges, residual),
                float(np.mean(z)),
                float(np.std(z)),
                float(np.min(z)),
                float(np.max(z)),
                float(np.mean(elevations)),
                float(np.std(elevations)),
            ],
            dtype=np.float64,
        )
        if features.shape != (len(FEATURE_NAMES),):
            raise AssertionError("Object feature schema changed unexpectedly.")
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class DualBranchCandidateExtractor:
    """Stationary OGM connected components plus moving XYZ DBSCAN."""

    def __init__(self, config: ObjectClusteringConfig | None = None) -> None:
        self.config = config or ObjectClusteringConfig()
        self.features = RadarObjectFeatureExtractor(self.config.static_cell_size_m)

    def extract(
        self,
        detections: Sequence[RadarDetection],
        motion_labels: Sequence[MotionLabel],
    ) -> list[ObjectCandidate]:
        if len(detections) != len(motion_labels):
            raise ValueError("detections and motion_labels must have equal length.")
        return self._static_candidates(detections, motion_labels) + self._dynamic_candidates(
            detections, motion_labels
        )

    def _static_candidates(
        self,
        detections: Sequence[RadarDetection],
        motion_labels: Sequence[MotionLabel],
    ) -> list[ObjectCandidate]:
        indices = np.asarray(
            [i for i, label in enumerate(motion_labels) if label == MotionLabel.STATIC],
            dtype=np.int64,
        )
        if indices.size == 0:
            return []
        xy = np.asarray([detections[int(i)].xyz_lidar_m[:2] for i in indices])
        base_cells = np.floor(xy / self.config.static_cell_size_m).astype(np.int64)
        occupied: set[tuple[int, int]] = set()
        radius = self.config.static_dilation_cells
        for x, y in base_cells:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    occupied.add((int(x + dx), int(y + dy)))

        component_by_cell: dict[tuple[int, int], int] = {}
        remaining = set(occupied)
        component_id = 0
        while remaining:
            seed = remaining.pop()
            stack = [seed]
            component_by_cell[seed] = component_id
            while stack:
                cell = stack.pop()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        neighbour = (cell[0] + dx, cell[1] + dy)
                        if neighbour in remaining:
                            remaining.remove(neighbour)
                            component_by_cell[neighbour] = component_id
                            stack.append(neighbour)
            component_id += 1

        grouped: dict[int, list[int]] = {}
        for original_index, cell in zip(indices, base_cells):
            group = component_by_cell[(int(cell[0]), int(cell[1]))]
            grouped.setdefault(group, []).append(int(original_index))

        candidates: list[ObjectCandidate] = []
        for members in grouped.values():
            if len(members) < self.config.static_min_points:
                continue
            member_array = np.asarray(members, dtype=np.int64)
            member_cells = np.asarray(
                [
                    np.floor(
                        detections[int(i)].xyz_lidar_m[:2]
                        / self.config.static_cell_size_m
                    ).astype(np.int64)
                    for i in member_array
                ]
            )
            if len(np.unique(member_cells, axis=0)) < self.config.static_min_cells:
                continue
            candidates.append(
                ObjectCandidate(
                    indices=member_array,
                    branch="static",
                    features=self.features.extract(
                        detections, motion_labels, member_array, "static"
                    ),
                )
            )
        return candidates

    def _dynamic_candidates(
        self,
        detections: Sequence[RadarDetection],
        motion_labels: Sequence[MotionLabel],
    ) -> list[ObjectCandidate]:
        indices = np.asarray(
            [i for i, label in enumerate(motion_labels) if label == MotionLabel.DYNAMIC],
            dtype=np.int64,
        )
        if indices.size < self.config.dynamic_min_points:
            return []
        xyz = np.asarray([detections[int(i)].xyz_lidar_m for i in indices])
        scaled = xyz / np.asarray(
            [
                self.config.dynamic_eps_xy_m,
                self.config.dynamic_eps_xy_m,
                self.config.dynamic_eps_z_m,
            ]
        )
        cluster_labels = self._dbscan(scaled, self.config.dynamic_min_points)
        candidates: list[ObjectCandidate] = []
        for cluster_id in range(int(cluster_labels.max()) + 1):
            members = indices[cluster_labels == cluster_id]
            candidates.append(
                ObjectCandidate(
                    indices=members,
                    branch="dynamic",
                    features=self.features.extract(
                        detections, motion_labels, members, "dynamic"
                    ),
                )
            )
        return candidates

    @staticmethod
    def _dbscan(points: np.ndarray, min_points: int) -> np.ndarray:
        """Small deterministic DBSCAN with unit radius in scaled space."""
        count = len(points)
        labels = np.full(count, -2, dtype=np.int64)
        if count == 0:
            return labels
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        neighbours = [np.flatnonzero(row <= 1.0) for row in distances]
        cluster_id = 0
        for seed in range(count):
            if labels[seed] != -2:
                continue
            if len(neighbours[seed]) < min_points:
                labels[seed] = -1
                continue
            labels[seed] = cluster_id
            queue = [int(i) for i in neighbours[seed] if i != seed]
            queued = set(queue)
            cursor = 0
            while cursor < len(queue):
                index = queue[cursor]
                cursor += 1
                if labels[index] == -1:
                    labels[index] = cluster_id
                if labels[index] != -2:
                    continue
                labels[index] = cluster_id
                if len(neighbours[index]) >= min_points:
                    for neighbour in neighbours[index]:
                        value = int(neighbour)
                        if value not in queued:
                            queued.add(value)
                            queue.append(value)
            cluster_id += 1
        return labels


class ObjectAwareSemanticClassifier(SemanticClassifier):
    """Random-Forest objectness; motion is evidence, not the semantic rule."""

    def __init__(self, estimator: Any, config: ObjectClusteringConfig | None = None) -> None:
        self.estimator = estimator
        self.config = config or ObjectClusteringConfig()
        self.candidates = DualBranchCandidateExtractor(self.config)
        self.last_diagnostics: dict[str, Any] = {}

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        config: ObjectClusteringConfig | None = None,
    ) -> "ObjectAwareSemanticClassifier":
        try:
            import joblib
        except ImportError as error:
            raise RuntimeError(
                "Object-aware semantics requires joblib and scikit-learn."
            ) from error
        model_path = Path(path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Object classifier model not found: {model_path}")
        bundle = joblib.load(model_path)
        if not isinstance(bundle, dict) or "estimator" not in bundle:
            raise ValueError(f"Invalid object classifier bundle: {model_path}")
        if tuple(bundle.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError(
                "Object model feature schema does not match this code version: "
                f"stored {len(bundle.get('feature_names', ()))} features; "
                f"expected {len(FEATURE_NAMES)} in the current order. "
                "Regenerate the training features and retrain with "
                "train_traditional_object_classifier.sh."
            )
        expected_count = len(FEATURE_NAMES)
        if getattr(bundle["estimator"], "n_features_in_", expected_count) != expected_count:
            raise ValueError(
                f"Object estimator must accept {expected_count} features. "
                "Regenerate the training features and retrain the object classifier."
            )
        instance = cls(bundle["estimator"], config=config)
        instance.last_diagnostics = {
            "model_path": str(model_path),
            "training_metadata": bundle.get("metadata", {}),
        }
        return instance

    def classify(
        self,
        detections: Sequence[RadarDetection],
        motion_labels: Sequence[MotionLabel],
    ) -> list[SemanticLabel]:
        labels, _ = self.classify_with_acceptance(detections, motion_labels)
        return labels

    def classify_with_acceptance(
        self,
        detections: Sequence[RadarDetection],
        motion_labels: Sequence[MotionLabel],
    ) -> tuple[list[SemanticLabel], np.ndarray]:
        if len(detections) != len(motion_labels):
            raise ValueError("detections and motion_labels must have equal length.")
        labels = [SemanticLabel.BACKGROUND] * len(detections)
        accepted = np.asarray(
            [label == MotionLabel.STATIC for label in motion_labels], dtype=bool
        )
        candidates = self.candidates.extract(detections, motion_labels)
        probabilities: list[float] = []
        foreground_clusters = 0
        for candidate in candidates:
            probability = self._foreground_probability(candidate.features)
            probabilities.append(probability)
            if candidate.branch == "static":
                accepted[candidate.indices] = True
            if probability < self.config.foreground_probability_threshold:
                continue
            foreground_clusters += 1
            accepted[candidate.indices] = True
            for index in candidate.indices:
                labels[int(index)] = SemanticLabel.FOREGROUND
        persistent = {
            key: value
            for key, value in self.last_diagnostics.items()
            if key in {"model_path", "training_metadata"}
        }
        self.last_diagnostics = {
            **persistent,
            "candidate_count": len(candidates),
            "static_candidate_count": sum(c.branch == "static" for c in candidates),
            "dynamic_candidate_count": sum(c.branch == "dynamic" for c in candidates),
            "foreground_cluster_count": foreground_clusters,
            "mean_foreground_probability": (
                float(np.mean(probabilities)) if probabilities else float("nan")
            ),
        }
        return labels, accepted

    def _foreground_probability(self, features: np.ndarray) -> float:
        probabilities = np.asarray(
            self.estimator.predict_proba(features.reshape(1, -1)), dtype=np.float64
        )
        classes = list(getattr(self.estimator, "classes_", [0, 1]))
        if 1 not in classes:
            return 0.0
        return float(probabilities[0, classes.index(1)])
