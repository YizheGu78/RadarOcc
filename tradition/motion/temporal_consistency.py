from __future__ import annotations

from collections import deque

import numpy as np

from tradition.core.config import MotionConfig, TemporalConfig
from tradition.core.geometry import relative_lidar_transform, transform_points
from tradition.core.interfaces import TemporalMotionClassifier
from tradition.core.types import (
    DopplerEvidence,
    MotionLabel,
    TemporalClassification,
    TemporalDetectionFrame,
)


class _RadiusIndex:
    """Small fixed-radius spatial hash used for per-frame point association."""

    def __init__(self, points: np.ndarray, radius_m: float) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        self.radius_m = float(radius_m)
        self.cells: dict[tuple[int, int, int], list[int]] = {}
        for index, point in enumerate(self.points):
            key = self._key(point)
            self.cells.setdefault(key, []).append(index)

    def _key(self, point: np.ndarray) -> tuple[int, int, int]:
        value = np.floor(point / self.radius_m).astype(np.int64)
        return int(value[0]), int(value[1]), int(value[2])

    def neighbours(self, point: np.ndarray) -> np.ndarray:
        base = self._key(point)
        candidates: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    candidates.extend(
                        self.cells.get(
                            (base[0] + dx, base[1] + dy, base[2] + dz), ()
                        )
                    )
        if not candidates:
            return np.empty(0, dtype=np.int64)
        indices = np.asarray(candidates, dtype=np.int64)
        distance = np.linalg.norm(self.points[indices] - point, axis=1)
        return indices[distance <= self.radius_m]


class PoseAlignedTemporalClassifier(TemporalMotionClassifier):
    """Fuse Doppler evidence over a causal pose-aligned rolling window."""

    def __init__(
        self,
        temporal_cfg: TemporalConfig | None = None,
        motion_cfg: MotionConfig | None = None,
    ) -> None:
        self.temporal_cfg = temporal_cfg or TemporalConfig()
        self.motion_cfg = motion_cfg or MotionConfig()
        self._frames: deque[TemporalDetectionFrame] = deque(
            maxlen=self.temporal_cfg.window_size
        )

    def reset(self) -> None:
        self._frames.clear()

    def update(
        self, frame: TemporalDetectionFrame
    ) -> TemporalClassification:
        self._frames.append(frame)
        frames = list(self._frames)
        required_static = min(
            self.temporal_cfg.min_static_support, len(frames)
        )
        required_dynamic = min(
            self.temporal_cfg.min_dynamic_support, len(frames)
        )

        transformed: list[np.ndarray] = []
        static_indices: list[_RadiusIndex] = []
        dynamic_indices: list[_RadiusIndex] = []
        for source in frames:
            xyz = np.asarray(
                [det.xyz_lidar_m for det in source.detections],
                dtype=np.float64,
            ).reshape(-1, 3)
            relative = relative_lidar_transform(
                source.pose_lidar_to_world,
                frame.pose_lidar_to_world,
            )
            points = transform_points(xyz, relative)
            transformed.append(points)
            static_indices.append(
                _RadiusIndex(points, self.temporal_cfg.static_match_radius_m)
            )
            dynamic_points = points[
                source.doppler_evidence == int(DopplerEvidence.DYNAMIC)
            ]
            dynamic_indices.append(
                _RadiusIndex(
                    dynamic_points,
                    self.temporal_cfg.dynamic_match_radius_m,
                )
            )

        current_points = transformed[-1]
        static_support = np.zeros(len(current_points), dtype=np.int16)
        dynamic_support = np.zeros(len(current_points), dtype=np.int16)
        current_labels: list[MotionLabel] = []
        current_indices: list[int] = []

        for index, point in enumerate(current_points):
            support, _ = self._static_support(
                point, frames, static_indices
            )
            static_support[index] = support
            dynamic_support[index] = sum(
                spatial.neighbours(point).size > 0
                for spatial in dynamic_indices
            )
            is_background = support >= required_static
            is_foreground = (
                not is_background
                and frame.doppler_evidence[index]
                == int(DopplerEvidence.DYNAMIC)
                and dynamic_support[index] >= required_dynamic
                and support < required_static
            )
            if is_background:
                current_indices.append(index)
                current_labels.append(MotionLabel.STATIC)
            elif is_foreground:
                current_indices.append(index)
                current_labels.append(MotionLabel.DYNAMIC)

        historic_background: list[np.ndarray] = []
        # Accumulate only confirmed static history; current accepted points are
        # already handled by the ordinary inverse sensor model.
        for frame_index in range(len(frames) - 1):
            for point in transformed[frame_index]:
                support, _ = self._static_support(
                    point, frames, static_indices
                )
                if support >= required_static:
                    historic_background.append(point)

        return TemporalClassification(
            current_indices=np.asarray(current_indices, dtype=np.int64),
            current_motion_labels=current_labels,
            historic_background_lidar_m=np.asarray(
                historic_background, dtype=np.float64
            ).reshape(-1, 3),
            static_support=static_support,
            dynamic_support=dynamic_support,
        )

    def _static_support(
        self,
        point: np.ndarray,
        frames: list[TemporalDetectionFrame],
        spatial_indices: list[_RadiusIndex],
    ) -> tuple[int, list[float]]:
        static_evidence_support = 0
        residuals: list[float] = []
        for source, spatial in zip(frames, spatial_indices):
            neighbours = spatial.neighbours(point)
            if neighbours.size == 0:
                continue
            frame_residuals = np.abs(source.doppler_residuals_mps[neighbours])
            finite = frame_residuals[np.isfinite(frame_residuals)]
            if finite.size == 0:
                continue
            best_residual = float(np.min(finite))
            residuals.append(best_residual)
            if best_residual <= self.motion_cfg.static_residual_threshold_mps:
                static_evidence_support += 1
        return static_evidence_support, residuals
