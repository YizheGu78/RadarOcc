from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from tradition.core.config import MotionConfig, TemporalConfig
from tradition.core.geometry import relative_lidar_transform, transform_points
from tradition.core.interfaces import TemporalMotionClassifier
from tradition.core.types import (
    DopplerEvidence,
    MotionLabel,
    RadarDetection,
    TemporalClassification,
    TemporalDetectionFrame,
)


class _RadiusIndex:
    """Small fixed-radius spatial hash used for per-frame point association."""

    def __init__(self, points: np.ndarray, radius_m: float) -> None:
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self.radius_m = float(radius_m)
        self.cells: dict[tuple[int, int, int], list[int]] = {}
        for index, point in enumerate(self.points):
            self.cells.setdefault(self._key(point), []).append(index)

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


@dataclass(frozen=True)
class PersistenceAssessment:
    """Read-only persistence result produced before velocity processing."""

    token: str
    world_points_m: np.ndarray
    support: np.ndarray
    persistent_mask: np.ndarray
    historic_detections: list[RadarDetection]


class OccupancyFirstTemporalClassifier(TemporalMotionClassifier):
    """World-grid persistence followed by selective motion confirmation.

    Persistence uses only prior pose-aligned frames; the current frame is
    committed after classification, so a point cannot support itself. Points
    outside persistent cells require validated velocity evidence before they
    enter the motion branch. Everything else remains explicitly unknown.
    """

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

    def assess_persistence(
        self,
        token: str,
        pose_lidar_to_world: np.ndarray,
        detections: Sequence[RadarDetection],
    ) -> PersistenceAssessment:
        current_xyz = self._detection_xyz(detections)
        current_world = transform_points(current_xyz, pose_lidar_to_world)
        current_cells = self._xy_cells(current_world)
        historical_cells = [
            set(map(tuple, self._xy_cells(self._world_points(frame))))
            for frame in self._frames
        ]

        support = np.asarray(
            [
                sum(
                    self._cell_has_support(tuple(cell), cells)
                    for cells in historical_cells
                )
                for cell in current_cells
            ],
            dtype=np.int16,
        )
        persistent_mask = support >= self.temporal_cfg.min_static_support
        historic_detections = self._historic_persistent_detections(
            pose_lidar_to_world,
            current_cells[persistent_mask],
        )
        return PersistenceAssessment(
            token=str(token),
            world_points_m=current_world,
            support=support,
            persistent_mask=persistent_mask,
            historic_detections=historic_detections,
        )

    def update(
        self,
        frame: TemporalDetectionFrame,
        persistence: PersistenceAssessment | None = None,
    ) -> TemporalClassification:
        assessment = persistence or self.assess_persistence(
            frame.token,
            frame.pose_lidar_to_world,
            frame.detections,
        )
        self._validate_assessment(frame, assessment)

        count = len(frame.detections)
        dynamic_support = np.zeros(count, dtype=np.int16)
        current_dynamic = frame.doppler_evidence == int(DopplerEvidence.DYNAMIC)
        historical_dynamic_indices = []
        for history in self._frames:
            history_world = self._world_points(history)
            historical_dynamic_indices.append(
                _RadiusIndex(
                    history_world[
                        history.doppler_evidence == int(DopplerEvidence.DYNAMIC)
                    ],
                    self.temporal_cfg.dynamic_match_radius_m,
                )
            )
        for index in np.flatnonzero(~assessment.persistent_mask & current_dynamic):
            support = 1
            point = assessment.world_points_m[int(index)]
            for spatial_index in historical_dynamic_indices:
                if spatial_index.neighbours(point).size:
                    support += 1
            dynamic_support[int(index)] = support

        motion_mask = (
            ~assessment.persistent_mask
            & current_dynamic
            & (dynamic_support >= self.temporal_cfg.min_dynamic_support)
        )
        unknown_mask = ~(assessment.persistent_mask | motion_mask)
        persistent_indices = np.flatnonzero(assessment.persistent_mask).astype(
            np.int64
        )
        motion_indices = np.flatnonzero(motion_mask).astype(np.int64)
        unknown_indices = np.flatnonzero(unknown_mask).astype(np.int64)
        current_indices = np.concatenate([persistent_indices, motion_indices])
        current_labels = (
            [MotionLabel.PERSISTENT] * len(persistent_indices)
            + [MotionLabel.MOTION] * len(motion_indices)
        )

        self._frames.append(frame)
        return TemporalClassification(
            current_indices=current_indices,
            current_motion_labels=current_labels,
            persistent_indices=persistent_indices,
            motion_indices=motion_indices,
            unknown_indices=unknown_indices,
            historic_persistent_lidar_m=self._detection_xyz(
                assessment.historic_detections
            ),
            historic_detections=assessment.historic_detections,
            static_support=assessment.support,
            dynamic_support=dynamic_support,
        )

    def _historic_persistent_detections(
        self,
        current_pose_lidar_to_world: np.ndarray,
        persistent_cells: np.ndarray,
    ) -> list[RadarDetection]:
        if persistent_cells.size == 0:
            return []
        accepted_cells = self._dilated_cells(persistent_cells)
        aligned: list[RadarDetection] = []
        for source in self._frames:
            world_points = self._world_points(source)
            source_cells = self._xy_cells(world_points)
            relative = relative_lidar_transform(
                source.pose_lidar_to_world,
                current_pose_lidar_to_world,
            )
            current_points = transform_points(
                self._detection_xyz(source.detections),
                relative,
            )
            for index, (point, cell) in enumerate(zip(current_points, source_cells)):
                if tuple(cell) not in accepted_cells:
                    continue
                residual = source.doppler_residuals_mps[index]
                aligned.append(
                    replace(
                        source.detections[index],
                        xyz_lidar_m=point.copy(),
                        radial_velocity_mps=(
                            float(residual) if np.isfinite(residual) else 0.0
                        ),
                    )
                )
        return aligned

    def _cell_has_support(
        self,
        cell: tuple[int, int],
        occupied: set[tuple[int, int]],
    ) -> bool:
        radius = self.temporal_cfg.occupancy_dilation_cells
        return any(
            (cell[0] + dx, cell[1] + dy) in occupied
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
        )

    def _dilated_cells(self, cells: np.ndarray) -> set[tuple[int, int]]:
        radius = self.temporal_cfg.occupancy_dilation_cells
        return {
            (int(cell[0] + dx), int(cell[1] + dy))
            for cell in cells
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
        }

    def _xy_cells(self, world_points: np.ndarray) -> np.ndarray:
        return np.floor(
            world_points[:, :2] / self.temporal_cfg.occupancy_cell_size_m
        ).astype(np.int64)

    @staticmethod
    def _detection_xyz(detections: Sequence[RadarDetection]) -> np.ndarray:
        return np.asarray(
            [item.xyz_lidar_m for item in detections],
            dtype=np.float64,
        ).reshape(-1, 3)

    def _world_points(self, frame: TemporalDetectionFrame) -> np.ndarray:
        return transform_points(
            self._detection_xyz(frame.detections),
            frame.pose_lidar_to_world,
        )

    @staticmethod
    def _validate_assessment(
        frame: TemporalDetectionFrame,
        assessment: PersistenceAssessment,
    ) -> None:
        if assessment.token != str(frame.token):
            raise ValueError("Persistence assessment token does not match the frame.")
        if len(assessment.persistent_mask) != len(frame.detections):
            raise ValueError("Persistence assessment size does not match detections.")


PoseAlignedTemporalClassifier = OccupancyFirstTemporalClassifier
