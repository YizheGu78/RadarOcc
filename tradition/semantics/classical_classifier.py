from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from tradition.core.config import SemanticConfig
from tradition.core.interfaces import SemanticClassifier
from tradition.core.types import MotionLabel, RadarDetection, SemanticLabel


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        root_a, root_b = self.find(first), self.find(second)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


class ClassicalClusterSemanticClassifier(SemanticClassifier):
    """Infer RadarOcc background/foreground without annotation leakage.

    Moving detections are foreground.  A deterministic 3-D connected-component
    pass then promotes compact, object-like clusters, allowing stationary road
    users such as parked vehicles to be foreground as well.  Extended or flat
    static returns remain background.
    """

    def __init__(self, config: SemanticConfig | None = None) -> None:
        self.config = config or SemanticConfig()

    def classify(
        self,
        detections: Sequence[RadarDetection],
        motion_labels: Sequence[MotionLabel],
    ) -> list[SemanticLabel]:
        if len(detections) != len(motion_labels):
            raise ValueError("detections and motion_labels must have equal length.")
        if not detections:
            return []

        xyz = np.asarray([det.xyz_lidar_m for det in detections], dtype=np.float64)
        labels = [
            SemanticLabel.FOREGROUND
            if motion == MotionLabel.DYNAMIC
            else SemanticLabel.BACKGROUND
            for motion in motion_labels
        ]
        for members in self._clusters(xyz):
            if self._is_object_cluster(xyz[members], motion_labels, members):
                for index in members:
                    labels[index] = SemanticLabel.FOREGROUND
        return labels

    def _clusters(self, xyz: np.ndarray) -> list[list[int]]:
        cfg = self.config
        scale = np.array(
            [cfg.neighbor_radius_xy_m, cfg.neighbor_radius_xy_m, cfg.neighbor_radius_z_m],
            dtype=np.float64,
        )
        cells = np.floor(xyz / scale).astype(np.int64)
        buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        dsu = _DisjointSet(len(xyz))

        for index, cell_array in enumerate(cells):
            cell = tuple(int(value) for value in cell_array)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                        for other in buckets.get(neighbor, ()):
                            delta = xyz[index] - xyz[other]
                            if (
                                float(np.hypot(delta[0], delta[1]))
                                <= cfg.neighbor_radius_xy_m
                                and abs(float(delta[2])) <= cfg.neighbor_radius_z_m
                            ):
                                dsu.union(index, other)
            buckets[cell].append(index)

        components: dict[int, list[int]] = defaultdict(list)
        for index in range(len(xyz)):
            components[dsu.find(index)].append(index)
        return list(components.values())

    def _is_object_cluster(
        self,
        points: np.ndarray,
        motion_labels: Sequence[MotionLabel],
        members: list[int],
    ) -> bool:
        cfg = self.config
        if len(members) < cfg.min_cluster_points:
            return False

        centered_xy = points[:, :2] - np.mean(points[:, :2], axis=0)
        covariance = centered_xy.T @ centered_xy / max(1, len(points) - 1)
        _, axes = np.linalg.eigh(covariance)
        projected = centered_xy @ axes
        xy_extents = np.ptp(projected, axis=0)
        width_m, length_m = sorted(float(value) for value in xy_extents)
        height_m = float(np.ptp(points[:, 2]))
        compact = (
            length_m <= cfg.max_object_length_m
            and width_m <= cfg.max_object_width_m
            and height_m <= cfg.max_object_height_m
        )
        if not compact:
            return False

        dynamic_fraction = sum(
            motion_labels[index] == MotionLabel.DYNAMIC for index in members
        ) / len(members)
        has_object_height = (
            height_m >= cfg.min_object_height_m
            or float(np.max(points[:, 2])) >= cfg.min_object_top_z_m
        )
        return (
            dynamic_fraction >= cfg.dynamic_cluster_min_fraction
            or has_object_height
        )
