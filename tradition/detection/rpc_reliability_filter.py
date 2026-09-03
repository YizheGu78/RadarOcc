from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from tradition.core.config import ReliabilityConfig
from tradition.core.interfaces import DetectionReliabilityFilter
from tradition.core.types import RadarDetection


class LocalPowerRPCFilter(DetectionReliabilityFilter):
    """Reject weak/isolated RPC sidelobes in a fixed polar-bin neighbourhood.

    This is a local measurement-quality gate, not a geometric clusterer.  It
    never creates object instances and never promotes a point to foreground.
    """

    def __init__(self, config: ReliabilityConfig | None = None) -> None:
        self.config = config or ReliabilityConfig()

    def filter(
        self, detections: Sequence[RadarDetection]
    ) -> list[RadarDetection]:
        if not detections:
            return []
        bins: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for index, det in enumerate(detections):
            bins[
                (det.range_index, det.azimuth_index, det.elevation_index)
            ].append(index)

        cfg = self.config
        kept: list[RadarDetection] = []
        for index, det in enumerate(detections):
            neighbours: list[int] = []
            for dr in range(-cfg.range_radius_bins, cfg.range_radius_bins + 1):
                for da in range(
                    -cfg.azimuth_radius_bins, cfg.azimuth_radius_bins + 1
                ):
                    for de in range(
                        -cfg.elevation_radius_bins,
                        cfg.elevation_radius_bins + 1,
                    ):
                        neighbours.extend(
                            bins.get(
                                (
                                    det.range_index + dr,
                                    det.azimuth_index + da,
                                    det.elevation_index + de,
                                ),
                                (),
                            )
                        )
            other_count = sum(neighbour != index for neighbour in neighbours)
            if other_count < cfg.min_local_neighbors:
                continue
            local_max = max(float(detections[j].power) for j in neighbours)
            if local_max <= 0.0:
                continue
            if float(det.power) / local_max < cfg.min_local_power_ratio:
                continue
            kept.append(det)
        return kept
