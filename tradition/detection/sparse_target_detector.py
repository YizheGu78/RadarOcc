from __future__ import annotations

import numpy as np

from tradition.core.config import KRadarConfig, SparseDetectionConfig
from tradition.core.geometry import (
    angle_from_index,
    doppler_from_index,
    radar_to_lidar,
    spherical_to_cartesian,
)
from tradition.core.interfaces import TargetDetector
from tradition.core.types import RadarDetection, SparseRadarFrame


class SparseCandidateTargetDetector(TargetDetector):
    """RadarOcc EAsparse candidates -> target list, without claiming CFAR.

    The full local reference cells have already been discarded, so CA/OS-CFAR
    cannot be reconstructed. Mean power is used only to thin the existing
    candidates per range; the strongest stored top-3 Doppler component supplies
    radial velocity.
    """

    def __init__(
        self,
        radar_cfg: KRadarConfig | None = None,
        sparse_cfg: SparseDetectionConfig | None = None,
    ) -> None:
        self.radar_cfg = radar_cfg or KRadarConfig()
        self.sparse_cfg = sparse_cfg or SparseDetectionConfig()

    def _selected_indices(self, frame: SparseRadarFrame) -> np.ndarray:
        score = np.asarray(frame.mean_power, dtype=np.float64)
        finite = np.isfinite(score)
        valid_geometry = (
            (frame.range_ind >= 0)
            & (frame.elevation_ind >= 0)
            & (frame.elevation_ind < self.radar_cfg.expected_elevation_bins)
            & (frame.azimuth_ind >= 0)
            & (frame.azimuth_ind < self.radar_cfg.expected_azimuth_bins)
        )
        valid = finite & valid_geometry

        kept: list[np.ndarray] = []
        for range_index in np.unique(frame.range_ind[valid]):
            group = np.flatnonzero(valid & (frame.range_ind == range_index))
            if group.size > self.sparse_cfg.max_per_range:
                order = np.argsort(score[group])[::-1]
                group = group[order[: self.sparse_cfg.max_per_range]]
            kept.append(group)

        if not kept:
            return np.empty((0,), dtype=np.int64)

        selected = np.concatenate(kept).astype(np.int64, copy=False)
        if selected.size > self.sparse_cfg.max_detections:
            order = np.argsort(score[selected])[::-1]
            selected = selected[order[: self.sparse_cfg.max_detections]]
        return selected

    def detect(self, measurement: SparseRadarFrame) -> list[RadarDetection]:
        if not isinstance(measurement, SparseRadarFrame):
            raise TypeError(
                "SparseCandidateTargetDetector expects SparseRadarFrame, "
                f"got {type(measurement).__name__}"
            )

        selected = self._selected_indices(measurement)
        cfg = self.radar_cfg
        detections: list[RadarDetection] = []

        for index in selected:
            top3_power = measurement.top3_power[:, index]
            top3_doppler = measurement.top3_doppler_ind[:, index]
            if not np.any(np.isfinite(top3_power)):
                continue
            local_peak = int(np.nanargmax(top3_power))
            doppler_index = int(round(float(top3_doppler[local_peak])))
            if not 0 <= doppler_index < cfg.expected_doppler_bins:
                continue

            r = int(measurement.range_ind[index])
            e = int(measurement.elevation_ind[index])
            a = int(measurement.azimuth_ind[index])
            power = float(top3_power[local_peak])

            range_m = r * cfg.range_resolution_m
            azimuth_rad = angle_from_index(
                a, cfg.azimuth_min_deg, cfg.azimuth_step_deg
            )
            elevation_rad = angle_from_index(
                e, cfg.elevation_min_deg, cfg.elevation_step_deg
            )
            radial_velocity = doppler_from_index(doppler_index, cfg)
            xyz_radar = spherical_to_cartesian(
                range_m,
                azimuth_rad,
                elevation_rad,
            )
            xyz_lidar = radar_to_lidar(xyz_radar, cfg)

            detections.append(
                RadarDetection(
                    range_index=r,
                    doppler_index=doppler_index,
                    elevation_index=e,
                    azimuth_index=a,
                    power=power,
                    range_m=range_m,
                    radial_velocity_mps=radial_velocity,
                    azimuth_rad=azimuth_rad,
                    elevation_rad=elevation_rad,
                    xyz_radar_m=xyz_radar,
                    xyz_lidar_m=xyz_lidar,
                )
            )

        return detections
