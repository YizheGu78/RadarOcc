from __future__ import annotations

import numpy as np

from tradition.core.config import KRadarConfig
from tradition.core.geometry import radar_to_lidar
from tradition.core.interfaces import TargetDetector
from tradition.core.types import RPCPointCloudFrame, RadarDetection


class RPCPointTargetDetector(TargetDetector):
    """Adapt an already density-reduced RPC point list to radar detections.

    No CFAR or further Top-K thinning is applied: ``pc01p`` is already the
    detection-like representation selected from the original 4-D tensor.
    """

    def __init__(self, radar_cfg: KRadarConfig | None = None) -> None:
        self.radar_cfg = radar_cfg or KRadarConfig()

    def detect(self, measurement: RPCPointCloudFrame) -> list[RadarDetection]:
        if not isinstance(measurement, RPCPointCloudFrame):
            raise TypeError(
                "RPCPointTargetDetector expects RPCPointCloudFrame, "
                f"got {type(measurement).__name__}"
            )

        cfg = self.radar_cfg
        finite = (
            np.all(np.isfinite(measurement.xyz_radar_m), axis=1)
            & np.isfinite(measurement.power)
            & np.isfinite(measurement.radial_velocity_mps)
            & np.isfinite(measurement.range_m)
            & np.isfinite(measurement.azimuth_rad)
            & np.isfinite(measurement.elevation_rad)
        )
        valid = (
            finite
            & (measurement.range_m > 0.0)
            & (measurement.range_ind >= 0)
            & (measurement.azimuth_ind >= 0)
            & (measurement.azimuth_ind < cfg.expected_azimuth_bins)
            & (measurement.elevation_ind >= 0)
            & (measurement.elevation_ind < cfg.expected_elevation_bins)
        )

        detections: list[RadarDetection] = []
        for index in np.flatnonzero(valid):
            xyz_radar = np.asarray(
                measurement.xyz_radar_m[index], dtype=np.float64
            )
            detections.append(
                RadarDetection(
                    range_index=int(measurement.range_ind[index]),
                    # RPC stores physical Doppler, not a Doppler-bin index.
                    doppler_index=-1,
                    elevation_index=int(measurement.elevation_ind[index]),
                    azimuth_index=int(measurement.azimuth_ind[index]),
                    power=float(measurement.power[index]),
                    range_m=float(measurement.range_m[index]),
                    radial_velocity_mps=float(
                        measurement.radial_velocity_mps[index]
                    ),
                    azimuth_rad=float(measurement.azimuth_rad[index]),
                    elevation_rad=float(measurement.elevation_rad[index]),
                    xyz_radar_m=xyz_radar,
                    xyz_lidar_m=radar_to_lidar(xyz_radar, cfg),
                )
            )
        return detections
