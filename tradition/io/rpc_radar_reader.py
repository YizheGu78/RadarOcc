from __future__ import annotations

from pathlib import Path

import numpy as np

from tradition.core.interfaces import RadarMeasurementReader
from tradition.core.types import RPCPointCloudFrame


class KRadarRPCReader(RadarMeasurementReader):
    """Read one Enhanced K-Radar ``pc01p``/RPC NPY point cloud."""

    NUM_COLUMNS = 11
    COLUMN_NAMES = (
        "x",
        "y",
        "z",
        "power",
        "doppler",
        "range",
        "azimuth",
        "elevation",
        "range_index",
        "azimuth_index",
        "elevation_index",
    )

    def read(self, path: str | Path) -> RPCPointCloudFrame:
        path = Path(path)
        if path.suffix.lower() != ".npy":
            raise ValueError(f"RPC reader expects .npy, got {path.suffix}: {path}")

        points = np.load(path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != self.NUM_COLUMNS:
            raise ValueError(
                "Enhanced K-Radar RPC must have shape [N,11] with columns "
                f"{self.COLUMN_NAMES}; got {points.shape} in {path}"
            )
        points = np.asarray(points, dtype=np.float64)

        return RPCPointCloudFrame(
            xyz_radar_m=points[:, 0:3],
            power=points[:, 3],
            radial_velocity_mps=points[:, 4],
            range_m=points[:, 5],
            azimuth_rad=points[:, 6],
            elevation_rad=points[:, 7],
            range_ind=np.rint(points[:, 8]).astype(np.int64),
            azimuth_ind=np.rint(points[:, 9]).astype(np.int64),
            elevation_ind=np.rint(points[:, 10]).astype(np.int64),
        )
