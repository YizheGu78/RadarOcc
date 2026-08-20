from __future__ import annotations

from pathlib import Path

import numpy as np

from tradition.core.interfaces import RadarTensorReader
from tradition.core.types import SparseRadarFrame


class RadarOccSparseReader(RadarTensorReader):
    """Read RadarOcc EAsparse_*.npz as a sparse radar measurement frame."""

    REQUIRED_KEYS = (
        "range_ind",
        "elevation_ind",
        "azimuth_ind",
        "power_val",
    )

    def read(self, path: str | Path) -> SparseRadarFrame:
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError(
                f"Sparse RadarOcc reader expects .npz, got {path.suffix}: {path}"
            )

        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in self.REQUIRED_KEYS if key not in data]
            if missing:
                raise KeyError(f"{path} is missing NPZ keys: {missing}")

            range_ind = np.asarray(data["range_ind"]).reshape(-1).astype(np.int64)
            elevation_ind = (
                np.asarray(data["elevation_ind"]).reshape(-1).astype(np.int64)
            )
            azimuth_ind = (
                np.asarray(data["azimuth_ind"]).reshape(-1).astype(np.int64)
            )
            descriptor = np.asarray(data["power_val"], dtype=np.float64)

        n = range_ind.size
        if elevation_ind.size != n or azimuth_ind.size != n:
            raise ValueError(
                "Sparse coordinate arrays must have equal length: "
                f"R={n}, E={elevation_ind.size}, A={azimuth_ind.size}"
            )

        if descriptor.ndim != 2:
            raise ValueError(
                f"Expected power_val to be 2-D, got {descriptor.shape} in {path}"
            )
        if descriptor.shape == (n, 8):
            descriptor = descriptor.T
        if descriptor.shape != (8, n):
            raise ValueError(
                f"Expected power_val shape [8,N], got {descriptor.shape} for N={n}"
            )

        return SparseRadarFrame(
            range_ind=range_ind,
            elevation_ind=elevation_ind,
            azimuth_ind=azimuth_ind,
            descriptor=descriptor,
        )
