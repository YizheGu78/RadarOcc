from __future__ import annotations

from pathlib import Path

import numpy as np

from tradition.core.config import KRadarConfig
from tradition.core.interfaces import RadarMeasurementReader


class KRadarTensorReader(RadarMeasurementReader):
    """Read K-Radar 4D power tensors without changing their source data."""

    def __init__(self, radar_cfg: KRadarConfig | None = None) -> None:
        self.radar_cfg = radar_cfg or KRadarConfig()

    def read(self, path: str | Path) -> np.ndarray:
        path = Path(path)

        if path.suffix.lower() == ".npy":
            cube = np.load(path, mmap_mode="r", allow_pickle=False)
        elif path.suffix.lower() == ".mat":
            from scipy.io import loadmat

            content = loadmat(path, variable_names=("arrDREA",))
            if "arrDREA" not in content:
                raise KeyError(f"{path} does not contain arrDREA")
            cube = content["arrDREA"]
        else:
            raise ValueError(f"Unsupported radar tensor format: {path.suffix}")

        cube = np.asarray(cube)
        if cube.ndim != 4:
            raise ValueError(f"Expected [D,R,E,A], got shape={cube.shape} for {path}")

        d, _, e, a = cube.shape
        cfg = self.radar_cfg
        warnings = []
        if d != cfg.expected_doppler_bins:
            warnings.append(f"D={d} (configured {cfg.expected_doppler_bins})")
        if e != cfg.expected_elevation_bins:
            warnings.append(f"E={e} (configured {cfg.expected_elevation_bins})")
        if a != cfg.expected_azimuth_bins:
            warnings.append(f"A={a} (configured {cfg.expected_azimuth_bins})")
        if warnings:
            raise ValueError(
                "Tensor geometry does not match KRadarConfig: "
                + ", ".join(warnings)
                + ". Adjust KRadarConfig instead of silently reinterpreting bins."
            )
        return cube
