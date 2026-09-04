from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from tradition.core.config import GridConfig
from tradition.core.geometry import xyz_to_voxel
from tradition.core.types import RadarDetection
from tradition.semantics.object_classifier import FEATURE_NAMES, ObjectCandidate


@dataclass(frozen=True)
class ClusterLabellingConfig:
    """Convert RadarOcc train GT into cluster-level objectness targets."""

    match_radius_voxels: int = 2
    positive_foreground_fraction: float = 0.20
    negative_foreground_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.match_radius_voxels < 0:
            raise ValueError("GT match radius must be non-negative.")
        if not (
            0.0
            <= self.negative_foreground_fraction
            < self.positive_foreground_fraction
            <= 1.0
        ):
            raise ValueError("Invalid positive/negative foreground fractions.")


class RadarOccClusterLabeller:
    """Label proposals using train-set semantic GT; never used at inference."""

    def __init__(
        self,
        grid_cfg: GridConfig | None = None,
        config: ClusterLabellingConfig | None = None,
    ) -> None:
        self.grid_cfg = grid_cfg or GridConfig()
        self.config = config or ClusterLabellingConfig()

    def label(
        self,
        candidate: ObjectCandidate,
        detections: Sequence[RadarDetection],
        gt_labels_xyz: np.ndarray,
    ) -> int | None:
        if gt_labels_xyz.shape != self.grid_cfg.shape_xyz:
            raise ValueError(
                f"GT shape {gt_labels_xyz.shape} != {self.grid_cfg.shape_xyz}."
            )
        matches: list[bool] = []
        radius = self.config.match_radius_voxels
        sx, sy, sz = self.grid_cfg.shape_xyz
        for index in candidate.indices:
            voxel = xyz_to_voxel(detections[int(index)].xyz_lidar_m, self.grid_cfg)
            if voxel is None:
                continue
            x, y, z = voxel
            region = gt_labels_xyz[
                max(0, x - radius) : min(sx, x + radius + 1),
                max(0, y - radius) : min(sy, y + radius + 1),
                max(0, z - radius) : min(sz, z + radius + 1),
            ]
            matches.append(bool(np.any(region == 2)))
        if not matches:
            return None
        foreground_fraction = float(np.mean(matches))
        if foreground_fraction >= self.config.positive_foreground_fraction:
            return 1
        if foreground_fraction <= self.config.negative_foreground_fraction:
            return 0
        return None


def train_random_forest_objectness(
    features: np.ndarray,
    targets: np.ndarray,
    output_path: str | Path,
    n_estimators: int = 200,
    random_state: int = 13,
    metadata: dict | None = None,
) -> Path:
    """Fit and persist the non-neural cluster objectness classifier."""
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as error:
        raise RuntimeError("Training requires scikit-learn and joblib.") from error

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int8)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected features [N,{len(FEATURE_NAMES)}], got {x.shape}.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Training data must contain background and foreground clusters.")
    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
        oob_score=True,
    )
    estimator.fit(x, y)
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "format": "radarocc-classical-objectness-v1",
            "feature_names": FEATURE_NAMES,
            "estimator": estimator,
            "metadata": {
                **(metadata or {}),
                "sample_count": int(len(y)),
                "background_samples": int(np.count_nonzero(y == 0)),
                "foreground_samples": int(np.count_nonzero(y == 1)),
                "oob_score": float(estimator.oob_score_),
                "n_estimators": int(n_estimators),
            },
        },
        path,
    )
    return path
