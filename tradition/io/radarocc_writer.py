from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tradition.core.interfaces import PredictionWriter
from tradition.core.types import FramePrediction


class RadarOccPredictionWriter(PredictionWriter):
    """Write predictions in the current repository's RadarOcc convention."""

    def write(
        self,
        prediction: FramePrediction,
        output_root: str | Path,
        token: str,
    ) -> Path:
        dense = np.asarray(prediction.dense_labels_xyz)
        if dense.shape != (128, 128, 14):
            raise ValueError(
                f"Expected RadarOcc dense shape (128,128,14), got {dense.shape}"
            )
        if not np.all(np.isin(np.unique(dense), [0, 1, 2])):
            raise ValueError("Prediction labels must be 0=free, 1=static, 2=dynamic.")

        frame_dir = Path(output_root) / str(token)
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.save(frame_dir / "pred_dense.npy", dense.astype(np.uint8), allow_pickle=False)

        xyz = np.argwhere(dense != 0).astype(np.int64)
        if xyz.size:
            labels = dense[tuple(xyz.T)].astype(np.int64)[:, None]
            zyx = xyz[:, [2, 1, 0]]
            sparse = np.concatenate([zyx, labels], axis=1)
        else:
            sparse = np.empty((0, 4), dtype=np.int64)

        pred_path = frame_dir / "pred_c.npy"
        np.save(pred_path, sparse, allow_pickle=False)

        metadata = dict(prediction.metadata)
        metadata.update(
            {
                "lidar_token": str(token),
                "dense_order": "xyz",
                "sparse_order": "zyx_class",
                "classes": {"0": "free", "1": "static/background", "2": "dynamic/foreground"},
                "occupied_voxels": int(sparse.shape[0]),
                "detections": len(prediction.detections),
            }
        )
        with (frame_dir / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

        return pred_path
