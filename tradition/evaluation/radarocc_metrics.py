from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


GRID_SHAPE_XYZ = (128, 128, 14)


def _simplify_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    result = np.zeros_like(labels, dtype=np.uint8)
    result[labels == 1] = 1
    result[(labels >= 2) & (labels != 255)] = 2
    return result


def load_prediction_sparse_zyx(path: str | Path) -> np.ndarray:
    sparse = np.load(path, allow_pickle=False)
    dense = np.zeros(GRID_SHAPE_XYZ, dtype=np.uint8)
    if sparse.size == 0:
        return dense
    if sparse.ndim != 2 or sparse.shape[1] < 4:
        raise ValueError(f"Expected [z,y,x,class] rows, got {sparse.shape}")
    zyx = np.rint(sparse[:, :3]).astype(np.int64)
    xyz = zyx[:, [2, 1, 0]]
    labels = _simplify_labels(np.rint(sparse[:, -1]).astype(np.int64))
    valid = np.all((xyz >= 0) & (xyz < np.asarray(GRID_SHAPE_XYZ)), axis=1)
    xyz, labels = xyz[valid], labels[valid]
    dense[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = labels
    return dense


def load_gt_sparse_xyz(path: str | Path, coordinate_order: str = "xyz") -> np.ndarray:
    sparse = np.load(path, allow_pickle=False)
    dense = np.zeros(GRID_SHAPE_XYZ, dtype=np.uint8)
    if sparse.size == 0:
        return dense
    if sparse.ndim != 2 or sparse.shape[1] < 4:
        raise ValueError(f"Expected sparse GT rows, got {sparse.shape}")
    coords = np.rint(sparse[:, :3]).astype(np.int64)
    if coordinate_order == "zyx":
        xyz = coords[:, [2, 1, 0]]
    elif coordinate_order == "xyz":
        xyz = coords
    else:
        raise ValueError("coordinate_order must be 'xyz' or 'zyx'.")
    labels = _simplify_labels(np.rint(sparse[:, -1]).astype(np.int64))
    valid = np.all((xyz >= 0) & (xyz < np.asarray(GRID_SHAPE_XYZ)), axis=1)
    xyz, labels = xyz[valid], labels[valid]
    dense[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = labels
    return dense


@dataclass
class MetricResult:
    range_m: float
    occupied_iou: float
    static_iou: float
    dynamic_iou: float
    mean_iou: float


class RadarOccMetrics:
    """Three-class metrics aligned with RadarOcc's 0/1/2 setup."""

    @staticmethod
    def _iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
        intersection = np.logical_and(mask_pred, mask_gt).sum()
        union = np.logical_or(mask_pred, mask_gt).sum()
        return float(intersection / union) if union else float("nan")

    def evaluate(
        self,
        prediction_xyz: np.ndarray,
        ground_truth_xyz: np.ndarray,
        ranges_m: tuple[float, ...] = (12.8, 25.6, 51.2),
    ) -> list[MetricResult]:
        pred = np.asarray(prediction_xyz)
        gt = np.asarray(ground_truth_xyz)
        if pred.shape != GRID_SHAPE_XYZ or gt.shape != GRID_SHAPE_XYZ:
            raise ValueError("Both dense grids must be (128,128,14).")

        results = []
        for range_m in ranges_m:
            x_bins = min(GRID_SHAPE_XYZ[0], int(round(range_m / 0.4)))
            pred_roi = pred[:x_bins]
            gt_roi = gt[:x_bins]
            occ_iou = self._iou(pred_roi > 0, gt_roi > 0)
            static_iou = self._iou(pred_roi == 1, gt_roi == 1)
            dynamic_iou = self._iou(pred_roi == 2, gt_roi == 2)
            mean_iou = float(np.nanmean([static_iou, dynamic_iou]))
            results.append(
                MetricResult(
                    range_m=range_m,
                    occupied_iou=occ_iou,
                    static_iou=static_iou,
                    dynamic_iou=dynamic_iou,
                    mean_iou=mean_iou,
                )
            )
        return results
