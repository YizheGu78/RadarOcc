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


def load_gt_sparse_xyz(path: str | Path, coordinate_order: str = "xyz") -> np.ndarray:
    """Load RadarOcc sparse GT into a dense [X,Y,Z] three-class grid."""
    sparse = np.load(path, allow_pickle=False)
    dense = np.zeros(GRID_SHAPE_XYZ, dtype=np.uint8)
    if sparse.size == 0:
        return dense
    if sparse.ndim != 2 or sparse.shape[1] < 4:
        raise ValueError(f"Expected sparse GT rows, got {sparse.shape}")

    coords = np.rint(sparse[:, :3]).astype(np.int64)
    if coordinate_order == "xyz":
        xyz = coords
    elif coordinate_order == "zyx":
        xyz = coords[:, [2, 1, 0]]
    else:
        raise ValueError("coordinate_order must be 'xyz' or 'zyx'.")

    labels = _simplify_labels(np.rint(sparse[:, -1]).astype(np.int64))
    valid = np.all((xyz >= 0) & (xyz < np.asarray(GRID_SHAPE_XYZ)), axis=1)
    xyz, labels = xyz[valid], labels[valid]
    if xyz.size:
        dense[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = labels
    return dense


@dataclass(frozen=True)
class MetricResult:
    range_m: float
    sc_iou: float
    ssc_miou: float
    background_iou: float
    foreground_iou: float


class RadarOccMetricAccumulator:
    """Accumulate confusion matrices exactly at dataset level.

    Labels follow the current RadarOcc K-Radar setup:
      0 = free, 1 = background/static occupied, 2 = foreground occupied.

    SC IoU collapses classes 1/2 into occupied. SSC mIoU is the mean of
    Background IoU and Foreground IoU, matching the table convention used in
    the RadarOcc paper/reproduction results.
    """

    def __init__(self, ranges_m: tuple[float, ...] = (12.8, 25.6, 51.2)) -> None:
        self.ranges_m = ranges_m
        self._confusions = {
            float(r): np.zeros((3, 3), dtype=np.int64) for r in ranges_m
        }
        self.frames = 0

    @staticmethod
    def _confusion(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        pred_flat = pred.reshape(-1).astype(np.int64)
        gt_flat = gt.reshape(-1).astype(np.int64)
        valid = (gt_flat >= 0) & (gt_flat <= 2) & (pred_flat >= 0) & (pred_flat <= 2)
        encoded = gt_flat[valid] * 3 + pred_flat[valid]
        return np.bincount(encoded, minlength=9).reshape(3, 3)

    def update(self, prediction_xyz: np.ndarray, ground_truth_xyz: np.ndarray) -> None:
        pred = np.asarray(prediction_xyz)
        gt = np.asarray(ground_truth_xyz)
        if pred.shape != GRID_SHAPE_XYZ or gt.shape != GRID_SHAPE_XYZ:
            raise ValueError(
                f"Both dense grids must be {GRID_SHAPE_XYZ}, got {pred.shape} and {gt.shape}."
            )

        for range_m in self.ranges_m:
            x_bins = min(GRID_SHAPE_XYZ[0], int(round(range_m / 0.4)))
            self._confusions[float(range_m)] += self._confusion(
                pred[:x_bins], gt[:x_bins]
            )
        self.frames += 1

    @staticmethod
    def _class_iou(cm: np.ndarray, cls: int) -> float:
        tp = float(cm[cls, cls])
        union = float(cm[cls, :].sum() + cm[:, cls].sum() - cm[cls, cls])
        return tp / union if union else float("nan")

    @staticmethod
    def _occupied_iou(cm: np.ndarray) -> float:
        tp = float(cm[1:, 1:].sum())
        fp = float(cm[0, 1:].sum())
        fn = float(cm[1:, 0].sum())
        union = tp + fp + fn
        return tp / union if union else float("nan")

    def results(self) -> list[MetricResult]:
        result: list[MetricResult] = []
        for range_m in self.ranges_m:
            cm = self._confusions[float(range_m)]
            bg = self._class_iou(cm, 1)
            fg = self._class_iou(cm, 2)
            result.append(
                MetricResult(
                    range_m=float(range_m),
                    sc_iou=self._occupied_iou(cm),
                    ssc_miou=float(np.nanmean([bg, fg])),
                    background_iou=bg,
                    foreground_iou=fg,
                )
            )
        return result


class RadarOccMetrics:
    """Single-call compatibility wrapper around the dataset accumulator."""

    def evaluate(
        self,
        prediction_xyz: np.ndarray,
        ground_truth_xyz: np.ndarray,
        ranges_m: tuple[float, ...] = (12.8, 25.6, 51.2),
    ) -> list[MetricResult]:
        accumulator = RadarOccMetricAccumulator(ranges_m)
        accumulator.update(prediction_xyz, ground_truth_xyz)
        return accumulator.results()
