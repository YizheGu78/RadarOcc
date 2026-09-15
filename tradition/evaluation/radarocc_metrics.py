from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


GRID_SHAPE_XYZ = (128, 128, 14)
IGNORE_LABEL = 255
RADAROCC_RANGES_M = (51.2, 25.6, 12.8)
RADAROCC_REGION_SLICES_XYZ = {
    51.2: (slice(0, 128), slice(0, 128), slice(0, 14)),
    25.6: (slice(0, 64), slice(32, 96), slice(0, 14)),
    12.8: (slice(0, 32), slice(48, 80), slice(0, 14)),
}


def _simplify_labels(
    labels: np.ndarray,
    *,
    preserve_ignore: bool = False,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    result = np.zeros_like(labels, dtype=np.uint8)
    result[labels == 1] = 1
    result[(labels >= 2) & (labels != IGNORE_LABEL)] = 2
    if preserve_ignore:
        result[labels == IGNORE_LABEL] = IGNORE_LABEL
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

    labels = _simplify_labels(
        np.rint(sparse[:, -1]).astype(np.int64),
        preserve_ignore=True,
    )
    valid = np.all((xyz >= 0) & (xyz < np.asarray(GRID_SHAPE_XYZ)), axis=1)
    xyz, labels = xyz[valid], labels[valid]
    if xyz.size:
        dense[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = labels
    return dense


def load_prediction_sparse_zyx(path: str | Path) -> np.ndarray:
    """Load a writer-produced ``[z,y,x,class]`` file into dense ``[X,Y,Z]``."""
    sparse = np.load(path, allow_pickle=False)
    dense = np.zeros(GRID_SHAPE_XYZ, dtype=np.uint8)
    if sparse.size == 0:
        return dense
    if sparse.ndim != 2 or sparse.shape[1] < 4:
        raise ValueError(f"Expected sparse prediction rows, got {sparse.shape}")
    xyz = np.rint(sparse[:, :3]).astype(np.int64)[:, [2, 1, 0]]
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
    free_iou: float
    background_iou: float
    foreground_iou: float


def radarocc_metric_dict(results: list[MetricResult]) -> dict[str, float]:
    """Return exactly the metric keys emitted by RadarOcc evaluation."""
    by_range = {result.range_m: result for result in results}
    missing = set(RADAROCC_RANGES_M) - set(by_range)
    if missing:
        raise ValueError(f"Missing RadarOcc metric ranges: {sorted(missing)}")

    full = by_range[51.2]
    range1 = by_range[25.6]
    range2 = by_range[12.8]
    return {
        "SC_non-empty": full.sc_iou,
        "SC1_non-empty": range1.sc_iou,
        "SC2_non-empty": range2.sc_iou,
        "SSC_free": full.free_iou,
        "SSC_Background": full.background_iou,
        "SSC_Foreground": full.foreground_iou,
        "SSC_mean": full.ssc_miou,
        "SSC1_free": range1.free_iou,
        "SSC1_Background": range1.background_iou,
        "SSC1_Foreground": range1.foreground_iou,
        "SSC1_mean": range1.ssc_miou,
        "SSC2_free": range2.free_iou,
        "SSC2_Background": range2.background_iou,
        "SSC2_Foreground": range2.foreground_iou,
        "SSC2_mean": range2.ssc_miou,
    }


class RadarOccMetricAccumulator:
    """Accumulate confusion matrices exactly at dataset level.

    Labels follow the current RadarOcc K-Radar setup:
      0 = free, 1 = background occupied, 2 = foreground occupied.

    SC IoU collapses classes 1/2 into occupied. SSC mIoU is the mean of
    Background IoU and Foreground IoU, matching the table convention used in
    the RadarOcc paper/reproduction results.
    """

    def __init__(
        self,
        ranges_m: tuple[float, ...] = RADAROCC_RANGES_M,
    ) -> None:
        unsupported = set(ranges_m) - set(RADAROCC_REGION_SLICES_XYZ)
        if unsupported:
            raise ValueError(
                f"Unsupported RadarOcc metric ranges: {sorted(unsupported)}"
            )
        self.ranges_m = ranges_m
        self._confusions = {
            float(r): np.zeros((3, 3), dtype=np.int64) for r in ranges_m
        }
        self.frames = 0

    @staticmethod
    def _confusion(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        pred_flat = pred.reshape(-1).astype(np.int64)
        gt_flat = gt.reshape(-1).astype(np.int64)
        valid = (
            (gt_flat != IGNORE_LABEL)
            & (gt_flat >= 0)
            & (gt_flat <= 2)
            & (pred_flat >= 0)
            & (pred_flat <= 2)
        )
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
            region = RADAROCC_REGION_SLICES_XYZ[float(range_m)]
            self._confusions[float(range_m)] += self._confusion(
                pred[region], gt[region]
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
            free = self._class_iou(cm, 0)
            bg = self._class_iou(cm, 1)
            fg = self._class_iou(cm, 2)
            result.append(
                MetricResult(
                    range_m=float(range_m),
                    sc_iou=self._occupied_iou(cm),
                    ssc_miou=(bg + fg) / 2.0,
                    free_iou=free,
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
        ranges_m: tuple[float, ...] = RADAROCC_RANGES_M,
    ) -> list[MetricResult]:
        accumulator = RadarOccMetricAccumulator(ranges_m)
        accumulator.update(prediction_xyz, ground_truth_xyz)
        return accumulator.results()
