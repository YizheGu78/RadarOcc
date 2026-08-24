from __future__ import annotations

import argparse
from pathlib import Path

from tradition.evaluation.radarocc_metrics import (
    RadarOccMetrics,
    load_gt_sparse_xyz,
    load_prediction_sparse_zyx,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one traditional pred_c.npy against RadarOcc GT."
    )
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--gt-order", choices=("xyz", "zyx"), default="xyz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred = load_prediction_sparse_zyx(args.prediction)
    gt = load_gt_sparse_xyz(args.ground_truth, coordinate_order=args.gt_order)
    results = RadarOccMetrics().evaluate(pred, gt)
    print(
        "range_m,SC_IoU,SSC_mIoU_BG_FG,ThreeClass_mIoU,"
        "Free_IoU,Background_IoU,Foreground_IoU"
    )
    for result in results:
        print(
            f"{result.range_m:.1f},"
            f"{100*result.sc_iou:.3f},"
            f"{100*result.ssc_miou:.3f},"
            f"{100*result.three_class_miou:.3f},"
            f"{100*result.free_iou:.3f},"
            f"{100*result.background_iou:.3f},"
            f"{100*result.foreground_iou:.3f}"
        )


if __name__ == "__main__":
    main()
