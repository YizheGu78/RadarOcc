from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tradition.evaluation.radarocc_metrics import (
    RadarOccMetrics,
    load_gt_sparse_xyz,
    load_prediction_sparse_zyx,
    radarocc_metric_dict,
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
        {
            key: float(np.round(value, 3))
            for key, value in radarocc_metric_dict(results).items()
        }
    )


if __name__ == "__main__":
    main()
