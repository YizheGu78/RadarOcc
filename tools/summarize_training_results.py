#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Summarize MMCV/MMDetection validation metrics by epoch.

Examples:
    python tools/summarize_training_results.py work_dirs/exp_a
    python tools/summarize_training_results.py work_dirs/exp_a --top-k 5
    python tools/summarize_training_results.py work_dirs/exp_a --save-csv
    python tools/summarize_training_results.py work_dirs/exp_a --all-metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

NON_METRIC_KEYS = {
    "mode",
    "epoch",
    "iter",
    "inner_iter",
    "lr",
    "time",
    "data_time",
    "memory",
    "loss",
    "grad_norm",
    "step",
}

DEFAULT_COLUMNS = [
    "SC_non-empty",
    "SC1_non-empty",
    "SC2_non-empty",
    "SSC_mean",
    "SSC1_mean",
    "SSC2_mean",
    "SSC_fine_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize MMCV/MMDetection validation metrics"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="work_dir, one .log.json file, or one .log file",
    )
    parser.add_argument(
        "--best-metric",
        default="SSC_mean",
        help="metric used to select the best epoch (default: SSC_mean)",
    )
    parser.add_argument(
        "--rule",
        choices=("greater", "less"),
        default="greater",
        help="whether a greater or smaller value is better",
    )
    parser.add_argument(
        "--columns",
        default=None,
        help="comma-separated metric names to display",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="display every validation metric found in the logs",
    )
    parser.add_argument(
        "--save-csv",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "export CSV; without a path it is written to "
            "work_dir/training_summary.csv"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="show the top K epochs according to --best-metric",
    )
    return parser.parse_args()


def find_json_logs(path: Path) -> tuple[Path, list[Path]]:
    """Return work_dir and all relevant JSON log files.

    Reading every JSON log is important when training was resumed because MMCV
    creates a new log file for each run.
    """
    path = path.expanduser().resolve()

    if path.is_file():
        if not path.name.endswith(".log.json"):
            raise ValueError(f"expected a .log.json file: {path}")
        return path.parent, [path]

    if not path.is_dir():
        raise FileNotFoundError(f"path does not exist: {path}")

    logs = sorted(
        path.glob("*.log.json"),
        key=lambda item: (item.stat().st_mtime, item.name),
    )
    if not logs:
        raise FileNotFoundError(f"no *.log.json files found in: {path}")

    return path, logs


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def looks_like_metric(key: str, value: Any) -> bool:
    if key in NON_METRIC_KEYS or not is_number(value):
        return False

    lower = key.lower()
    if lower.startswith(("raw_", "loss_", "loss")):
        return False

    radarocc_prefixes = (
        "sc_",
        "sc1_",
        "sc2_",
        "ssc_",
        "ssc1_",
        "ssc2_",
        "ssc_fine_",
    )
    common_tokens = (
        "iou",
        "miou",
        "map",
        "accuracy",
        "acc",
        "recall",
        "precision",
        "f1",
        "wer",
        "cer",
        "dice",
        "auc",
        "epe",
    )

    # Use the lower-cased key here. RadarOcc writes upper-case keys such as
    # SC_non-empty and SSC_mean.
    return lower.startswith(radarocc_prefixes) or any(
        token in lower for token in common_tokens
    )


def load_epoch_metrics(
    json_logs: list[Path],
) -> tuple[dict[int, dict[str, float]], set[str]]:
    """Read all JSON-lines logs and merge validation metrics by epoch.

    Later log files overwrite the same metric for the same epoch. This makes
    resumed runs behave as expected.
    """
    epochs: dict[int, dict[str, float]] = {}
    observed_keys: set[str] = set()

    for json_log in json_logs:
        with json_log.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"warning: skipped malformed JSON at "
                        f"{json_log}:{line_number}",
                        file=sys.stderr,
                    )
                    continue

                observed_keys.update(record.keys())
                epoch = record.get("epoch")

                if isinstance(epoch, bool):
                    continue
                if isinstance(epoch, float) and epoch.is_integer():
                    epoch = int(epoch)
                if not isinstance(epoch, int):
                    continue

                metrics: dict[str, float] = {}
                for key, value in record.items():
                    if not looks_like_metric(key, value):
                        continue
                    value = float(value)
                    if math.isfinite(value):
                        metrics[key] = value

                if metrics:
                    epochs.setdefault(epoch, {}).update(metrics)

    return epochs, observed_keys


def all_metric_names(
    epochs: dict[int, dict[str, float]],
) -> list[str]:
    names: set[str] = set()
    for metrics in epochs.values():
        names.update(metrics)
    return sorted(names)


def choose_columns(
    args: argparse.Namespace,
    available: list[str],
) -> list[str]:
    if args.all_metrics:
        return available

    if args.columns:
        requested = [
            item.strip()
            for item in args.columns.split(",")
            if item.strip()
        ]
    else:
        requested = DEFAULT_COLUMNS

    columns = [name for name in requested if name in available]
    if not columns and args.best_metric in available:
        columns = [args.best_metric]
    return columns or available


def print_table(
    epochs: dict[int, dict[str, float]],
    columns: list[str],
) -> None:
    headers = ["Epoch", *columns]
    rows: list[list[str]] = []

    for epoch in sorted(epochs):
        rows.append(
            [
                str(epoch),
                *[
                    "-"
                    if column not in epochs[epoch]
                    else f"{epochs[epoch][column]:.4f}"
                    for column in columns
                ],
            ]
        )

    widths = [
        max(
            len(headers[index]),
            max((len(row[index]) for row in rows), default=0),
        )
        for index in range(len(headers))
    ]

    print(
        "  ".join(
            headers[index].rjust(widths[index])
            for index in range(len(headers))
        )
    )
    print("  ".join("-" * width for width in widths))

    for row in rows:
        print(
            "  ".join(
                row[index].rjust(widths[index])
                for index in range(len(row))
            )
        )


def rank_epochs(
    epochs: dict[int, dict[str, float]],
    metric: str,
    rule: str,
) -> list[tuple[int, float]]:
    values = [
        (epoch, metrics[metric])
        for epoch, metrics in epochs.items()
        if metric in metrics
    ]
    if not values:
        available = ", ".join(all_metric_names(epochs))
        raise KeyError(
            f"metric not found: {metric}; available metrics: {available}"
        )

    return sorted(
        values,
        key=lambda item: item[1],
        reverse=(rule == "greater"),
    )


def find_checkpoint(
    work_dir: Path,
    epoch: int,
    metric: str,
) -> Path | None:
    normalized_metric = metric.replace("/", "_")

    exact = work_dir / f"best_{normalized_metric}_epoch_{epoch}.pth"
    if exact.exists():
        return exact

    for checkpoint in sorted(work_dir.glob("best_*.pth")):
        if f"epoch_{epoch}" in checkpoint.name:
            return checkpoint

    epoch_checkpoint = work_dir / f"epoch_{epoch}.pth"
    return epoch_checkpoint if epoch_checkpoint.exists() else None


def save_csv(
    path: Path,
    epochs: dict[int, dict[str, float]],
    names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", *names])
        writer.writeheader()
        for epoch in sorted(epochs):
            row: dict[str, Any] = {"epoch": epoch}
            row.update(epochs[epoch])
            writer.writerow(row)


def main() -> int:
    args = parse_args()

    try:
        work_dir, json_logs = find_json_logs(args.path)
        epochs, observed_keys = load_epoch_metrics(json_logs)

        if not epochs:
            key_preview = ", ".join(sorted(observed_keys)[:30])
            raise RuntimeError(
                "no validation metrics were found in the JSON logs. "
                f"Observed keys include: {key_preview}"
            )

        names = all_metric_names(epochs)
        columns = choose_columns(args, names)
        ranking = rank_epochs(epochs, args.best_metric, args.rule)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"experiment directory: {work_dir}")
    print("JSON logs:")
    for json_log in json_logs:
        print(f"  {json_log.name}")
    print(f"epochs found: {min(epochs)} - {max(epochs)}\n")

    print_table(epochs, columns)

    print(f"\nRanking by {args.best_metric} (rule={args.rule}):")
    top_k = max(1, min(args.top_k, len(ranking)))

    for index, (epoch, value) in enumerate(ranking[:top_k], start=1):
        checkpoint = find_checkpoint(
            work_dir,
            epoch,
            args.best_metric,
        )
        print(
            f"{index}. epoch={epoch}, "
            f"{args.best_metric}={value:.4f}"
        )
        print(
            f"   checkpoint: "
            f"{checkpoint if checkpoint else 'not found'}"
        )

    best_epoch, best_value = ranking[0]
    print("\nAll metrics for the best epoch:")
    print(f"epoch: {best_epoch}")
    print(f"{args.best_metric}: {best_value:.4f}")

    for key in sorted(epochs[best_epoch]):
        if key != args.best_metric:
            print(f"{key}: {epochs[best_epoch][key]:.4f}")

    if args.save_csv is not None:
        csv_path = (
            work_dir / "training_summary.csv"
            if args.save_csv == "auto"
            else Path(args.save_csv).expanduser().resolve()
        )
        save_csv(csv_path, epochs, names)
        print(f"\nCSV saved to: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
