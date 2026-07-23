#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional


SECTION_RE = re.compile(
    r"(SSC2|SSC1|SSC|SC2|SC1|SC)\s+Evaluation"
)

ROW_RE = re.compile(
    r"\|\s*(non-empty|free|Background|Foreground|mean)\s*"
    r"\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|"
)

EPOCH_RE = re.compile(r"epoch_(\d+)\.log$")


ALL_FIELDS = [
    "epoch",

    # 12.8 m
    "SC2_non-empty",
    "SSC2_free",
    "SSC2_Background",
    "SSC2_Foreground",
    "SSC2_mean",

    # 25.6 m
    "SC1_non-empty",
    "SSC1_free",
    "SSC1_Background",
    "SSC1_Foreground",
    "SSC1_mean",

    # 51.2 m
    "SC_non-empty",
    "SSC_free",
    "SSC_Background",
    "SSC_Foreground",
    "SSC_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize RadarOcc text evaluation logs containing "
            "SC/SC1/SC2 and SSC/SSC1/SSC2 metrics."
        )
    )

    parser.add_argument(
        "log_dir",
        type=Path,
        help="Directory containing epoch_*.log files.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top epochs to display.",
    )

    parser.add_argument(
        "--sort-by",
        default="SSC_mean",
        choices=[
            "SSC_mean",
            "SSC1_mean",
            "SSC2_mean",
            "SC_non-empty",
            "SC1_non-empty",
            "SC2_non-empty",
            "SSC_Foreground",
            "SSC1_Foreground",
            "SSC2_Foreground",
        ],
        help="Metric used to rank checkpoints.",
    )

    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save all parsed results as CSV.",
    )

    parser.add_argument(
        "--csv-name",
        default="official_eval_summary.csv",
        help="CSV output filename.",
    )

    return parser.parse_args()


def parse_epoch(path: Path) -> int:
    match = EPOCH_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot extract epoch from filename: {path.name}")

    return int(match.group(1))


def parse_log(path: Path) -> Dict[str, Optional[float]]:
    text = path.read_text(encoding="utf-8", errors="replace")

    row: Dict[str, Optional[float]] = {
        field: None for field in ALL_FIELDS
    }
    row["epoch"] = parse_epoch(path)

    section_matches = list(SECTION_RE.finditer(text))

    for index, section_match in enumerate(section_matches):
        section_name = section_match.group(1)

        section_start = section_match.end()
        if index + 1 < len(section_matches):
            section_end = section_matches[index + 1].start()
        else:
            section_end = len(text)

        section_text = text[section_start:section_end]
        values = {
            label: float(value)
            for label, value in ROW_RE.findall(section_text)
        }

        if section_name in {"SC", "SC1", "SC2"}:
            non_empty = values.get("non-empty")
            if non_empty is not None:
                row[f"{section_name}_non-empty"] = non_empty

        elif section_name in {"SSC", "SSC1", "SSC2"}:
            for class_name in (
                "free",
                "Background",
                "Foreground",
                "mean",
            ):
                value = values.get(class_name)
                if value is not None:
                    row[f"{section_name}_{class_name}"] = value

    return row


def format_value(value: Optional[float]) -> str:
    if value is None:
        return "   -   "

    return f"{value:7.3f}"


def print_epoch_table(rows: List[Dict[str, Optional[float]]]) -> None:
    print()
    print("Metrics ordered according to the paper:")
    print("SC2/SSC2 = 12.8 m, SC1/SSC1 = 25.6 m, SC/SSC = 51.2 m")
    print()

    header = (
        f"{'Epoch':>5}  "
        f"{'SC2@12.8':>10} "
        f"{'SC1@25.6':>10} "
        f"{'SC@51.2':>10}  "
        f"{'mIoU2':>8} "
        f"{'mIoU1':>8} "
        f"{'mIoU':>8}"
    )

    print(header)
    print("-" * len(header))

    for row in sorted(rows, key=lambda item: int(item["epoch"])):
        print(
            f"{int(row['epoch']):5d}  "
            f"{format_value(row['SC2_non-empty']):>10} "
            f"{format_value(row['SC1_non-empty']):>10} "
            f"{format_value(row['SC_non-empty']):>10}  "
            f"{format_value(row['SSC2_mean']):>8} "
            f"{format_value(row['SSC1_mean']):>8} "
            f"{format_value(row['SSC_mean']):>8}"
        )


def print_ranking(
    rows: List[Dict[str, Optional[float]]],
    metric: str,
    top_k: int,
    experiment_dir: Path,
) -> List[Dict[str, Optional[float]]]:
    valid_rows = [
        row for row in rows
        if row.get(metric) is not None
    ]

    ranked = sorted(
        valid_rows,
        key=lambda item: float(item[metric]),
        reverse=True,
    )

    print()
    print(f"Ranking by {metric} (greater is better):")

    for rank, row in enumerate(ranked[:top_k], start=1):
        epoch = int(row["epoch"])
        checkpoint = experiment_dir / f"epoch_{epoch}.pth"

        print(
            f"{rank}. epoch={epoch}, "
            f"{metric}={float(row[metric]):.3f}"
        )
        print(f"   checkpoint: {checkpoint}")

    return ranked


def print_best_details(
    best: Dict[str, Optional[float]],
) -> None:
    epoch = int(best["epoch"])

    print()
    print(f"All metrics for the best epoch: {epoch}")
    print()

    print(
        f"{'Range':>8} "
        f"{'SC IoU':>9} "
        f"{'Free':>9} "
        f"{'BG IoU':>9} "
        f"{'FG IoU':>9} "
        f"{'mIoU':>9}"
    )
    print("-" * 58)

    ranges = [
        (
            "12.8 m",
            "SC2_non-empty",
            "SSC2_free",
            "SSC2_Background",
            "SSC2_Foreground",
            "SSC2_mean",
        ),
        (
            "25.6 m",
            "SC1_non-empty",
            "SSC1_free",
            "SSC1_Background",
            "SSC1_Foreground",
            "SSC1_mean",
        ),
        (
            "51.2 m",
            "SC_non-empty",
            "SSC_free",
            "SSC_Background",
            "SSC_Foreground",
            "SSC_mean",
        ),
    ]

    for (
        range_name,
        sc_key,
        free_key,
        bg_key,
        fg_key,
        mean_key,
    ) in ranges:
        print(
            f"{range_name:>8} "
            f"{format_value(best[sc_key]):>9} "
            f"{format_value(best[free_key]):>9} "
            f"{format_value(best[bg_key]):>9} "
            f"{format_value(best[fg_key]):>9} "
            f"{format_value(best[mean_key]):>9}"
        )


def save_csv(
    rows: List[Dict[str, Optional[float]]],
    csv_path: Path,
) -> None:
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=ALL_FIELDS,
        )
        writer.writeheader()

        for row in sorted(
            rows,
            key=lambda item: int(item["epoch"]),
        ):
            writer.writerow(row)

    print()
    print(f"CSV saved to: {csv_path}")


def main() -> None:
    args = parse_args()

    log_dir = args.log_dir.resolve()

    if not log_dir.is_dir():
        raise FileNotFoundError(
            f"Evaluation log directory not found: {log_dir}"
        )

    log_paths = sorted(
        log_dir.glob("epoch_*.log"),
        key=parse_epoch,
    )

    if not log_paths:
        raise FileNotFoundError(
            f"No epoch_*.log files found in: {log_dir}"
        )

    rows = [parse_log(path) for path in log_paths]

    print(f"Evaluation directory: {log_dir}")
    print(f"Logs found: {len(log_paths)}")
    print(
        "Epochs found:",
        min(int(row["epoch"]) for row in rows),
        "-",
        max(int(row["epoch"]) for row in rows),
    )

    print_epoch_table(rows)

    experiment_dir = log_dir.parent
    ranked = print_ranking(
        rows=rows,
        metric=args.sort_by,
        top_k=args.top_k,
        experiment_dir=experiment_dir,
    )

    if ranked:
        print_best_details(ranked[0])
    else:
        raise RuntimeError(
            f"No valid values found for metric: {args.sort_by}"
        )

    missing_counts = {
        field: sum(row.get(field) is None for row in rows)
        for field in ALL_FIELDS
        if field != "epoch"
    }

    missing_fields = {
        field: count
        for field, count in missing_counts.items()
        if count > 0
    }

    if missing_fields:
        print()
        print("Warning: some metrics were not found:")
        for field, count in missing_fields.items():
            print(f"  {field}: missing in {count} log(s)")

    if args.save_csv:
        save_csv(
            rows,
            log_dir / args.csv_name,
        )


if __name__ == "__main__":
    main()
