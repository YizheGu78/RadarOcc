from pathlib import Path
from decimal import Decimal
import argparse
import csv
import math
import statistics


def scene_sort_key(path: Path):
    """
    让场景按 1, 2, 3, ..., 10, 11 排序，
    而不是字符串的 1, 10, 11, 2...
    """
    try:
        return (0, int(path.name))
    except ValueError:
        return (1, path.name)


def find_timestamp_file(scene_dir: Path, target_file: str):
    """
    在场景目录下递归寻找指定 timestamp 文件。

    支持例如：
        scene/1/os2-64.txt
        scene/1/time_info/os2-64.txt
    """
    candidates = list(scene_dir.rglob(target_file))

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:
        print(
            f"[WARNING] Scene {scene_dir.name}: "
            f"found multiple {target_file}, using {candidates[0]}"
        )

    return candidates[0]


def read_timestamps(file_path: Path):
    """
    读取 K-Radar time_info timestamp。

    官方格式类似：
        os2-64_0.pcd, 1643184534.6934597
        os2-64_1.pcd, 1643184534.7934512

    返回 Decimal timestamp 列表。
    """
    timestamps = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                # K-Radar 官方格式：filename, timestamp
                if "," in line:
                    timestamp_str = line.split(",")[-1].strip()

                # 顺便兼容只有 timestamp 一列的情况
                else:
                    timestamp_str = line.split()[-1].strip()

                timestamp = Decimal(timestamp_str)
                timestamps.append(timestamp)

            except Exception:
                print(
                    f"[WARNING] Skip invalid line: "
                    f"{file_path}:{line_number} -> {line}"
                )

    return timestamps


def calculate_statistics(timestamps):
    """
    计算与截图相同的统计量。
    """

    n_frames = len(timestamps)

    if n_frames < 2:
        return {
            "frame_count": n_frames,
            "interval_count": 0,
            "mean_dt_s": math.nan,
            "median_dt_s": math.nan,
            "population_std_dt_s": math.nan,
            "sample_std_dt_s": math.nan,
            "min_dt_s": math.nan,
            "max_dt_s": math.nan,
            "mean_fps_hz": math.nan,
        }

    # 用 Decimal 先做时间戳差，避免 1.6e9 量级 timestamp
    # 直接 float 相减带来的额外精度损失
    dt = [
        float(timestamps[i + 1] - timestamps[i])
        for i in range(n_frames - 1)
    ]

    mean_dt = statistics.mean(dt)
    median_dt = statistics.median(dt)

    # 总体标准差，对应 numpy.std(..., ddof=0)
    population_std = statistics.pstdev(dt)

    # 样本标准差，对应 numpy.std(..., ddof=1)
    sample_std = (
        statistics.stdev(dt)
        if len(dt) >= 2
        else math.nan
    )

    min_dt = min(dt)
    max_dt = max(dt)

    # 与你截图中的计算方式一致
    mean_fps = 1.0 / mean_dt if mean_dt > 0 else math.nan

    return {
        "frame_count": n_frames,
        "interval_count": len(dt),
        "mean_dt_s": mean_dt,
        "median_dt_s": median_dt,
        "population_std_dt_s": population_std,
        "sample_std_dt_s": sample_std,
        "min_dt_s": min_dt,
        "max_dt_s": max_dt,
        "mean_fps_hz": mean_fps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate per-scene K-Radar timestamp statistics."
    )

    parser.add_argument(
        "--root",
        type=str,
        default="data/K-Radar-timeinfo",
        help="Root directory containing all K-Radar scenes.",
    )

    parser.add_argument(
        "--target-file",
        type=str,
        default="os2-64.txt",
        help=(
            "Timestamp file used for statistics. "
            "Default: os2-64.txt"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/K-Radar-timeinfo/scene_time_statistics.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(
            f"Root directory does not exist: {root}"
        )

    scene_dirs = [
        p for p in root.iterdir()
        if p.is_dir()
    ]

    scene_dirs.sort(key=scene_sort_key)

    results = []

    print("=" * 90)
    print(f"K-Radar time statistics")
    print(f"Root       : {root}")
    print(f"Target file: {args.target_file}")
    print("=" * 90)

    for scene_dir in scene_dirs:

        timestamp_file = find_timestamp_file(
            scene_dir,
            args.target_file
        )

        if timestamp_file is None:
            print(
                f"[SKIP] Scene {scene_dir.name}: "
                f"{args.target_file} not found"
            )
            continue

        timestamps = read_timestamps(timestamp_file)

        stats = calculate_statistics(timestamps)

        row = {
            "scene": scene_dir.name,
            **stats,
        }

        results.append(row)

        print(
            f"Scene {scene_dir.name:>3} | "
            f"frames={stats['frame_count']:>5} | "
            f"intervals={stats['interval_count']:>5} | "
            f"mean_dt={stats['mean_dt_s']:.7f} s | "
            f"median_dt={stats['median_dt_s']:.7f} s | "
            f"std={stats['population_std_dt_s']:.7f} s | "
            f"min={stats['min_dt_s']:.7f} s | "
            f"max={stats['max_dt_s']:.7f} s | "
            f"fps={stats['mean_fps_hz']:.5f} Hz"
        )

    if len(results) == 0:
        print("\nNo valid scenes found.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "scene",
        "frame_count",
        "interval_count",
        "mean_dt_s",
        "median_dt_s",
        "population_std_dt_s",
        "sample_std_dt_s",
        "min_dt_s",
        "max_dt_s",
        "mean_fps_hz",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 90)
    print(f"Finished.")
    print(f"Scenes processed: {len(results)}")
    print(f"CSV saved to: {output_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()