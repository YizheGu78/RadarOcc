#!/usr/bin/env python3
"""
Visualize RadarOcc mean-power top-250 preprocessing with four RD panels:

1. Top-1 Doppler peak of every selected spatial token
2. Top-2 Doppler peak of every selected spatial token
3. Top-3 Doppler peak of every selected spatial token
4. Mean Doppler spectrum of the same 250 selected tokens per range

Important interpretation
------------------------
"Top-1" is defined independently for every selected (range, elevation,
azimuth) token. Since there are 250 selected tokens per range, the Top-1
panel still contains 250 token projections per range. Multiple tokens can
have the same Top-1 Doppler bin and therefore overlap in the same RD cell.

The EAsparse files created with np.argpartition do not necessarily store
the three peaks in descending order. This script sorts the stored top-3
power/index pairs for every token before drawing Top-1, Top-2 and Top-3.

The Mean panel is not a point overlay. Mean power stored in the NPZ is a
scalar and has no Doppler coordinate. Therefore this script returns to the
raw tensor, extracts the complete 64-bin Doppler spectrum of the selected
250 tokens, and averages those 250 spectra for each range.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.io import loadmat


EPS = np.finfo(np.float32).tiny
EXPECTED_D = 64
EXPECTED_R = 256
EXPECTED_E = 37
EXPECTED_A = 107


@dataclass(frozen=True)
class SparseRadar:
    range_ind: np.ndarray
    elevation_ind: np.ndarray
    azimuth_ind: np.ndarray
    descriptor: np.ndarray  # [8, N]

    @property
    def top3_power(self) -> np.ndarray:
        return self.descriptor[0:3]

    @property
    def top3_doppler_ind(self) -> np.ndarray:
        return np.rint(self.descriptor[3:6]).astype(np.int64)

    @property
    def mean_power(self) -> np.ndarray:
        return self.descriptor[6]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw RadarOcc selected-token Top-1/Top-2/Top-3 RD projections "
            "and selected-token mean Doppler spectrum."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--raw-dir",
        type=Path,
        help="Directory containing matching raw .mat/.npy files.",
    )
    source.add_argument(
        "--raw",
        type=Path,
        nargs="+",
        help="Explicit raw .mat/.npy files.",
    )
    parser.add_argument(
        "--sparse-dir",
        type=Path,
        required=True,
        help="Directory containing EAsparse_*.npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--frames",
        nargs="*",
        default=None,
        help="Optional numeric frame IDs, e.g. 34 80 101.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=250,
        help="Expected selected spatial tokens per range. Default: 250.",
    )
    parser.add_argument(
        "--model-ranges",
        type=int,
        default=175,
        help=(
            "Also draw the first N ranges used by the current Small model. "
            "Set 0 to disable. Default: 175."
        ),
    )
    parser.add_argument(
        "--projection",
        choices=("max", "mean"),
        default="max",
        help=(
            "Reduction over elevation/azimuth for the original RD background. "
            "Default: max."
        ),
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help=(
            "Marker-area scale. Actual area is point_size multiplied by the "
            "number of selected tokens in the same RD cell."
        ),
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.82,
        help="Opacity of hollow red circles.",
    )
    parser.add_argument(
        "--min-rd-count",
        type=int,
        default=2,
        help=(
            "Minimum projected-token count required to draw an RD circle in "
            "Top-1/Top-2/Top-3. Default: 2, so singleton cells are hidden. "
            "RA and Mean RD are unchanged."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI.",
    )
    return parser.parse_args()


def frame_id(path: Path) -> int:
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise ValueError(f"Cannot extract frame ID from {path.name}")
    return int(numbers[-1])


def collect_raw_files(args: argparse.Namespace) -> list[Path]:
    if args.raw is not None:
        files = [path.expanduser().resolve() for path in args.raw]
    else:
        raw_dir = args.raw_dir.expanduser().resolve()
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
        files = sorted(
            path
            for path in raw_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".mat", ".npy"}
            and not path.name.startswith("arr_")
            and path.name != "info_arr.mat"
        )

    if args.frames:
        requested = {int(value) for value in args.frames}
        files = [path for path in files if frame_id(path) in requested]

    if not files:
        raise FileNotFoundError("No matching raw radar files were found.")
    return files


def index_sparse_files(directory: Path) -> dict[int, Path]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Sparse directory does not exist: {directory}")

    result: dict[int, Path] = {}
    for path in directory.rglob("EAsparse_*.npz"):
        current_id = frame_id(path)
        if current_id in result:
            raise RuntimeError(
                f"Multiple sparse files found for frame {current_id}: "
                f"{result[current_id]} and {path}"
            )
        result[current_id] = path

    if not result:
        raise FileNotFoundError(f"No EAsparse_*.npz files found under {directory}")
    return result


def normalize_to_drea(array: np.ndarray, path: Path) -> np.ndarray:
    """Normalize raw radar tensor to [Doppler, Range, Elevation, Azimuth]."""
    array = np.asarray(array).squeeze()
    if array.ndim != 4:
        raise ValueError(f"Expected 4-D radar tensor, got {array.shape} in {path}")

    expected = (EXPECTED_D, EXPECTED_R, EXPECTED_E, EXPECTED_A)
    if array.shape == expected:
        return array

    common_drae = (EXPECTED_D, EXPECTED_R, EXPECTED_A, EXPECTED_E)
    if array.shape == common_drae:
        return np.transpose(array, (0, 1, 3, 2))

    sizes = list(array.shape)
    required = list(expected)
    if all(sizes.count(size) == 1 for size in required):
        permutation = tuple(sizes.index(size) for size in required)
        return np.transpose(array, permutation)

    raise ValueError(
        f"Cannot infer D-R-E-A axis order for {path}; got {array.shape}"
    )


def load_raw(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".mat":
        content = loadmat(path)
        if "arrDREA" in content:
            array = np.asarray(content["arrDREA"])
        else:
            candidates = [
                np.asarray(value)
                for key, value in content.items()
                if not key.startswith("__")
                and isinstance(value, np.ndarray)
                and value.ndim == 4
            ]
            if len(candidates) != 1:
                raise KeyError(
                    f"{path} has no arrDREA and not exactly one 4-D array."
                )
            array = candidates[0]
    elif path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        raise ValueError(f"Unsupported raw file type: {path.suffix}")

    return normalize_to_drea(array, path)


def load_sparse(path: Path) -> SparseRadar:
    with np.load(path) as data:
        required = (
            "range_ind",
            "elevation_ind",
            "azimuth_ind",
            "power_val",
        )
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{path.name} is missing arrays: {missing}")

        descriptor = np.asarray(data["power_val"])
        if descriptor.ndim != 2:
            raise ValueError(
                f"power_val must be 2-D, got {descriptor.shape}"
            )
        if descriptor.shape[0] != 8 and descriptor.shape[1] == 8:
            descriptor = descriptor.T
        if descriptor.shape[0] != 8:
            raise ValueError(
                f"Expected power_val shape [8, N], got {descriptor.shape}"
            )

        sparse = SparseRadar(
            range_ind=np.asarray(data["range_ind"]).reshape(-1).astype(np.int64),
            elevation_ind=np.asarray(data["elevation_ind"]).reshape(-1).astype(np.int64),
            azimuth_ind=np.asarray(data["azimuth_ind"]).reshape(-1).astype(np.int64),
            descriptor=descriptor,
        )

    number_points = sparse.descriptor.shape[1]
    if not (
        sparse.range_ind.size
        == sparse.elevation_ind.size
        == sparse.azimuth_ind.size
        == number_points
    ):
        raise ValueError("Sparse coordinate lengths do not match power_val.")
    return sparse


def db10(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(
        np.maximum(np.asarray(values, dtype=np.float64), EPS)
    )


def sort_top3_per_token(
    top3_power: np.ndarray,
    top3_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort each token's three stored peak pairs from strongest to weakest."""
    order = np.argsort(top3_power, axis=0)[::-1]
    sorted_power = np.take_along_axis(top3_power, order, axis=0)
    sorted_index = np.take_along_axis(top3_index, order, axis=0)
    return sorted_power, sorted_index


def collapse_cells(
    x_index: np.ndarray,
    y_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unique 2-D cells and number of projected tokens per cell."""
    pairs = np.column_stack(
        (
            np.asarray(y_index, dtype=np.int64).reshape(-1),
            np.asarray(x_index, dtype=np.int64).reshape(-1),
        )
    )
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    return unique_pairs[:, 1], unique_pairs[:, 0], counts.astype(np.float64)


def check_250_per_range(
    y_index: np.ndarray,
    counts: np.ndarray,
    range_limit: int,
    expected_k: int,
    label: str,
) -> None:
    count_per_range = np.bincount(
        y_index,
        weights=counts,
        minlength=range_limit,
    )
    bad = np.where(
        count_per_range[:range_limit] != expected_k
    )[0]
    if bad.size:
        preview = {
            int(current_range): float(count_per_range[current_range])
            for current_range in bad[:10]
        }
        raise RuntimeError(
            f"{label} does not preserve {expected_k} token projections per "
            f"range. First mismatches: {preview}"
        )


def original_rd_map(raw: np.ndarray, range_limit: int, projection: str) -> np.ndarray:
    raw_used = raw[:, :range_limit, :, :]
    if projection == "max":
        result = raw_used.max(axis=(2, 3)).T
    else:
        result = raw_used.mean(axis=(2, 3)).T
    return db10(result)


def original_ra_map(raw: np.ndarray, range_limit: int, projection: str) -> np.ndarray:
    # Same mean-over-Doppler score used by the preprocessing.
    mean_doppler = raw[:, :range_limit, :, :].mean(axis=0)  # [R, E, A]
    if projection == "max":
        result = mean_doppler.max(axis=1)
    else:
        result = mean_doppler.mean(axis=1)
    return db10(result)


def selected_mean_rd_map(
    raw: np.ndarray,
    range_ind: np.ndarray,
    elevation_ind: np.ndarray,
    azimuth_ind: np.ndarray,
    range_limit: int,
    expected_k: int,
) -> np.ndarray:
    """Average the complete 64-bin spectra of selected tokens at each range."""
    result = np.full(
        (range_limit, raw.shape[0]),
        np.nan,
        dtype=np.float64,
    )

    for current_range in range(range_limit):
        mask = range_ind == current_range
        number_tokens = int(mask.sum())
        if number_tokens != expected_k:
            raise RuntimeError(
                f"Range {current_range} contains {number_tokens} selected "
                f"tokens; expected {expected_k}."
            )

        spectra = raw[
            :,
            current_range,
            elevation_ind[mask],
            azimuth_ind[mask],
        ]  # [D, K]
        result[current_range] = spectra.mean(axis=1)

    return db10(result)


def plot_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    x_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Range bin")
    return image


def plot_top_peak_panel(
    ax: plt.Axes,
    rd_background: np.ndarray,
    peak_index: np.ndarray,
    range_ind: np.ndarray,
    rank: int,
    point_size: float,
    point_alpha: float,
    min_rd_count: int,
    vmin: float,
    vmax: float,
) -> tuple[int, int, float]:
    """Draw one Top-k RD panel while hiding low-count cells visually."""
    x_all, y_all, counts_all = collapse_cells(peak_index, range_ind)

    # Count all 250 token projections per range first, then filter only the
    # displayed circles. This does not change preprocessing data.
    keep = counts_all >= min_rd_count
    x_cell = x_all[keep]
    y_cell = y_all[keep]
    counts = counts_all[keep]

    plot_heatmap(
        ax,
        rd_background,
        (
            f"Sorted Top-{rank} Doppler peak of each selected token\n"
            f"show count >= {min_rd_count}; circle area = token multiplicity"
        ),
        "Doppler bin",
        vmin=vmin,
        vmax=vmax,
    )
    ax.scatter(
        x_cell,
        y_cell,
        s=point_size * counts,
        facecolors="none",
        edgecolors="red",
        linewidths=0.45,
        alpha=point_alpha,
        marker="o",
        rasterized=True,
        zorder=3,
    )

    legend_handle = Line2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        markersize=5,
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=0.8,
        alpha=point_alpha,
    )
    ax.legend(
        [legend_handle],
        [f"Top-{rank}: count >= {min_rd_count}"],
        loc="upper right",
        fontsize=7,
        framealpha=0.85,
        borderpad=0.3,
    )

    maximum_count = float(counts_all.max()) if counts_all.size else 0.0
    hidden_cells = int(np.sum(~keep))
    return int(np.sum(keep)), hidden_cells, maximum_count

def create_figure(
    raw: np.ndarray,
    sparse: SparseRadar,
    raw_path: Path,
    sparse_path: Path,
    output_path: Path,
    range_limit: int,
    expected_k: int,
    projection: str,
    point_size: float,
    point_alpha: float,
    min_rd_count: int,
    dpi: int,
) -> None:
    selected = sparse.range_ind < range_limit
    selected_range = sparse.range_ind[selected]
    selected_elevation = sparse.elevation_ind[selected]
    selected_azimuth = sparse.azimuth_ind[selected]

    expected_total = range_limit * expected_k
    if selected_range.size != expected_total:
        raise RuntimeError(
            f"Selected {selected_range.size} tokens for {range_limit} ranges; "
            f"expected {expected_total}."
        )

    selected_top3_power = sparse.top3_power[:, selected]
    selected_top3_index = sparse.top3_doppler_ind[:, selected]

    sorted_power, sorted_index = sort_top3_per_token(
        selected_top3_power,
        selected_top3_index,
    )

    if np.any((sorted_index < 0) | (sorted_index >= raw.shape[0])):
        raise IndexError("Top-3 descriptor contains invalid Doppler indices.")

    rd_background = original_rd_map(raw, range_limit, projection)
    ra_background = original_ra_map(raw, range_limit, projection)
    mean_selected_rd = selected_mean_rd_map(
        raw,
        selected_range,
        selected_elevation,
        selected_azimuth,
        range_limit,
        expected_k,
    )

    # Shared RD color scale makes raw backgrounds and selected-spectrum mean
    # visually comparable.
    scale_values = np.concatenate(
        (
            rd_background[np.isfinite(rd_background)],
            mean_selected_rd[np.isfinite(mean_selected_rd)],
        )
    )
    vmin = float(np.percentile(scale_values, 1.0))
    vmax = float(np.percentile(scale_values, 99.5))

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(22, 12),
        constrained_layout=True,
    )

    # RA context panel: same 250 mean-power-selected tokens.
    ra_x, ra_y, ra_counts = collapse_cells(
        selected_azimuth,
        selected_range,
    )
    check_250_per_range(
        ra_y,
        ra_counts,
        range_limit,
        expected_k,
        "RA projection",
    )

    ra_image = plot_heatmap(
        axes[0, 0],
        ra_background,
        (
            "RA context: mean-power-selected spatial tokens\n"
            "circle area = selected elevation count at each (R, A)"
        ),
        "Azimuth bin",
    )
    axes[0, 0].scatter(
        ra_x,
        ra_y,
        s=point_size * ra_counts,
        facecolors="none",
        edgecolors="red",
        linewidths=0.45,
        alpha=point_alpha,
        marker="o",
        rasterized=True,
        zorder=3,
    )
    fig.colorbar(ra_image, ax=axes[0, 0], label="Power [dB]")

    occupied_cells: list[int] = []
    hidden_low_count_cells: list[int] = []
    maximum_multiplicities: list[float] = []

    # Top-1.
    cells, hidden_cells, max_count = plot_top_peak_panel(
        axes[0, 1],
        rd_background,
        sorted_index[0],
        selected_range,
        rank=1,
        point_size=point_size,
        point_alpha=point_alpha,
        min_rd_count=min_rd_count,
        vmin=vmin,
        vmax=vmax,
    )
    check_x, check_y, check_counts = collapse_cells(
        sorted_index[0], selected_range
    )
    check_250_per_range(
        check_y, check_counts, range_limit, expected_k, "Top-1 RD projection"
    )
    occupied_cells.append(cells)
    hidden_low_count_cells.append(hidden_cells)
    maximum_multiplicities.append(max_count)
    fig.colorbar(
        axes[0, 1].images[0],
        ax=axes[0, 1],
        label="Raw RD power [dB]",
    )

    # Top-2.
    cells, hidden_cells, max_count = plot_top_peak_panel(
        axes[0, 2],
        rd_background,
        sorted_index[1],
        selected_range,
        rank=2,
        point_size=point_size,
        point_alpha=point_alpha,
        min_rd_count=min_rd_count,
        vmin=vmin,
        vmax=vmax,
    )
    check_x, check_y, check_counts = collapse_cells(
        sorted_index[1], selected_range
    )
    check_250_per_range(
        check_y, check_counts, range_limit, expected_k, "Top-2 RD projection"
    )
    occupied_cells.append(cells)
    hidden_low_count_cells.append(hidden_cells)
    maximum_multiplicities.append(max_count)
    fig.colorbar(
        axes[0, 2].images[0],
        ax=axes[0, 2],
        label="Raw RD power [dB]",
    )

    # Top-3.
    cells, hidden_cells, max_count = plot_top_peak_panel(
        axes[1, 0],
        rd_background,
        sorted_index[2],
        selected_range,
        rank=3,
        point_size=point_size,
        point_alpha=point_alpha,
        min_rd_count=min_rd_count,
        vmin=vmin,
        vmax=vmax,
    )
    check_x, check_y, check_counts = collapse_cells(
        sorted_index[2], selected_range
    )
    check_250_per_range(
        check_y, check_counts, range_limit, expected_k, "Top-3 RD projection"
    )
    occupied_cells.append(cells)
    hidden_low_count_cells.append(hidden_cells)
    maximum_multiplicities.append(max_count)
    fig.colorbar(
        axes[1, 0].images[0],
        ax=axes[1, 0],
        label="Raw RD power [dB]",
    )

    # Mean spectrum of the selected 250 tokens.
    mean_image = plot_heatmap(
        axes[1, 1],
        mean_selected_rd,
        (
            "Mean complete Doppler spectrum of the same 250 tokens\n"
            "average over selected elevation/azimuth tokens"
        ),
        "Doppler bin",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(
        mean_image,
        ax=axes[1, 1],
        label="Selected-token mean power [dB]",
    )

    # Summary.
    axes[1, 2].axis("off")
    summary = (
        f"Frame: {frame_id(raw_path):05d}\n"
        f"Raw: {raw_path.name}\n"
        f"Sparse: {sparse_path.name}\n\n"
        f"Raw shape D-R-E-A: {tuple(raw.shape)}\n"
        f"Plotted ranges: {range_limit}\n"
        f"Selected tokens/range: {expected_k}\n"
        f"Total selected tokens: {expected_total:,}\n\n"
        "Top-k definition:\n"
        "  one independently sorted Doppler peak per selected\n"
        "  (range, elevation, azimuth) token.\n\n"
        f"Displayed RD cells (count >= {min_rd_count}):\n"
        f"  Top-1: {occupied_cells[0]:,}\n"
        f"  Top-2: {occupied_cells[1]:,}\n"
        f"  Top-3: {occupied_cells[2]:,}\n\n"
        f"Hidden RD cells (count < {min_rd_count}):\n"
        f"  Top-1: {hidden_low_count_cells[0]:,}\n"
        f"  Top-2: {hidden_low_count_cells[1]:,}\n"
        f"  Top-3: {hidden_low_count_cells[2]:,}\n\n"
        f"Maximum tokens sharing one RD cell:\n"
        f"  Top-1: {maximum_multiplicities[0]:.0f}\n"
        f"  Top-2: {maximum_multiplicities[1]:.0f}\n"
        f"  Top-3: {maximum_multiplicities[2]:.0f}\n\n"
        "Each Top-k panel is computed from exactly 250 token\n"
        "projections per range. RD cells below the display\n"
        f"threshold count < {min_rd_count} are hidden only in the\n"
        "plot; circle AREA represents multiplicity. RA unchanged.\n\n"
        "The Mean panel is a heatmap, not points, because\n"
        "mean power alone has no Doppler index."
    )
    axes[1, 2].text(
        0.0,
        1.0,
        summary,
        va="top",
        family="monospace",
        fontsize=9,
    )

    fig.suptitle(
        (
            "RadarOcc mean-power top-250 preprocessing: "
            "RA context and four RD views"
        ),
        fontsize=16,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def process_frame(
    raw_path: Path,
    sparse_path: Path,
    args: argparse.Namespace,
) -> None:
    print(f"\n[frame {frame_id(raw_path):05d}] raw={raw_path}")
    print(f"[frame {frame_id(raw_path):05d}] sparse={sparse_path}")

    raw = load_raw(raw_path)
    sparse = load_sparse(sparse_path)

    print(f"[frame {frame_id(raw_path):05d}] raw D-R-E-A shape={raw.shape}")
    print(
        f"[frame {frame_id(raw_path):05d}] sparse points="
        f"{sparse.range_ind.size}"
    )

    range_limits = [raw.shape[1]]
    if 0 < args.model_ranges < raw.shape[1]:
        range_limits.append(args.model_ranges)

    for range_limit in range_limits:
        tag = (
            f"all{raw.shape[1]}"
            if range_limit == raw.shape[1]
            else f"model{range_limit}"
        )
        output_path = (
            args.output_dir
            / f"frame_{frame_id(raw_path):05d}_rd_top3_mean_{tag}.png"
        )
        create_figure(
            raw=raw,
            sparse=sparse,
            raw_path=raw_path,
            sparse_path=sparse_path,
            output_path=output_path,
            range_limit=range_limit,
            expected_k=args.k,
            projection=args.projection,
            point_size=args.point_size,
            point_alpha=args.point_alpha,
            min_rd_count=args.min_rd_count,
            dpi=args.dpi,
        )
        print(f"[frame {frame_id(raw_path):05d}] wrote {output_path}")


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = collect_raw_files(args)
    sparse_index = index_sparse_files(args.sparse_dir)

    matched = 0
    for raw_path in raw_files:
        current_id = frame_id(raw_path)
        sparse_path = sparse_index.get(current_id)
        if sparse_path is None:
            print(
                f"[warning] no matching EAsparse file for "
                f"{raw_path.name}; skipping"
            )
            continue

        process_frame(raw_path, sparse_path, args)
        matched += 1

    if matched == 0:
        raise RuntimeError("No raw files had matching EAsparse files.")

    print(f"\nDone. Processed {matched} frame(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
