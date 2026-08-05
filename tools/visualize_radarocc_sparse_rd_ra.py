#!/usr/bin/env python3
"""Visualize RadarOcc sparse preprocessing on range–azimuth (RA) and
range–Doppler (RD) maps.

The script supports:
  * K-Radar raw ``tesseract_*.mat`` files containing ``arrDREA``.
  * RadarOcc-converted ``*.npy`` files.
  * RadarOcc sparse ``EAsparse_*.npz`` files containing
    ``range_ind``, ``elevation_ind``, ``azimuth_ind`` and ``power_val``.

It intentionally keeps the raw array in D-R-E-A order, because that is the
index convention used by RadarOcc's ``generate_4d_polar_doppler.py``.
No azimuth/elevation flip is applied before matching sparse indices.

Example:
    python tools/visualize_radarocc_sparse_rd_ra.py \
      --raw-dir "/mnt/elab-share/.../kradar raw/11" \
      --sparse-dir "/home/user1/projects/RadarOcc/data/11/radar_tensor_8doppler" \
      --output-dir work_dirs/seq11_rd_ra \
      --model-ranges 175

Optional physical axes:
    --info-arr /path/to/K-Radar/resources/info_arr.mat \
    --arr-doppler /path/to/K-Radar/resources/arr_doppler.mat
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.io import loadmat


EPS = np.finfo(np.float32).tiny
EXPECTED_AXIS_SIZES = {"doppler": 64, "range": 256, "elevation": 37, "azimuth": 107}


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

    @property
    def variance(self) -> np.ndarray:
        return self.descriptor[7]


@dataclass(frozen=True)
class PhysicalAxes:
    range_values: np.ndarray
    azimuth_values_deg: np.ndarray
    doppler_values: np.ndarray


@dataclass
class FrameStats:
    frame_id: str
    raw_path: str
    sparse_path: str
    raw_shape: str
    sparse_points: int
    plotted_ranges: int
    coordinate_match_ratio: float
    top3_bin_match_ratio: float
    mean_abs_error: float
    within_3db_ratio: float
    db_3_to_10_ratio: float
    db_10_to_20_ratio: float
    below_20db_ratio: float
    output_png: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay RadarOcc sparse selections on raw RA and RD maps."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--raw-dir",
        type=Path,
        help="Directory searched recursively for raw .mat/.npy radar files.",
    )
    source.add_argument(
        "--raw",
        type=Path,
        nargs="+",
        help="One or more explicit raw .mat/.npy radar files.",
    )
    parser.add_argument(
        "--sparse-dir",
        type=Path,
        required=True,
        help="Directory searched recursively for EAsparse_*.npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which PNG figures and summary.csv are written.",
    )
    parser.add_argument(
        "--frames",
        nargs="*",
        default=None,
        help="Optional frame IDs to process, for example 000123 000124 000125.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=250,
        help="Expected number of selected elevation-azimuth cells per range.",
    )
    parser.add_argument(
        "--model-ranges",
        type=int,
        default=175,
        help=(
            "Also generate a figure limited to the first N ranges actually used by "
            "the current RadarOcc-Small code. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--projection",
        choices=("max", "mean"),
        default="max",
        help=(
            "Reduction for hidden dimensions in the background maps. 'max' makes "
            "peaks easier to inspect; 'mean' gives a conventional averaged projection."
        ),
    )
    parser.add_argument(
        "--info-arr",
        type=Path,
        default=None,
        help="Optional K-Radar resources/info_arr.mat for range and azimuth axes.",
    )
    parser.add_argument(
        "--arr-doppler",
        type=Path,
        default=None,
        help="Optional K-Radar resources/arr_doppler.mat for Doppler values.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip reconstruction checks against the current RadarOcc preprocessing.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=3.0,
        help=(
            "Marker area scale. Marker area is point_size multiplied by the "
            "number of selected 3D tokens projected into the same 2D cell."
        ),
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.85,
        help="Opacity of the red hollow-circle overlays.",
    )
    parser.add_argument(
        "--point-color",
        type=str,
        default="red",
        help="Color of the hollow-circle overlays. Default: red.",
    )
    return parser.parse_args()


def trailing_frame_id(path: Path) -> str:
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise ValueError(f"Cannot extract a numeric frame ID from {path.name}")
    return numbers[-1]


def normalized_frame_id(value: str) -> int:
    return int(value)


def collect_raw_files(args: argparse.Namespace) -> list[Path]:
    if args.raw is not None:
        files = [p.expanduser().resolve() for p in args.raw]
    else:
        raw_dir = args.raw_dir.expanduser().resolve()
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
        files = sorted(
            [
                p
                for p in raw_dir.rglob("*")
                if p.is_file()
                and p.suffix.lower() in {".mat", ".npy"}
                and not p.name.startswith("arr_")
                and p.name != "info_arr.mat"
            ]
        )

    if args.frames:
        requested = {normalized_frame_id(frame) for frame in args.frames}
        files = [p for p in files if normalized_frame_id(trailing_frame_id(p)) in requested]

    if not files:
        raise FileNotFoundError("No matching raw .mat or .npy radar files were found.")
    return files


def index_sparse_files(sparse_dir: Path) -> dict[int, Path]:
    sparse_dir = sparse_dir.expanduser().resolve()
    if not sparse_dir.is_dir():
        raise FileNotFoundError(f"Sparse directory does not exist: {sparse_dir}")

    mapping: dict[int, Path] = {}
    for path in sparse_dir.rglob("EAsparse_*.npz"):
        frame = normalized_frame_id(trailing_frame_id(path))
        if frame in mapping:
            raise RuntimeError(
                f"Multiple sparse files map to frame {frame}: {mapping[frame]} and {path}"
            )
        mapping[frame] = path
    if not mapping:
        raise FileNotFoundError(f"No EAsparse_*.npz files found under {sparse_dir}")
    return mapping


def _find_single_4d_array(mat_dict: dict[str, object], path: Path) -> np.ndarray:
    if "arrDREA" in mat_dict:
        return np.asarray(mat_dict["arrDREA"])

    candidates = [
        np.asarray(value)
        for key, value in mat_dict.items()
        if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 4
    ]
    if len(candidates) != 1:
        raise KeyError(
            f"{path} has no arrDREA key and does not contain exactly one 4-D array."
        )
    return candidates[0]


def normalize_to_drea(array: np.ndarray, path: Path) -> np.ndarray:
    """Return raw radar tensor in D-R-E-A order without physical-axis flipping."""
    array = np.asarray(array).squeeze()
    if array.ndim != 4:
        raise ValueError(f"Expected a 4-D radar tensor in {path}, got shape {array.shape}")

    shape = array.shape
    target = (
        EXPECTED_AXIS_SIZES["doppler"],
        EXPECTED_AXIS_SIZES["range"],
        EXPECTED_AXIS_SIZES["elevation"],
        EXPECTED_AXIS_SIZES["azimuth"],
    )
    if shape == target:
        return array

    # Common alternative used by K-Radar visualization code: D-R-A-E.
    if shape == (target[0], target[1], target[3], target[2]):
        return np.transpose(array, (0, 1, 3, 2))

    # Generic detection by the known axis sizes.
    sizes = list(shape)
    required = list(target)
    if all(sizes.count(value) == 1 for value in required):
        permutation = tuple(sizes.index(value) for value in required)
        return np.transpose(array, permutation)

    raise ValueError(
        f"Cannot infer D-R-E-A axes for {path}; got shape {shape}, expected a permutation "
        f"of {target}."
    )


def load_raw(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        raw = np.load(path, mmap_mode="r")
    elif suffix == ".mat":
        raw = _find_single_4d_array(loadmat(path), path)
    else:
        raise ValueError(f"Unsupported raw format: {path}")
    return normalize_to_drea(raw, path)


def load_sparse(path: Path) -> SparseRadar:
    with np.load(path) as data:
        required = {"range_ind", "elevation_ind", "azimuth_ind", "power_val"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} is missing keys: {sorted(missing)}")

        descriptor = np.asarray(data["power_val"])
        if descriptor.ndim != 2:
            raise ValueError(f"power_val must be 2-D, got {descriptor.shape} in {path}")
        if descriptor.shape[0] != 8 and descriptor.shape[1] == 8:
            descriptor = descriptor.T
        if descriptor.shape[0] != 8:
            raise ValueError(f"Expected power_val shape [8, N], got {descriptor.shape} in {path}")

        range_ind = np.asarray(data["range_ind"]).reshape(-1).astype(np.int64)
        elevation_ind = np.asarray(data["elevation_ind"]).reshape(-1).astype(np.int64)
        azimuth_ind = np.asarray(data["azimuth_ind"]).reshape(-1).astype(np.int64)

    n = descriptor.shape[1]
    if not (len(range_ind) == len(elevation_ind) == len(azimuth_ind) == n):
        raise ValueError(f"Coordinate and descriptor lengths differ in {path}")

    return SparseRadar(range_ind, elevation_ind, azimuth_ind, descriptor)


def load_physical_axes(
    info_arr_path: Path | None,
    arr_doppler_path: Path | None,
    raw_shape: Sequence[int],
) -> PhysicalAxes | None:
    if info_arr_path is None and arr_doppler_path is None:
        return None
    if info_arr_path is None or arr_doppler_path is None:
        raise ValueError("Both --info-arr and --arr-doppler are required for physical axes.")

    info = loadmat(info_arr_path.expanduser().resolve())
    doppler = loadmat(arr_doppler_path.expanduser().resolve())
    range_values = np.asarray(info["arrRange"]).reshape(-1)
    azimuth_values_deg = np.asarray(info["arrAzimuth"]).reshape(-1)
    doppler_values = np.asarray(doppler["arr_doppler"]).reshape(-1)

    _, r_dim, _, a_dim = raw_shape
    d_dim = raw_shape[0]
    if len(range_values) != r_dim or len(azimuth_values_deg) != a_dim or len(doppler_values) != d_dim:
        raise ValueError(
            "Physical-axis lengths do not match the normalized D-R-E-A tensor: "
            f"range={len(range_values)}/{r_dim}, azimuth={len(azimuth_values_deg)}/{a_dim}, "
            f"doppler={len(doppler_values)}/{d_dim}."
        )
    return PhysicalAxes(range_values, azimuth_values_deg, doppler_values)


def db10(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(values, dtype=np.float64), EPS))


def validate_sparse_against_raw(
    raw: np.ndarray,
    sparse: SparseRadar,
    k: int,
) -> tuple[float, float, float]:
    """Reconstruct the current preprocessing and return validation metrics.

    Returns:
        coordinate_match_ratio: fraction of ranges whose selected E-A set exactly matches.
        top3_bin_match_ratio: fraction of points whose three Doppler-bin IDs match as a set.
        mean_abs_error: mean absolute error between stored and recomputed mean power.
    """
    d_dim, r_dim, e_dim, a_dim = raw.shape
    if k > e_dim * a_dim:
        raise ValueError(f"k={k} exceeds E*A={e_dim*a_dim}")

    expected_flat = np.argpartition(
        raw.mean(axis=0).reshape(r_dim, -1), -k, axis=1
    )[:, -k:]

    per_range_match: list[bool] = []
    for range_idx in range(r_dim):
        saved_mask = sparse.range_ind == range_idx
        saved_flat = sparse.elevation_ind[saved_mask] * a_dim + sparse.azimuth_ind[saved_mask]
        per_range_match.append(
            len(saved_flat) == k
            and np.array_equal(np.sort(saved_flat), np.sort(expected_flat[range_idx]))
        )
    coordinate_match_ratio = float(np.mean(per_range_match))

    valid = (
        (sparse.range_ind >= 0)
        & (sparse.range_ind < r_dim)
        & (sparse.elevation_ind >= 0)
        & (sparse.elevation_ind < e_dim)
        & (sparse.azimuth_ind >= 0)
        & (sparse.azimuth_ind < a_dim)
    )
    if not np.all(valid):
        bad = int((~valid).sum())
        raise IndexError(f"Sparse file contains {bad} out-of-bounds coordinates.")

    selected_raw = np.asarray(
        raw[:, sparse.range_ind, sparse.elevation_ind, sparse.azimuth_ind]
    )
    recomputed_mean = selected_raw.mean(axis=0)
    mean_abs_error = float(np.mean(np.abs(recomputed_mean - sparse.mean_power)))

    recomputed_bins = np.argpartition(selected_raw, -3, axis=0)[-3:]
    stored_bins = sparse.top3_doppler_ind
    top3_match = np.all(
        np.sort(recomputed_bins, axis=0) == np.sort(stored_bins, axis=0), axis=0
    )
    top3_bin_match_ratio = float(np.mean(top3_match))

    return coordinate_match_ratio, top3_bin_match_ratio, mean_abs_error


def build_maps(
    raw: np.ndarray,
    sparse: SparseRadar,
    range_limit: int,
    projection: str,
) -> dict[str, np.ndarray | float]:
    d_dim, r_dim, _, a_dim = raw.shape
    range_limit = min(max(range_limit, 1), r_dim)

    mean_doppler_cube = raw.mean(axis=0)  # [R, E, A], identical score used for top-k.
    if projection == "max":
        ra_power = mean_doppler_cube.max(axis=1)  # [R, A]
        rd_power = raw.max(axis=(2, 3)).T  # [R, D]
    else:
        ra_power = mean_doppler_cube.mean(axis=1)
        rd_power = raw.mean(axis=(2, 3)).T

    selected = sparse.range_ind < range_limit
    r = sparse.range_ind[selected]
    e = sparse.elevation_ind[selected]
    a = sparse.azimuth_ind[selected]
    mean_power = sparse.mean_power[selected]
    top3_doppler = sparse.top3_doppler_ind[:, selected]

    ra_count = np.zeros((r_dim, a_dim), dtype=np.int32)
    np.add.at(ra_count, (r, a), 1)

    rd_count = np.zeros((r_dim, d_dim), dtype=np.int32)
    for row in top3_doppler:
        valid_d = (row >= 0) & (row < d_dim)
        np.add.at(rd_count, (r[valid_d], row[valid_d]), 1)

    # Two relative-strength diagnostics:
    # 1) mean-power point strength against the strongest E-A cell at the same range;
    # 2) the strongest stored Doppler component against the strongest angular cell at
    #    the same (range, Doppler). The second diagnostic is more directly related to
    #    angular sidelobes, because a sidelobe is a weaker off-angle copy at the same
    #    range and Doppler as a dominant return.
    peak_per_range = mean_doppler_cube.reshape(r_dim, -1).max(axis=1)
    mean_range_delta_db = db10(mean_power) - db10(peak_per_range[r])

    selected_top3_power = sparse.top3_power[:, selected]
    strongest_descriptor_row = np.argmax(selected_top3_power, axis=0)
    point_indices = np.arange(r.size)
    primary_power = selected_top3_power[strongest_descriptor_row, point_indices]
    primary_doppler = top3_doppler[strongest_descriptor_row, point_indices]
    valid_primary = (primary_doppler >= 0) & (primary_doppler < d_dim)
    angular_delta_db = np.full(r.size, np.nan, dtype=np.float64)
    rd_angular_peak = raw.max(axis=(2, 3))  # [D, R]
    angular_delta_db[valid_primary] = (
        db10(primary_power[valid_primary])
        - db10(rd_angular_peak[primary_doppler[valid_primary], r[valid_primary]])
    )

    return {
        "range_limit": float(range_limit),
        "ra_db": db10(ra_power),
        "rd_db": db10(rd_power),
        "ra_count": ra_count,
        "rd_count": rd_count,
        "mean_range_delta_db": mean_range_delta_db,
        "angular_delta_db": angular_delta_db,
        "selected_range": r,
        "selected_elevation": e,
        "selected_azimuth": a,
        "top3_doppler": top3_doppler,
    }


def diagnostic_ratios(delta_db: np.ndarray) -> tuple[float, float, float, float]:
    if delta_db.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(delta_db >= -3.0)),
        float(np.mean((delta_db < -3.0) & (delta_db >= -10.0))),
        float(np.mean((delta_db < -10.0) & (delta_db >= -20.0))),
        float(np.mean(delta_db < -20.0)),
    )



def collapse_projected_cells(
    x_index: np.ndarray,
    y_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse repeated integer grid coordinates to one point per occupied cell.

    Returns:
        unique_x, unique_y, multiplicity
    """
    x_index = np.asarray(x_index, dtype=np.int64).reshape(-1)
    y_index = np.asarray(y_index, dtype=np.int64).reshape(-1)
    if x_index.size != y_index.size:
        raise ValueError(
            f"Projected x/y lengths differ: {x_index.size} vs {y_index.size}"
        )
    if x_index.size == 0:
        return x_index, y_index, np.empty(0, dtype=np.int64)

    pairs = np.column_stack((y_index, x_index))
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    return unique_pairs[:, 1], unique_pairs[:, 0], counts


def _plot_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    x_values: np.ndarray | None,
    y_values: np.ndarray | None,
    x_label: str,
    y_label: str,
):
    if x_values is not None and y_values is not None:
        mesh = ax.pcolormesh(x_values, y_values, matrix, shading="auto")
    else:
        mesh = ax.imshow(matrix, origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return mesh


def create_figure(
    raw: np.ndarray,
    sparse: SparseRadar,
    maps: dict[str, np.ndarray | float],
    frame_id: str,
    raw_path: Path,
    sparse_path: Path,
    output_path: Path,
    projection: str,
    axes: PhysicalAxes | None,
    validation: tuple[float, float, float],
    point_size: float,
    point_alpha: float,
    point_color: str,
) -> tuple[float, float, float, float]:
    """Overlay the 250 mean-power-selected tokens from every range on raw maps.

    The preprocessing selects 250 distinct (elevation, azimuth) locations per
    range using mean power over Doppler.

    RA overlay:
        Project those same 250 tokens to (range, azimuth). Tokens from different
        elevation bins can land in the same RA cell. Marker area represents the
        number of selected elevation bins in that cell.

    RD overlay:
        For each of the same 250 selected tokens, choose exactly one Doppler bin:
        the strongest of its stored top-3 Doppler components. Project that token
        to (range, strongest Doppler). Marker area represents how many of the 250
        tokens land in the same RD cell.

    Therefore, for every plotted range, RA multiplicities sum to 250 and RD
    multiplicities also sum to 250.
    """
    range_limit = int(maps["range_limit"])
    ra_db = np.asarray(maps["ra_db"])[:range_limit]
    rd_db = np.asarray(maps["rd_db"])[:range_limit]
    mean_range_delta_db = np.asarray(maps["mean_range_delta_db"])
    angular_delta_db = np.asarray(maps["angular_delta_db"])
    angular_delta_db = angular_delta_db[np.isfinite(angular_delta_db)]

    selected_range = np.asarray(maps["selected_range"], dtype=np.int64)
    selected_azimuth = np.asarray(maps["selected_azimuth"], dtype=np.int64)
    top3_doppler = np.asarray(maps["top3_doppler"], dtype=np.int64)

    # ------------------------------------------------------------------
    # The exact 250 mean-power-selected tokens per range
    # ------------------------------------------------------------------
    # RA projection: every selected token contributes one (range, azimuth)
    # sample. Elevation is collapsed only for visualization.
    ra_x_index = selected_azimuth
    ra_y_index = selected_range

    # RD projection: every selected token must also contribute exactly one
    # sample, otherwise plotting all top-3 indices would create 750 RD samples
    # per range and no longer visualize "the remaining 250 points".
    #
    # Select the strongest of the stored top-3 Doppler components for each token.
    selected_mask = sparse.range_ind < range_limit
    selected_top3_power = sparse.top3_power[:, selected_mask]
    strongest_row = np.argmax(selected_top3_power, axis=0)
    token_id = np.arange(selected_range.size)
    rd_x_index = top3_doppler[strongest_row, token_id]
    rd_y_index = selected_range

    valid_rd_index = (
        (rd_x_index >= 0)
        & (rd_x_index < raw.shape[0])
    )
    rd_x_index = rd_x_index[valid_rd_index]
    rd_y_index = rd_y_index[valid_rd_index]

    # Multiple 3D tokens can project to the same 2D cell. Collapse those
    # duplicate 2D coordinates, but preserve their multiplicity in marker area.
    ra_x_index, ra_y_index, ra_counts = collapse_projected_cells(
        ra_x_index, ra_y_index
    )
    rd_x_index, rd_y_index, rd_counts = collapse_projected_cells(
        rd_x_index, rd_y_index
    )

    # Matplotlib scatter's 's' is marker area. Therefore area proportional to
    # count directly visualizes how many of the 250 selected tokens occupy the
    # projected cell.
    ra_marker_sizes = point_size * ra_counts.astype(np.float64)
    rd_marker_sizes = point_size * rd_counts.astype(np.float64)

    # Verify that the projected multiplicities preserve exactly 250 tokens per
    # range. This catches indexing mistakes immediately.
    ra_count_per_range = np.bincount(
        ra_y_index,
        weights=ra_counts,
        minlength=range_limit,
    )
    rd_count_per_range = np.bincount(
        rd_y_index,
        weights=rd_counts,
        minlength=range_limit,
    )
    if not np.all(ra_count_per_range[:range_limit] == 250):
        bad = np.where(ra_count_per_range[:range_limit] != 250)[0][:10]
        raise RuntimeError(
            f"RA projection does not preserve 250 tokens for ranges: {bad.tolist()}"
        )
    if not np.all(rd_count_per_range[:range_limit] == 250):
        bad = np.where(rd_count_per_range[:range_limit] != 250)[0][:10]
        raise RuntimeError(
            f"RD projection does not preserve 250 tokens for ranges: {bad.tolist()}"
        )

    if axes is None:
        range_axis = None
        azimuth_axis = None
        doppler_axis = None
        ra_x_label = "Azimuth bin"
        rd_x_label = "Doppler bin"
        y_label = "Range bin"

        ra_point_x = ra_x_index
        ra_point_y = ra_y_index
        rd_point_x = rd_x_index
        rd_point_y = rd_y_index
    else:
        range_axis = axes.range_values[:range_limit]
        azimuth_axis = axes.azimuth_values_deg
        doppler_axis = axes.doppler_values
        ra_x_label = "Azimuth [deg]"
        rd_x_label = "Doppler [m/s]"
        y_label = "Range [m]"

        ra_point_x = axes.azimuth_values_deg[ra_x_index]
        ra_point_y = axes.range_values[ra_y_index]
        rd_point_x = axes.doppler_values[rd_x_index]
        rd_point_y = axes.range_values[rd_y_index]

    fig, grid = plt.subplots(2, 2, figsize=(17, 11), constrained_layout=True)

    # Top left: original RA heatmap with RadarOcc-selected azimuth indices.
    m0 = _plot_heatmap(
        grid[0, 0],
        ra_db,
        f"Raw RA power ({projection} over elevation)\n"
        "+ RadarOcc selected (range, azimuth) indices",
        azimuth_axis,
        range_axis,
        ra_x_label,
        y_label,
    )
    fig.colorbar(m0, ax=grid[0, 0], label="Power [dB]")
    grid[0, 0].scatter(
        ra_point_x,
        ra_point_y,
        s=ra_marker_sizes,
        facecolors="none",
        edgecolors=point_color,
        alpha=point_alpha,
        marker="o",
        linewidths=0.35,
        rasterized=True,
        zorder=3,
        label="250 mean-power-selected tokens/range projected to RA",
    )
    ra_legend_handle = Line2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        markersize=5,
        markerfacecolor="none",
        markeredgecolor=point_color,
        markeredgewidth=0.8,
        alpha=point_alpha,
    )
    grid[0, 0].legend(
        [ra_legend_handle],
        ["Mean-power top-250 projection"],
        loc="upper right",
        fontsize=8,
        framealpha=0.85,
        borderpad=0.35,
        handletextpad=0.45,
    )

    # Top right: keep the previous relative-power diagnostic.
    grid[0, 1].hist(angular_delta_db, bins=80)
    grid[0, 1].axvline(-3.0)
    grid[0, 1].axvline(-10.0)
    grid[0, 1].axvline(-20.0)
    grid[0, 1].set_title(
        "Strongest stored Doppler component relative to\n"
        "angular maximum at the same range and Doppler"
    )
    grid[0, 1].set_xlabel("Relative power at fixed (range, Doppler) [dB]")
    grid[0, 1].set_ylabel("Number of selected points")

    # Bottom left: original RD heatmap with all saved top-3 Doppler indices.
    m2 = _plot_heatmap(
        grid[1, 0],
        rd_db,
        f"Raw RD power ({projection} over azimuth/elevation)\n"
        "+ same 250 selected tokens via strongest stored Doppler index",
        doppler_axis,
        range_axis,
        rd_x_label,
        y_label,
    )
    fig.colorbar(m2, ax=grid[1, 0], label="Power [dB]")
    grid[1, 0].scatter(
        rd_point_x,
        rd_point_y,
        s=rd_marker_sizes,
        facecolors="none",
        edgecolors=point_color,
        alpha=point_alpha,
        marker="o",
        linewidths=0.35,
        rasterized=True,
        zorder=3,
        label="250 mean-power-selected tokens/range projected to RD",
    )
    rd_legend_handle = Line2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        markersize=5,
        markerfacecolor="none",
        markeredgecolor=point_color,
        markeredgewidth=0.8,
        alpha=point_alpha,
    )
    grid[1, 0].legend(
        [rd_legend_handle],
        ["Selected-token projection"],
        loc="upper right",
        fontsize=8,
        framealpha=0.85,
        borderpad=0.35,
        handletextpad=0.45,
    )

    within_3, db_3_10, db_10_20, below_20 = diagnostic_ratios(angular_delta_db)
    mean_range_ratios = diagnostic_ratios(mean_range_delta_db)
    coordinate_match, top3_match, mean_error = validation
    summary = (
        f"Frame: {frame_id}\n"
        f"Raw: {raw_path.name}\n"
        f"Sparse: {sparse_path.name}\n\n"
        f"Raw D-R-E-A shape: {tuple(raw.shape)}\n"
        f"Sparse points in file: {sparse.range_ind.size:,}\n"
        f"Plotted ranges: {range_limit}\n"
        f"Selected tokens: {selected_range.size:,} "
        f"({range_limit} ranges x 250)\n"
        f"RA occupied cells: {ra_point_x.size:,}\n"
        f"RD occupied cells: {rd_point_x.size:,}\n"
        f"RA projected count/range: exactly 250\n"
        f"RD projected count/range: exactly 250\n"
        f"RD uses strongest of stored top-3 Doppler bins\n\n"
        f"Preprocessing coordinate-set match: {coordinate_match:.2%}\n"
        f"Top-3 Doppler-bin set match: {top3_match:.2%}\n"
        f"Mean-power absolute error: {mean_error:.6g}\n\n"
        f"Fixed-(range,Doppler) angular diagnostic:\n"
        f"  within 3 dB of angular maximum: {within_3:.2%}\n"
        f"  3–10 dB below angular maximum: {db_3_10:.2%}\n"
        f"  10–20 dB below angular maximum: {db_10_20:.2%}\n"
        f"  more than 20 dB below: {below_20:.2%}\n\n"
        f"Mean-power vs strongest E-A at same range:\n"
        f"  within 3 dB: {mean_range_ratios[0]:.2%}\n"
        f"  more than 20 dB below: {mean_range_ratios[3]:.2%}\n\n"
        "Red-circle AREA represents how many of the 250 selected tokens\n"
        "project into that RA/RD cell. Every range contributes exactly 250\n"
        "tokens to each projection. These are not sidelobe labels."
    )
    grid[1, 1].axis("off")
    grid[1, 1].text(0.0, 1.0, summary, va="top", family="monospace")

    fig.suptitle(
        f"RadarOcc mean-power top-250 tokens overlaid on original RA and RD maps — "
        f"frame {frame_id}",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return within_3, db_3_10, db_10_20, below_20

def process_frame(
    raw_path: Path,
    sparse_path: Path,
    args: argparse.Namespace,
) -> list[FrameStats]:
    frame_id = trailing_frame_id(raw_path)
    print(f"\n[frame {frame_id}] raw={raw_path}")
    print(f"[frame {frame_id}] sparse={sparse_path}")

    raw = load_raw(raw_path)
    sparse = load_sparse(sparse_path)
    print(f"[frame {frame_id}] normalized raw shape D-R-E-A = {raw.shape}")
    print(f"[frame {frame_id}] sparse points = {sparse.range_ind.size}")

    if args.skip_validation:
        validation = (float("nan"), float("nan"), float("nan"))
    else:
        validation = validate_sparse_against_raw(raw, sparse, args.k)
        print(
            f"[frame {frame_id}] coordinate match={validation[0]:.2%}, "
            f"top3-bin match={validation[1]:.2%}, mean abs error={validation[2]:.6g}"
        )

    physical_axes = load_physical_axes(args.info_arr, args.arr_doppler, raw.shape)
    range_limits = [raw.shape[1]]
    if 0 < args.model_ranges < raw.shape[1]:
        range_limits.append(args.model_ranges)

    results: list[FrameStats] = []
    for range_limit in range_limits:
        tag = f"all{raw.shape[1]}" if range_limit == raw.shape[1] else f"model{range_limit}"
        maps = build_maps(raw, sparse, range_limit, args.projection)
        output_png = args.output_dir / f"frame_{frame_id}_rd_ra_{tag}.png"
        ratios = create_figure(
            raw=raw,
            sparse=sparse,
            maps=maps,
            frame_id=frame_id,
            raw_path=raw_path,
            sparse_path=sparse_path,
            output_path=output_png,
            projection=args.projection,
            axes=physical_axes,
            validation=validation,
            point_size=args.point_size,
            point_alpha=args.point_alpha,
            point_color=args.point_color,
        )
        print(f"[frame {frame_id}] wrote {output_png}")
        results.append(
            FrameStats(
                frame_id=frame_id,
                raw_path=str(raw_path),
                sparse_path=str(sparse_path),
                raw_shape=str(tuple(raw.shape)),
                sparse_points=int(sparse.range_ind.size),
                plotted_ranges=int(range_limit),
                coordinate_match_ratio=float(validation[0]),
                top3_bin_match_ratio=float(validation[1]),
                mean_abs_error=float(validation[2]),
                within_3db_ratio=ratios[0],
                db_3_to_10_ratio=ratios[1],
                db_10_to_20_ratio=ratios[2],
                below_20db_ratio=ratios[3],
                output_png=str(output_png),
            )
        )
    return results


def write_summary(path: Path, rows: Iterable[FrameStats]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].__dict__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = collect_raw_files(args)
    sparse_index = index_sparse_files(args.sparse_dir)

    all_stats: list[FrameStats] = []
    for raw_path in raw_files:
        frame_number = normalized_frame_id(trailing_frame_id(raw_path))
        sparse_path = sparse_index.get(frame_number)
        if sparse_path is None:
            print(
                f"[warning] no EAsparse file matched raw frame {raw_path.name}; skipping.",
                file=sys.stderr,
            )
            continue
        all_stats.extend(process_frame(raw_path, sparse_path, args))

    if not all_stats:
        raise RuntimeError("No raw/sparse frame pairs were processed.")

    summary_path = args.output_dir / "summary.csv"
    write_summary(summary_path, all_stats)
    print(f"\nWrote summary: {summary_path}")
    print(
        "Interpretation reminder: RA/RD projections can reveal repeated angular bands and "
        "weak selected points, but they cannot by themselves prove that every selected point "
        "is a sidelobe."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
