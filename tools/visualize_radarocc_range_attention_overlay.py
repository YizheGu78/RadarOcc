#!/usr/bin/env python3
"""
Overlay RadarOcc range-attention behavior on the original RA and RD maps.

Purpose
-------
The RadarOcc preprocessing keeps exactly 250 (elevation, azimuth) tokens per
range using Doppler mean power. Range-wise attention does not delete or move
those tokens. Therefore, "after attention" must be visualized as a change in
how much attention mass is assigned to the same 250 tokens.

This script draws, with the same projection logic:

RA before:
    circle area = number of selected 3D tokens projected into each (R, A) cell.

RA after:
    circle area = sum of attention-received scores of those tokens in each
    (R, A) cell.

RD before:
    each selected token contributes exactly one RD sample using the strongest
    of its stored top-3 Doppler components. Circle area = token count.

RD after:
    same RD coordinates, but circle area = summed attention-received score.

For every range:
    sum(before token weights) = 250
    sum(after attention weights) = 250

Thus the before/after marker areas are directly comparable. The change maps
show log2(after_mass / before_count):
    < 0: relatively suppressed by attention routing
    = 0: unchanged
    > 0: relatively enhanced

Important limitation
--------------------
Range attention is feature mixing, not hard point filtering. Low received
attention is evidence that a token contributes less to other tokens in the
chosen attention diagnostic; it is not a definitive sidelobe label.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.io import loadmat
import torch


NUM_RANGES = 175
TOKENS_PER_RANGE = 250
NUM_ELEVATION = 37
NUM_AZIMUTH = 107
NUM_DOPPLER = 64
EPS = np.finfo(np.float32).tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RadarOcc mean-power top-250 token projections before and "
            "after trained range-wise attention."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="RadarOcc repository root. Default: current directory.",
    )
    parser.add_argument(
        "--raw-file",
        type=Path,
        required=True,
        help="Matching K-Radar raw tesseract_*.mat or normalized .npy file.",
    )
    parser.add_argument(
        "--sparse-file",
        type=Path,
        required=True,
        help="Matching RadarOcc EAsparse_*.npz file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trained RadarOcc checkpoint containing range_attn weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for PNG and NPZ outputs.",
    )
    parser.add_argument(
        "--index-mode",
        choices=("mixed", "fixed"),
        default="mixed",
        help=(
            "Use 'mixed' for a checkpoint trained with the currently active "
            "local-index gather bug; use 'fixed' for a checkpoint trained with "
            "range offsets."
        ),
    )
    parser.add_argument(
        "--after-score",
        choices=("received", "rollout", "feature_norm"),
        default="received",
        help=(
            "'received': attention mass received in the last layer; "
            "'rollout': four-layer attention rollout mass; "
            "'feature_norm': final 5-D feature norm normalized to mean 1 per range."
        ),
    )
    parser.add_argument(
        "--projection",
        choices=("max", "mean"),
        default="max",
        help="How hidden dimensions are reduced for raw RA/RD backgrounds.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device. Default: cuda.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.2,
        help=(
            "Marker-area scale. Actual marker area equals point_size multiplied "
            "by projected count/attention mass."
        ),
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.80,
        help="Opacity of hollow red circles.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output PNG resolution.",
    )
    return parser.parse_args()


def trailing_frame_id(path: Path) -> int:
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise ValueError(f"Cannot infer frame ID from {path.name}")
    return int(numbers[-1])


def normalize_to_drea(array: np.ndarray, path: Path) -> np.ndarray:
    """Return radar tensor in [Doppler, Range, Elevation, Azimuth] order."""
    array = np.asarray(array).squeeze()
    if array.ndim != 4:
        raise ValueError(f"Expected a 4-D tensor in {path}, got {array.shape}")

    target = (
        NUM_DOPPLER,
        256,
        NUM_ELEVATION,
        NUM_AZIMUTH,
    )
    if array.shape == target:
        return array

    # Common K-Radar visualization order: D-R-A-E.
    alternative = (
        NUM_DOPPLER,
        256,
        NUM_AZIMUTH,
        NUM_ELEVATION,
    )
    if array.shape == alternative:
        return np.transpose(array, (0, 1, 3, 2))

    sizes = list(array.shape)
    required = list(target)
    if all(sizes.count(value) == 1 for value in required):
        permutation = tuple(sizes.index(value) for value in required)
        return np.transpose(array, permutation)

    raise ValueError(
        f"Cannot infer D-R-E-A axes for {path}; got shape {array.shape}"
    )


def load_raw(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Raw file does not exist: {path}")

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


def load_sparse(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Sparse file does not exist: {path}")

    with np.load(path) as data:
        required = ("power_val", "elevation_ind", "azimuth_ind")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{path.name} is missing: {missing}")

        power = np.asarray(data["power_val"])
        elevation = np.asarray(data["elevation_ind"]).reshape(-1)
        azimuth = np.asarray(data["azimuth_ind"]).reshape(-1)

    if power.ndim != 2:
        raise ValueError(f"power_val must be 2-D, got {power.shape}")
    if power.shape[0] != 8 and power.shape[1] == 8:
        power = power.T
    if power.shape[0] != 8:
        raise ValueError(f"Expected power_val [8, N], got {power.shape}")

    required_points = NUM_RANGES * TOKENS_PER_RANGE
    if power.shape[1] < required_points:
        raise ValueError(
            f"Need at least {required_points} sparse points, got {power.shape[1]}"
        )
    if elevation.size != power.shape[1] or azimuth.size != power.shape[1]:
        raise ValueError("Sparse coordinate lengths do not match power_val.")

    return power, elevation, azimuth


def load_attention_class(repo_root: Path) -> Any:
    source = (
        repo_root
        / "projects/occ_plugin/occupancy/voxel_encoder/sparse_lidar_enc.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Cannot find source file: {source}")

    spec = importlib.util.spec_from_file_location(
        "radarocc_sparse_encoder_attention_vis",
        source,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import source file: {source}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "MultiLayerRangeAttention"):
        raise AttributeError(
            f"MultiLayerRangeAttention not found in {source}"
        )
    return module.MultiLayerRangeAttention


def load_range_attention_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict.")

    marker = "pts_middle_encoder.range_attn."
    extracted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if marker in key:
            extracted[key.split(marker, 1)[1]] = value

    if not extracted:
        candidates = [key for key in state_dict if "range_attn" in key][:30]
        raise KeyError(
            "No range-attention weights found. "
            f"Candidate keys: {candidates}"
        )
    return extracted


def prepare_exact_attention_input(
    power_np: np.ndarray,
    elevation_np: np.ndarray,
    azimuth_np: np.ndarray,
    index_mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce radarocc_self_small.py immediately before range_attn."""
    power = torch.from_numpy(power_np).to(device=device)
    elevation = torch.from_numpy(elevation_np).to(
        device=device,
        dtype=torch.long,
    )
    azimuth = torch.from_numpy(azimuth_np).to(
        device=device,
        dtype=torch.long,
    )

    number_used = NUM_RANGES * TOKENS_PER_RANGE

    # Exact current detector ranking: descriptor channel 2.
    ranking = power[2, :number_used].reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
    )
    order = torch.argsort(ranking, dim=1, descending=True)

    elevation_grouped = elevation[:number_used].reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
    )
    azimuth_grouped = azimuth[:number_used].reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
    )

    elevation_out = elevation_grouped.gather(1, order).reshape(-1)
    azimuth_out = azimuth_grouped.gather(1, order).reshape(-1)
    range_out = (
        torch.arange(NUM_RANGES, device=device)
        .unsqueeze(1)
        .expand(NUM_RANGES, TOKENS_PER_RANGE)
        .reshape(-1)
    )

    if index_mode == "mixed":
        # Exact active local-index gather in the current branch.
        gather_index = order.reshape(-1)
    else:
        range_offsets = (
            torch.arange(NUM_RANGES, device=device).unsqueeze(1)
            * TOKENS_PER_RANGE
        )
        gather_index = (order + range_offsets).reshape(-1)

    descriptors = power.T[gather_index].to(torch.float32)
    return descriptors, range_out, elevation_out, azimuth_out


@torch.no_grad()
def trace_range_attention(
    attention: torch.nn.Module,
    descriptors: torch.Tensor,
    elevation: torch.Tensor,
    azimuth: torch.Tensor,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Run exact repository attention and retain per-layer matrices."""
    top_values = torch.log10(
        descriptors[:, :3]
    ).reshape(NUM_RANGES, TOKENS_PER_RANGE, 3)
    top_indices = descriptors[:, 3:6].reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
        3,
    ).to(torch.long)
    mean_variance = torch.log10(
        descriptors[:, -2:]
    ).reshape(NUM_RANGES, TOKENS_PER_RANGE, 2)

    if (
        not torch.isfinite(top_values).all()
        or not torch.isfinite(mean_variance).all()
    ):
        raise FloatingPointError(
            "torch.log10 produced NaN/Inf. Inspect non-positive descriptor values."
        )

    doppler_embedding = attention.doppler_embed(
        top_indices
    ).flatten(2).to(torch.float32)
    elevation_embedding = attention.elevation_embed(
        elevation.reshape(NUM_RANGES, TOKENS_PER_RANGE)
    ).to(torch.float32)
    azimuth_embedding = attention.azimuth_embed(
        azimuth.reshape(NUM_RANGES, TOKENS_PER_RANGE)
    ).to(torch.float32)

    x = torch.cat([top_values, mean_variance], dim=-1)
    probability_matrices: list[torch.Tensor] = []

    for layer in attention.attention_layers:
        layer_input = torch.cat(
            [
                x,
                elevation_embedding,
                azimuth_embedding,
                doppler_embedding,
            ],
            dim=-1,
        ).to(torch.float32)

        query = layer.query(layer_input)
        key = layer.key(layer_input).transpose(-2, -1)
        value = layer.value(layer_input)

        probability = torch.softmax(
            torch.matmul(query, key) / math.sqrt(query.size(-1)),
            dim=-1,
        )
        x = torch.matmul(probability, value)
        probability_matrices.append(probability)

    output = torch.cat([x, top_indices], dim=-1).reshape(
        NUM_RANGES * TOKENS_PER_RANGE,
        8,
    )

    repository_output = attention(descriptors, elevation, azimuth)
    maximum_error = float(
        (output - repository_output).abs().max().item()
    )
    if maximum_error > 1e-5:
        raise RuntimeError(
            "Traced attention does not match repository forward: "
            f"max_abs_error={maximum_error}"
        )

    last_probability = probability_matrices[-1]
    last_received = last_probability.sum(dim=1)

    # Attention rollout heuristic through the four row-stochastic matrices.
    rollout = probability_matrices[0]
    for probability in probability_matrices[1:]:
        rollout = torch.matmul(probability, rollout)
    rollout_received = rollout.sum(dim=1)

    entropy = -(
        last_probability.clamp_min(1e-12)
        * last_probability.clamp_min(1e-12).log()
    ).sum(dim=-1) / math.log(TOKENS_PER_RANGE)

    return output, probability_matrices, last_received, rollout_received, entropy


def db10(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(
        np.maximum(np.asarray(values, dtype=np.float64), EPS)
    )


def build_raw_maps(
    raw: np.ndarray,
    projection: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw_used = raw[:, :NUM_RANGES, :, :]

    mean_doppler = raw_used.mean(axis=0)  # [R, E, A]
    if projection == "max":
        ra = mean_doppler.max(axis=1)  # [R, A]
        rd = raw_used.max(axis=(2, 3)).T  # [R, D]
    else:
        ra = mean_doppler.mean(axis=1)
        rd = raw_used.mean(axis=(2, 3)).T

    return db10(ra), db10(rd)


def strongest_doppler_per_token(
    descriptors: np.ndarray,
) -> np.ndarray:
    top3_power = descriptors[:, 0:3]
    top3_index = np.rint(descriptors[:, 3:6]).astype(np.int64)
    strongest_row = np.argmax(top3_power, axis=1)
    token_id = np.arange(descriptors.shape[0])
    strongest = top3_index[token_id, strongest_row]

    if np.any((strongest < 0) | (strongest >= NUM_DOPPLER)):
        bad = int(np.sum((strongest < 0) | (strongest >= NUM_DOPPLER)))
        raise IndexError(f"Found {bad} invalid strongest Doppler indices.")
    return strongest


def normalize_feature_score_per_range(
    feature_norm: np.ndarray,
) -> np.ndarray:
    score = feature_norm.reshape(NUM_RANGES, TOKENS_PER_RANGE)
    mean = score.mean(axis=1, keepdims=True)
    normalized = score / np.maximum(mean, EPS)
    return normalized.reshape(-1)


def collapse_weighted_projection(
    x_index: np.ndarray,
    y_index: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse repeated 2-D coordinates.

    Returns:
        unique_x,
        unique_y,
        number_of_tokens,
        summed_weight
    """
    x_index = np.asarray(x_index, dtype=np.int64).reshape(-1)
    y_index = np.asarray(y_index, dtype=np.int64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)

    if not (x_index.size == y_index.size == weights.size):
        raise ValueError("Projection arrays have inconsistent lengths.")

    pairs = np.column_stack((y_index, x_index))
    unique_pairs, inverse, counts = np.unique(
        pairs,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    summed_weight = np.zeros(unique_pairs.shape[0], dtype=np.float64)
    np.add.at(summed_weight, inverse, weights)

    return (
        unique_pairs[:, 1],
        unique_pairs[:, 0],
        counts.astype(np.float64),
        summed_weight,
    )


def projection_ratio_map(
    width: int,
    x_index: np.ndarray,
    y_index: np.ndarray,
    before_count: np.ndarray,
    after_mass: np.ndarray,
) -> np.ndarray:
    result = np.full(
        (NUM_RANGES, width),
        np.nan,
        dtype=np.float64,
    )
    ratio = after_mass / np.maximum(before_count, EPS)
    result[y_index, x_index] = np.log2(np.maximum(ratio, EPS))
    return result


def plot_background_with_circles(
    ax: plt.Axes,
    background: np.ndarray,
    x_index: np.ndarray,
    y_index: np.ndarray,
    marker_mass: np.ndarray,
    title: str,
    x_label: str,
    point_size: float,
    point_alpha: float,
) -> None:
    image = ax.imshow(
        background,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Power [dB]")

    ax.scatter(
        x_index,
        y_index,
        s=point_size * marker_mass,
        facecolors="none",
        edgecolors="red",
        linewidths=0.45,
        alpha=point_alpha,
        marker="o",
        rasterized=True,
        zorder=3,
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Range bin")

    fixed_legend = Line2D(
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
        [fixed_legend],
        ["Projected selected-token mass"],
        loc="upper right",
        fontsize=7,
        framealpha=0.85,
        borderpad=0.3,
        handletextpad=0.4,
    )


def plot_ratio_map(
    ax: plt.Axes,
    ratio_map: np.ndarray,
    title: str,
    x_label: str,
) -> None:
    finite = ratio_map[np.isfinite(ratio_map)]
    if finite.size:
        absolute_limit = max(
            0.5,
            float(np.nanpercentile(np.abs(finite), 98)),
        )
        absolute_limit = min(absolute_limit, 3.0)
    else:
        absolute_limit = 1.0

    image = ax.imshow(
        ratio_map,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-absolute_limit,
        vmax=absolute_limit,
    )
    plt.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="log2(after attention mass / before token count)",
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Range bin")


def summarize_ratio(
    before_count: np.ndarray,
    after_mass: np.ndarray,
) -> tuple[float, float, float]:
    ratio = after_mass / np.maximum(before_count, EPS)
    suppressed = float(np.mean(ratio < 0.8))
    similar = float(np.mean((ratio >= 0.8) & (ratio <= 1.25)))
    enhanced = float(np.mean(ratio > 1.25))
    return suppressed, similar, enhanced


def main() -> int:
    args = parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    raw_file = args.raw_file.expanduser().resolve()
    sparse_file = args.sparse_file.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if trailing_frame_id(raw_file) != trailing_frame_id(sparse_file):
        raise ValueError(
            f"Frame mismatch: raw={raw_file.name}, sparse={sparse_file.name}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    print(f"[1/7] Loading raw: {raw_file}")
    raw = load_raw(raw_file)
    print(f"      raw D-R-E-A shape: {raw.shape}")

    print(f"[2/7] Loading sparse: {sparse_file}")
    power, elevation, azimuth = load_sparse(sparse_file)

    print("[3/7] Reproducing exact detector input")
    (
        descriptors,
        range_index_t,
        elevation_index_t,
        azimuth_index_t,
    ) = prepare_exact_attention_input(
        power,
        elevation,
        azimuth,
        args.index_mode,
        device,
    )

    print("[4/7] Loading trained range attention")
    AttentionClass = load_attention_class(repo_root)
    attention = AttentionClass(
        num_layers=4,
        num_ranges=NUM_RANGES,
        K=TOKENS_PER_RANGE,
        doppler=3,
    ).to(device).eval()
    attention.load_state_dict(
        load_range_attention_state(checkpoint),
        strict=True,
    )

    print("[5/7] Running and tracing range attention")
    (
        attention_output,
        probability_matrices,
        last_received_t,
        rollout_received_t,
        entropy_t,
    ) = trace_range_attention(
        attention,
        descriptors,
        elevation_index_t,
        azimuth_index_t,
    )

    descriptors_np = descriptors.detach().cpu().numpy()
    output_np = attention_output.detach().cpu().numpy()
    range_index = range_index_t.detach().cpu().numpy().astype(np.int64)
    azimuth_index = azimuth_index_t.detach().cpu().numpy().astype(np.int64)

    if args.after_score == "received":
        after_score = last_received_t.detach().cpu().numpy().reshape(-1)
        score_description = "last-layer received attention"
    elif args.after_score == "rollout":
        after_score = rollout_received_t.detach().cpu().numpy().reshape(-1)
        score_description = "four-layer attention rollout"
    else:
        feature_norm = np.linalg.norm(output_np[:, :5], axis=1)
        after_score = normalize_feature_score_per_range(feature_norm)
        score_description = "normalized final 5-D feature norm"

    # Before every token has unit weight. After weights are normalized so each
    # range still contributes exactly 250 total mass.
    before_score = np.ones_like(after_score, dtype=np.float64)

    before_per_range = before_score.reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
    ).sum(axis=1)
    after_per_range = after_score.reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
    ).sum(axis=1)

    if not np.allclose(before_per_range, TOKENS_PER_RANGE, atol=1e-5):
        raise RuntimeError("Before projection does not contain 250 tokens/range.")
    if not np.allclose(after_per_range, TOKENS_PER_RANGE, atol=1e-3):
        raise RuntimeError(
            "After attention score is not normalized to 250 mass/range. "
            f"min={after_per_range.min()}, max={after_per_range.max()}"
        )

    strongest_doppler = strongest_doppler_per_token(descriptors_np)
    ra_background, rd_background = build_raw_maps(raw, args.projection)

    (
        ra_x,
        ra_y,
        ra_before_count,
        ra_before_mass,
    ) = collapse_weighted_projection(
        azimuth_index,
        range_index,
        before_score,
    )
    (
        ra_x_after,
        ra_y_after,
        _,
        ra_after_mass,
    ) = collapse_weighted_projection(
        azimuth_index,
        range_index,
        after_score,
    )

    (
        rd_x,
        rd_y,
        rd_before_count,
        rd_before_mass,
    ) = collapse_weighted_projection(
        strongest_doppler,
        range_index,
        before_score,
    )
    (
        rd_x_after,
        rd_y_after,
        _,
        rd_after_mass,
    ) = collapse_weighted_projection(
        strongest_doppler,
        range_index,
        after_score,
    )

    if not (
        np.array_equal(ra_x, ra_x_after)
        and np.array_equal(ra_y, ra_y_after)
        and np.array_equal(rd_x, rd_x_after)
        and np.array_equal(rd_y, rd_y_after)
    ):
        raise RuntimeError(
            "Projected coordinates changed after attention, which should not happen."
        )

    ra_ratio = projection_ratio_map(
        NUM_AZIMUTH,
        ra_x,
        ra_y,
        ra_before_count,
        ra_after_mass,
    )
    rd_ratio = projection_ratio_map(
        NUM_DOPPLER,
        rd_x,
        rd_y,
        rd_before_count,
        rd_after_mass,
    )

    print("[6/7] Creating before/after overlay figure")
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(22, 12),
        constrained_layout=True,
    )

    plot_background_with_circles(
        axes[0, 0],
        ra_background,
        ra_x,
        ra_y,
        ra_before_count,
        "RA before range attention\ncircle area = selected-token count",
        "Azimuth bin",
        args.point_size,
        args.point_alpha,
    )
    plot_background_with_circles(
        axes[0, 1],
        ra_background,
        ra_x,
        ra_y,
        ra_after_mass,
        f"RA after range attention\ncircle area = {score_description} mass",
        "Azimuth bin",
        args.point_size,
        args.point_alpha,
    )
    plot_ratio_map(
        axes[0, 2],
        ra_ratio,
        "RA relative redistribution\nblue suppressed, red enhanced",
        "Azimuth bin",
    )

    plot_background_with_circles(
        axes[1, 0],
        rd_background,
        rd_x,
        rd_y,
        rd_before_count,
        "RD before range attention\nsame 250 tokens via strongest stored Doppler",
        "Doppler bin",
        args.point_size,
        args.point_alpha,
    )
    plot_background_with_circles(
        axes[1, 1],
        rd_background,
        rd_x,
        rd_y,
        rd_after_mass,
        f"RD after range attention\ncircle area = {score_description} mass",
        "Doppler bin",
        args.point_size,
        args.point_alpha,
    )
    plot_ratio_map(
        axes[1, 2],
        rd_ratio,
        "RD relative redistribution\nblue suppressed, red enhanced",
        "Doppler bin",
    )

    frame = trailing_frame_id(sparse_file)
    figure.suptitle(
        "RadarOcc mean-power top-250 tokens before/after range attention "
        f"— frame {frame:05d}\n"
        f"checkpoint={checkpoint.name}, index_mode={args.index_mode}, "
        f"after_score={args.after_score}",
        fontsize=15,
    )

    output_png = output_dir / (
        f"frame_{frame:05d}_range_attention_overlay_"
        f"{args.index_mode}_{args.after_score}.png"
    )
    figure.savefig(output_png, dpi=args.dpi)
    plt.close(figure)

    ra_suppressed, ra_similar, ra_enhanced = summarize_ratio(
        ra_before_count,
        ra_after_mass,
    )
    rd_suppressed, rd_similar, rd_enhanced = summarize_ratio(
        rd_before_count,
        rd_after_mass,
    )

    entropy = entropy_t.detach().cpu().numpy().reshape(-1)
    output_npz = output_dir / (
        f"frame_{frame:05d}_range_attention_overlay_"
        f"{args.index_mode}_{args.after_score}.npz"
    )
    np.savez_compressed(
        output_npz,
        range_ind=range_index,
        azimuth_ind=azimuth_index,
        strongest_doppler_ind=strongest_doppler,
        descriptor_before=descriptors_np,
        features_after=output_np,
        after_score=after_score,
        last_attention_received=last_received_t.detach().cpu().numpy(),
        rollout_attention_received=rollout_received_t.detach().cpu().numpy(),
        last_attention_entropy=entropy,
        ra_x=ra_x,
        ra_y=ra_y,
        ra_before_count=ra_before_count,
        ra_after_mass=ra_after_mass,
        ra_log2_ratio=ra_ratio,
        rd_x=rd_x,
        rd_y=rd_y,
        rd_before_count=rd_before_count,
        rd_after_mass=rd_after_mass,
        rd_log2_ratio=rd_ratio,
    )

    print("[7/7] Done")
    print(f"Saved figure: {output_png}")
    print(f"Saved arrays: {output_npz}")
    print()
    print("Per-range mass check:")
    print(
        f"  before: min={before_per_range.min():.6f}, "
        f"max={before_per_range.max():.6f}"
    )
    print(
        f"  after:  min={after_per_range.min():.6f}, "
        f"max={after_per_range.max():.6f}"
    )
    print()
    print("Projected-cell redistribution:")
    print(
        f"  RA suppressed (<0.8x): {ra_suppressed:.2%}, "
        f"similar: {ra_similar:.2%}, enhanced (>1.25x): {ra_enhanced:.2%}"
    )
    print(
        f"  RD suppressed (<0.8x): {rd_suppressed:.2%}, "
        f"similar: {rd_similar:.2%}, enhanced (>1.25x): {rd_enhanced:.2%}"
    )
    print(
        f"  Last-layer entropy: {entropy.mean():.6f} ± {entropy.std():.6f}"
    )
    print()
    print(
        "Interpretation: coordinates and Doppler IDs are unchanged. "
        "Only attention/feature mass is redistributed among the same 250 "
        "tokens per range. Broad angular structures that retain large circles "
        "after attention are residual candidates, not confirmed sidelobe labels."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
