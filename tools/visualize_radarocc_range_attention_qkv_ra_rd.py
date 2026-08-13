#!/usr/bin/env python3
"""
RadarOcc Range-Attention visualization for fixed K-Radar frames.

Default experiment / data in this script
----------------------------------------
Project:
    /home/user1/projects/RadarOcc

Raw scene-11 radar:
    /home/user1/projects/RadarOcc/data/K-Radar/11/radar_tensor_8doppler

RadarOcc EAsparse scene-11:
    /home/user1/projects/RadarOcc/data/RadarOcc_8doppler/11/radar_tensor_8doppler

Checkpoint:
    /home/user1/projects/RadarOcc/work_dirs/
    radarocc_small_fp32_idfix_timealign_v2/epoch_4.pth

Frames:
    34, 80, 101

What this script reproduces
---------------------------
It reproduces the exact RadarOcc_small preprocessing immediately before
RadarEncV8small.range_attn:

    EAsparse
      -> first 175 ranges
      -> 250 tokens/range
      -> detector's channel-2 sorting
      -> fixed or mixed descriptor gather
      -> trained MultiLayerRangeAttention (4 layers)

For the ID-fixed/time-aligned checkpoint, use:
    --index-mode fixed

The repository Range-Attention block computes, in every layer:

    S = Q K^T / sqrt(d)
    A = softmax(S)
    O = A V

where d = 32 for the current SelfAttentionBlock.

To map attention back to each original selected token j, this script uses:

    received_j = sum_i A[i, j]

and defines the total source-token feature contribution:

    contribution_j
        = received_j * ||V_j||_2
        = sum_i || A[i,j] V_j ||_2

because A[i,j] >= 0.

Since Range Attention is computed independently inside each range, the
contribution is normalized to mean 1 inside each range for visualization:

    relative_j = contribution_j / mean_range(contribution)

This makes "before" and "after" directly comparable:
    - before: every selected token has mass 1
    - after: every selected token has relative contribution mass
    - each range has total mass 250 in both cases

Outputs per frame
-----------------
1) frame_XXXXX_RA_before_after_attention.png
      Selected mean-power RA before attention
      Attention-weighted power RA after attention
      Power-change heatmap in dB

2) frame_XXXXX_RD_attention_top3_mean.png
      Top-1 Doppler after attention
      Top-2 Doppler after attention
      Top-3 Doppler after attention
      Attention-weighted mean Doppler spectrum

3) frame_XXXXX_attention_data.npz
      Coordinates, sorted top-3 Doppler bins, per-layer received attention,
      per-layer contribution, final-layer contribution, RA/RD projections,
      and weighted mean RD.

Optional:
    --save-full-matrices
stores final-layer Q/K/V, QK^T/sqrt(d), and softmax(A). This is large.

Important interpretation
------------------------
This is a Range-Attention contribution diagnostic, not an automatic sidelobe
classifier. A location becoming relatively weaker after attention indicates
lower feature contribution within that range; it does not by itself prove
that the point is a physical radar sidelobe.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.io import loadmat
import torch


# ---------------------------------------------------------------------------
# RadarOcc / K-Radar constants used by the current Small model
# ---------------------------------------------------------------------------
NUM_DOPPLER = 64
NUM_RAW_RANGES = 256
NUM_RANGES = 175
NUM_ELEVATION = 37
NUM_AZIMUTH = 107
TOKENS_PER_RANGE = 250
DESCRIPTOR_DIM = 8
NUM_ATTENTION_LAYERS = 4
EPS = np.finfo(np.float32).tiny

DEFAULT_REPO_ROOT = Path("/home/user1/projects/RadarOcc")
DEFAULT_RAW_DIR = DEFAULT_REPO_ROOT / "data/K-Radar/11/radar_tensor_8doppler"
DEFAULT_SPARSE_DIR = (
    DEFAULT_REPO_ROOT / "data/RadarOcc_8doppler/11/radar_tensor_8doppler"
)
DEFAULT_CHECKPOINT = (
    DEFAULT_REPO_ROOT
    / "work_dirs/radarocc_small_fp32_idfix_timealign_v2/epoch_4.pth"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_REPO_ROOT
    / "work_dirs/radarocc_small_fp32_idfix_timealign_v2"
    / "range_attention_qkv_ra_rd"
)
DEFAULT_FRAMES = [34, 80, 101]


@dataclass
class SparseFrame:
    power_val: np.ndarray       # [8, N]
    range_ind: np.ndarray       # [N]
    elevation_ind: np.ndarray   # [N]
    azimuth_ind: np.ndarray     # [N]


@dataclass
class AttentionTrace:
    output: torch.Tensor
    layer_received: np.ndarray
    layer_value_norm: np.ndarray
    layer_contribution_raw: np.ndarray

    final_query: np.ndarray
    final_key: np.ndarray
    final_value: np.ndarray
    final_logits: np.ndarray
    final_probs: np.ndarray

    final_received: np.ndarray
    final_value_norm: np.ndarray
    final_contribution_raw: np.ndarray
    final_contribution_relative: np.ndarray

    max_forward_error: float
    query_dim: int
    attention_scale: float
    row_sum_error: float
    received_sum_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize trained RadarOcc Range Attention on raw RA/RD maps."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
    )
    parser.add_argument(
        "--sparse-dir",
        type=Path,
        default=DEFAULT_SPARSE_DIR,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=DEFAULT_FRAMES,
    )
    parser.add_argument(
        "--index-mode",
        choices=("fixed", "mixed"),
        default="fixed",
        help=(
            "Descriptor gather used before Range Attention. "
            "'fixed' uses range offsets and is the correct choice for "
            "radarocc_small_fp32_idfix_timealign_v2."
        ),
    )
    parser.add_argument(
        "--projection",
        choices=("max", "mean"),
        default="max",
        help=(
            "Projection used only for the raw RA/RD background. "
            "Default: max."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=3.0,
        help=(
            "Marker area scale. Before attention: area = point_size * count. "
            "After attention: area = point_size * relative contribution mass."
        ),
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--min-rd-count",
        type=int,
        default=2,
        help=(
            "Visualization-only filter for Top-1/2/3 RD panels. "
            "RD cells hit by fewer than this many selected tokens are hidden. "
            "Default 2, matching the previous requested display."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
    )
    parser.add_argument(
        "--save-full-matrices",
        action="store_true",
        help=(
            "Also save final Q, K, V, scaled logits and softmax attention "
            "matrix. This considerably increases NPZ size."
        ),
    )
    parser.add_argument(
        "--top-csv",
        type=int,
        default=50,
        help="Save the top-N final-layer source-token contributions to CSV.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help=(
            "Overwrite previously generated RA/RD PNGs. "
            "By default existing original PNGs are preserved."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# File discovery / loading
# ---------------------------------------------------------------------------
def frame_id_from_path(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        raise ValueError(f"Cannot infer frame ID from: {path}")
    return int(matches[-1])


def find_frame_file(
    directory: Path,
    frame: int,
    prefix: str,
    suffixes: tuple[str, ...],
) -> Path:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)

    exact_candidates = []
    for suffix in suffixes:
        exact_candidates.extend(
            [
                directory / f"{prefix}_{frame:05d}{suffix}",
                directory / f"{prefix}_{frame}{suffix}",
            ]
        )
    for candidate in exact_candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in suffixes
        and frame_id_from_path(path) == frame
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {prefix} file for frame {frame} in "
            f"{directory}, found: {matches}"
        )
    return matches[0].resolve()


def normalize_to_drea(array: np.ndarray, path: Path) -> np.ndarray:
    """Normalize raw K-Radar tensor to [D, R, E, A]."""
    array = np.asarray(array).squeeze()
    if array.ndim != 4:
        raise ValueError(
            f"{path}: expected a 4-D raw radar tensor, got {array.shape}"
        )

    target = (
        NUM_DOPPLER,
        NUM_RAW_RANGES,
        NUM_ELEVATION,
        NUM_AZIMUTH,
    )
    if array.shape == target:
        return array

    common_drae = (
        NUM_DOPPLER,
        NUM_RAW_RANGES,
        NUM_AZIMUTH,
        NUM_ELEVATION,
    )
    if array.shape == common_drae:
        return np.transpose(array, (0, 1, 3, 2))

    sizes = list(array.shape)
    required = list(target)
    if all(sizes.count(size) == 1 for size in required):
        permutation = tuple(sizes.index(size) for size in required)
        return np.transpose(array, permutation)

    raise ValueError(
        f"Cannot infer D-R-E-A axis order for {path}; shape={array.shape}"
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
                    f"{path}: no arrDREA and not exactly one 4-D ndarray."
                )
            array = candidates[0]

    elif path.suffix.lower() == ".npy":
        array = np.load(path)

    else:
        raise ValueError(f"Unsupported raw file type: {path.suffix}")

    return normalize_to_drea(array, path)


def load_sparse(path: Path) -> SparseFrame:
    path = path.expanduser().resolve()

    with np.load(path) as data:
        required = {
            "power_val",
            "range_ind",
            "elevation_ind",
            "azimuth_ind",
        }
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path.name}: missing arrays {sorted(missing)}")

        power = np.asarray(data["power_val"])
        range_ind = np.asarray(data["range_ind"]).reshape(-1).astype(np.int64)
        elevation_ind = (
            np.asarray(data["elevation_ind"]).reshape(-1).astype(np.int64)
        )
        azimuth_ind = (
            np.asarray(data["azimuth_ind"]).reshape(-1).astype(np.int64)
        )

    if power.ndim != 2:
        raise ValueError(f"power_val must be 2-D, got {power.shape}")
    if power.shape[0] != DESCRIPTOR_DIM and power.shape[1] == DESCRIPTOR_DIM:
        power = power.T
    if power.shape[0] != DESCRIPTOR_DIM:
        raise ValueError(
            f"Expected power_val shape [8, N], got {power.shape}"
        )

    n = power.shape[1]
    if not (
        range_ind.size
        == elevation_ind.size
        == azimuth_ind.size
        == n
    ):
        raise ValueError(
            "Coordinate-array lengths do not match power_val point count."
        )

    required_points = NUM_RANGES * TOKENS_PER_RANGE
    if n < required_points:
        raise ValueError(
            f"Need >= {required_points} sparse tokens, got {n}"
        )

    return SparseFrame(
        power_val=power,
        range_ind=range_ind,
        elevation_ind=elevation_ind,
        azimuth_ind=azimuth_ind,
    )


# ---------------------------------------------------------------------------
# Load exact repository Range-Attention class + checkpoint weights
# ---------------------------------------------------------------------------
def load_attention_class(repo_root: Path) -> Any:
    source = (
        repo_root.expanduser().resolve()
        / "projects/occ_plugin/occupancy/voxel_encoder/sparse_lidar_enc.py"
    )
    if not source.is_file():
        raise FileNotFoundError(
            f"Cannot find RadarOcc attention source: {source}"
        )

    spec = importlib.util.spec_from_file_location(
        "radarocc_range_attention_analysis_module",
        source,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import: {source}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "MultiLayerRangeAttention"):
        raise AttributeError(
            f"MultiLayerRangeAttention is missing in {source}"
        )
    return module.MultiLayerRangeAttention


def load_range_attention_state(
    checkpoint_path: Path,
) -> dict[str, torch.Tensor]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a state_dict.")

    markers = (
        "pts_middle_encoder.range_attn.",
        "range_attn.",
    )

    extracted: dict[str, torch.Tensor] = {}
    used_marker = None

    # Prefer the fully qualified path.
    for marker in markers:
        candidate: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if marker in key:
                suffix = key.split(marker, 1)[1]
                candidate[suffix] = value
        if candidate:
            extracted = candidate
            used_marker = marker
            break

    if not extracted:
        nearby = [
            key
            for key in state_dict.keys()
            if "attn" in key.lower()
        ][:50]
        raise KeyError(
            "Could not find Range-Attention parameters in checkpoint. "
            f"Nearby keys: {nearby}"
        )

    print(
        f"Loaded {len(extracted)} Range-Attention tensors "
        f"using checkpoint marker '{used_marker}'"
    )
    return extracted


# ---------------------------------------------------------------------------
# Reproduce RadarOcc_small.extract_pts_feat immediately before range_attn
# ---------------------------------------------------------------------------
def validate_sparse_grouping(sparse: SparseFrame) -> None:
    """Check the preprocessing assumption: 250 entries per range."""
    n = NUM_RANGES * TOKENS_PER_RANGE
    observed = sparse.range_ind[:n].reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
    )
    expected = np.arange(NUM_RANGES, dtype=np.int64)[:, None]

    if not np.all(observed == expected):
        bad = np.argwhere(observed != expected)
        first = bad[0].tolist() if bad.size else None
        raise RuntimeError(
            "The first 175*250 EAsparse entries are not grouped as "
            "250 tokens per range. "
            f"First mismatch [range,row]={first}."
        )


def prepare_exact_attention_input(
    sparse: SparseFrame,
    index_mode: str,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Mirror radarocc_self_small.py before pts_middle_encoder.range_attn.

    Current detector behavior:
        original_k = 250
        n_ranges   = 175
        ranking descriptor = power_val[2]
        sort all 250 tokens within every range
        gather coordinates by local top_k_indices

    Descriptor gather:
        fixed: add range offsets before flattening
        mixed: reproduce the historical local/global ID bug
    """
    validate_sparse_grouping(sparse)

    power = torch.from_numpy(sparse.power_val).to(
        device=device,
        dtype=torch.float32,
    )
    elevation = torch.from_numpy(sparse.elevation_ind).to(
        device=device,
        dtype=torch.long,
    )
    azimuth = torch.from_numpy(sparse.azimuth_ind).to(
        device=device,
        dtype=torch.long,
    )

    number_used = NUM_RANGES * TOKENS_PER_RANGE

    ranking = power[2, :number_used].reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
    )

    # torch.sort(..., descending=True) in detector.
    _, order = torch.sort(
        ranking,
        dim=1,
        descending=True,
    )

    elevation_grouped = elevation[:number_used].reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
    )
    azimuth_grouped = azimuth[:number_used].reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
    )

    selected_elevation = elevation_grouped.gather(
        1,
        order,
    ).reshape(-1)

    selected_azimuth = azimuth_grouped.gather(
        1,
        order,
    ).reshape(-1)

    selected_range = (
        torch.arange(
            NUM_RANGES,
            device=device,
            dtype=torch.long,
        )
        .unsqueeze(1)
        .expand(NUM_RANGES, TOKENS_PER_RANGE)
        .reshape(-1)
    )

    if index_mode == "fixed":
        range_offsets = (
            torch.arange(
                NUM_RANGES,
                device=device,
                dtype=torch.long,
            ).unsqueeze(1)
            * TOKENS_PER_RANGE
        )
        gather_index = (order + range_offsets).reshape(-1)

    elif index_mode == "mixed":
        gather_index = order.reshape(-1)

    else:
        raise ValueError(index_mode)

    descriptors = power.T[gather_index].to(torch.float32)

    return (
        descriptors,
        selected_range,
        selected_elevation,
        selected_azimuth,
        gather_index,
    )


# ---------------------------------------------------------------------------
# Trace Q/K/V and the exact 4-layer attention
# ---------------------------------------------------------------------------
@torch.no_grad()
def trace_range_attention(
    attention: torch.nn.Module,
    descriptors: torch.Tensor,
    elevation: torch.Tensor,
    azimuth: torch.Tensor,
    zero_azimuth_embedding: bool = False,
) -> AttentionTrace:
    """Run Range Attention and capture Q/K/V/weights.

    zero_azimuth_embedding=True is an inference-time diagnostic ablation:
    only the learned azimuth embedding is replaced by zeros.
    """
    # Exact MultiLayerRangeAttention forward for doppler=3.
    top_values = torch.log10(
        descriptors[:, :3]
    ).reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
        3,
    )

    top_indices = descriptors[:, 3:6].reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
        3,
    ).to(torch.long)

    mean_variance = torch.log10(
        descriptors[:, -2:]
    ).reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
        2,
    )

    if not torch.isfinite(top_values).all():
        raise FloatingPointError(
            "Non-finite log10(top-3 power). "
            "This would also break the repository forward."
        )
    if not torch.isfinite(mean_variance).all():
        raise FloatingPointError(
            "Non-finite log10(mean/variance). "
            "This would also break the repository forward."
        )

    if torch.any(top_indices < 0) or torch.any(top_indices >= NUM_DOPPLER):
        minimum = int(top_indices.min().item())
        maximum = int(top_indices.max().item())
        raise IndexError(
            f"Invalid Doppler indices: min={minimum}, max={maximum}"
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
    if zero_azimuth_embedding:
        azimuth_embedding = torch.zeros_like(azimuth_embedding)

    # First layer gets [log top-3 power, log mean, log variance].
    x = torch.cat(
        [top_values, mean_variance],
        dim=-1,
    )

    layer_received: list[np.ndarray] = []
    layer_value_norm: list[np.ndarray] = []
    layer_contribution: list[np.ndarray] = []

    final_query = None
    final_key = None
    final_value = None
    final_logits = None
    final_probs = None

    for layer_index, layer in enumerate(attention.attention_layers):
        layer_input = torch.cat(
            [
                x,
                elevation_embedding,
                azimuth_embedding,
                doppler_embedding,
            ],
            dim=-1,
        ).to(torch.float32)

        query = layer.query(layer_input)        # [R, K, 32]
        key_untransposed = layer.key(layer_input)  # [R, K, 32]
        value = layer.value(layer_input)        # [R, K, 5]

        query_dim = int(query.shape[-1])
        scale = math.sqrt(query_dim)

        logits = torch.matmul(
            query,
            key_untransposed.transpose(-2, -1),
        ) / scale

        probs = torch.softmax(
            logits,
            dim=-1,
        )

        # Exact repository attended output for the next layer.
        x = torch.matmul(
            probs,
            value,
        )

        # For each source/key token j:
        # received_j = sum over all output-query i of A[i,j].
        received = probs.sum(dim=1)
        value_norm = torch.linalg.vector_norm(
            value,
            ord=2,
            dim=-1,
        )

        # Total source-token contribution to all outputs:
        # sum_i || A_ij * V_j || = (sum_i A_ij) * ||V_j||.
        contribution = received * value_norm

        layer_received.append(
            received.detach().cpu().numpy()
        )
        layer_value_norm.append(
            value_norm.detach().cpu().numpy()
        )
        layer_contribution.append(
            contribution.detach().cpu().numpy()
        )

        if layer_index == len(attention.attention_layers) - 1:
            final_query = query.detach().cpu().numpy()
            final_key = key_untransposed.detach().cpu().numpy()
            final_value = value.detach().cpu().numpy()
            final_logits = logits.detach().cpu().numpy()
            final_probs = probs.detach().cpu().numpy()

    if (
        final_query is None
        or final_key is None
        or final_value is None
        or final_logits is None
        or final_probs is None
    ):
        raise RuntimeError("No Range-Attention layers were executed.")

    output = torch.cat(
        [x, top_indices],
        dim=-1,
    ).reshape(
        NUM_RANGES * TOKENS_PER_RANGE,
        8,
    )

    # Exact repository-equivalence check applies only to the original model.
    if not zero_azimuth_embedding:
        repository_output = attention(
            descriptors,
            elevation,
            azimuth,
        )
        max_forward_error = float(
            torch.max(
                torch.abs(output - repository_output)
            ).item()
        )
        if max_forward_error > 1e-5:
            raise RuntimeError(
                "Trace does not match repository Range-Attention forward. "
                f"max_abs_error={max_forward_error:.8g}"
            )
    else:
        max_forward_error = float("nan")

    layer_received_np = np.stack(layer_received, axis=0)
    layer_value_norm_np = np.stack(layer_value_norm, axis=0)
    layer_contribution_np = np.stack(layer_contribution, axis=0)

    final_received_np = layer_received_np[-1]
    final_value_norm_np = layer_value_norm_np[-1]
    final_contribution_raw_np = layer_contribution_np[-1]

    # Range Attention is independent inside each range. Normalize each range
    # to mean contribution 1 so before/after mass is directly comparable.
    range_mean = final_contribution_raw_np.mean(
        axis=1,
        keepdims=True,
    )
    if np.any(range_mean <= 0):
        bad = np.where(range_mean.reshape(-1) <= 0)[0]
        raise RuntimeError(
            f"Non-positive mean final contribution in ranges: {bad.tolist()}"
        )

    final_contribution_relative_np = (
        final_contribution_raw_np
        / range_mean
    )

    # Numerical sanity checks.
    row_sum_error = float(
        np.max(
            np.abs(
                final_probs.sum(axis=-1) - 1.0
            )
        )
    )

    received_sum_error = float(
        np.max(
            np.abs(
                final_received_np.sum(axis=-1)
                - TOKENS_PER_RANGE
            )
        )
    )

    return AttentionTrace(
        output=output,
        layer_received=layer_received_np,
        layer_value_norm=layer_value_norm_np,
        layer_contribution_raw=layer_contribution_np,
        final_query=final_query,
        final_key=final_key,
        final_value=final_value,
        final_logits=final_logits,
        final_probs=final_probs,
        final_received=final_received_np,
        final_value_norm=final_value_norm_np,
        final_contribution_raw=final_contribution_raw_np,
        final_contribution_relative=final_contribution_relative_np,
        max_forward_error=max_forward_error,
        query_dim=int(final_query.shape[-1]),
        attention_scale=math.sqrt(int(final_query.shape[-1])),
        row_sum_error=row_sum_error,
        received_sum_error=received_sum_error,
    )


# ---------------------------------------------------------------------------
# Raw RA/RD + projection helpers
# ---------------------------------------------------------------------------
def db10(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(
        np.maximum(
            np.asarray(values, dtype=np.float64),
            EPS,
        )
    )


def build_raw_maps(
    raw: np.ndarray,
    projection: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw_used = raw[:, :NUM_RANGES, :, :]

    # RA: preprocessing-relevant Doppler mean, then collapse elevation.
    doppler_mean = raw_used.mean(axis=0)  # [R, E, A]

    if projection == "max":
        ra_linear = doppler_mean.max(axis=1)  # [R, A]
        rd_linear = raw_used.max(axis=(2, 3)).T  # [R, D]
    elif projection == "mean":
        ra_linear = doppler_mean.mean(axis=1)
        rd_linear = raw_used.mean(axis=(2, 3)).T
    else:
        raise ValueError(projection)

    return db10(ra_linear), db10(rd_linear)


def sort_top3_per_token(
    descriptors_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort each token's stored top-3 pairs strongest -> weakest.

    EAsparse was generated with np.argpartition, whose returned top-3 order is
    not guaranteed to be strongest/second/third. We explicitly sort the three
    stored power/index pairs here.
    """
    top3_power = descriptors_np[:, 0:3]
    top3_doppler = np.rint(
        descriptors_np[:, 3:6]
    ).astype(np.int64)

    order = np.argsort(
        top3_power,
        axis=1,
    )[:, ::-1]

    sorted_power = np.take_along_axis(
        top3_power,
        order,
        axis=1,
    )

    sorted_doppler = np.take_along_axis(
        top3_doppler,
        order,
        axis=1,
    )

    if np.any(
        (sorted_doppler < 0)
        | (sorted_doppler >= NUM_DOPPLER)
    ):
        raise IndexError(
            "Sorted top-3 contains invalid Doppler-bin indices."
        )

    return sorted_power, sorted_doppler


def collapse_projection(
    x_index: np.ndarray,
    y_index: np.ndarray,
    weights: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Collapse repeated 2-D cells.

    Returns:
        x_unique
        y_unique
        token_count
        summed_weight
    """
    x_index = np.asarray(
        x_index,
        dtype=np.int64,
    ).reshape(-1)

    y_index = np.asarray(
        y_index,
        dtype=np.int64,
    ).reshape(-1)

    weights = np.asarray(
        weights,
        dtype=np.float64,
    ).reshape(-1)

    if not (
        x_index.size
        == y_index.size
        == weights.size
    ):
        raise ValueError(
            "Projection arrays have inconsistent lengths."
        )

    pairs = np.column_stack(
        (y_index, x_index)
    )

    unique_pairs, inverse, counts = np.unique(
        pairs,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    summed_weight = np.zeros(
        unique_pairs.shape[0],
        dtype=np.float64,
    )
    np.add.at(
        summed_weight,
        inverse,
        weights,
    )

    return (
        unique_pairs[:, 1],
        unique_pairs[:, 0],
        counts.astype(np.float64),
        summed_weight,
    )


def build_change_map(
    width: int,
    x: np.ndarray,
    y: np.ndarray,
    before_count: np.ndarray,
    after_mass: np.ndarray,
) -> np.ndarray:
    """log2(after relative mass / before token count)."""
    result = np.full(
        (NUM_RANGES, width),
        np.nan,
        dtype=np.float64,
    )

    ratio = (
        after_mass
        / np.maximum(before_count, EPS)
    )

    result[y, x] = np.log2(
        np.maximum(ratio, EPS)
    )
    return result


def build_attention_weighted_power_ra(
    range_index: np.ndarray,
    azimuth_index: np.ndarray,
    mean_power: np.ndarray,
    relative_contribution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build three directly comparable RA heatmaps.

    Before:
        P_before(r,a) = sum_j mean_power_j

    After:
        P_after(r,a)  = sum_j mean_power_j * relative_contribution_j

    Change:
        Delta_dB(r,a) = 10*log10(P_after / P_before)

    j runs over the selected elevation tokens that project to the same
    (range, azimuth) cell.

    IMPORTANT:
    P_after is an attention-weighted diagnostic quantity. It is NOT a new
    physical radar measurement after the neural network.
    """
    range_index = np.asarray(range_index, dtype=np.int64).reshape(-1)
    azimuth_index = np.asarray(azimuth_index, dtype=np.int64).reshape(-1)
    mean_power = np.asarray(mean_power, dtype=np.float64).reshape(-1)
    relative_contribution = np.asarray(
        relative_contribution,
        dtype=np.float64,
    ).reshape(-1)

    if not (
        range_index.size
        == azimuth_index.size
        == mean_power.size
        == relative_contribution.size
    ):
        raise ValueError("RA power inputs have inconsistent lengths.")

    valid = (
        (range_index >= 0)
        & (range_index < NUM_RANGES)
        & (azimuth_index >= 0)
        & (azimuth_index < NUM_AZIMUTH)
        & np.isfinite(mean_power)
        & np.isfinite(relative_contribution)
        & (mean_power > 0.0)
        & (relative_contribution >= 0.0)
    )

    r = range_index[valid]
    a = azimuth_index[valid]
    p = mean_power[valid]
    w = relative_contribution[valid]

    before_linear = np.zeros(
        (NUM_RANGES, NUM_AZIMUTH),
        dtype=np.float64,
    )
    after_linear = np.zeros_like(before_linear)

    # Collapse all elevations that land at the same (range, azimuth) cell.
    np.add.at(before_linear, (r, a), p)
    np.add.at(after_linear, (r, a), p * w)

    support = before_linear > 0.0

    before_db = np.full_like(before_linear, np.nan)
    after_db = np.full_like(after_linear, np.nan)
    change_db = np.full_like(before_linear, np.nan)

    before_db[support] = 10.0 * np.log10(
        np.maximum(before_linear[support], EPS)
    )
    after_db[support] = 10.0 * np.log10(
        np.maximum(after_linear[support], EPS)
    )
    change_db[support] = 10.0 * np.log10(
        np.maximum(after_linear[support], EPS)
        / np.maximum(before_linear[support], EPS)
    )

    return before_db, after_db, change_db


def attention_weighted_mean_rd(
    raw: np.ndarray,
    range_index: np.ndarray,
    elevation_index: np.ndarray,
    azimuth_index: np.ndarray,
    relative_contribution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted average of complete 64-bin spectra of the same 250 tokens."""
    result_linear = np.zeros(
        (NUM_RANGES, NUM_DOPPLER),
        dtype=np.float64,
    )

    relative = relative_contribution.reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
    )

    range_index = range_index.reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
    )
    elevation_index = elevation_index.reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
    )
    azimuth_index = azimuth_index.reshape(
        NUM_RANGES,
        TOKENS_PER_RANGE,
    )

    for current_range in range(NUM_RANGES):
        r_values = range_index[current_range]
        if not np.all(r_values == current_range):
            raise RuntimeError(
                f"Range indexing mismatch at range {current_range}"
            )

        elev = elevation_index[current_range]
        azi = azimuth_index[current_range]
        weights = relative[current_range].astype(np.float64)

        # Advanced indexing result: [D, K]
        spectra = raw[
            :,
            current_range,
            elev,
            azi,
        ].astype(np.float64)

        result_linear[current_range] = np.average(
            spectra,
            axis=1,
            weights=weights,
        )

    return result_linear, db10(result_linear)



def normalize_metric_per_range(metric: np.ndarray) -> np.ndarray:
    """Normalize [R,K] token metric to mean 1 independently per range."""
    metric = np.asarray(metric, dtype=np.float64)
    mean = metric.mean(axis=1, keepdims=True)
    return metric / np.maximum(mean, EPS)


def project_token_metric_mean_to_ra(
    range_index: np.ndarray,
    azimuth_index: np.ndarray,
    metric: np.ndarray,
) -> np.ndarray:
    """Project token metric to RA using MEAN over tokens/elevations per cell."""
    r = np.asarray(range_index, dtype=np.int64).reshape(-1)
    a = np.asarray(azimuth_index, dtype=np.int64).reshape(-1)
    values = np.asarray(metric, dtype=np.float64).reshape(-1)

    valid = (
        (r >= 0) & (r < NUM_RANGES)
        & (a >= 0) & (a < NUM_AZIMUTH)
        & np.isfinite(values)
    )
    r, a, values = r[valid], a[valid], values[valid]

    total = np.zeros((NUM_RANGES, NUM_AZIMUTH), dtype=np.float64)
    count = np.zeros_like(total)
    np.add.at(total, (r, a), values)
    np.add.at(count, (r, a), 1.0)

    result = np.full_like(total, np.nan)
    support = count > 0
    result[support] = total[support] / count[support]
    return result


def azimuth_profile_from_tokens(
    azimuth_index: np.ndarray,
    metric: np.ndarray,
) -> np.ndarray:
    """Average one token metric over all selected tokens for each azimuth."""
    a = np.asarray(azimuth_index, dtype=np.int64).reshape(-1)
    values = np.asarray(metric, dtype=np.float64).reshape(-1)

    valid = (
        (a >= 0) & (a < NUM_AZIMUTH)
        & np.isfinite(values)
    )
    a, values = a[valid], values[valid]

    total = np.zeros(NUM_AZIMUTH, dtype=np.float64)
    count = np.zeros(NUM_AZIMUTH, dtype=np.float64)
    np.add.at(total, a, values)
    np.add.at(count, a, 1.0)

    result = np.full(NUM_AZIMUTH, np.nan, dtype=np.float64)
    support = count > 0
    result[support] = total[support] / count[support]
    return result


def plot_attention_source_diagnostics(
    frame: int,
    range_index: np.ndarray,
    azimuth_index: np.ndarray,
    trace: AttentionTrace,
    checkpoint_name: str,
    dpi: int,
    output_path: Path,
) -> dict[str, np.ndarray]:
    """Separate received-attention bias from value-feature magnitude bias."""
    received_rel = normalize_metric_per_range(trace.final_received)
    value_rel = normalize_metric_per_range(trace.final_value_norm)
    contribution_rel = trace.final_contribution_relative.astype(np.float64)

    received_ra = project_token_metric_mean_to_ra(
        range_index, azimuth_index, received_rel
    )
    value_ra = project_token_metric_mean_to_ra(
        range_index, azimuth_index, value_rel
    )
    contribution_ra = project_token_metric_mean_to_ra(
        range_index, azimuth_index, contribution_rel
    )

    received_profile = azimuth_profile_from_tokens(
        azimuth_index, received_rel
    )
    value_profile = azimuth_profile_from_tokens(
        azimuth_index, value_rel
    )
    contribution_profile = azimuth_profile_from_tokens(
        azimuth_index, contribution_rel
    )

    fig, axes = plt.subplots(
        1, 4, figsize=(23, 6.2), constrained_layout=True
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad((0.82, 0.82, 0.82, 1.0))

    panels = [
        (received_ra,
         r"Received Attention $\sum_i A_{ij}$"
         "\n(range-normalized; RA cell mean)"),
        (value_ra,
         r"Value Magnitude $\|V_j\|_2$"
         "\n(range-normalized; RA cell mean)"),
        (contribution_ra,
         r"Final Contribution $(\sum_i A_{ij})\|V_j\|_2$"
         "\n(range-normalized; RA cell mean)"),
    ]

    for ax, (array, title) in zip(axes[:3], panels):
        finite = array[np.isfinite(array)]
        if finite.size:
            lo = float(np.percentile(finite, 2.0))
            hi = float(np.percentile(finite, 98.0))
            span = max(1.0 - lo, hi - 1.0, 0.15)
            vmin, vmax = max(0.0, 1.0 - span), 1.0 + span
        else:
            vmin, vmax = 0.0, 2.0

        image = ax.imshow(
            np.ma.masked_invalid(array),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Azimuth bin")
        ax.set_ylabel("Range bin")
        fig.colorbar(
            image, ax=ax,
            label="Relative metric (range mean = 1)",
            shrink=0.92,
        )

    x = np.arange(NUM_AZIMUTH)
    axes[3].plot(x, received_profile,
                 label=r"Received $\sum_i A_{ij}$", linewidth=1.5)
    axes[3].plot(x, value_profile,
                 label=r"$\|V_j\|_2$", linewidth=1.5)
    axes[3].plot(x, contribution_profile,
                 label="Contribution", linewidth=1.8)
    axes[3].axhline(1.0, linestyle="--", linewidth=1.0,
                    color="black", alpha=0.7)
    axes[3].axvline(80, linestyle=":", linewidth=1.0,
                    color="black", alpha=0.7)
    axes[3].set_xlim(0, NUM_AZIMUTH - 1)
    axes[3].set_xlabel("Azimuth bin")
    axes[3].set_ylabel("Mean relative metric")
    axes[3].set_title("Azimuth Profile\nmean over all selected tokens")
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.2)

    fig.suptitle(
        f"Range Attention Source Diagnosis — "
        f"frame {frame:05d} — {checkpoint_name}",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "received_relative": received_rel,
        "value_relative": value_rel,
        "contribution_relative": contribution_rel,
        "received_ra": received_ra,
        "value_ra": value_ra,
        "contribution_ra": contribution_ra,
        "received_profile": received_profile,
        "value_profile": value_profile,
        "contribution_profile": contribution_profile,
    }


def plot_zero_azimuth_ablation(
    frame: int,
    range_index: np.ndarray,
    azimuth_index: np.ndarray,
    original_trace: AttentionTrace,
    zero_azimuth_trace: AttentionTrace,
    checkpoint_name: str,
    dpi: int,
    output_path: Path,
) -> dict[str, np.ndarray]:
    """Zero only the learned azimuth embedding and compare contribution."""
    original_rel = original_trace.final_contribution_relative.astype(np.float64)
    zero_rel = zero_azimuth_trace.final_contribution_relative.astype(np.float64)

    original_ra = project_token_metric_mean_to_ra(
        range_index, azimuth_index, original_rel
    )
    zero_ra = project_token_metric_mean_to_ra(
        range_index, azimuth_index, zero_rel
    )

    difference = np.full_like(original_ra, np.nan)
    support = np.isfinite(original_ra) & np.isfinite(zero_ra)
    difference[support] = np.log2(
        np.maximum(zero_ra[support], EPS)
        / np.maximum(original_ra[support], EPS)
    )

    original_profile = azimuth_profile_from_tokens(
        azimuth_index, original_rel
    )
    zero_profile = azimuth_profile_from_tokens(
        azimuth_index, zero_rel
    )

    fig, axes = plt.subplots(
        1, 4, figsize=(23, 6.2), constrained_layout=True
    )

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad((0.82, 0.82, 0.82, 1.0))

    finite = np.concatenate([
        original_ra[np.isfinite(original_ra)],
        zero_ra[np.isfinite(zero_ra)],
    ])
    if finite.size:
        lo = float(np.percentile(finite, 2.0))
        hi = float(np.percentile(finite, 98.0))
        span = max(1.0 - lo, hi - 1.0, 0.15)
        vmin, vmax = max(0.0, 1.0 - span), 1.0 + span
    else:
        vmin, vmax = 0.0, 2.0

    image0 = axes[0].imshow(
        np.ma.masked_invalid(original_ra),
        origin="lower", aspect="auto", interpolation="nearest",
        cmap=cmap, vmin=vmin, vmax=vmax,
    )
    axes[0].set_title("Original Model\nFinal Contribution")
    axes[0].set_xlabel("Azimuth bin")
    axes[0].set_ylabel("Range bin")

    image1 = axes[1].imshow(
        np.ma.masked_invalid(zero_ra),
        origin="lower", aspect="auto", interpolation="nearest",
        cmap=cmap, vmin=vmin, vmax=vmax,
    )
    axes[1].set_title("Azimuth Embedding = 0\nFinal Contribution")
    axes[1].set_xlabel("Azimuth bin")
    axes[1].set_ylabel("Range bin")
    fig.colorbar(
        image1, ax=[axes[0], axes[1]],
        label="Relative contribution (range mean = 1)",
        shrink=0.92,
    )

    finite_diff = difference[np.isfinite(difference)]
    if finite_diff.size:
        limit = float(np.percentile(np.abs(finite_diff), 98.0))
        limit = min(max(limit, 0.5), 3.0)
    else:
        limit = 1.0

    diff_cmap = plt.get_cmap("coolwarm").copy()
    diff_cmap.set_bad((0.82, 0.82, 0.82, 1.0))
    image2 = axes[2].imshow(
        np.ma.masked_invalid(difference),
        origin="lower", aspect="auto", interpolation="nearest",
        cmap=diff_cmap, vmin=-limit, vmax=limit,
    )
    axes[2].set_title(
        "Effect of Removing Azimuth Embedding\n"
        r"$\log_2(C_{\rm zero-azi}/C_{\rm original})$"
    )
    axes[2].set_xlabel("Azimuth bin")
    axes[2].set_ylabel("Range bin")
    fig.colorbar(
        image2, ax=axes[2],
        label="log2 contribution ratio",
        shrink=0.92,
    )

    x = np.arange(NUM_AZIMUTH)
    axes[3].plot(x, original_profile, label="Original", linewidth=1.8)
    axes[3].plot(x, zero_profile,
                 label="Azimuth embedding = 0", linewidth=1.8)
    axes[3].axhline(1.0, linestyle="--", linewidth=1.0,
                    color="black", alpha=0.7)
    axes[3].axvline(80, linestyle=":", linewidth=1.0,
                    color="black", alpha=0.7)
    axes[3].set_xlim(0, NUM_AZIMUTH - 1)
    axes[3].set_xlabel("Azimuth bin")
    axes[3].set_ylabel("Mean relative contribution")
    axes[3].set_title(
        "Azimuth Contribution Profile\n"
        "Does the ~80-bin peak disappear?"
    )
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.2)

    fig.suptitle(
        f"Azimuth-Embedding Ablation — "
        f"frame {frame:05d} — {checkpoint_name}",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "original_contribution_ra": original_ra,
        "zero_azimuth_contribution_ra": zero_ra,
        "zero_over_original_log2": difference,
        "original_azimuth_profile": original_profile,
        "zero_azimuth_profile": zero_profile,
    }


def save_attention_source_npz(
    path: Path,
    source_data: dict[str, np.ndarray],
    ablation_data: dict[str, np.ndarray],
) -> None:
    payload = {}
    payload.update({f"source_{k}": v for k, v in source_data.items()})
    payload.update({f"ablation_{k}": v for k, v in ablation_data.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def fixed_circle_legend(
    color: str,
    label: str,
) -> Line2D:
    return Line2D(
        [0],
        [0],
        linestyle="none",
        marker="o",
        markersize=5,
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=0.8,
        label=label,
    )


def plot_ra_figure(
    frame: int,
    before_power_db: np.ndarray,
    after_power_db: np.ndarray,
    power_change_db: np.ndarray,
    checkpoint_name: str,
    dpi: int,
    output_path: Path,
) -> None:
    """Plot the requested three-panel power view.

    Left:
        selected-token mean power before Range Attention.

    Middle:
        the same power reweighted by final-layer attention-feature
        contribution.

    Right:
        dB change = after - before.

    The first two panels always use exactly the SAME color limits.
    """
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6.3),
        constrained_layout=True,
    )

    # Explicit masks make cells without selected tokens visually distinct
    # from cells whose change is close to zero.
    before_masked = np.ma.masked_invalid(before_power_db)
    after_masked = np.ma.masked_invalid(after_power_db)
    change_masked = np.ma.masked_invalid(power_change_db)

    finite_power = np.concatenate(
        [
            before_power_db[np.isfinite(before_power_db)],
            after_power_db[np.isfinite(after_power_db)],
        ]
    )

    if finite_power.size == 0:
        raise RuntimeError("No finite RA power values to plot.")

    # Shared absolute color scale: the same color means the same dB value
    # before and after attention.
    power_vmin = float(np.percentile(finite_power, 1.0))
    power_vmax = float(np.percentile(finite_power, 99.5))
    if power_vmax <= power_vmin:
        power_vmax = power_vmin + 1.0

    power_cmap = plt.get_cmap("viridis").copy()
    power_cmap.set_bad("white")

    image_before = axes[0].imshow(
        before_masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=power_cmap,
        vmin=power_vmin,
        vmax=power_vmax,
    )
    axes[0].set_title(
        "Before Range Attention\n"
        "Selected mean-power RA"
    )
    axes[0].set_xlabel("Azimuth bin")
    axes[0].set_ylabel("Range bin")

    image_after = axes[1].imshow(
        after_masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=power_cmap,
        vmin=power_vmin,
        vmax=power_vmax,
    )
    axes[1].set_title(
        "After Range Attention (layer 4)\n"
        "Attention-weighted power RA"
    )
    axes[1].set_xlabel("Azimuth bin")
    axes[1].set_ylabel("Range bin")

    # One shared power colorbar for both absolute-power panels.
    fig.colorbar(
        image_after,
        ax=[axes[0], axes[1]],
        label="Power-like diagnostic [dB]",
        shrink=0.93,
    )

    finite_change = power_change_db[
        np.isfinite(power_change_db)
    ]

    if finite_change.size:
        # Symmetric color scale, robust to a few extreme cells.
        change_limit = float(
            np.percentile(
                np.abs(finite_change),
                98.0,
            )
        )
        change_limit = max(change_limit, 1.0)
        change_limit = min(change_limit, 12.0)
    else:
        change_limit = 1.0

    change_cmap = plt.get_cmap("coolwarm").copy()
    # No selected token = grey, rather than white (white means ~0 dB change).
    change_cmap.set_bad((0.75, 0.75, 0.75, 1.0))

    image_change = axes[2].imshow(
        change_masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=change_cmap,
        vmin=-change_limit,
        vmax=change_limit,
    )
    axes[2].set_title(
        "Attention Effect\n"
        r"$10\log_{10}(P_{\rm weighted}/P_{\rm before})$"
    )
    axes[2].set_xlabel("Azimuth bin")
    axes[2].set_ylabel("Range bin")

    fig.colorbar(
        image_change,
        ax=axes[2],
        label="Power change [dB]",
        shrink=0.93,
    )

    fig.suptitle(
        f"RadarOcc Range Attention Power — "
        f"frame {frame:05d} — {checkpoint_name}",
        fontsize=15,
    )

    # Small note to avoid interpreting the weighted map as sensor power.
    fig.text(
        0.5,
        0.008,
        "Middle panel is an attention-weighted diagnostic: "
        "original selected mean power × normalized attention-feature "
        "contribution; it is not a new physical radar measurement.",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)



def plot_rd_top_panel(
    ax: plt.Axes,
    raw_rd: np.ndarray,
    doppler_index: np.ndarray,
    range_index: np.ndarray,
    relative_contribution: np.ndarray,
    rank: int,
    min_rd_count: int,
    point_size: float,
    point_alpha: float,
    vmin: float,
    vmax: float,
) -> tuple[int, int, float]:
    x_all, y_all, counts_all, after_mass_all = collapse_projection(
        doppler_index,
        range_index,
        relative_contribution,
    )

    keep = counts_all >= min_rd_count

    x = x_all[keep]
    y = y_all[keep]
    counts = counts_all[keep]
    after_mass = after_mass_all[keep]

    ax.imshow(
        raw_rd,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    ax.scatter(
        x,
        y,
        s=point_size * after_mass,
        facecolors="none",
        edgecolors="red",
        linewidths=0.5,
        alpha=point_alpha,
        rasterized=True,
        zorder=3,
    )

    ax.set_title(
        f"Top-{rank} Doppler after Range Attention\n"
        f"show projected count >= {min_rd_count}; "
        "area = contribution"
    )
    ax.set_xlabel("Doppler bin")
    ax.set_ylabel("Range bin")
    ax.legend(
        handles=[
            fixed_circle_legend(
                "red",
                "Attention-weighted contribution",
            )
        ],
        loc="upper right",
        fontsize=7,
    )

    return (
        int(np.sum(keep)),
        int(np.sum(~keep)),
        float(
            np.max(after_mass_all)
            if after_mass_all.size
            else 0.0
        ),
    )


def plot_rd_figure(
    frame: int,
    raw_rd: np.ndarray,
    sorted_doppler: np.ndarray,
    range_index: np.ndarray,
    relative_contribution: np.ndarray,
    weighted_mean_rd_db: np.ndarray,
    checkpoint_name: str,
    min_rd_count: int,
    point_size: float,
    point_alpha: float,
    dpi: int,
    output_path: Path,
) -> list[tuple[int, int, float]]:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.5, 11),
        constrained_layout=True,
    )

    raw_vmin = float(
        np.nanpercentile(raw_rd, 1.0)
    )
    raw_vmax = float(
        np.nanpercentile(raw_rd, 99.5)
    )

    stats: list[tuple[int, int, float]] = []

    for rank in range(3):
        ax = axes.flat[rank]
        stats.append(
            plot_rd_top_panel(
                ax=ax,
                raw_rd=raw_rd,
                doppler_index=sorted_doppler[:, rank],
                range_index=range_index,
                relative_contribution=relative_contribution,
                rank=rank + 1,
                min_rd_count=min_rd_count,
                point_size=point_size,
                point_alpha=point_alpha,
                vmin=raw_vmin,
                vmax=raw_vmax,
            )
        )

    image = axes[1, 1].imshow(
        weighted_mean_rd_db,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=raw_vmin,
        vmax=raw_vmax,
    )
    axes[1, 1].set_title(
        "Attention-weighted mean Doppler spectrum\n"
        "same 250 selected tokens per range"
    )
    axes[1, 1].set_xlabel("Doppler bin")
    axes[1, 1].set_ylabel("Range bin")

    fig.colorbar(
        image,
        ax=axes[1, 1],
        label="Weighted mean power [dB]",
    )

    # One shared raw-RD colorbar for the Top-1/2/3 panels.
    mappable = axes[0, 0].images[0]
    fig.colorbar(
        mappable,
        ax=[
            axes[0, 0],
            axes[0, 1],
            axes[1, 0],
        ],
        label="Raw RD power [dB]",
        shrink=0.84,
    )

    fig.suptitle(
        f"RadarOcc Range Attention RD — frame {frame:05d} — {checkpoint_name}",
        fontsize=15,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.savefig(
        output_path,
        dpi=dpi,
    )
    plt.close(fig)

    return stats


# ---------------------------------------------------------------------------
# Save numeric results
# ---------------------------------------------------------------------------
def save_top_tokens_csv(
    path: Path,
    frame: int,
    range_index: np.ndarray,
    elevation_index: np.ndarray,
    azimuth_index: np.ndarray,
    sorted_doppler: np.ndarray,
    received: np.ndarray,
    value_norm: np.ndarray,
    contribution_raw: np.ndarray,
    contribution_relative: np.ndarray,
    top_n: int,
) -> None:
    if top_n <= 0:
        return

    relative_flat = contribution_relative.reshape(-1)
    order = np.argsort(relative_flat)[::-1]
    order = order[: min(top_n, order.size)]

    received_flat = received.reshape(-1)
    value_norm_flat = value_norm.reshape(-1)
    raw_flat = contribution_raw.reshape(-1)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "token_flat_index",
                "range",
                "elevation",
                "azimuth",
                "top1_doppler",
                "top2_doppler",
                "top3_doppler",
                "received_attention",
                "value_norm",
                "contribution_raw",
                "contribution_relative",
            ]
        )

        for token in order:
            writer.writerow(
                [
                    frame,
                    int(token),
                    int(range_index[token]),
                    int(elevation_index[token]),
                    int(azimuth_index[token]),
                    int(sorted_doppler[token, 0]),
                    int(sorted_doppler[token, 1]),
                    int(sorted_doppler[token, 2]),
                    float(received_flat[token]),
                    float(value_norm_flat[token]),
                    float(raw_flat[token]),
                    float(relative_flat[token]),
                ]
            )


def save_npz(
    path: Path,
    descriptors_np: np.ndarray,
    gather_index: np.ndarray,
    range_index: np.ndarray,
    elevation_index: np.ndarray,
    azimuth_index: np.ndarray,
    sorted_power: np.ndarray,
    sorted_doppler: np.ndarray,
    trace: AttentionTrace,
    ra_x: np.ndarray,
    ra_y: np.ndarray,
    ra_count: np.ndarray,
    ra_after_mass: np.ndarray,
    ra_change: np.ndarray,
    ra_power_before_db: np.ndarray,
    ra_power_after_weighted_db: np.ndarray,
    ra_power_change_db: np.ndarray,
    weighted_mean_rd_linear: np.ndarray,
    weighted_mean_rd_db: np.ndarray,
    save_full_matrices: bool,
) -> None:
    payload: dict[str, Any] = {
        "descriptor_after_detector_gather": descriptors_np,
        "descriptor_gather_index": gather_index,
        "range_ind": range_index,
        "elevation_ind": elevation_index,
        "azimuth_ind": azimuth_index,
        "sorted_top3_power": sorted_power,
        "sorted_top3_doppler": sorted_doppler,

        "layer_received_attention": trace.layer_received,
        "layer_value_norm": trace.layer_value_norm,
        "layer_contribution_raw": trace.layer_contribution_raw,

        "final_received_attention": trace.final_received,
        "final_value_norm": trace.final_value_norm,
        "final_contribution_raw": trace.final_contribution_raw,
        "final_contribution_relative": trace.final_contribution_relative,

        "query_dim": np.array(trace.query_dim, dtype=np.int64),
        "attention_scale_sqrt_d": np.array(
            trace.attention_scale,
            dtype=np.float32,
        ),
        "max_forward_error": np.array(
            trace.max_forward_error,
            dtype=np.float32,
        ),
        "attention_row_sum_error": np.array(
            trace.row_sum_error,
            dtype=np.float32,
        ),
        "received_sum_error": np.array(
            trace.received_sum_error,
            dtype=np.float32,
        ),

        "ra_x": ra_x,
        "ra_y": ra_y,
        "ra_before_count": ra_count,
        "ra_after_contribution_mass": ra_after_mass,
        "ra_log2_after_over_before": ra_change,

        "ra_power_before_db": ra_power_before_db,
        "ra_power_after_weighted_db": ra_power_after_weighted_db,
        "ra_power_change_db": ra_power_change_db,

        "weighted_mean_rd_linear": weighted_mean_rd_linear,
        "weighted_mean_rd_db": weighted_mean_rd_db,
    }

    if save_full_matrices:
        payload.update(
            {
                "final_query": trace.final_query,
                "final_key": trace.final_key,
                "final_value": trace.final_value,
                "final_scaled_logits_qk_sqrt_d": trace.final_logits,
                "final_attention_probs_softmax": trace.final_probs,
            }
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        **payload,
    )


# ---------------------------------------------------------------------------
# Process one frame
# ---------------------------------------------------------------------------
def process_frame(
    frame: int,
    raw_path: Path,
    sparse_path: Path,
    checkpoint_path: Path,
    attention: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    print()
    print("=" * 78)
    print(f"Frame {frame:05d}")
    print(f"Raw:    {raw_path}")
    print(f"Sparse: {sparse_path}")
    print("=" * 78)

    raw = load_raw(raw_path)
    sparse = load_sparse(sparse_path)

    print(f"Raw normalized D-R-E-A: {raw.shape}")
    print(f"Sparse descriptor shape: {sparse.power_val.shape}")

    (
        descriptors,
        selected_range_t,
        selected_elevation_t,
        selected_azimuth_t,
        gather_index_t,
    ) = prepare_exact_attention_input(
        sparse=sparse,
        index_mode=args.index_mode,
        device=device,
    )

    trace = trace_range_attention(
        attention=attention,
        descriptors=descriptors,
        elevation=selected_elevation_t,
        azimuth=selected_azimuth_t,
        zero_azimuth_embedding=False,
    )

    # Diagnostic ablation: same checkpoint/input, but replace only the learned
    # azimuth embedding entering every Range-Attention layer with zeros.
    zero_azimuth_trace = trace_range_attention(
        attention=attention,
        descriptors=descriptors,
        elevation=selected_elevation_t,
        azimuth=selected_azimuth_t,
        zero_azimuth_embedding=True,
    )

    descriptors_np = descriptors.detach().cpu().numpy()
    gather_index = gather_index_t.detach().cpu().numpy().astype(np.int64)
    range_index = selected_range_t.detach().cpu().numpy().astype(np.int64)
    elevation_index = (
        selected_elevation_t.detach().cpu().numpy().astype(np.int64)
    )
    azimuth_index = (
        selected_azimuth_t.detach().cpu().numpy().astype(np.int64)
    )

    relative_contribution = (
        trace.final_contribution_relative.reshape(-1)
    )

    # Every range has mean relative contribution 1 -> sum exactly 250.
    relative_per_range = trace.final_contribution_relative.sum(axis=1)
    relative_sum_error = float(
        np.max(
            np.abs(
                relative_per_range - TOKENS_PER_RANGE
            )
        )
    )

    sorted_power, sorted_doppler = sort_top3_per_token(
        descriptors_np
    )

    raw_ra, raw_rd = build_raw_maps(
        raw,
        projection=args.projection,
    )

    # RA before and after use the same exact selected-token coordinates.
    ra_x, ra_y, ra_count, ra_after_mass = collapse_projection(
        x_index=azimuth_index,
        y_index=range_index,
        weights=relative_contribution,
    )

    ra_change = build_change_map(
        width=NUM_AZIMUTH,
        x=ra_x,
        y=ra_y,
        before_count=ra_count,
        after_mass=ra_after_mass,
    )

    # Descriptor layout:
    # [top1_power, top2_power, top3_power,
    #  doppler1, doppler2, doppler3,
    #  mean_power, variance]
    #
    # Use the selected token's ORIGINAL mean power as P_before.
    # P_after is a visualization diagnostic:
    # original mean power * final normalized attention-feature contribution.
    selected_mean_power = descriptors_np[:, 6].astype(np.float64)

    (
        ra_power_before_db,
        ra_power_after_weighted_db,
        ra_power_change_db,
    ) = build_attention_weighted_power_ra(
        range_index=range_index,
        azimuth_index=azimuth_index,
        mean_power=selected_mean_power,
        relative_contribution=relative_contribution,
    )

    weighted_mean_rd_linear, weighted_mean_rd_db = (
        attention_weighted_mean_rd(
            raw=raw,
            range_index=range_index,
            elevation_index=elevation_index,
            azimuth_index=azimuth_index,
            relative_contribution=relative_contribution,
        )
    )

    checkpoint_name = checkpoint_path.name

    ra_png = (
        args.output_dir
        / f"frame_{frame:05d}_RA_before_after_attention.png"
    )
    rd_png = (
        args.output_dir
        / f"frame_{frame:05d}_RD_attention_top3_mean.png"
    )
    npz_path = (
        args.output_dir
        / f"frame_{frame:05d}_attention_data.npz"
    )
    csv_path = (
        args.output_dir
        / f"frame_{frame:05d}_top_attention_tokens.csv"
    )

    source_png = (
        args.output_dir
        / f"frame_{frame:05d}_attention_source_diagnostics.png"
    )
    zero_azi_png = (
        args.output_dir
        / f"frame_{frame:05d}_zero_azimuth_embedding_ablation.png"
    )
    source_npz = (
        args.output_dir
        / f"frame_{frame:05d}_attention_source_diagnostics.npz"
    )

    # Preserve the original figures already in this folder unless explicitly
    # requested otherwise.
    if args.overwrite_existing or not ra_png.exists():
        plot_ra_figure(
            frame=frame,
            before_power_db=ra_power_before_db,
            after_power_db=ra_power_after_weighted_db,
            power_change_db=ra_power_change_db,
            checkpoint_name=checkpoint_name,
            dpi=args.dpi,
            output_path=ra_png,
        )
    else:
        print(f"Preserved existing original RA PNG: {ra_png}")

    if args.overwrite_existing or not rd_png.exists():
        rd_stats = plot_rd_figure(
            frame=frame,
            raw_rd=raw_rd,
            sorted_doppler=sorted_doppler,
            range_index=range_index,
            relative_contribution=relative_contribution,
            weighted_mean_rd_db=weighted_mean_rd_db,
            checkpoint_name=checkpoint_name,
            min_rd_count=args.min_rd_count,
            point_size=args.point_size,
            point_alpha=args.point_alpha,
            dpi=args.dpi,
            output_path=rd_png,
        )
    else:
        print(f"Preserved existing original RD PNG: {rd_png}")
        rd_stats = []

    # New figure 1: determine whether the suspicious azimuth stripe comes
    # from softmax attention, V magnitude, or their product.
    source_data = plot_attention_source_diagnostics(
        frame=frame,
        range_index=range_index,
        azimuth_index=azimuth_index,
        trace=trace,
        checkpoint_name=checkpoint_name,
        dpi=args.dpi,
        output_path=source_png,
    )

    # New figure 2: test the learned azimuth embedding directly.
    ablation_data = plot_zero_azimuth_ablation(
        frame=frame,
        range_index=range_index,
        azimuth_index=azimuth_index,
        original_trace=trace,
        zero_azimuth_trace=zero_azimuth_trace,
        checkpoint_name=checkpoint_name,
        dpi=args.dpi,
        output_path=zero_azi_png,
    )

    save_attention_source_npz(
        path=source_npz,
        source_data=source_data,
        ablation_data=ablation_data,
    )

    save_npz(
        path=npz_path,
        descriptors_np=descriptors_np,
        gather_index=gather_index,
        range_index=range_index,
        elevation_index=elevation_index,
        azimuth_index=azimuth_index,
        sorted_power=sorted_power,
        sorted_doppler=sorted_doppler,
        trace=trace,
        ra_x=ra_x,
        ra_y=ra_y,
        ra_count=ra_count,
        ra_after_mass=ra_after_mass,
        ra_change=ra_change,
        ra_power_before_db=ra_power_before_db,
        ra_power_after_weighted_db=ra_power_after_weighted_db,
        ra_power_change_db=ra_power_change_db,
        weighted_mean_rd_linear=weighted_mean_rd_linear,
        weighted_mean_rd_db=weighted_mean_rd_db,
        save_full_matrices=args.save_full_matrices,
    )

    save_top_tokens_csv(
        path=csv_path,
        frame=frame,
        range_index=range_index,
        elevation_index=elevation_index,
        azimuth_index=azimuth_index,
        sorted_doppler=sorted_doppler,
        received=trace.final_received,
        value_norm=trace.final_value_norm,
        contribution_raw=trace.final_contribution_raw,
        contribution_relative=trace.final_contribution_relative,
        top_n=args.top_csv,
    )

    print()
    print("Attention sanity checks")
    print("-----------------------")
    print(
        f"Q/K dimension d:             {trace.query_dim}"
    )
    print(
        f"sqrt(d):                     {trace.attention_scale:.6f}"
    )
    print(
        f"Trace vs repository forward: {trace.max_forward_error:.3e}"
    )
    print(
        f"max |sum_j A_ij - 1|:        {trace.row_sum_error:.3e}"
    )
    print(
        f"max |sum_j received_j-250|:  {trace.received_sum_error:.3e}"
    )
    print(
        f"max |sum relative-250|:      {relative_sum_error:.3e}"
    )
    print(
        "Final relative contribution: "
        f"min={relative_contribution.min():.4f}, "
        f"median={np.median(relative_contribution):.4f}, "
        f"max={relative_contribution.max():.4f}"
    )

    if rd_stats:
        print()
        print("RD display stats")
        print("----------------")
        for rank, stat in enumerate(rd_stats, start=1):
            displayed, hidden, max_mass = stat
            print(
                f"Top-{rank}: displayed cells={displayed:,}, "
                f"hidden count<{args.min_rd_count}={hidden:,}, "
                f"max contribution mass={max_mass:.3f}"
            )

    profile = source_data["contribution_profile"]
    finite_profile = np.where(np.isfinite(profile), profile, -np.inf)
    top_bins = np.argsort(finite_profile)[::-1][:10]

    print()
    print("Top azimuth bins by mean final contribution")
    print("-------------------------------------------")
    print(
        ", ".join(
            f"{int(bin_id)}:{float(profile[bin_id]):.3f}"
            for bin_id in top_bins
            if np.isfinite(profile[bin_id])
        )
    )

    print()
    print("Saved")
    print("-----")
    print(ra_png)
    print(rd_png)
    print(source_png)
    print(zero_azi_png)
    print(source_npz)
    print(npz_path)
    print(csv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    args.repo_root = args.repo_root.expanduser().resolve()
    args.raw_dir = args.raw_dir.expanduser().resolve()
    args.sparse_dir = args.sparse_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False."
        )

    print("RadarOcc Range-Attention RA/RD visualization")
    print("============================================")
    print(f"Repo root:   {args.repo_root}")
    print(f"Raw dir:     {args.raw_dir}")
    print(f"Sparse dir:  {args.sparse_dir}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Frames:      {args.frames}")
    print(f"Index mode:  {args.index_mode}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Device:      {device}")

    AttentionClass = load_attention_class(
        args.repo_root
    )

    attention = AttentionClass(
        num_layers=NUM_ATTENTION_LAYERS,
        num_ranges=NUM_RANGES,
        K=TOKENS_PER_RANGE,
        doppler=3,
    ).to(device)

    attention_state = load_range_attention_state(
        args.checkpoint
    )

    missing, unexpected = attention.load_state_dict(
        attention_state,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint Range-Attention state does not exactly match model. "
            f"missing={missing}, unexpected={unexpected}"
        )

    attention.eval()

    print(
        f"Range Attention loaded: "
        f"{len(attention.attention_layers)} layers"
    )

    for frame in args.frames:
        raw_path = find_frame_file(
            directory=args.raw_dir,
            frame=frame,
            prefix="tesseract",
            suffixes=(".mat", ".npy"),
        )

        sparse_path = find_frame_file(
            directory=args.sparse_dir,
            frame=frame,
            prefix="EAsparse",
            suffixes=(".npz",),
        )

        process_frame(
            frame=frame,
            raw_path=raw_path,
            sparse_path=sparse_path,
            checkpoint_path=args.checkpoint,
            attention=attention,
            args=args,
            device=device,
        )

    print()
    print("All requested frames finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
