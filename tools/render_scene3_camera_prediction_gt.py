#!/usr/bin/env python3
"""
Create a synchronized scene comparison:

    Camera | Simple occupancy overlay

The occupancy panel keeps the earlier simple RadarOcc appearance:
blue base occupancy with red prediction foreground. Ground-truth foreground
is added in yellow, and overlapping foreground between prediction and GT is
shown in orange. This avoids FP/FN-style clutter.

Repository format note:
    raw GT coordinates       = [x, y, z]
    save_occ prediction data = [z, y, x]

Overlay colors:
    blue   = prediction static/background occupancy
    red    = prediction foreground only
    yellow = GT foreground only
    orange = overlapping prediction+GT foreground

The prediction-to-GT relationship is read from the official test annotation
PKL. Camera frames are paired by their natural-sorted ordinal position inside
the scene because their filename indices use a different numbering system.

Run this script through xvfb-run in the radarocc-vis environment.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--camera-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene", default="3")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--camera-offset",
        type=int,
        default=0,
        help=(
            "Camera ordinal offset relative to scene annotation order. "
            "Default 0. Use only after inspecting the generated manifest."
        ),
    )
    parser.add_argument("--panel-size", type=int, default=600)
    parser.add_argument("--azimuth", type=float, default=180.0)
    parser.add_argument("--elevation", type=float, default=65.0)
    parser.add_argument("--distance", type=float, default=82.0)
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Do not rotate occupancy screenshots 90 degrees counter-clockwise.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build and print the synchronization manifest.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep intermediate combined PNG frames.",
    )
    return parser.parse_args()


def natural_key(value: str | Path) -> list[Any]:
    text = str(value)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def load_infos(annotation: Path) -> list[dict[str, Any]]:
    with annotation.open("rb") as handle:
        content = pickle.load(handle)

    if isinstance(content, dict):
        if isinstance(content.get("infos"), list):
            infos = content["infos"]
        elif isinstance(content.get("data_list"), list):
            infos = content["data_list"]
        else:
            candidates = [value for value in content.values() if isinstance(value, list)]
            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot identify the info list in annotation: {annotation}"
                )
            infos = candidates[0]
    elif isinstance(content, list):
        infos = content
    else:
        raise TypeError(f"Unsupported annotation type: {type(content)!r}")

    return sorted(infos, key=lambda info: info.get("timestamp", 0))


def resolve_gt_path(
    info: dict[str, Any],
    repo_root: Path,
    gt_root: Path,
) -> Path:
    raw = Path(str(info["occ_path"])).expanduser()
    candidates = [
        raw,
        repo_root / raw,
        gt_root / raw.name,
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not resolve GT file from annotation occ_path="
        f"{info['occ_path']!r}. Tried: {candidates}"
    )


def prediction_index(prediction_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in prediction_root.rglob("pred_c.npy"):
        token = path.parent.name
        if token in result:
            raise RuntimeError(f"Duplicate prediction token {token}: {path}")
        result[token] = path.resolve()
    if not result:
        raise FileNotFoundError(
            f"No */pred_c.npy files found under {prediction_root}"
        )
    return result


def camera_files(camera_dir: Path) -> list[Path]:
    files = sorted(
        (
            path.resolve()
            for path in camera_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_key,
    )
    if not files:
        raise FileNotFoundError(f"No camera images found in {camera_dir}")
    return files


def build_mapping(args: argparse.Namespace) -> list[dict[str, Any]]:
    infos = load_infos(args.annotation)
    scene_infos = [
        info
        for info in infos
        if str(info.get("scene_token")) == str(args.scene)
    ]
    if not scene_infos:
        raise RuntimeError(
            f"No annotation entries found for scene {args.scene}"
        )

    predictions = prediction_index(args.prediction_root)
    cameras = camera_files(args.camera_dir)

    mapping: list[dict[str, Any]] = []
    missing_predictions: list[str] = []

    for scene_ordinal, info in enumerate(scene_infos):
        lidar_token = str(info["lidar_token"])
        prediction = predictions.get(lidar_token)
        if prediction is None:
            missing_predictions.append(lidar_token)
            continue

        camera_ordinal = scene_ordinal + args.camera_offset
        if camera_ordinal < 0 or camera_ordinal >= len(cameras):
            raise IndexError(
                f"Camera ordinal {camera_ordinal} is outside 0..{len(cameras)-1}. "
                "Adjust --camera-offset after checking the file counts."
            )

        mapping.append(
            {
                "scene_ordinal": scene_ordinal,
                "lidar_token": lidar_token,
                "prediction": prediction,
                "gt": resolve_gt_path(info, args.repo_root, args.gt_root),
                "camera_ordinal": camera_ordinal,
                "camera": cameras[camera_ordinal],
                "timestamp": info.get("timestamp", ""),
            }
        )

    if not mapping:
        preview = ", ".join(missing_predictions[:10])
        raise RuntimeError(
            "No prediction folders matched annotation lidar_token values. "
            f"First unmatched tokens: {preview}"
        )

    start = args.start
    stop = min(len(mapping), start + args.max_frames)
    selected = mapping[start:stop]
    if not selected:
        raise IndexError(
            f"Requested start={start}, but only {len(mapping)} mapped frames exist."
        )
    return selected


def write_manifest(mapping: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "output_index",
                "scene_ordinal",
                "lidar_token",
                "prediction",
                "ground_truth",
                "camera_ordinal",
                "camera",
                "timestamp",
            ]
        )
        for output_index, item in enumerate(mapping):
            writer.writerow(
                [
                    output_index,
                    item["scene_ordinal"],
                    item["lidar_token"],
                    item["prediction"],
                    item["gt"],
                    item["camera_ordinal"],
                    item["camera"],
                    item["timestamp"],
                ]
            )


def print_mapping_summary(mapping: list[dict[str, Any]], manifest: Path) -> None:
    print(f"Mapped frames: {len(mapping)}")
    print(f"Manifest: {manifest}")
    print()
    print("First mappings:")
    for output_index, item in enumerate(mapping[:10]):
        print(
            f"{output_index:03d}  token={item['lidar_token']}  "
            f"GT={Path(item['gt']).name}  "
            f"camera={Path(item['camera']).name}"
        )


def load_sparse_occupancy(
    path: Path,
    is_gt: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Load occupancy and return coordinates in saved-prediction [z, y, x].

    Important repository-specific format difference
    -----------------------------------------------
    Raw K-Radar GT files use [x, y, z, class] and fit the model grid
    [128, 128, 14].

    Prediction files written by save_occ() are explicitly reordered before
    saving and therefore use [z, y, x, class].

    The two formats must not be interpreted identically.
    """
    array = np.load(path, allow_pickle=False)

    if array.ndim == 3:
        # Dense model occupancy arrays are indexed [x, y, z].
        occupied_xyz = np.argwhere((array != 0) & (array != 255))
        labels = array[tuple(occupied_xyz.T)].astype(np.int64)
        coordinates_zyx = occupied_xyz[:, [2, 1, 0]].astype(np.int64)

    elif array.ndim == 2 and array.shape[1] >= 4:
        raw_coordinates = np.rint(array[:, :3]).astype(np.int64)

        # GT files:         [x, y, z, ..., class]
        # save_occ output: [z, y, x, class]
        if is_gt:
            coordinates_zyx = raw_coordinates[:, [2, 1, 0]]
        else:
            coordinates_zyx = raw_coordinates

        class_column = 3 if array.shape[1] == 4 else array.shape[1] - 1
        labels = np.rint(array[:, class_column]).astype(np.int64)

    else:
        raise ValueError(
            f"Unsupported occupancy shape {array.shape} in {path}"
        )

    if coordinates_zyx.size == 0:
        raise RuntimeError(f"No occupancy coordinates found in {path}")

    raw_shape = tuple(array.shape)
    raw_label_values, raw_label_counts = np.unique(
        labels,
        return_counts=True,
    )

    coord_min = coordinates_zyx.min(axis=0)
    coord_max = coordinates_zyx.max(axis=0)

    # Internal comparison/rendering order is [Z, Y, X].
    model_shape_zyx = np.array([14, 128, 128], dtype=np.int64)

    # Optional support for an exactly 4x finer grid.
    if np.any(coord_max >= model_shape_zyx):
        fine_shape_zyx = model_shape_zyx * 4
        if np.all(coord_min >= 0) and np.all(coord_max < fine_shape_zyx):
            coordinates_zyx = coordinates_zyx // 4
            print(
                f"  [{'GT' if is_gt else 'Prediction'} conversion] "
                "downsampled 4x-fine coordinates to model grid"
            )
        else:
            raise RuntimeError(
                f"Coordinates in {path.name} do not fit [Z,Y,X]="
                f"{model_shape_zyx.tolist()} or a 4x fine grid after format "
                f"conversion. min={coord_min.tolist()}, "
                f"max={coord_max.tolist()}, array_shape={raw_shape}, "
                f"is_gt={is_gt}"
            )

    # Current three-class setup:
    # 0/255 = empty/invalid; 1 = static occupied; >=2 = foreground occupied.
    valid = (labels != 0) & (labels != 255)
    coordinates_zyx = coordinates_zyx[valid]
    labels = labels[valid]
    labels = np.where(labels >= 2, 2, 1).astype(np.int64)

    in_bounds = (
        (coordinates_zyx[:, 0] >= 0)
        & (coordinates_zyx[:, 0] < 14)
        & (coordinates_zyx[:, 1] >= 0)
        & (coordinates_zyx[:, 1] < 128)
        & (coordinates_zyx[:, 2] >= 0)
        & (coordinates_zyx[:, 2] < 128)
    )
    coordinates_zyx = coordinates_zyx[in_bounds]
    labels = labels[in_bounds]

    if coordinates_zyx.shape[0] == 0:
        label_summary = dict(
            zip(raw_label_values.tolist(), raw_label_counts.tolist())
        )
        raise RuntimeError(
            f"{'GT' if is_gt else 'Prediction'} became empty after filtering: "
            f"path={path}, raw_shape={raw_shape}, "
            f"converted_coordinate_min={coord_min.tolist()}, "
            f"converted_coordinate_max={coord_max.tolist()}, "
            f"raw_labels={label_summary}"
        )

    # Collapse duplicate voxels. Foreground class 2 wins over class 1.
    linear = np.ravel_multi_index(
        coordinates_zyx.T,
        dims=(14, 128, 128),
    )
    order = np.argsort(linear)
    linear = linear[order]
    labels = labels[order]

    unique_linear, first = np.unique(linear, return_index=True)
    reduced_labels = np.maximum.reduceat(labels, first)
    reduced_coordinates = np.column_stack(
        np.unravel_index(unique_linear, (14, 128, 128))
    ).astype(np.int64)

    return reduced_coordinates, reduced_labels

def build_simple_overlay(
    pred_coordinates: np.ndarray,
    pred_labels: np.ndarray,
    gt_coordinates: np.ndarray,
    gt_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build a simple readable overlay.

    Category IDs:
        1: prediction static/background occupancy (blue)
        2: prediction foreground only (red)
        3: GT foreground only (yellow)
        4: overlapping prediction+GT foreground (orange)

    Notes:
    - GT static/background occupancy is intentionally not shown to keep the
      panel visually simple.
    - Prediction static/background occupancy stays as the blue base layer,
      matching the user's earlier preferred appearance.
    """
    grid_shape = (14, 128, 128)
    number_voxels = int(np.prod(grid_shape))

    pred_dense = np.zeros(number_voxels, dtype=np.uint8)
    gt_dense = np.zeros(number_voxels, dtype=np.uint8)

    pred_linear = np.ravel_multi_index(
        pred_coordinates.T,
        dims=grid_shape,
    )
    gt_linear = np.ravel_multi_index(
        gt_coordinates.T,
        dims=grid_shape,
    )

    pred_dense[pred_linear] = pred_labels.astype(np.uint8)
    gt_dense[gt_linear] = gt_labels.astype(np.uint8)

    pred_static = pred_dense == 1
    pred_foreground = pred_dense == 2
    gt_foreground = gt_dense == 2

    overlap_foreground = pred_foreground & gt_foreground
    pred_foreground_only = pred_foreground & (~gt_foreground)
    gt_foreground_only = gt_foreground & (~pred_foreground)

    # Keep the blue base occupancy, but do not overwrite the more important
    # red/yellow/orange foreground overlay categories.
    pred_static_only = pred_static & (~pred_foreground) & (~gt_foreground)

    category = np.zeros(number_voxels, dtype=np.uint8)
    category[pred_static_only] = 1
    category[pred_foreground_only] = 2
    category[gt_foreground_only] = 3
    category[overlap_foreground] = 4

    used = np.flatnonzero(category != 0)
    coordinates = np.column_stack(
        np.unravel_index(used, grid_shape)
    ).astype(np.int64)
    categories = category[used]

    stats = {
        "blue_base": int(np.sum(category == 1)),
        "prediction_red": int(np.sum(category == 2)),
        "gt_yellow": int(np.sum(category == 3)),
        "overlap_orange": int(np.sum(category == 4)),
    }
    return coordinates, categories, stats


def draw_box(mlab: Any) -> None:
    """Draw the real RadarOcc physical range in metres."""
    x0, x1 = 0.0, 51.2
    y0, y1 = -25.6, 25.6
    z0, z1 = -2.6, 3.0

    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for first, second in edges:
        a, b = corners[first], corners[second]
        mlab.plot3d(
            [a[0], b[0]],
            [a[1], b[1]],
            [a[2], b[2]],
            color=(0.2, 0.2, 0.2),
            tube_radius=None,
            line_width=1.0,
        )

def autocrop_white(image: Image.Image, margin: int = 8) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    mask = np.any(rgb < 248, axis=2)
    rows, cols = np.where(mask)
    if rows.size == 0:
        return image
    left = max(0, int(cols.min()) - margin)
    upper = max(0, int(rows.min()) - margin)
    right = min(image.width, int(cols.max()) + margin + 1)
    lower = min(image.height, int(rows.max()) + margin + 1)
    return image.crop((left, upper, right, lower))


def render_simple_overlay(
    coordinates_zyx: np.ndarray,
    categories: np.ndarray,
    args: argparse.Namespace,
) -> Image.Image:
    """Render the simple blue/red/yellow/orange overlay panel."""
    try:
        from mayavi import mlab
    except ImportError as exc:
        raise RuntimeError(
            "Mayavi is required. Activate the radarocc-vis environment."
        ) from exc

    size = args.panel_size
    figure = mlab.figure(
        size=(size, size),
        bgcolor=(1.0, 1.0, 1.0),
        fgcolor=(0.0, 0.0, 0.0),
    )

    # Convert [z, y, x] model-grid indices to physical voxel centres in metres.
    z = -2.6 + (coordinates_zyx[:, 0].astype(np.float32) + 0.5) * 0.4
    y = -25.6 + (coordinates_zyx[:, 1].astype(np.float32) + 0.5) * 0.4
    x = 0.0 + (coordinates_zyx[:, 2].astype(np.float32) + 0.5) * 0.4

    styles = {
        1: ((0.10, 0.45, 0.95), 0.70, 0.38),  # blue base
        2: ((0.95, 0.15, 0.10), 0.95, 0.42),  # red prediction fg only
        3: ((1.00, 0.82, 0.10), 0.95, 0.42),  # yellow GT fg only
        4: ((1.00, 0.45, 0.05), 1.00, 0.46),  # orange overlap
    }

    # Draw blue base first, then the overlays.
    for category_id in (1, 2, 3, 4):
        mask = categories == category_id
        if not np.any(mask):
            continue

        color, opacity, scale_factor = styles[category_id]
        mlab.points3d(
            x[mask],
            y[mask],
            z[mask],
            mode="cube",
            scale_factor=scale_factor,
            color=color,
            opacity=opacity,
        )

    draw_box(mlab)

    mlab.points3d(
        [0.0],
        [0.0],
        [0.0],
        mode="cube",
        scale_factor=0.9,
        color=(0.0, 0.0, 0.0),
    )

    mlab.view(
        azimuth=args.azimuth,
        elevation=args.elevation,
        distance=args.distance,
        focalpoint=(25.6, 0.0, 0.0),
        figure=figure,
    )
    figure.scene.parallel_projection = True

    screenshot = mlab.screenshot(
        figure=figure,
        mode="rgb",
        antialiased=True,
    )
    mlab.close(figure)

    image = Image.fromarray(screenshot)
    image = autocrop_white(image)

    if not args.no_rotate:
        # Rotate clockwise 90 degrees so the ego vehicle appears to move forward
        # rather than to the left.
        image = image.transpose(Image.Transpose.ROTATE_90)

    return image


def fit_image(
    image: Image.Image,
    width: int,
    height: int,
    background: tuple[int, int, int],
) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), background)
    x = (width - image.width) // 2
    y = (height - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def get_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def compose_frame(
    camera_path: Path,
    overlay_panel: Image.Image,
    token: str,
    output_index: int,
    args: argparse.Namespace,
) -> Image.Image:
    panel = args.panel_size
    title_height = 58
    legend_height = 62
    footer_height = 42
    total_height = title_height + panel + legend_height + footer_height
    total_width = panel * 2

    camera = Image.open(camera_path).convert("RGB")
    camera = fit_image(camera, panel, panel, (0, 0, 0))
    overlay_panel = fit_image(
        overlay_panel,
        panel,
        panel,
        (255, 255, 255),
    )

    canvas = Image.new(
        "RGB",
        (total_width, total_height),
        (255, 255, 255),
    )

    # Swap positions: occupancy overlay on the left, camera on the right.
    canvas.paste(overlay_panel, (0, title_height))
    canvas.paste(camera, (panel, title_height))

    draw = ImageDraw.Draw(canvas)
    title_font = get_font(27)
    legend_font = get_font(18)
    footer_font = get_font(17)

    titles = ["Prediction + GT overlay", "Camera"]
    for index, title in enumerate(titles):
        box = draw.textbbox((0, 0), title, font=title_font)
        text_width = box[2] - box[0]
        x = index * panel + (panel - text_width) // 2
        draw.text((x, 14), title, fill=(0, 0, 0), font=title_font)

    # Keep the legend placement logic unchanged.
    legend_items = [
        ("Base occupancy", (26, 115, 242)),
        ("Prediction", (242, 38, 26)),
        ("GT", (255, 209, 26)),
        ("Overlap", (255, 115, 13)),
    ]

    legend_y = title_height + panel + 14
    x = 36
    for label, color in legend_items:
        draw.rectangle(
            (x, legend_y + 3, x + 20, legend_y + 23),
            fill=color,
            outline=(0, 0, 0),
        )
        draw.text(
            (x + 30, legend_y),
            label,
            fill=(0, 0, 0),
            font=legend_font,
        )
        text_box = draw.textbbox((x + 30, legend_y), label, font=legend_font)
        x = text_box[2] + 42

    footer = (
        f"Scene {args.scene} | frame {output_index:03d} | "
        f"token {token} | {camera_path.name}"
    )
    box = draw.textbbox((0, 0), footer, font=footer_font)
    footer_width = box[2] - box[0]
    draw.text(
        (
            (total_width - footer_width) // 2,
            title_height + panel + legend_height + 7,
        ),
        footer,
        fill=(0, 0, 0),
        font=footer_font,
    )
    return canvas



def encode_video(
    frames_dir: Path,
    output_dir: Path,
    fps: int,
) -> tuple[Path, Path]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed or not on PATH.")

    mp4 = output_dir / "scene_camera_simple_overlay.mp4"
    gif = output_dir / "scene_camera_simple_overlay.gif"

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%05d.png"),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "medium",
            str(mp4),
        ],
        check=True,
    )

    filter_graph = (
        f"fps={fps},scale=1800:-2:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=256:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=sierra2_4a"
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(mp4),
            "-filter_complex",
            filter_graph,
            "-loop",
            "0",
            str(gif),
        ],
        check=True,
    )
    return mp4, gif


def main() -> int:
    args = parse_args()

    args.repo_root = args.repo_root.expanduser().resolve()
    args.prediction_root = args.prediction_root.expanduser().resolve()
    args.annotation = args.annotation.expanduser().resolve()
    args.gt_root = args.gt_root.expanduser().resolve()
    args.camera_dir = args.camera_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "synchronization_manifest.tsv"

    mapping = build_mapping(args)
    write_manifest(mapping, manifest)
    print_mapping_summary(mapping, manifest)

    if args.dry_run:
        print("\nDry run complete; no images were rendered.")
        return 0

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for output_index, item in enumerate(mapping):
        print(
            f"[{output_index + 1:03d}/{len(mapping):03d}] "
            f"token={item['lidar_token']}"
        )

        pred_coordinates, pred_labels = load_sparse_occupancy(
            item["prediction"], is_gt=False
        )
        gt_coordinates, gt_labels = load_sparse_occupancy(
            item["gt"], is_gt=True
        )

        print(
            f"    prediction voxels={len(pred_coordinates):,}, "
            f"classes={dict(zip(*np.unique(pred_labels, return_counts=True)))}"
        )
        print(
            f"    GT voxels={len(gt_coordinates):,}, "
            f"classes={dict(zip(*np.unique(gt_labels, return_counts=True)))}"
        )

        overlay_coordinates, overlay_categories, stats = (
            build_simple_overlay(
                pred_coordinates,
                pred_labels,
                gt_coordinates,
                gt_labels,
            )
        )

        print(
            "    overlay: "
            f"blue_base={stats['blue_base']:,}, "
            f"prediction_red={stats['prediction_red']:,}, "
            f"gt_yellow={stats['gt_yellow']:,}, "
            f"overlap_orange={stats['overlap_orange']:,}"
        )

        overlay_panel = render_simple_overlay(
            overlay_coordinates,
            overlay_categories,
            args,
        )

        combined = compose_frame(
            item["camera"],
            overlay_panel,
            item["lidar_token"],
            output_index,
            args,
        )
        combined.save(frames_dir / f"frame_{output_index:05d}.png")

    mp4, gif = encode_video(frames_dir, args.output_dir, args.fps)

    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    print("\nFinished:")
    print(f"MP4: {mp4}")
    print(f"GIF: {gif}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
