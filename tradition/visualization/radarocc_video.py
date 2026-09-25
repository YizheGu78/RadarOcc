from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from tools.render_scene3_camera_prediction_gt import (
    build_simple_overlay,
    encode_video,
    fit_image,
    get_font,
    load_sparse_occupancy,
    render_simple_overlay,
)


SEMANTIC_LEGEND = [
    ("Background", (26, 115, 242)),
    ("Prediction FG", (242, 38, 26)),
    ("GT FG", (255, 209, 26)),
    ("FG overlap", (255, 115, 13)),
]

OCCUPANCY_LEGEND = [
    ("Free", (26, 115, 242)),
    ("Occupied", (242, 38, 26)),
    ("GT occupied", (255, 209, 26)),
    ("Overlap", (255, 115, 13)),
    ("Unknown", (150, 150, 150)),
]


def _dense_xyz_to_sparse_zyx(dense_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.argwhere(dense_xyz != 0).astype(np.int64)
    if xyz.size == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int64)
    labels = dense_xyz[tuple(xyz.T)].astype(np.int64)
    return xyz[:, [2, 1, 0]], labels


def _index_radarocc_predictions(
    prediction_root: str | Path,
) -> dict[str, Path]:
    root = Path(prediction_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"RadarOcc background prediction root not found: {root}"
        )

    predictions: dict[str, Path] = {}
    for path in root.rglob("pred_c.npy"):
        token = path.parent.name
        if token in predictions:
            raise RuntimeError(
                f"Duplicate RadarOcc pred_c.npy for token {token}: "
                f"{predictions[token]} and {path}"
            )
        predictions[token] = path.resolve()

    if not predictions:
        raise FileNotFoundError(f"No */pred_c.npy files found under {root}")
    return predictions


def _replace_background_for_visualization(
    traditional_prediction_xyz: np.ndarray,
    radarocc_prediction_path: Path,
) -> np.ndarray:
    """Use RadarOcc label-1 voxels as blue video base only.

    Traditional object-aware foreground retains visual priority. This runs
    inside the renderer after metric accumulation, so evaluation remains a
    pure traditional prediction.
    """
    radarocc_coords_zyx, radarocc_labels = load_sparse_occupancy(
        radarocc_prediction_path,
        is_gt=False,
    )
    radarocc_background_zyx = radarocc_coords_zyx[radarocc_labels == 1]

    visual_prediction = np.zeros_like(traditional_prediction_xyz)
    if radarocc_background_zyx.size:
        radarocc_background_xyz = radarocc_background_zyx[:, [2, 1, 0]]
        visual_prediction[tuple(radarocc_background_xyz.T)] = 1

    visual_prediction[traditional_prediction_xyz >= 2] = 2
    return visual_prediction


def _native_bev_categories(
    native_prediction_xyz: np.ndarray,
    ground_truth_xyz: np.ndarray,
) -> np.ndarray:
    """Project exact 3D OctoMap/GT comparisons into a readable BEV.

    Category IDs:
        0 unknown, 1 free, 2 occupied-only, 3 GT-only, 4 exact overlap.

    Overlap is computed per 3D voxel before projection. The BEV precedence is
    overlap > occupied-only > GT-only > free > unknown, so voxels at different
    heights are never counted as overlaps merely because they share an XY cell.
    Unknown is displayed only when an XY column has no stronger evidence.
    """
    native = np.asarray(native_prediction_xyz)
    gt = np.asarray(ground_truth_xyz)
    if native.shape != gt.shape or native.ndim != 3:
        raise ValueError(
            "Native prediction and GT must be matching [X,Y,Z] arrays; "
            f"got {native.shape} and {gt.shape}"
        )
    unexpected = np.setdiff1d(np.unique(native), np.array([0, 1, 255]))
    if unexpected.size:
        raise ValueError(
            f"Native OctoMap labels must be 0/1/255; got {unexpected.tolist()}"
        )

    free = native == 0
    occupied = native == 1
    gt_occupied = (gt != 0) & (gt != 255)
    overlap = occupied & gt_occupied
    occupied_only = occupied & ~gt_occupied
    gt_only = gt_occupied & ~occupied

    bev = np.zeros(native.shape[:2], dtype=np.uint8)
    bev[np.any(free, axis=2)] = 1
    bev[np.any(gt_only, axis=2)] = 3
    bev[np.any(occupied_only, axis=2)] = 2
    bev[np.any(overlap, axis=2)] = 4
    return bev


def _render_native_bev(
    native_prediction_xyz: np.ndarray,
    ground_truth_xyz: np.ndarray,
    panel_size: int,
) -> Image.Image:
    # RGB colors follow the requested video legend exactly.
    colors = np.array(
        [
            [150, 150, 150],  # unknown grey
            [26, 115, 242],   # free blue
            [242, 38, 26],    # occupied red
            [255, 209, 26],   # GT occupied yellow
            [255, 115, 13],   # exact occupied/GT overlap orange
        ],
        dtype=np.uint8,
    )
    bev = _native_bev_categories(native_prediction_xyz, ground_truth_xyz)
    # X: far range at the top, ego at the bottom. Reverse Y as well so
    # image left/right matches the camera and the existing 3D panel.
    image = Image.fromarray(colors[bev[::-1, ::-1]], mode="RGB")
    image = image.resize((panel_size, panel_size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, panel_size - 1, panel_size - 1), outline=(40, 40, 40), width=2)
    marker_x = panel_size // 2
    draw.polygon(
        [(marker_x, panel_size - 16), (marker_x - 8, panel_size - 3),
         (marker_x + 8, panel_size - 3)],
        fill=(0, 0, 0),
    )
    return image


def _draw_legend_row(
    draw: ImageDraw.ImageDraw,
    heading: str,
    items: list[tuple[str, tuple[int, int, int]]],
    y: int,
    width: int,
) -> None:
    font = get_font(16)
    heading_font = get_font(17)
    gap = 34
    swatch = 18
    heading_width = draw.textbbox((0, 0), heading, font=heading_font)[2]
    item_widths = [
        swatch + 9 + (draw.textbbox((0, 0), label, font=font)[2])
        for label, _ in items
    ]
    total = heading_width + gap + sum(item_widths) + gap * (len(items) - 1)
    x = max(18, (width - total) // 2)
    draw.text((x, y), heading, fill=(0, 0, 0), font=heading_font)
    x += heading_width + gap
    for (label, color), item_width in zip(items, item_widths):
        draw.rectangle((x, y + 1, x + swatch, y + swatch + 1), fill=color, outline=(0, 0, 0))
        draw.text((x + swatch + 9, y), label, fill=(0, 0, 0), font=font)
        x += item_width + gap


def _compose_triptych(
    camera_path: Path,
    semantic_panel: Image.Image,
    occupancy_panel: Image.Image,
    token: str,
    output_index: int,
    args: SimpleNamespace,
) -> Image.Image:
    panel = args.panel_size
    title_height = 58
    legend_height = 82
    footer_height = 42
    total_width = panel * 3
    total_height = title_height + panel + legend_height + footer_height

    semantic_panel = fit_image(semantic_panel, panel, panel, (255, 255, 255))
    occupancy_panel = fit_image(occupancy_panel, panel, panel, (255, 255, 255))
    camera = fit_image(Image.open(camera_path).convert("RGB"), panel, panel, (0, 0, 0))

    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    canvas.paste(semantic_panel, (0, title_height))
    canvas.paste(occupancy_panel, (panel, title_height))
    canvas.paste(camera, (panel * 2, title_height))

    draw = ImageDraw.Draw(canvas)
    title_font = get_font(24)
    titles = [
        "Semantic prediction + GT (3D)",
        "OctoMap native + GT (BEV)",
        "RGB camera",
    ]
    for index, title in enumerate(titles):
        box = draw.textbbox((0, 0), title, font=title_font)
        x = index * panel + (panel - (box[2] - box[0])) // 2
        draw.text((x, 14), title, fill=(0, 0, 0), font=title_font)

    legend_top = title_height + panel + 7
    _draw_legend_row(draw, "Left 3D:", SEMANTIC_LEGEND, legend_top, total_width)
    _draw_legend_row(draw, "Middle BEV:", OCCUPANCY_LEGEND, legend_top + 35, total_width)

    footer_font = get_font(17)
    footer = (
        f"Scene {args.scene} | frame {output_index:03d} | "
        f"token {token} | {camera_path.name}"
    )
    box = draw.textbbox((0, 0), footer, font=footer_font)
    draw.text(
        ((total_width - (box[2] - box[0])) // 2,
         title_height + panel + legend_height + 7),
        footer,
        fill=(0, 0, 0),
        font=footer_font,
    )
    return canvas


class RadarOccStyleVideoRenderer:
    """Render semantic 3D, native OctoMap BEV, and RGB panels from RAM.

    No prediction .npy is written. Only temporary PNG frames are used for
    ffmpeg; they are deleted after encoding unless keep_frames=True.
    """

    def __init__(
        self,
        output_dir: str | Path,
        scene: str,
        fps: int = 10,
        panel_size: int = 600,
        azimuth: float = 180.0,
        elevation: float = 65.0,
        distance: float = 82.0,
        no_rotate: bool = False,
        keep_frames: bool = False,
        background_prediction_root: str | Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / "semantic_octomap_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        for stale_frame in self.frames_dir.glob("frame_*.png"):
            stale_frame.unlink()
        self.keep_frames = keep_frames
        self.fps = fps
        self.frame_count = 0
        self.background_predictions = (
            _index_radarocc_predictions(background_prediction_root)
            if background_prediction_root is not None
            else None
        )
        self.args = SimpleNamespace(
            scene=str(scene),
            panel_size=panel_size,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            no_rotate=no_rotate,
        )

    def add_frame(
        self,
        prediction_xyz: np.ndarray,
        native_prediction_xyz: np.ndarray,
        ground_truth_xyz: np.ndarray,
        camera_path: str | Path,
        token: str,
    ) -> None:
        visual_prediction_xyz = prediction_xyz
        if self.background_predictions is not None:
            background_path = self.background_predictions.get(str(token))
            if background_path is None:
                raise FileNotFoundError(
                    f"No RadarOcc pred_c.npy found for video token {token}"
                )
            visual_prediction_xyz = _replace_background_for_visualization(
                prediction_xyz,
                background_path,
            )

        pred_coords, pred_labels = _dense_xyz_to_sparse_zyx(visual_prediction_xyz)
        gt_coords, gt_labels = _dense_xyz_to_sparse_zyx(ground_truth_xyz)
        overlay_coords, overlay_categories, _ = build_simple_overlay(
            pred_coords, pred_labels, gt_coords, gt_labels
        )
        semantic_panel = render_simple_overlay(
            overlay_coords, overlay_categories, self.args
        )
        occupancy_panel = _render_native_bev(
            native_prediction_xyz, ground_truth_xyz, self.args.panel_size
        )
        frame = _compose_triptych(
            Path(camera_path), semantic_panel, occupancy_panel,
            str(token), self.frame_count, self.args
        )
        frame.save(self.frames_dir / f"frame_{self.frame_count:05d}.png")
        self.frame_count += 1

    def finish(self) -> tuple[Path, Path] | None:
        if self.frame_count == 0:
            return None
        mp4, gif = encode_video(self.frames_dir, self.output_dir, self.fps)
        output_mp4 = self.output_dir / f"scene_{self.args.scene}_semantic_octomap_rgb.mp4"
        output_gif = self.output_dir / f"scene_{self.args.scene}_semantic_octomap_rgb.gif"
        mp4.replace(output_mp4)
        gif.replace(output_gif)
        if not self.keep_frames:
            shutil.rmtree(self.frames_dir, ignore_errors=True)
        return output_mp4, output_gif
