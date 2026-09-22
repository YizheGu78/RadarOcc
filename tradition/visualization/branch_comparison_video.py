from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from tools.render_scene3_camera_prediction_gt import (
    encode_video,
    fit_image,
    get_font,
    render_simple_overlay,
)
from tradition.core.config import GridConfig


def _points_to_sparse_zyx(
    points_lidar_m: np.ndarray,
    grid: GridConfig,
) -> np.ndarray:
    points = np.asarray(points_lidar_m, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return np.empty((0, 3), dtype=np.int64)
    lower = np.asarray(grid.min_xyz, dtype=np.float64)
    indices = np.floor((points - lower) / grid.voxel_size_m).astype(np.int64)
    shape = np.asarray(grid.shape_xyz, dtype=np.int64)
    valid = np.all((indices >= 0) & (indices < shape), axis=1)
    indices = np.unique(indices[valid], axis=0)
    return indices[:, [2, 1, 0]]


class BranchComparisonVideoRenderer:
    """Render the pipeline's true static/dynamic branches beside RGB."""

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
        grid: GridConfig | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / "branch_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        for stale_frame in self.frames_dir.glob("frame_*.png"):
            stale_frame.unlink()
        self.scene = str(scene)
        self.fps = int(fps)
        self.keep_frames = bool(keep_frames)
        self.frame_count = 0
        self.grid = grid or GridConfig()
        self.args = SimpleNamespace(
            scene=self.scene,
            panel_size=panel_size,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            no_rotate=no_rotate,
        )

    def add_frame(
        self,
        static_points_lidar_m: np.ndarray,
        dynamic_points_lidar_m: np.ndarray,
        camera_path: str | Path,
        token: str,
    ) -> None:
        static = _points_to_sparse_zyx(
            static_points_lidar_m,
            self.grid,
        )
        dynamic = _points_to_sparse_zyx(dynamic_points_lidar_m, self.grid)
        static_panel = render_simple_overlay(
            static,
            np.ones(len(static), dtype=np.int8),
            self.args,
        )
        dynamic_panel = render_simple_overlay(
            dynamic,
            np.full(len(dynamic), 2, dtype=np.int8),
            self.args,
        )
        frame = self._compose(
            static_panel,
            dynamic_panel,
            Path(camera_path),
            str(token),
        )
        frame.save(self.frames_dir / f"frame_{self.frame_count:05d}.png")
        self.frame_count += 1

    def _compose(
        self,
        static_panel: Image.Image,
        dynamic_panel: Image.Image,
        camera_path: Path,
        token: str,
    ) -> Image.Image:
        panel_size = self.args.panel_size
        title_height = 60
        footer_height = 44
        canvas = Image.new(
            "RGB",
            (panel_size * 3, title_height + panel_size + footer_height),
            (255, 255, 255),
        )
        with Image.open(camera_path) as camera_file:
            camera = camera_file.convert("RGB")
        panels = (
            fit_image(static_panel, panel_size, panel_size, (255, 255, 255)),
            fit_image(dynamic_panel, panel_size, panel_size, (255, 255, 255)),
            fit_image(
                camera,
                panel_size,
                panel_size,
                (0, 0, 0),
            ),
        )
        for index, panel in enumerate(panels):
            canvas.paste(panel, (index * panel_size, title_height))

        draw = ImageDraw.Draw(canvas)
        title_font = get_font(25)
        footer_font = get_font(17)
        titles = (
            "Static branch (current + history)",
            "Dynamic branch (current only)",
            "RGB original",
        )
        for index, title in enumerate(titles):
            box = draw.textbbox((0, 0), title, font=title_font)
            width = box[2] - box[0]
            x = index * panel_size + (panel_size - width) // 2
            draw.text((x, 15), title, fill=(0, 0, 0), font=title_font)

        footer = (
            f"Scene {self.scene} | frame {self.frame_count:03d} | "
            f"token {token} | blue=static | red=dynamic | raw Doppler bins"
        )
        box = draw.textbbox((0, 0), footer, font=footer_font)
        width = box[2] - box[0]
        draw.text(
            ((canvas.width - width) // 2, title_height + panel_size + 10),
            footer,
            fill=(0, 0, 0),
            font=footer_font,
        )
        return canvas

    def finish(self) -> tuple[Path, Path] | None:
        if self.frame_count == 0:
            return None
        mp4, gif = encode_video(self.frames_dir, self.output_dir, self.fps)
        branch_mp4 = self.output_dir / f"scene_{self.scene}_branches_rgb.mp4"
        branch_gif = self.output_dir / f"scene_{self.scene}_branches_rgb.gif"
        mp4.replace(branch_mp4)
        gif.replace(branch_gif)
        if not self.keep_frames:
            shutil.rmtree(self.frames_dir, ignore_errors=True)
        return branch_mp4, branch_gif
