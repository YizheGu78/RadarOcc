from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.render_scene3_camera_prediction_gt import (
    build_simple_overlay,
    compose_frame,
    encode_video,
    render_simple_overlay,
)


def _dense_xyz_to_sparse_zyx(dense_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.argwhere(dense_xyz != 0).astype(np.int64)
    if xyz.size == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int64)
    labels = dense_xyz[tuple(xyz.T)].astype(np.int64)
    return xyz[:, [2, 1, 0]], labels


class RadarOccStyleVideoRenderer:
    """Render the existing RadarOcc camera/occupancy overlay directly from RAM.

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
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.keep_frames = keep_frames
        self.fps = fps
        self.frame_count = 0
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
        ground_truth_xyz: np.ndarray,
        camera_path: str | Path,
        token: str,
    ) -> None:
        pred_coords, pred_labels = _dense_xyz_to_sparse_zyx(prediction_xyz)
        gt_coords, gt_labels = _dense_xyz_to_sparse_zyx(ground_truth_xyz)
        overlay_coords, overlay_categories, _ = build_simple_overlay(
            pred_coords, pred_labels, gt_coords, gt_labels
        )
        panel = render_simple_overlay(overlay_coords, overlay_categories, self.args)
        frame = compose_frame(
            Path(camera_path), panel, str(token), self.frame_count, self.args
        )
        frame.save(self.frames_dir / f"frame_{self.frame_count:05d}.png")
        self.frame_count += 1

    def finish(self) -> tuple[Path, Path] | None:
        if self.frame_count == 0:
            return None
        mp4, gif = encode_video(self.frames_dir, self.output_dir, self.fps)
        if not self.keep_frames:
            shutil.rmtree(self.frames_dir, ignore_errors=True)
        return mp4, gif
