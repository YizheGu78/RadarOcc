from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.render_scene3_camera_prediction_gt import (
    build_simple_overlay,
    compose_frame,
    encode_video,
    load_sparse_occupancy,
    render_simple_overlay,
)


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
        raise FileNotFoundError(
            f"No */pred_c.npy files found under {root}"
        )
    return predictions


def _replace_background_for_visualization(
    traditional_prediction_xyz: np.ndarray,
    radarocc_prediction_path: Path,
) -> np.ndarray:
    """Use RadarOcc label-1 voxels as the video's blue base only.

    Traditional foreground remains label 2 and takes visual priority. This
    function is intentionally used after metric accumulation, so the classical
    baseline scores never include RadarOcc predictions.
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
        background_prediction_root: str | Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
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

        pred_coords, pred_labels = _dense_xyz_to_sparse_zyx(
            visual_prediction_xyz
        )
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
