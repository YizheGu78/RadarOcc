from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from tradition.evaluation.radarocc_metrics import (
    RadarOccMetricAccumulator,
    load_gt_sparse_xyz,
)
from tradition.pipeline.traditional_radar_pipeline import TraditionalRadarPipeline
from tradition.reporting.report_writer import MetricsReportWriter
from tradition.visualization.radarocc_video import RadarOccStyleVideoRenderer


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _load_infos(annotation: Path) -> list[dict[str, Any]]:
    with annotation.open("rb") as handle:
        content = pickle.load(handle)
    if isinstance(content, dict) and isinstance(content.get("infos"), list):
        infos = content["infos"]
    elif isinstance(content, dict) and isinstance(content.get("data_list"), list):
        infos = content["data_list"]
    elif isinstance(content, list):
        infos = content
    else:
        raise TypeError(f"Unsupported annotation structure in {annotation}")
    return sorted(infos, key=lambda info: info.get("timestamp", 0))


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_gt(info: dict[str, Any], repo_root: Path, gt_root: Path | None) -> Path:
    raw = Path(str(info["occ_path"]))
    candidates = [raw, repo_root / raw]
    if gt_root is not None:
        candidates.extend([gt_root / raw.name, gt_root / str(info.get("scene_token", "")) / raw.name])
    resolved = _first_existing(candidates)
    if resolved is None:
        raise FileNotFoundError(f"Cannot resolve GT occ_path={raw}; tried {candidates}")
    return resolved


def _radar_value(info: dict[str, Any]) -> str | None:
    for key in ("radar_tensor_path", "rdr_tensor_path", "radar_path"):
        value = info.get(key)
        if value:
            return str(value)
    curr = info.get("curr")
    if isinstance(curr, dict):
        for key in ("radar_tensor_path", "rdr_tensor_path", "radar_path"):
            value = curr.get(key)
            if value:
                return str(value)
    return None


def _resolve_radar(info: dict[str, Any], repo_root: Path, radar_root: Path | None) -> Path:
    value = _radar_value(info)
    if value is None:
        raise KeyError(
            "Annotation entry has no raw radar tensor path. Expected radar_tensor_path/rdr_tensor_path/radar_path."
        )
    raw = Path(value)
    scene = str(info.get("scene_token", ""))
    candidates = [raw, repo_root / raw]
    if radar_root is not None:
        candidates.extend(
            [
                radar_root / raw,
                radar_root / scene / "radar_tesseract" / raw.name,
                radar_root / scene / "radar_polar_cube" / raw.name,
                radar_root / scene / raw.name,
            ]
        )
    resolved = _first_existing(candidates)
    if resolved is None:
        raise FileNotFoundError(f"Cannot resolve raw radar tensor {raw}; tried {candidates}")
    return resolved


def _camera_map(
    infos: list[dict[str, Any]],
    scene: str,
    camera_dir: Path,
    offset: int = 0,
) -> dict[str, Path]:
    scene_infos = [info for info in infos if str(info.get("scene_token")) == str(scene)]
    cameras = sorted(
        [p.resolve() for p in camera_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS],
        key=_natural_key,
    )
    if not cameras:
        raise FileNotFoundError(f"No camera images found in {camera_dir}")

    mapping: dict[str, Path] = {}
    for ordinal, info in enumerate(scene_infos):
        camera_index = ordinal + offset
        if camera_index < 0 or camera_index >= len(cameras):
            raise IndexError(
                f"Camera index {camera_index} outside 0..{len(cameras)-1}; adjust --camera-offset."
            )
        mapping[str(info["lidar_token"])] = cameras[camera_index]
    return mapping


class TraditionalDatasetRunner:
    """Direct raw-radar -> metrics/video runner; predictions remain in memory."""

    def __init__(self, pipeline: TraditionalRadarPipeline) -> None:
        self.pipeline = pipeline

    def run(
        self,
        annotation: str | Path,
        output_dir: str | Path,
        repo_root: str | Path = ".",
        radar_root: str | Path | None = None,
        gt_root: str | Path | None = None,
        gt_order: str = "xyz",
        scene: str | None = None,
        ego_speed_mps: float = 0.0,
        max_frames: int | None = None,
        video_scene: str | None = None,
        camera_dir: str | Path | None = None,
        camera_offset: int = 0,
        max_video_frames: int = 100,
        fps: int = 10,
        keep_frames: bool = False,
    ) -> dict[str, Path]:
        annotation = Path(annotation).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        repo_root = Path(repo_root).expanduser().resolve()
        radar_root_p = Path(radar_root).expanduser().resolve() if radar_root else None
        gt_root_p = Path(gt_root).expanduser().resolve() if gt_root else None
        infos = _load_infos(annotation)
        if scene is not None:
            infos = [info for info in infos if str(info.get("scene_token")) == str(scene)]
        if max_frames is not None:
            infos = infos[:max_frames]
        if not infos:
            raise RuntimeError("No annotation entries selected.")

        camera_by_token: dict[str, Path] = {}
        renderer: RadarOccStyleVideoRenderer | None = None
        if video_scene is not None:
            if camera_dir is None:
                raise ValueError("--camera-dir is required when --video-scene is used.")
            camera_by_token = _camera_map(
                _load_infos(annotation), str(video_scene), Path(camera_dir).expanduser().resolve(), camera_offset
            )
            renderer = RadarOccStyleVideoRenderer(
                output_dir=output_dir,
                scene=str(video_scene),
                fps=fps,
                keep_frames=keep_frames,
            )

        accumulator = RadarOccMetricAccumulator()
        rendered = 0
        for index, info in enumerate(infos, start=1):
            token = str(info["lidar_token"])
            radar_path = _resolve_radar(info, repo_root, radar_root_p)
            gt_path = _resolve_gt(info, repo_root, gt_root_p)
            prediction = self.pipeline.predict_file(radar_path, ego_speed_mps=ego_speed_mps)
            gt = load_gt_sparse_xyz(gt_path, coordinate_order=gt_order)
            accumulator.update(prediction.dense_labels_xyz, gt)

            if (
                renderer is not None
                and str(info.get("scene_token")) == str(video_scene)
                and token in camera_by_token
                and rendered < max_video_frames
            ):
                renderer.add_frame(
                    prediction.dense_labels_xyz,
                    gt,
                    camera_by_token[token],
                    token,
                )
                rendered += 1

            print(
                f"[{index}/{len(infos)}] scene={info.get('scene_token')} token={token} "
                f"detections={len(prediction.detections)}"
            )

        outputs = MetricsReportWriter().write(accumulator.results(), output_dir)
        if renderer is not None:
            video = renderer.finish()
            if video is not None:
                outputs["mp4"], outputs["gif"] = video
        return outputs
