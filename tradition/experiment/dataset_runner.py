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
_RAW_RADAR_EXTENSIONS = {".mat", ".npy"}
_RAW_RADAR_DIRS = (
    "radar_tesseract",
    "radar_tensor_8doppler",
    "radar_polar_cube",
)
_RPC_RADAR_DIRS = (
    "",
    "rpc",
    "pc01p",
    "radar_pc",
    "radar_point_cloud",
)


def _natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


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


def _first_raw_radar(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        candidate = candidate.expanduser()
        if (
            candidate.suffix.lower() in _RAW_RADAR_EXTENSIONS
            and candidate.is_file()
        ):
            return candidate.resolve()
    return None


def _resolve_gt(
    info: dict[str, Any],
    repo_root: Path,
    gt_root: Path | None,
) -> Path:
    raw = Path(str(info["occ_path"]))
    candidates = [raw, repo_root / raw]
    if gt_root is not None:
        candidates.extend(
            [
                gt_root / raw.name,
                gt_root / str(info.get("scene_token", "")) / raw.name,
            ]
        )
    resolved = _first_existing(candidates)
    if resolved is None:
        raise FileNotFoundError(
            f"Cannot resolve GT occ_path={raw}; tried {candidates}"
        )
    return resolved


def _radar_value(info: dict[str, Any]) -> str | None:
    for key in (
        "radar_tensor_path",
        "rdr_tensor_path",
        "radar_path",
    ):
        value = info.get(key)
        if value:
            return str(value)
    curr = info.get("curr")
    if isinstance(curr, dict):
        for key in (
                "radar_tensor_path",
            "rdr_tensor_path",
            "radar_path",
        ):
            value = curr.get(key)
            if value:
                return str(value)
    return None


def _frame_tokens(info: dict[str, Any], radar_value: str) -> list[str]:
    raw = Path(radar_value)
    sources = [
        str(info.get("radar_frame_idx", "")),
        raw.stem,
        str(info.get("lidar_token", "")),
    ]
    tokens: list[str] = []
    for source in sources:
        groups = re.findall(r"\d+", source)
        if not groups:
            continue
        token = groups[-1]
        variants = [token]
        try:
            number = int(token)
        except ValueError:
            number = None
        if number is not None:
            variants.extend([f"{number:05d}", f"{number:06d}", str(number)])
        for variant in variants:
            if variant not in tokens:
                tokens.append(variant)
    return tokens


def _raw_names(frame_tokens: list[str]) -> list[str]:
    names: list[str] = []
    for frame in frame_tokens:
        for name in (
            f"tesseract_{frame}.mat",
            f"tesseract_{frame}.npy",
            f"DREA_{frame}.npy",
            f"radar_{frame}.mat",
            f"radar_{frame}.npy",
            f"{frame}.mat",
            f"{frame}.npy",
        ):
            if name not in names:
                names.append(name)
    return names


def _resolve_raw_radar(
    info: dict[str, Any],
    repo_root: Path,
    radar_root: Path | None,
) -> Path:
    value = _radar_value(info)
    if value is None:
        raise KeyError("Annotation entry has no radar-related path.")

    raw = Path(value)
    scene = str(info.get("scene_token", ""))
    frame_tokens = _frame_tokens(info, value)

    resolved = _first_raw_radar([raw, repo_root / raw])
    if resolved is not None:
        return resolved

    candidates: list[Path] = []
    if radar_root is not None:
        scene_root = radar_root / scene
        if raw.suffix.lower() in _RAW_RADAR_EXTENSIONS:
            candidates.append(radar_root / raw)
            for folder in _RAW_RADAR_DIRS:
                candidates.append(scene_root / folder / raw.name)
            candidates.append(scene_root / raw.name)

        for name in _raw_names(frame_tokens):
            for folder in _RAW_RADAR_DIRS:
                candidates.append(scene_root / folder / name)
            candidates.append(scene_root / name)

        resolved = _first_raw_radar(candidates)
        if resolved is not None:
            return resolved

    raise FileNotFoundError(
        "Cannot resolve the original 4-D K-Radar tensor. "
        f"annotation={raw}, scene={scene}, root={radar_root}"
    )


def _rpc_names(frame_tokens: list[str]) -> list[str]:
    names: list[str] = []
    for frame in frame_tokens:
        for name in (
            f"rpc_{frame}.npy",
            f"pc01p_{frame}.npy",
            f"radar_pc_{frame}.npy",
            f"{frame}.npy",
        ):
            if name not in names:
                names.append(name)
    return names


def _scene_variants(scene: str) -> list[str]:
    variants = [scene]
    try:
        number = int(scene)
    except ValueError:
        return variants
    for value in (str(number), f"{number:02d}"):
        if value not in variants:
            variants.append(value)
    return variants


def _resolve_rpc_radar(
    info: dict[str, Any],
    repo_root: Path,
    radar_root: Path | None,
) -> Path:
    value = _radar_value(info)
    if value is None:
        value = str(info.get("radar_frame_idx", ""))
    if not value:
        raise KeyError("Annotation entry has no radar path or radar_frame_idx.")

    raw = Path(value)
    scene = str(info.get("scene_token", ""))
    frame_tokens = _frame_tokens(info, value)
    names = _rpc_names(frame_tokens)

    direct = _first_raw_radar([raw, repo_root / raw])
    if direct is not None and direct.suffix.lower() == ".npy":
        return direct

    candidates: list[Path] = []
    if radar_root is not None:
        for scene_name in _scene_variants(scene):
            scene_root = radar_root / scene_name
            for folder in _RPC_RADAR_DIRS:
                base = scene_root / folder if folder else scene_root
                candidates.extend(base / name for name in names)
        for folder in _RPC_RADAR_DIRS:
            base = radar_root / folder if folder else radar_root
            candidates.extend(base / name for name in names)

    resolved = _first_raw_radar(candidates)
    if resolved is not None and resolved.suffix.lower() == ".npy":
        return resolved

    raise FileNotFoundError(
        "Cannot resolve Enhanced K-Radar RPC point cloud. Expected an "
        "rpc_*.npy/pc01p_*.npy [N,11] file.\n"
        f"Annotation radar path: {raw}\n"
        f"Radar frame candidates: {frame_tokens}\n"
        f"Scene: {scene}\n"
        f"Input root: {radar_root}\n"
        f"Tried: {candidates}"
    )


def _camera_map(
    infos: list[dict[str, Any]],
    scene: str,
    camera_dir: Path,
    offset: int = 0,
) -> dict[str, Path]:
    scene_infos = [
        info
        for info in infos
        if str(info.get("scene_token")) == str(scene)
    ]
    cameras = sorted(
        [
            p.resolve()
            for p in camera_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        ],
        key=_natural_key,
    )
    if not cameras:
        raise FileNotFoundError(f"No camera images found in {camera_dir}")

    mapping: dict[str, Path] = {}
    for ordinal, info in enumerate(scene_infos):
        camera_index = ordinal + offset
        if camera_index < 0 or camera_index >= len(cameras):
            raise IndexError(
                f"Camera index {camera_index} outside 0..{len(cameras)-1}; "
                "adjust --camera-offset."
            )
        mapping[str(info["lidar_token"])] = cameras[camera_index]
    return mapping


class TraditionalDatasetRunner:
    """Direct radar -> metrics/video runner; predictions stay in memory."""

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
        ego_speed_mps: float | None = None,
        max_frames: int | None = None,
        video_scene: str | None = None,
        camera_dir: str | Path | None = None,
        camera_offset: int = 0,
        max_video_frames: int = 100,
        fps: int = 10,
        no_rotate: bool = False,
        keep_frames: bool = False,
        video_background_prediction_root: str | Path | None = None,
        input_mode: str = "rpc",
    ) -> dict[str, Path]:
        if input_mode not in {"rpc", "raw"}:
            raise ValueError("input_mode must be 'rpc' or 'raw'.")

        annotation = Path(annotation).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        repo_root = Path(repo_root).expanduser().resolve()
        radar_root_p = (
            Path(radar_root).expanduser().resolve() if radar_root else None
        )
        gt_root_p = Path(gt_root).expanduser().resolve() if gt_root else None

        infos = _load_infos(annotation)
        if scene is not None:
            infos = [
                info
                for info in infos
                if str(info.get("scene_token")) == str(scene)
            ]
        if max_frames is not None:
            infos = infos[:max_frames]
        if not infos:
            raise RuntimeError("No annotation entries selected.")

        camera_by_token: dict[str, Path] = {}
        renderer: RadarOccStyleVideoRenderer | None = None
        if video_scene is not None:
            if camera_dir is None:
                raise ValueError(
                    "--camera-dir is required when --video-scene is used."
                )
            camera_by_token = _camera_map(
                _load_infos(annotation),
                str(video_scene),
                Path(camera_dir).expanduser().resolve(),
                camera_offset,
            )
            renderer = RadarOccStyleVideoRenderer(
                output_dir=output_dir,
                scene=str(video_scene),
                fps=fps,
                no_rotate=no_rotate,
                keep_frames=keep_frames,
                background_prediction_root=video_background_prediction_root,
            )

        accumulator = RadarOccMetricAccumulator()
        rendered = 0
        for index, info in enumerate(infos, start=1):
            token = str(info["lidar_token"])
            if input_mode == "rpc":
                radar_path = _resolve_rpc_radar(info, repo_root, radar_root_p)
            else:
                radar_path = _resolve_raw_radar(info, repo_root, radar_root_p)

            gt_path = _resolve_gt(info, repo_root, gt_root_p)
            prediction = self.pipeline.predict_file(
                radar_path,
                ego_speed_mps=ego_speed_mps,
            )
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
                f"[{index}/{len(infos)}] mode={input_mode} "
                f"scene={info.get('scene_token')} token={token} "
                f"input={radar_path.name} detections={len(prediction.detections)} "
                f"ego_speed={prediction.metadata['ego_speed_mps']:.2f}m/s "
                f"background={sum(int(label) == 1 for label in prediction.semantic_labels)} "
                f"foreground={sum(int(label) == 2 for label in prediction.semantic_labels)}"
            )

        outputs = MetricsReportWriter().write(
            accumulator.results(),
            output_dir,
        )
        if renderer is not None:
            video = renderer.finish()
            if video is not None:
                outputs["mp4"], outputs["gif"] = video
        return outputs
