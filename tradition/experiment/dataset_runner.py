from __future__ import annotations

import json
import math
import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from tradition.core.config import KRadarConfig, PoseConfig
from tradition.evaluation.radarocc_metrics import (
    RadarOccMetricAccumulator,
    load_gt_sparse_xyz,
)
from tradition.io.pose_reader import RadarOccPoseReader
from tradition.motion.pose_ego_motion import PoseEgoMotionEstimator
from tradition.pipeline.traditional_radar_pipeline import TraditionalRadarPipeline
from tradition.reporting.report_writer import MetricsReportWriter


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
_CALIBRATION_FILENAME = "calib_radar_lidar.txt"


class RPCFrameNotFoundError(FileNotFoundError):
    """An aligned annotation frame has no corresponding local RPC file."""

    def __init__(
        self,
        message: str,
        *,
        scene: str,
        aligned_frame: int,
        frame_difference: int,
        rpc_frame: int,
    ) -> None:
        super().__init__(message)
        self.scene = str(scene)
        self.aligned_frame = int(aligned_frame)
        self.frame_difference = int(frame_difference)
        self.rpc_frame = int(rpc_frame)


def _natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _info_sequence_key(info: dict[str, Any]) -> tuple[int, int | float]:
    """Chronological key inside one scene; never used to match cross-sensor IDs."""
    token = str(info.get("lidar_token", ""))
    groups = re.findall(r"\d+", token)
    if groups:
        return 0, int(groups[-1])
    try:
        return 1, float(info.get("timestamp", 0))
    except (TypeError, ValueError):
        return 1, 0.0


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

    infos = sorted(infos, key=lambda info: info.get("timestamp", 0))

    # RPC/RGB file numbers belong to different sensor clocks and are not
    # comparable to annotation/LiDAR frame numbers.  Store only the ordinal
    # position of each annotation entry inside its scene.  RPC resolution then
    # pairs the N-th annotation frame with the N-th RPC file after each side is
    # independently sorted in chronological filename order.
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for info in infos:
        by_scene.setdefault(str(info.get("scene_token", "")), []).append(info)
    for scene_infos in by_scene.values():
        ordered = sorted(scene_infos, key=_info_sequence_key)
        count = len(ordered)
        for ordinal, info in enumerate(ordered):
            info["_rpc_sequence_ordinal"] = ordinal
            info["_rpc_sequence_count"] = count
    return infos


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


def _is_rpc_file(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".npy":
        return False
    return re.fullmatch(
        r"(?:rpc_|pc01p_|radar_pc_)?\d+\.npy",
        path.name,
        flags=re.IGNORECASE,
    ) is not None


@lru_cache(maxsize=None)
def _scene_rpc_files(radar_root_text: str, scene: str) -> tuple[Path, ...]:
    """Return one scene's RPC files in sensor-local chronological order."""
    radar_root = Path(radar_root_text).expanduser().resolve()
    searched: list[Path] = []

    # Prefer an explicit scene directory (4 before 04), then support callers
    # that pass a directory already pointing at one scene.
    bases: list[Path] = []
    for scene_name in _scene_variants(scene):
        scene_root = radar_root / scene_name
        for folder in _RPC_RADAR_DIRS:
            bases.append(scene_root / folder if folder else scene_root)
    for folder in _RPC_RADAR_DIRS:
        bases.append(radar_root / folder if folder else radar_root)

    seen_bases: set[Path] = set()
    for base in bases:
        if base in seen_bases:
            continue
        seen_bases.add(base)
        searched.append(base)
        if not base.is_dir():
            continue
        files = sorted(
            (path.resolve() for path in base.iterdir() if _is_rpc_file(path)),
            key=_natural_key,
        )
        if files:
            return tuple(files)

    raise FileNotFoundError(
        "Cannot find any Enhanced K-Radar RPC files for "
        f"scene={scene} under {radar_root}. Searched: {searched}"
    )


def _resolve_rpc_radar(
    info: dict[str, Any],
    repo_root: Path,
    radar_root: Path | None,
    calib_root: Path | None = None,
) -> Path:
    """Resolve RPC by scene-local order, never by cross-sensor frame number.

    The N-th annotation entry in a scene is paired with the N-th RPC file after
    the annotation entries and RPC files are independently sorted.  This is
    intentional: radar, LiDAR/annotation and RGB filenames use different frame
    numbering/rates, so fixed ID equality or a fixed frame-difference offset is
    not a valid correspondence rule.
    """
    value = _radar_value(info)
    if value is None:
        value = str(info.get("radar_frame_idx", ""))
    if not value:
        raise KeyError("Annotation entry has no radar path or radar_frame_idx.")

    raw = Path(value)
    scene = str(info.get("scene_token", ""))

    # A direct NPY is already an explicit RPC path and needs no ordinal lookup.
    direct = _first_raw_radar([raw, repo_root / raw])
    if direct is not None and direct.suffix.lower() == ".npy":
        return direct

    if radar_root is None:
        raise FileNotFoundError(
            f"RPC input requires radar_root; scene={scene}, annotation={raw}"
        )

    ordinal_value = info.get("_rpc_sequence_ordinal")
    count_value = info.get("_rpc_sequence_count")
    if ordinal_value is None:
        raise RuntimeError(
            "RPC sequence ordinal is missing. Load annotation entries with "
            "_load_infos() before resolving RPC files."
        )
    ordinal = int(ordinal_value)
    annotation_count = int(count_value) if count_value is not None else None
    rpc_files = _scene_rpc_files(str(radar_root.resolve()), scene)

    if annotation_count is not None and annotation_count != len(rpc_files):
        raise RPCFrameNotFoundError(
            "Scene-local ordered RPC matching requires equal sequence lengths. "
            f"Scene {scene}: annotation frames={annotation_count}, "
            f"RPC files={len(rpc_files)}. "
            "Cross-sensor frame IDs are intentionally ignored.",
            scene=scene,
            aligned_frame=ordinal,
            frame_difference=0,
            rpc_frame=ordinal,
        )

    if ordinal < 0 or ordinal >= len(rpc_files):
        raise RPCFrameNotFoundError(
            "Cannot resolve RPC by scene-local order. "
            f"Scene {scene}: annotation ordinal={ordinal}, "
            f"RPC files={len(rpc_files)}. "
            "Cross-sensor frame IDs are intentionally ignored.",
            scene=scene,
            aligned_frame=ordinal,
            frame_difference=0,
            rpc_frame=ordinal,
        )

    return rpc_files[ordinal]


def _camera_map(
    infos: list[dict[str, Any]],
    scene: str,
    camera_dir: Path,
    offset: int = 0,
) -> dict[str, Path]:
    scene_infos = sorted(
        (
            info
            for info in infos
            if str(info.get("scene_token")) == str(scene)
        ),
        key=lambda info: int(info.get("_rpc_sequence_ordinal", 0)),
    )
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
        calib_root: str | Path | None = None,
        gt_root: str | Path | None = None,
        gt_order: str = "xyz",
        scene: str | None = None,
        ego_speed_mps: float | None = None,
        max_frames: int | None = None,
        video_scene: str | None = None,
        video_layout: str = "both",
        camera_dir: str | Path | None = None,
        camera_offset: int = 0,
        video_background_prediction_root: str | Path | None = None,
        max_video_frames: int = 100,
        fps: int = 10,
        no_rotate: bool = False,
        keep_frames: bool = False,
        input_mode: str = "rpc",
        pose_root: str | Path | None = None,
        pose_dt_s: float = 0.10,
    ) -> dict[str, Path]:
        if input_mode not in {"rpc", "raw"}:
            raise ValueError("input_mode must be 'rpc' or 'raw'.")

        unwrapper = self.pipeline.velocity_unwrapper
        if unwrapper is not None:
            if input_mode != "rpc" or pose_root is None:
                raise ValueError("Range-Kalman unwrapping requires RPC with poses.")
            if not math.isclose(unwrapper.config.frame_dt_s, pose_dt_s):
                raise ValueError("Unwrapping and pose frame intervals must match.")
        annotation = Path(annotation).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        repo_root = Path(repo_root).expanduser().resolve()
        radar_root_p = (
            Path(radar_root).expanduser().resolve() if radar_root else None
        )
        if calib_root:
            calib_root_path = Path(calib_root).expanduser()
            calib_root_p = (
                calib_root_path.resolve()
                if calib_root_path.is_absolute()
                else (repo_root / calib_root_path).resolve()
            )
        else:
            calib_root_p = (repo_root / "data" / "K-Radar_calib").resolve()
        gt_root_p = Path(gt_root).expanduser().resolve() if gt_root else None
        pose_root_p = (
            Path(pose_root).expanduser().resolve() if pose_root else None
        )
        if pose_root_p is not None and input_mode != "rpc":
            raise ValueError("Pose-temporal processing currently supports RPC only.")
        pose_reader = (
            RadarOccPoseReader(pose_root_p) if pose_root_p is not None else None
        )
        pose_estimator = (
            PoseEgoMotionEstimator(
                radar_cfg=KRadarConfig(),
                pose_cfg=PoseConfig(frame_dt_s=pose_dt_s),
            )
            if pose_reader is not None
            else None
        )

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
        branch_renderer = None
        semantic_renderer = None
        if video_scene is not None:
            if camera_dir is None:
                raise ValueError(
                    "--camera-dir is required when --video-scene is used."
                )
            if video_layout not in {
                "branch-comparison",
                "semantic-overlay",
                "both",
            }:
                raise ValueError("Unsupported video layout.")
            if (
                video_layout in {"branch-comparison", "both"}
                and pose_reader is None
            ):
                raise ValueError(
                    "Branch-comparison video requires temporal RPC processing "
                    "with --pose-root."
                )
            camera_by_token = _camera_map(
                _load_infos(annotation),
                str(video_scene),
                Path(camera_dir).expanduser().resolve(),
                camera_offset,
            )
            # Keep video-only Mayavi/Qt dependencies out of metrics-only runs.
            if video_layout in {"branch-comparison", "both"}:
                from tradition.visualization.branch_comparison_video import (
                    BranchComparisonVideoRenderer,
                )

                branch_renderer = BranchComparisonVideoRenderer(
                    output_dir=output_dir,
                    scene=str(video_scene),
                    fps=fps,
                    no_rotate=no_rotate,
                    keep_frames=keep_frames,
                )
            if video_layout in {"semantic-overlay", "both"}:
                from tradition.visualization.radarocc_video import (
                    RadarOccStyleVideoRenderer,
                )

                semantic_renderer = RadarOccStyleVideoRenderer(
                    output_dir=output_dir,
                    scene=str(video_scene),
                    fps=fps,
                    no_rotate=no_rotate,
                    keep_frames=keep_frames,
                    background_prediction_root=video_background_prediction_root,
                )

        accumulator = RadarOccMetricAccumulator()
        rendered = 0
        processed_frames = 0
        skipped_rpc_frames: list[dict[str, object]] = []
        active_scene: str | None = None
        self.pipeline.reset_sequence()
        for index, info in enumerate(infos, start=1):
            token = str(info["lidar_token"])
            scene_token = str(info.get("scene_token"))
            if scene_token != active_scene:
                self.pipeline.reset_sequence()
                active_scene = scene_token
            if input_mode == "rpc":
                try:
                    radar_path = _resolve_rpc_radar(
                        info, repo_root, radar_root_p, calib_root_p
                    )
                except RPCFrameNotFoundError as error:
                    skipped_rpc_frames.append(
                        {
                            "scene": scene_token,
                            "token": token,
                            "aligned_frame": error.aligned_frame,
                            "frame_difference": error.frame_difference,
                            "rpc_frame": error.rpc_frame,
                        }
                    )
                    print(
                        f"[{index}/{len(infos)}] SKIP missing RPC "
                        f"scene={scene_token} token={token} "
                        f"aligned={error.aligned_frame:05d} "
                        f"offset={error.frame_difference:+d} "
                        f"rpc={error.rpc_frame:05d}"
                    )
                    continue
            else:
                radar_path = _resolve_raw_radar(info, repo_root, radar_root_p)

            processed_frames += 1
            gt_path = _resolve_gt(info, repo_root, gt_root_p)
            if pose_reader is not None and pose_estimator is not None:
                pose_index = pose_reader.frame_index(token)
                current_pose, current_pose_path = pose_reader.read_index(
                    scene_token, pose_index
                )
                previous = pose_reader.try_read_index(
                    scene_token, pose_index - 1
                )
                following = (
                    pose_reader.try_read_index(scene_token, pose_index + 1)
                    if previous is None
                    else None
                )
                ego_motion = pose_estimator.estimate(
                    current_pose=current_pose,
                    previous_pose=previous[0] if previous is not None else None,
                    next_pose=following[0] if following is not None else None,
                )
                prediction = self.pipeline.predict_temporal_file(
                    radar_path=radar_path,
                    token=token,
                    ego_motion=ego_motion,
                )
                prediction.metadata["pose_path"] = str(current_pose_path)
                prediction.metadata["pose_index"] = pose_index
                prediction.metadata["pose_dt_s"] = pose_dt_s
                if previous is not None:
                    prediction.metadata["adjacent_pose_path"] = str(previous[1])
                elif following is not None:
                    prediction.metadata["adjacent_pose_path"] = str(following[1])
            else:
                prediction = self.pipeline.predict_file(
                    radar_path,
                    ego_speed_mps=ego_speed_mps,
                )
            if unwrapper is not None:
                diagnostics_dir = output_dir / "velocity_unwrapping"
                diagnostics_dir.mkdir(parents=True, exist_ok=True)
                (diagnostics_dir / f"{token}.json").write_text(
                    json.dumps(prediction.metadata["velocity_unwrapping"], allow_nan=False),
                    encoding="utf-8",
                )
            if pose_reader is not None:
                diagnostics_dir = output_dir / "branch_diagnostics"
                diagnostics_dir.mkdir(parents=True, exist_ok=True)
                (diagnostics_dir / f"{token}.json").write_text(
                    json.dumps(
                        {
                            "occupancy_persistence": prediction.metadata[
                                "occupancy_persistence"
                            ],
                            "persistent_current_count": prediction.metadata[
                                "persistent_current_count"
                            ],
                            "motion_confirmed_count": prediction.metadata[
                                "motion_confirmed_count"
                            ],
                            "unknown_count": prediction.metadata["unknown_count"],
                        },
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )
            gt = load_gt_sparse_xyz(gt_path, coordinate_order=gt_order)
            accumulator.update(prediction.dense_labels_xyz, gt)

            if (
                (branch_renderer is not None or semantic_renderer is not None)
                and str(info.get("scene_token")) == str(video_scene)
                and token in camera_by_token
                and rendered < max_video_frames
            ):
                camera_path = camera_by_token[token]
                if branch_renderer is not None:
                    branches = prediction.branch_points_lidar_m or {}
                    branch_renderer.add_frame(
                        branches.get("persistent", np.empty((0, 3))),
                        branches.get("motion", np.empty((0, 3))),
                        camera_path,
                        token,
                    )
                if semantic_renderer is not None:
                    semantic_renderer.add_frame(
                        prediction.dense_labels_xyz,
                        gt,
                        camera_path,
                        token,
                    )
                rendered += 1

            background_count = sum(
                int(label) == 1 for label in prediction.semantic_labels
            )
            foreground_count = sum(
                int(label) == 2 for label in prediction.semantic_labels
            )
            if pose_reader is not None:
                velocity = prediction.metadata["ego_velocity_lidar_mps"]
                print(
                    f"[{index}/{len(infos)}] mode=rpc-temporal "
                    f"scene={scene_token} token={token} input={radar_path.name} "
                    f"pose={prediction.metadata['pose_index']} "
                    f"raw={prediction.metadata['raw_detection_count']} "
                    f"reliable={prediction.metadata['reliable_detection_count']} "
                    f"temporal={prediction.metadata.get('temporal_accepted_detection_count', prediction.metadata['accepted_detection_count'])} "
                    f"persistent/motion/unknown="
                    f"{prediction.metadata.get('persistent_current_count', 0)}/"
                    f"{prediction.metadata.get('motion_confirmed_count', 0)}/"
                    f"{prediction.metadata.get('unknown_count', 0)} "
                    f"accepted={prediction.metadata['accepted_detection_count']} "
                    f"history_bg={prediction.metadata['historic_background_count']} "
                    f"history_fg={prediction.metadata.get('historic_foreground_count', 0)} "
                    f"objects={prediction.metadata.get('object_classifier', {}).get('candidate_count', 0)} "
                    f"fallback_bg={prediction.metadata.get('object_classifier', {}).get('motion_background_fallback_point_count', 0)} "
                    f"velocity_s/u/d="
                    f"{prediction.metadata['doppler_static_evidence_count']}/"
                    f"{prediction.metadata['doppler_uncertain_evidence_count']}/"
                    f"{prediction.metadata['doppler_dynamic_evidence_count']} "
                    f"residual_med={prediction.metadata['doppler_residual_median_mps']:.2f}m/s "
                    f"vx={velocity[0]:.2f}m/s vy={velocity[1]:.2f}m/s "
                    f"yaw_rate={math.degrees(prediction.metadata['yaw_rate_rps']):.2f}deg/s "
                    f"background={background_count} foreground={foreground_count}"
                )
            else:
                print(
                    f"[{index}/{len(infos)}] mode={input_mode} "
                    f"scene={scene_token} token={token} "
                    f"input={radar_path.name} detections={len(prediction.detections)} "
                    f"ego_speed={prediction.metadata['ego_speed_mps']:.2f}m/s "
                    f"background={background_count} foreground={foreground_count}"
                )

        if processed_frames == 0:
            raise RuntimeError(
                "No frames were processed. Check the RPC root and frame alignment."
            )

        skipped_rpc_path = None
        if skipped_rpc_frames:
            output_dir.mkdir(parents=True, exist_ok=True)
            skipped_rpc_path = output_dir / "skipped_rpc_frames.json"
            skipped_rpc_path.write_text(
                json.dumps(skipped_rpc_frames, indent=2),
                encoding="utf-8",
            )
        print(
            "Frame totals: "
            f"selected={len(infos)} processed={processed_frames} "
            f"skipped_missing_rpc={len(skipped_rpc_frames)}"
        )

        outputs = MetricsReportWriter().write(
            accumulator.results(),
            output_dir,
        )
        if skipped_rpc_path is not None:
            outputs["skipped_rpc_frames"] = skipped_rpc_path
        if branch_renderer is not None:
            branch_video = branch_renderer.finish()
            if branch_video is not None:
                outputs["branches_mp4"], outputs["branches_gif"] = branch_video
        if semantic_renderer is not None:
            semantic_video = semantic_renderer.finish()
            if semantic_video is not None:
                outputs["overlay_mp4"], outputs["overlay_gif"] = semantic_video
        return outputs
