#!/usr/bin/env python3
"""
Build canonical frame-level metadata PKLs for DG-STF temporal RadarOcc.

Design decisions:
- Keep the existing aligned RadarOcc PKL as the source of truth.
- Do NOT pack T=2/4/8 sequences into the PKL.
- Add only frame-level temporal metadata; TemporalDataset will construct
  sequence indices dynamically at runtime.
- Preserve every original field unchanged so frame_nums=1 can regress against
  the current single-frame RadarOcc pipeline.
- Ego pose is required by default.
- Real os2-64 timestamp is optional metadata and is NOT a core model input.

Expected local inputs:
  data/annotations/kradar_dict_{train,val,test}_official_doppler8.pkl
  data/K-RadarOcc/train/<scene>/pose/lidar_ego_pose<occ_idx>.npy

Optional time info:
  data/K-Radar-timeinfo/<scene>/**/os2-64.txt

Typical usage:
  python tools/local/build_kradar_temporal_pkls.py

Require real timestamp for every frame:
  python tools/local/build_kradar_temporal_pkls.py --require-timestamp
"""

from __future__ import annotations

import argparse
import copy
import pickle
import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ANN_DIR = ROOT / "data" / "annotations"
DEFAULT_POSE_ROOT = ROOT / "data" / "K-RadarOcc" / "train"
DEFAULT_TIME_ROOT = ROOT / "data" / "K-Radar-timeinfo"

INPUT_NAMES = {
    "train": "kradar_dict_train_official_doppler8.pkl",
    "val": "kradar_dict_val_official_doppler8.pkl",
    "test": "kradar_dict_test_official_doppler8.pkl",
}

OUTPUT_NAMES = {
    "train": "kradar_dict_train_official_temporal_doppler8.pkl",
    "val": "kradar_dict_val_official_temporal_doppler8.pkl",
    "test": "kradar_dict_test_official_temporal_doppler8.pkl",
}

DEBUG_NAME = "kradar_dict_debug_official_temporal_doppler8.pkl"


def _last_int(text: str) -> int:
    numbers = re.findall(r"\d+", str(text))
    if not numbers:
        raise ValueError(f"No integer found in: {text!r}")
    return int(numbers[-1])


def _scene_id(info: dict) -> int:
    if "scene_id" in info:
        return int(info["scene_id"])
    if "scene_token" in info:
        return int(info["scene_token"])
    raise KeyError("info has neither 'scene_id' nor 'scene_token'")


def _occ_frame_id(info: dict) -> int:
    for key in ("occ_frame_idx", "frame_idx"):
        if key in info:
            return int(info[key])

    if "occ_path" in info:
        return _last_int(Path(info["occ_path"]).stem)

    raise KeyError(
        "Cannot determine occupancy/frame id. Expected one of "
        "'occ_frame_idx', 'frame_idx', or 'occ_path'."
    )


def _radar_frame_id(info: dict) -> int:
    if "radar_frame_idx" in info:
        return int(info["radar_frame_idx"])

    for key in ("sparse_radar_path", "radar_path"):
        if key in info and info[key]:
            return _last_int(Path(info[key]).stem)

    raise KeyError(
        "Cannot determine radar frame id. Expected 'radar_frame_idx', "
        "'sparse_radar_path', or 'radar_path'."
    )


def _load_pose(pose_root: Path, scene_id: int, occ_frame_id: int) -> Tuple[np.ndarray, Path]:
    pose_path = (
        pose_root
        / str(scene_id)
        / "pose"
        / f"lidar_ego_pose{occ_frame_id}.npy"
    )

    if not pose_path.exists():
        raise FileNotFoundError(f"Missing ego pose: {pose_path}")

    pose = np.load(pose_path, allow_pickle=True)
    pose = np.asarray(pose, dtype=np.float64).squeeze()

    if pose.shape != (4, 4):
        raise ValueError(
            f"Pose must be 4x4, got {pose.shape} at {pose_path}"
        )

    if not np.isfinite(pose).all():
        raise ValueError(f"Pose contains NaN/Inf: {pose_path}")

    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(
            f"Unexpected homogeneous last row at {pose_path}: {pose[3]}"
        )

    rotation = pose[:3, :3]
    should_be_identity = rotation.T @ rotation
    if not np.allclose(should_be_identity, np.eye(3), atol=5e-3):
        raise ValueError(
            f"Pose rotation is not approximately orthonormal: {pose_path}"
        )

    return pose, pose_path


def _find_time_file(time_root: Path, scene_id: int) -> Optional[Path]:
    scene_dir = time_root / str(scene_id)
    if not scene_dir.exists():
        return None

    direct_candidates = [
        scene_dir / "os2-64.txt",
        scene_dir / "time_info" / "os2-64.txt",
    ]

    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    candidates = sorted(scene_dir.rglob("os2-64.txt"))
    if not candidates:
        return None

    if len(candidates) > 1:
        print(
            f"[WARNING] scene {scene_id}: multiple os2-64.txt files found; "
            f"using {candidates[0]}"
        )

    return candidates[0]


def _read_time_records(path: Path) -> List[Tuple[int, str, float]]:
    """
    Return sorted (source_frame_id, source_filename, timestamp_sec).

    K-Radar time_info lines look like:
      os2-64_00001.pcd, 1643292946.710046076

    Occupancy/pose frame_idx is a zero-based processed index. K-Radar
    preprocessing sorted the original os2-64 files then renamed them to
    pc0.npy, pc1.npy, ... . Therefore temporal metadata maps occ_frame_id=i to
    the i-th sorted os2-64 time record, not by equating numeric IDs.
    """
    records: List[Tuple[int, str, float]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                if "," in line:
                    filename, timestamp_text = line.rsplit(",", 1)
                    filename = filename.strip()
                    timestamp_text = timestamp_text.strip()
                else:
                    parts = line.split()
                    if len(parts) < 2:
                        raise ValueError("expected filename and timestamp")
                    filename = parts[0]
                    timestamp_text = parts[-1]

                source_frame_id = _last_int(Path(filename).stem)
                timestamp_sec = float(Decimal(timestamp_text))
                records.append(
                    (source_frame_id, filename, timestamp_sec)
                )
            except Exception as exc:
                raise ValueError(
                    f"Invalid time_info line {path}:{line_number}: {line}"
                ) from exc

    records.sort(key=lambda item: item[0])

    if not records:
        raise ValueError(f"No valid timestamps found in {path}")

    return records


def _load_annotation(path: Path) -> Tuple[dict, List[dict]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input annotation: {path}")

    with path.open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "infos" in data:
        infos = data["infos"]
        if not isinstance(infos, list):
            raise TypeError(f"{path}: data['infos'] is not a list")
        return data, infos

    if isinstance(data, list):
        # Compatibility fallback. Output will still use the standard
        # {'infos': ..., 'metadata': ...} structure.
        return {"infos": data, "metadata": {}}, data

    raise TypeError(
        f"Unsupported PKL structure in {path}; expected dict with 'infos' or list"
    )


def _group_and_sort(infos: Sequence[dict]) -> Dict[int, List[Tuple[int, dict]]]:
    grouped: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)

    for original_index, info in enumerate(infos):
        grouped[_scene_id(info)].append((original_index, info))

    for scene_id in grouped:
        grouped[scene_id].sort(
            key=lambda pair: (
                _occ_frame_id(pair[1]),
                _radar_frame_id(pair[1]),
                pair[0],
            )
        )

    return grouped


def _augment_infos(
    infos: Sequence[dict],
    pose_root: Path,
    time_root: Path,
    require_pose: bool,
    require_timestamp: bool,
) -> Tuple[List[dict], dict]:
    grouped = _group_and_sort(infos)

    augmented_by_original_index: Dict[int, dict] = {}

    num_pose = 0
    num_timestamp = 0
    scene_summaries = {}

    for scene_id in sorted(grouped):
        scene_entries = grouped[scene_id]

        time_file = _find_time_file(time_root, scene_id)
        time_records = None

        if time_file is not None:
            time_records = _read_time_records(time_file)
        elif require_timestamp:
            raise FileNotFoundError(
                f"scene {scene_id}: os2-64.txt not found below {time_root}"
            )

        seen_occ = set()
        seen_radar = set()

        first_occ = None
        last_occ = None

        scene_pose_count = 0
        scene_timestamp_count = 0

        for order_in_scene, (original_index, original_info) in enumerate(scene_entries):
            info = copy.deepcopy(original_info)

            occ_frame_id = _occ_frame_id(info)
            radar_frame_id = _radar_frame_id(info)

            if occ_frame_id in seen_occ:
                raise ValueError(
                    f"scene {scene_id}: duplicate occ_frame_id={occ_frame_id}"
                )
            if radar_frame_id in seen_radar:
                raise ValueError(
                    f"scene {scene_id}: duplicate radar_frame_id={radar_frame_id}"
                )

            seen_occ.add(occ_frame_id)
            seen_radar.add(radar_frame_id)

            first_occ = occ_frame_id if first_occ is None else first_occ
            last_occ = occ_frame_id

            # Canonical temporal identifiers.
            info["scene_id"] = int(scene_id)
            info["frame_id"] = int(occ_frame_id)
            info["order_in_scene"] = int(order_in_scene)
            info["occ_frame_id"] = int(occ_frame_id)
            info["radar_frame_id"] = int(radar_frame_id)

            # Explicit aliases make the temporal contract self-describing while
            # preserving all original RadarOcc fields.
            info["gt_path"] = info.get("occ_path", "")
            if not info.get("radar_path"):
                info["radar_path"] = info.get("sparse_radar_path", "")

            # Pose: required for the main DG-STF ego alignment path.
            try:
                ego_pose, _ = _load_pose(
                    pose_root=pose_root,
                    scene_id=scene_id,
                    occ_frame_id=occ_frame_id,
                )
                info["ego_pose"] = ego_pose.astype(np.float64)
                info["ego_pose_valid"] = True
                scene_pose_count += 1
                num_pose += 1
            except FileNotFoundError:
                if require_pose:
                    raise
                info["ego_pose"] = np.eye(4, dtype=np.float64)
                info["ego_pose_valid"] = False

            # Optional real timestamp. Do not overwrite legacy info['timestamp'];
            # keeping it unchanged protects frame_nums=1 regression behavior.
            info["timestamp_sec"] = None
            info["timestamp_valid"] = False
            info["os2_source_frame_id"] = None
            info["os2_source_filename"] = None

            if time_records is not None:
                # occ_frame_id is the zero-based processed LiDAR index.
                if 0 <= occ_frame_id < len(time_records):
                    source_frame_id, source_filename, timestamp_sec = (
                        time_records[occ_frame_id]
                    )
                    info["timestamp_sec"] = float(timestamp_sec)
                    info["timestamp_valid"] = True
                    info["os2_source_frame_id"] = int(source_frame_id)
                    info["os2_source_filename"] = source_filename
                    scene_timestamp_count += 1
                    num_timestamp += 1
                elif require_timestamp:
                    raise IndexError(
                        f"scene {scene_id}: occ_frame_id={occ_frame_id} but "
                        f"{time_file} has only {len(time_records)} records"
                    )

            # Convenience aliases for provenance/debug.
            if "frame_difference" in info:
                info["radar_lidar_frame_difference"] = int(
                    info["frame_difference"]
                )

            augmented_by_original_index[original_index] = info

        scene_summaries[scene_id] = {
            "samples": len(scene_entries),
            "first_occ_frame_id": first_occ,
            "last_occ_frame_id": last_occ,
            "pose_count": scene_pose_count,
            "timestamp_count": scene_timestamp_count,
            "time_file": str(time_file) if time_file is not None else None,
        }

    # Keep exactly the original split/sample ordering in the dumped PKL.
    augmented_infos = [
        augmented_by_original_index[i]
        for i in range(len(infos))
    ]

    summary = {
        "samples": len(augmented_infos),
        "pose_count": num_pose,
        "timestamp_count": num_timestamp,
        "scene_count": len(scene_summaries),
        "scenes": scene_summaries,
    }

    return augmented_infos, summary


def _dump_output(
    source_data: dict,
    infos: List[dict],
    out_path: Path,
    split: str,
    source_path: Path,
    pose_root: Path,
    time_root: Path,
) -> None:
    metadata = copy.deepcopy(source_data.get("metadata", {}))

    metadata.update(
        {
            "temporal_metadata_version": "dgstf-frame-level-v1",
            "temporal_split": split,
            "temporal_source_pkl": str(source_path),
            "temporal_sequence_policy": (
                "dynamic_in_dataset; do_not_prepack_T2_T4_T8"
            ),
            "temporal_order_key": "order_in_scene",
            "temporal_frame_gap_policy": (
                "relative_frame_gap; timestamp_sec is optional metadata"
            ),
            "ego_pose_field": "ego_pose",
            "ego_pose_convention": (
                "matrix copied from K-RadarOcc pose/lidar_ego_pose*.npy; "
                "treated as T_world_from_lidar by the original occupancy "
                "preprocessing"
            ),
            "pose_root": str(pose_root),
            "time_root": str(time_root),
            "legacy_timestamp_preserved": True,
        }
    )

    data_out = copy.deepcopy(source_data)
    data_out["infos"] = infos
    data_out["metadata"] = metadata

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as f:
        pickle.dump(
            data_out,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _print_summary(split: str, out_path: Path, summary: dict) -> None:
    print("\n" + "=" * 100)
    print(f"{split.upper()} -> {out_path}")
    print(
        f"samples={summary['samples']} | "
        f"scenes={summary['scene_count']} | "
        f"poses={summary['pose_count']} | "
        f"real_timestamps={summary['timestamp_count']}"
    )

    for scene_id, scene in summary["scenes"].items():
        print(
            f"  scene {scene_id:>3}: "
            f"samples={scene['samples']:>5} | "
            f"occ={scene['first_occ_frame_id']}..{scene['last_occ_frame_id']} | "
            f"pose={scene['pose_count']:>5} | "
            f"time={scene['timestamp_count']:>5}"
        )

    print("=" * 100)


def _validate_temporal_fields(infos: Iterable[dict]) -> None:
    required = (
        "scene_id",
        "frame_id",
        "order_in_scene",
        "occ_frame_id",
        "radar_frame_id",
        "radar_path",
        "gt_path",
        "ego_pose",
        "ego_pose_valid",
        "timestamp_sec",
        "timestamp_valid",
    )

    for i, info in enumerate(infos):
        missing = [key for key in required if key not in info]
        if missing:
            raise KeyError(
                f"output info[{i}] missing temporal fields: {missing}"
            )

        pose = np.asarray(info["ego_pose"])
        if pose.shape != (4, 4):
            raise ValueError(
                f"output info[{i}] ego_pose has shape {pose.shape}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Augment aligned K-Radar RadarOcc PKLs with canonical "
            "frame-level temporal metadata."
        )
    )

    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=DEFAULT_ANN_DIR,
        help="Directory containing the existing aligned official Doppler8 PKLs.",
    )
    parser.add_argument(
        "--pose-root",
        type=Path,
        default=DEFAULT_POSE_ROOT,
        help=(
            "Root containing <scene>/pose/lidar_ego_pose<idx>.npy. "
            "Default: data/K-RadarOcc/train"
        ),
    )
    parser.add_argument(
        "--time-root",
        type=Path,
        default=DEFAULT_TIME_ROOT,
        help=(
            "Optional root containing <scene>/**/os2-64.txt. "
            "Default: data/K-Radar-timeinfo"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ANN_DIR,
        help="Output directory for temporal frame-level PKLs.",
    )
    parser.add_argument(
        "--allow-missing-pose",
        action="store_true",
        help=(
            "Allow missing poses and write identity pose with "
            "ego_pose_valid=False. Not recommended for DG-STF."
        ),
    )
    parser.add_argument(
        "--require-timestamp",
        action="store_true",
        help=(
            "Fail if a scene/frame has no real os2-64 timestamp. "
            "By default timestamp is optional metadata."
        ),
    )
    parser.add_argument(
        "--no-debug-pkl",
        action="store_true",
        help="Do not create a 32-sample train debug PKL.",
    )

    args = parser.parse_args()

    pose_root = args.pose_root.resolve()
    time_root = args.time_root.resolve()
    ann_dir = args.ann_dir.resolve()
    output_dir = args.output_dir.resolve()

    all_outputs = {}

    for split in ("train", "val", "test"):
        source_path = ann_dir / INPUT_NAMES[split]
        out_path = output_dir / OUTPUT_NAMES[split]

        source_data, infos = _load_annotation(source_path)

        augmented_infos, summary = _augment_infos(
            infos=infos,
            pose_root=pose_root,
            time_root=time_root,
            require_pose=not args.allow_missing_pose,
            require_timestamp=args.require_timestamp,
        )

        _validate_temporal_fields(augmented_infos)

        _dump_output(
            source_data=source_data,
            infos=augmented_infos,
            out_path=out_path,
            split=split,
            source_path=source_path,
            pose_root=pose_root,
            time_root=time_root,
        )

        _print_summary(split, out_path, summary)
        all_outputs[split] = (source_data, augmented_infos)

    if not args.no_debug_pkl:
        train_source, train_infos = all_outputs["train"]
        debug_infos = train_infos[:32]
        debug_path = output_dir / DEBUG_NAME

        _dump_output(
            source_data=train_source,
            infos=debug_infos,
            out_path=debug_path,
            split="debug",
            source_path=ann_dir / INPUT_NAMES["train"],
            pose_root=pose_root,
            time_root=time_root,
        )

        print(f"Saved debug PKL: {debug_path} ({len(debug_infos)} samples)")

    print("\nTemporal frame-level metadata generation completed.")
    print(
        "Next step: TemporalDataset should build T=1/2/4/8 sequence_indices "
        "dynamically from scene_id + order_in_scene."
    )


if __name__ == "__main__":
    main()
