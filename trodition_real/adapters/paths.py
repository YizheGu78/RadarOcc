"""K-Radar scene-order IO snapshot; local adaptation, not Autoware."""

from __future__ import annotations

import pickle, re

from pathlib import Path

from functools import lru_cache

from typing import Any

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

_RAW_RADAR_EXTENSIONS = {".mat", ".npy"}

_RPC_RADAR_DIRS = ("", "rpc", "pc01p", "radar_pc", "radar_point_cloud")

class RPCFrameNotFoundError(FileNotFoundError):
    """A scene-order annotation entry has no corresponding RPC file."""

    def __init__(
        self,
        message: str,
        *,
        scene: str,
        annotation_ordinal: int,
        annotation_count: int | None,
        rpc_count: int,
    ) -> None:
        super().__init__(message)
        self.scene = str(scene)
        self.annotation_ordinal = int(annotation_ordinal)
        self.annotation_count = (
            None if annotation_count is None else int(annotation_count)
        )
        self.rpc_count = int(rpc_count)

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
            annotation_ordinal=ordinal,
            annotation_count=annotation_count,
            rpc_count=len(rpc_files),
        )

    if ordinal < 0 or ordinal >= len(rpc_files):
        raise RPCFrameNotFoundError(
            "Cannot resolve RPC by scene-local order. "
            f"Scene {scene}: annotation ordinal={ordinal}, "
            f"RPC files={len(rpc_files)}. "
            "Cross-sensor frame IDs are intentionally ignored.",
            scene=scene,
            annotation_ordinal=ordinal,
            annotation_count=annotation_count,
            rpc_count=len(rpc_files),
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
