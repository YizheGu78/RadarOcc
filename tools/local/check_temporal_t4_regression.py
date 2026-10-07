#!/usr/bin/env python3
"""
Validate DG-STF temporal frame metadata and T=4 current-only regression inputs.

This script does NOT run the neural network. It verifies:
  1) old/new PKLs have the same number of samples;
  2) every original RadarOcc field/value needed for current-frame inference is
     unchanged;
  3) temporal fields are valid;
  4) T=4 sequences never cross scene boundaries;
  5) scene-start valid_mask is correct;
  6) selected sparse radar NPZ and occupancy NPY files can be loaded.

After this passes, run the old and T=4 current-only configs with the SAME
checkpoint and compare the evaluation metrics.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

OLD_NAMES = {
    "train": "kradar_dict_train_official_doppler8.pkl",
    "val": "kradar_dict_val_official_doppler8.pkl",
    "test": "kradar_dict_test_official_doppler8.pkl",
}

NEW_NAMES = {
    "train": "kradar_dict_train_official_temporal_doppler8.pkl",
    "val": "kradar_dict_val_official_temporal_doppler8.pkl",
    "test": "kradar_dict_test_official_temporal_doppler8.pkl",
}


def load_infos(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "infos" in data:
        infos = data["infos"]
    elif isinstance(data, list):
        infos = data
    else:
        raise TypeError(
            f"Unsupported PKL format: {path}"
        )

    if not isinstance(infos, list):
        raise TypeError(
            f"infos is not a list: {path}"
        )

    return infos


def scene_id(info: dict) -> str:
    if "scene_id" in info:
        return str(info["scene_id"])
    return str(info["scene_token"])


def order_in_scene(info: dict) -> int:
    return int(info["order_in_scene"])


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return np.array_equal(
                np.asarray(a),
                np.asarray(b),
                equal_nan=True,
            )
        except TypeError:
            return np.array_equal(
                np.asarray(a),
                np.asarray(b),
            )

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(
            values_equal(a[key], b[key])
            for key in a
        )

    if isinstance(a, (list, tuple)) and isinstance(
        b, (list, tuple)
    ):
        return (
            len(a) == len(b)
            and all(
                values_equal(x, y)
                for x, y in zip(a, b)
            )
        )

    return a == b


def compare_original_fields(
    old_infos: Sequence[dict],
    new_infos: Sequence[dict],
) -> None:
    """
    Temporal PKL generation must preserve all fields from the old aligned PKL.
    Additional temporal fields are allowed.
    """
    for i, (old, new) in enumerate(
        zip(old_infos, new_infos)
    ):
        for key, old_value in old.items():
            if key not in new:
                raise AssertionError(
                    f"sample {i}: old field missing in new PKL: {key}"
                )

            if not values_equal(
                old_value,
                new[key],
            ):
                raise AssertionError(
                    f"sample {i}: original field changed: {key}\n"
                    f"old={old_value}\nnew={new[key]}"
                )


def validate_temporal_fields(
    infos: Sequence[dict],
) -> None:
    required = (
        "scene_id",
        "order_in_scene",
        "ego_pose",
        "ego_pose_valid",
        "radar_frame_id",
        "occ_frame_id",
        "radar_path",
        "gt_path",
    )

    seen = set()

    for i, info in enumerate(infos):
        missing = [
            key for key in required
            if key not in info
        ]
        if missing:
            raise AssertionError(
                f"sample {i}: missing temporal fields {missing}"
            )

        key = (
            scene_id(info),
            order_in_scene(info),
        )
        if key in seen:
            raise AssertionError(
                f"duplicate scene/order: {key}"
            )
        seen.add(key)

        pose = np.asarray(info["ego_pose"])
        if pose.shape != (4, 4):
            raise AssertionError(
                f"sample {i}: ego_pose shape={pose.shape}"
            )

        if not bool(info["ego_pose_valid"]):
            raise AssertionError(
                f"sample {i}: ego_pose_valid=False"
            )


def build_t4_and_validate(
    infos: Sequence[dict],
    frame_nums: int,
) -> None:
    """
    Emulate TemporalKRadarDataset ordering after NuscOCCDataset timestamp sort.
    """
    ordered_indices = sorted(
        range(len(infos)),
        key=lambda idx: infos[idx]["timestamp"],
    )

    scene_to_indices: Dict[str, List[int]] = {}

    for idx in ordered_indices:
        scene_to_indices.setdefault(
            scene_id(infos[idx]),
            [],
        ).append(idx)

    total_sequences = 0

    for scene, indices in scene_to_indices.items():
        indices.sort(
            key=lambda idx: order_in_scene(
                infos[idx]
            )
        )

        orders = [
            order_in_scene(infos[idx])
            for idx in indices
        ]

        if len(set(orders)) != len(orders):
            raise AssertionError(
                f"scene {scene}: duplicate order_in_scene"
            )

        first_idx = indices[0]

        for pos, current_idx in enumerate(indices):
            seq = []
            mask = []

            start = pos - frame_nums + 1

            for relative_pos in range(
                start,
                pos + 1,
            ):
                if relative_pos < 0:
                    seq.append(first_idx)
                    mask.append(False)
                else:
                    seq.append(
                        indices[relative_pos]
                    )
                    mask.append(True)

            if seq[-1] != current_idx:
                raise AssertionError(
                    f"scene {scene}: current frame is not final slot"
                )

            for hist_idx, valid in zip(seq, mask):
                if (
                    valid
                    and scene_id(infos[hist_idx])
                    != scene
                ):
                    raise AssertionError(
                        f"scene {scene}: sequence crossed scene boundary"
                    )

            expected_valid = min(
                pos + 1,
                frame_nums,
            )
            if sum(mask) != expected_valid:
                raise AssertionError(
                    f"scene {scene}, pos={pos}: "
                    f"mask={mask}, expected {expected_valid} valid"
                )

            total_sequences += 1

        # Print beginning examples because these are where bugs are common.
        examples = min(frame_nums, len(indices))
        print(
            f"scene {scene:>3}: samples={len(indices):>5}, "
            f"first orders={orders[:examples]}"
        )

    if total_sequences != len(infos):
        raise AssertionError(
            f"sequence count {total_sequences} != sample count {len(infos)}"
        )


def check_file_loads(
    infos: Sequence[dict],
    num_samples: int,
) -> None:
    if not infos:
        return

    sample_indices = np.linspace(
        0,
        len(infos) - 1,
        num=min(num_samples, len(infos)),
        dtype=int,
    )

    checked = set()

    for idx in sample_indices:
        idx = int(idx)
        if idx in checked:
            continue
        checked.add(idx)

        info = infos[idx]

        radar_path = Path(
            info.get(
                "radar_path",
                info.get("sparse_radar_path", ""),
            )
        )
        gt_path = Path(
            info.get(
                "gt_path",
                info.get("occ_path", ""),
            )
        )

        if not radar_path.exists():
            raise FileNotFoundError(
                f"sample {idx}: radar file missing: {radar_path}"
            )
        if not gt_path.exists():
            raise FileNotFoundError(
                f"sample {idx}: GT file missing: {gt_path}"
            )

        with np.load(radar_path) as radar:
            if not radar.files:
                raise AssertionError(
                    f"sample {idx}: empty radar NPZ: {radar_path}"
                )
            for key in radar.files:
                arr = radar[key]
                if arr.size == 0:
                    raise AssertionError(
                        f"sample {idx}: empty radar array '{key}'"
                    )

        gt = np.load(gt_path, mmap_mode="r")
        if gt.size == 0:
            raise AssertionError(
                f"sample {idx}: empty GT: {gt_path}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
    )
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=ROOT / "data" / "annotations",
    )
    parser.add_argument(
        "--frame-nums",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--file-check-samples",
        type=int,
        default=32,
    )
    args = parser.parse_args()

    old_path = (
        args.ann_dir
        / OLD_NAMES[args.split]
    )
    new_path = (
        args.ann_dir
        / NEW_NAMES[args.split]
    )

    print("=" * 90)
    print("DG-STF T=4 current-only dataset regression check")
    print(f"split : {args.split}")
    print(f"old   : {old_path}")
    print(f"new   : {new_path}")
    print("=" * 90)

    old_infos = load_infos(old_path)
    new_infos = load_infos(new_path)

    if len(old_infos) != len(new_infos):
        raise AssertionError(
            f"sample count mismatch: "
            f"old={len(old_infos)}, new={len(new_infos)}"
        )

    print(f"[OK] sample count: {len(new_infos)}")

    compare_original_fields(
        old_infos,
        new_infos,
    )
    print("[OK] every original PKL field is preserved")

    validate_temporal_fields(new_infos)
    print("[OK] temporal metadata fields")

    build_t4_and_validate(
        new_infos,
        frame_nums=args.frame_nums,
    )
    print(
        f"[OK] T={args.frame_nums} sequence indices / "
        "scene boundaries / valid masks"
    )

    check_file_loads(
        new_infos,
        num_samples=args.file_check_samples,
    )
    print(
        f"[OK] loaded {min(args.file_check_samples, len(new_infos))} "
        "distributed radar/GT samples"
    )

    print("\nPASS: temporal PKL is ready for current-only checkpoint inference.")
    print(
        "Next: evaluate old config and temporal T=4 current-only config "
        "with the SAME checkpoint."
    )


if __name__ == "__main__":
    main()
