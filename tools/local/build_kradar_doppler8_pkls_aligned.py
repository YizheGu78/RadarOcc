import pickle
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

RADAR_ROOT = ROOT / "data" / "RadarOcc_8doppler"
OCC_ROOT = ROOT / "data" / "K-RadarOcc" / "train"
CALIB_ROOT = ROOT / "data" / "K-Radar_calib"
OUT_DIR = ROOT / "data" / "annotations"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# RadarOcc论文/官方notebook使用的sequence划分
SPLITS = {
    "train": [
        4, 6, 7, 8, 9, 10, 11, 12, 13,
        14, 16, 17, 18, 19, 20, 27, 56,
    ],
    "val": [1, 2],
    "test": [3, 15, 22, 23, 55],
}

OUTPUT_NAMES = {
    "train": "kradar_dict_train_official_doppler8.pkl",
    "val": "kradar_dict_val_official_doppler8.pkl",
    "test": "kradar_dict_test_official_doppler8.pkl",
}


def idx_from_name(path: Path) -> int:
    numbers = re.findall(r"\d+", path.stem)

    if not numbers:
        raise ValueError(f"文件名中没有数字编号：{path}")

    return int(numbers[-1])


def read_calibration(seq: int):
    calib_path = (
        CALIB_ROOT
        / str(seq)
        / "info_calib"
        / "calib_radar_lidar.txt"
    )

    if not calib_path.exists():
        raise FileNotFoundError(
            f"Sequence {seq} 缺少 calibration：{calib_path}"
        )

    lines = [
        line.strip()
        for line in calib_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        raise ValueError(
            f"Calibration格式错误：{calib_path}"
        )

    values = [
        value.strip()
        for value in lines[1].split(",")
    ]

    if len(values) < 3:
        raise ValueError(
            f"Calibration第二行不足3列：{calib_path}"
        )

    frame_difference = int(float(values[0]))
    calib_x = float(values[1])
    calib_y = float(values[2])

    return frame_difference, calib_x, calib_y


def empty_annos():
    return {
        "gt_boxes": np.zeros((0, 7), dtype=np.float32),
        "gt_names": np.array([], dtype="<U32"),
        "num_lidar_pts": np.zeros((0,), dtype=np.int64),
        "num_radar_pts": np.zeros((0,), dtype=np.int64),
        "valid_flag": np.zeros((0,), dtype=bool),
        "gt_velocity": np.zeros((0, 2), dtype=np.float32),
    }


def build_sequence_infos(seq: int):
    radar_dir = (
        RADAR_ROOT
        / str(seq)
        / "radar_tensor_8doppler"
    )

    occ_dir = (
        OCC_ROOT
        / str(seq)
        / "semantic_occupancy_gt_fov"
    )

    if not radar_dir.exists():
        raise FileNotFoundError(
            f"缺少Radar目录：{radar_dir}"
        )

    if not occ_dir.exists():
        raise FileNotFoundError(
            f"缺少GT目录：{occ_dir}"
        )

    frame_difference, calib_x, calib_y = (
        read_calibration(seq)
    )

    expected_delta = 1 + frame_difference

    radar_files = sorted(
        radar_dir.glob("EAsparse_*.npz"),
        key=idx_from_name,
    )

    occ_files = sorted(
        occ_dir.glob(
            "occupancy_gt_with_semantic_fov*.npy"
        ),
        key=idx_from_name,
    )

    radar_map = {
        idx_from_name(path): path
        for path in radar_files
    }

    infos = []
    missing_pairs = []

    for occ_path in occ_files:
        occ_idx = idx_from_name(occ_path)

        # 官方convert_kradar.ipynb中的对齐规则
        radar_idx = (
            occ_idx
            + 1
            + frame_difference
        )

        radar_path = radar_map.get(radar_idx)

        if radar_path is None:
            missing_pairs.append(
                (occ_idx, radar_idx)
            )
            continue

        # 最终防止再次出现同编号误配
        assert (
            radar_idx - occ_idx
            == expected_delta
        )

        token = f"{seq}_{occ_idx:05d}"

        info = {
            "token": token,
            "sample_idx": token,

            "lidar_path": "",
            "sweeps": [],
            "cams": {},

            "lidar2ego_translation": np.zeros(
                3,
                dtype=np.float32,
            ),
            "lidar2ego_rotation": np.array(
                [1, 0, 0, 0],
                dtype=np.float32,
            ),
            "ego2global_translation": np.zeros(
                3,
                dtype=np.float32,
            ),
            "ego2global_rotation": np.array(
                [1, 0, 0, 0],
                dtype=np.float32,
            ),

            "prev": "",
            "next": "",
            "scene_token": str(seq),

            "radar_path": str(
                radar_path.resolve()
            ),
            "sparse_radar_path": str(
                radar_path.resolve()
            ),
            "occ_path": str(
                occ_path.resolve()
            ),

            "timestamp": (
                int(seq) * 100000000
                + int(occ_idx)
            ),

            "lidar_token": token,
            "lidarseg": "",

            # frame_idx保持为GT/occupancy帧编号
            "frame_idx": int(occ_idx),

            # 下面字段用于检查，不影响模型读取
            "occ_frame_idx": int(occ_idx),
            "radar_frame_idx": int(radar_idx),
            "frame_difference": int(
                frame_difference
            ),
            "calib_x": float(calib_x),
            "calib_y": float(calib_y),
        }

        info.update(empty_annos())
        infos.append(info)

    print(
        f"seq {seq:>2}: "
        f"fd={frame_difference:>3}, "
        f"delta={expected_delta:>3}, "
        f"radar={len(radar_files):>4}, "
        f"occ={len(occ_files):>4}, "
        f"matched={len(infos):>4}, "
        f"missing={len(missing_pairs):>3}"
    )

    if infos:
        first = infos[0]
        last = infos[-1]

        print(
            "       first: "
            f"occ {first['occ_frame_idx']} "
            f"-> radar {first['radar_frame_idx']}"
        )
        print(
            "       last : "
            f"occ {last['occ_frame_idx']} "
            f"-> radar {last['radar_frame_idx']}"
        )

    # 不允许静默丢弃GT
    if missing_pairs:
        print(
            f"\nSequence {seq} 找不到对应Radar文件："
        )

        for occ_idx, radar_idx in missing_pairs[:20]:
            print(
                f"  GT {occ_idx} "
                f"需要 EAsparse_{radar_idx:05d}.npz"
            )

        raise RuntimeError(
            f"Sequence {seq} 有 "
            f"{len(missing_pairs)} 个未匹配样本，"
            "停止生成PKL。"
        )

    if not infos:
        raise RuntimeError(
            f"Sequence {seq} 没有生成任何样本。"
        )

    return infos


def build_split(split_name: str, sequences):
    infos = []

    print("\n" + "=" * 90)
    print(
        f"Building {split_name}: "
        f"sequences={sequences}"
    )
    print("=" * 90)

    for seq in sequences:
        infos.extend(
            build_sequence_infos(seq)
        )

    infos.sort(
        key=lambda item: (
            int(item["scene_token"]),
            int(item["frame_idx"]),
        )
    )

    return infos


def dump_pkl(split_name: str, infos):
    out_path = OUT_DIR / OUTPUT_NAMES[split_name]

    data = {
        "infos": infos,
        "metadata": {
            "version": "kradar-radarocc-aligned",
            "split": split_name,
            "alignment_rule": (
                "radar_idx = "
                "occ_idx + 1 + frame_difference"
            ),
        },
    }

    with open(out_path, "wb") as file:
        pickle.dump(
            data,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"\nSaved {out_path}: "
        f"{len(infos)} samples\n"
    )


def main():
    all_split_infos = {}

    for split_name, sequences in SPLITS.items():
        all_split_infos[split_name] = (
            build_split(
                split_name,
                sequences,
            )
        )

    for split_name, infos in all_split_infos.items():
        dump_pkl(split_name, infos)

    # 用正式训练集前32帧创建debug文件
    debug_path = (
        OUT_DIR
        / "kradar_dict_debug_official_aligned_doppler8.pkl"
    )

    with open(debug_path, "wb") as file:
        pickle.dump(
            {
                "infos": all_split_infos["train"][:32],
                "metadata": {
                    "version": "kradar-radarocc-aligned-debug",
                },
            },
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"Saved {debug_path}: 32 samples"
    )


if __name__ == "__main__":
    main()
