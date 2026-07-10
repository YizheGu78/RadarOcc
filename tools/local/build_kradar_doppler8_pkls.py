import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RADAR_ROOT = ROOT / "data/RadarOcc_8doppler"
OCC_ROOT = ROOT / "data/K-RadarOcc/train"
OUT_DIR = ROOT / "data/annotations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 官方 README 给出的 sequence 列表，先按这个做复现测试
TRAIN_SEQS = [4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 27, 56]
VAL_SEQS   = [3, 15, 22, 23, 55]
TEST_SEQS  = [1, 2, 21, 46, 54]

def idx_from_name(path: Path) -> int:
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        raise ValueError(f"No index in filename: {path}")
    return int(nums[-1])

def empty_annos():
    return {
        "gt_boxes": np.zeros((0, 7), dtype=np.float32),
        "gt_names": np.array([], dtype="<U32"),
        "num_lidar_pts": np.zeros((0,), dtype=np.int64),
        "num_radar_pts": np.zeros((0,), dtype=np.int64),
        "valid_flag": np.zeros((0,), dtype=bool),
        "gt_velocity": np.zeros((0, 2), dtype=np.float32),
    }

def build_infos(seqs):
    infos = []

    for seq in seqs:
        seq = str(seq)
        radar_dir = RADAR_ROOT / seq / "radar_tensor_8doppler"
        occ_dir = OCC_ROOT / seq / "semantic_occupancy_gt_fov"

        if not radar_dir.exists():
            print(f"[WARN] missing radar dir: {radar_dir}")
            continue

        if not occ_dir.exists():
            print(f"[WARN] missing occ dir: {occ_dir}")
            continue

        radar_files = sorted(radar_dir.glob("EAsparse_*.npz"), key=idx_from_name)
        occ_files = sorted(occ_dir.glob("occupancy_gt_with_semantic_fov*.npy"), key=idx_from_name)

        radar_map = {idx_from_name(p): p for p in radar_files}
        occ_map = {idx_from_name(p): p for p in occ_files}
        common = sorted(set(radar_map.keys()) & set(occ_map.keys()))

        print(f"seq {seq}: radar={len(radar_files)}, occ={len(occ_files)}, matched={len(common)}")

        for i in common:
            radar_path = str(radar_map[i])
            occ_path = str(occ_map[i])

            info = {
                "token": f"{seq}_{i:05d}",
                "sample_idx": f"{seq}_{i:05d}",
                "lidar_path": "",
                "sweeps": [],
                "cams": {},
                "lidar2ego_translation": np.zeros(3, dtype=np.float32),
                "lidar2ego_rotation": np.array([1, 0, 0, 0], dtype=np.float32),
                "ego2global_translation": np.zeros(3, dtype=np.float32),
                "ego2global_rotation": np.array([1, 0, 0, 0], dtype=np.float32),
                "prev": "",
                "next": "",
                "scene_token": seq,
                "radar_path": radar_path,
                "sparse_radar_path": radar_path,
                "occ_path": occ_path,
                "timestamp": int(seq) * 100000000 + int(i),
                "lidar_token": f"{seq}_{i:05d}",
                "lidarseg": "",
                "frame_idx": int(i),
            }

            info.update(empty_annos())
            infos.append(info)

    infos = sorted(infos, key=lambda x: (int(x["scene_token"]), x["frame_idx"]))
    return infos

def dump(name, infos):
    out = OUT_DIR / name
    obj = {
        "infos": infos,
        "metadata": {
            "version": "v1.0-trainval"
        }
    }
    with open(out, "wb") as f:
        pickle.dump(obj, f)

    print(f"\nSaved {out}: {len(infos)} infos\n")

train_infos = build_infos(TRAIN_SEQS)
val_infos = build_infos(VAL_SEQS)
test_infos = build_infos(TEST_SEQS)

dump("kradar_dict_train_doppler8.pkl", train_infos)
dump("kradar_dict_val_doppler8.pkl", val_infos)
dump("kradar_dict_test_doppler8.pkl", test_infos)
dump("kradar_dict_debug_doppler8.pkl", train_infos[:32])
