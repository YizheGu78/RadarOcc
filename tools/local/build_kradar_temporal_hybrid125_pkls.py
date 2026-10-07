#!/usr/bin/env python3
"""Build DG-STF frame-level temporal PKLs for the Hybrid125 radar representation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.local import build_kradar_temporal_pkls as base


base.INPUT_NAMES = {
    "train": "kradar_dict_train_official_hybrid125.pkl",
    "val": "kradar_dict_val_official_hybrid125.pkl",
    "test": "kradar_dict_test_official_hybrid125.pkl",
}

base.OUTPUT_NAMES = {
    "train": "kradar_dict_train_official_temporal_hybrid125.pkl",
    "val": "kradar_dict_val_official_temporal_hybrid125.pkl",
    "test": "kradar_dict_test_official_temporal_hybrid125.pkl",
}

base.DEBUG_NAME = "kradar_dict_debug_official_temporal_hybrid125.pkl"


if __name__ == "__main__":
    base.main()
