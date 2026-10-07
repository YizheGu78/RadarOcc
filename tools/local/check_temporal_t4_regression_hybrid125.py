#!/usr/bin/env python3
"""Run the T=4 current-only metadata regression checker for Hybrid125."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.local import check_temporal_t4_regression as base


base.OLD_NAMES = {
    "train": "kradar_dict_train_official_hybrid125.pkl",
    "val": "kradar_dict_val_official_hybrid125.pkl",
    "test": "kradar_dict_test_official_hybrid125.pkl",
}

base.NEW_NAMES = {
    "train": "kradar_dict_train_official_temporal_hybrid125.pkl",
    "val": "kradar_dict_val_official_temporal_hybrid125.pkl",
    "test": "kradar_dict_test_official_temporal_hybrid125.pkl",
}


if __name__ == "__main__":
    base.main()
