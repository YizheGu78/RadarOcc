from __future__ import annotations

import argparse
import re
from pathlib import Path

from tradition.pipeline.traditional_radar_pipeline import build_default_pipeline


def _natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(
        list(path.glob("*.mat")) + list(path.glob("*.npy")), key=_natural_key
    )
    if not files:
        raise FileNotFoundError(f"No .mat/.npy radar tensors under {path}")
    return files


def _default_token(path: Path) -> str:
    matches = re.findall(r"\d+", path.stem)
    return matches[-1] if matches else path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traditional CFAR + Doppler + 3D OGM baseline for RadarOcc."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cfar-backend", choices=("numpy", "openradar"), default="numpy"
    )
    parser.add_argument(
        "--ego-speed-mps",
        type=float,
        default=0.0,
        help="Supply synchronized ego speed for a meaningful static/dynamic split.",
    )
    parser.add_argument("--token-prefix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = build_default_pipeline(cfar_backend=args.cfar_backend)
    files = _input_files(args.input)
    for index, radar_path in enumerate(files, start=1):
        token = args.token_prefix + _default_token(radar_path)
        pred_path = pipeline.predict_and_write(
            radar_path=radar_path,
            output_root=args.output,
            token=token,
            ego_speed_mps=args.ego_speed_mps,
        )
        print(f"[{index}/{len(files)}] {radar_path.name} -> {pred_path}")


if __name__ == "__main__":
    main()
