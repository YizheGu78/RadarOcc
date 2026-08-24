from __future__ import annotations

import argparse
from pathlib import Path

from tradition.cli.arguments import parse_ego_speed
from tradition.experiment.dataset_runner import TraditionalDatasetRunner
from tradition.pipeline.traditional_radar_pipeline import (
    build_raw_pipeline,
    build_rpc_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a non-learning radar occupancy baseline directly to "
            "RadarOcc-style metrics and optional video."
        )
    )
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--input-mode",
        choices=("rpc", "raw"),
        default="rpc",
        help=(
            "rpc: use Enhanced K-Radar pc01p [N,11] point clouds directly; "
            "raw: use full 4DRT and run CFAR. Default: rpc."
        ),
    )
    parser.add_argument(
        "--radar-root",
        type=Path,
        help=(
            "Input root: data/K-Radar_rpc for rpc, or the K-Radar root for raw."
        ),
    )
    parser.add_argument("--gt-root", type=Path)
    parser.add_argument("--gt-order", choices=("xyz", "zyx"), default="xyz")
    parser.add_argument("--scene", help="Evaluate only one scene/sequence.")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--cfar-backend", choices=("numpy", "openradar"), default="numpy"
    )
    parser.add_argument(
        "--ego-speed-mps",
        type=parse_ego_speed,
        default=None,
        metavar="MPS|auto",
        help=(
            "Fixed synchronized ego speed, or 'auto' to estimate it from "
            "each frame's wrapped Doppler consensus. Default: auto."
        ),
    )

    parser.add_argument(
        "--video-scene",
        help="Also render this scene using the existing RadarOcc overlay style.",
    )
    parser.add_argument("--camera-dir", type=Path)
    parser.add_argument("--camera-offset", type=int, default=0)
    parser.add_argument("--max-video-frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Keep the occupancy panel in its original, unrotated orientation.",
    )
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_mode == "rpc":
        pipeline = build_rpc_pipeline()
    else:
        pipeline = build_raw_pipeline(cfar_backend=args.cfar_backend)

    outputs = TraditionalDatasetRunner(pipeline).run(
        annotation=args.annotation,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        radar_root=args.radar_root,
        gt_root=args.gt_root,
        gt_order=args.gt_order,
        scene=args.scene,
        ego_speed_mps=args.ego_speed_mps,
        max_frames=args.max_frames,
        video_scene=args.video_scene,
        camera_dir=args.camera_dir,
        camera_offset=args.camera_offset,
        max_video_frames=args.max_video_frames,
        fps=args.fps,
        no_rotate=args.no_rotate,
        keep_frames=args.keep_frames,
        input_mode=args.input_mode,
    )
    print("\nOutputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
