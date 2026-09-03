from __future__ import annotations

import argparse
from pathlib import Path

from tradition.cli.arguments import parse_ego_speed
from tradition.core.config import (
    MotionConfig,
    PoseConfig,
    ReliabilityConfig,
    TemporalConfig,
)
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
        "--pose-root",
        type=Path,
        help=(
            "Root containing RadarOcc lidar_ego_pose*.npy files. When set, "
            "RPC processing uses pose-derived ego motion and causal temporal "
            "consistency instead of scalar/automatic single-frame speed."
        ),
    )
    parser.add_argument(
        "--pose-dt-s",
        type=float,
        default=0.10,
        help="Time interval between consecutive LiDAR poses. Default: 0.10 s.",
    )
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
        "--static-residual-threshold-mps",
        type=float,
        default=0.50,
        help=(
            "Maximum absolute ego-compensated Doppler residual classified "
            "as static evidence. Default: 0.50 m/s."
        ),
    )
    parser.add_argument(
        "--dynamic-residual-threshold-mps",
        type=float,
        default=0.80,
        help=(
            "Minimum absolute wrapped Doppler residual treated as dynamic "
            "evidence. Values between static/dynamic thresholds are discarded. "
            "Default: 0.80 m/s."
        ),
    )
    parser.add_argument(
        "--stationary-velocity-sign",
        choices=(-1.0, 1.0),
        type=float,
        default=1.0,
        help=(
            "Doppler sign multiplying the static ego-velocity projection. "
            "The supplied K-Radar RPC uses +1. Default: +1."
        ),
    )
    parser.add_argument("--temporal-window", type=int, default=3)
    parser.add_argument("--min-static-support", type=int, default=2)
    parser.add_argument("--min-dynamic-support", type=int, default=2)
    parser.add_argument("--static-match-radius-m", type=float, default=0.60)
    parser.add_argument("--dynamic-match-radius-m", type=float, default=2.00)
    parser.add_argument("--min-local-power-ratio", type=float, default=0.25)
    parser.add_argument("--min-local-neighbors", type=int, default=1)

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
    motion_cfg = MotionConfig(
        static_residual_threshold_mps=args.static_residual_threshold_mps,
        dynamic_residual_threshold_mps=args.dynamic_residual_threshold_mps,
        stationary_velocity_sign=args.stationary_velocity_sign,
    )
    pose_cfg = PoseConfig(frame_dt_s=args.pose_dt_s)
    reliability_cfg = ReliabilityConfig(
        min_local_power_ratio=args.min_local_power_ratio,
        min_local_neighbors=args.min_local_neighbors,
    )
    temporal_cfg = TemporalConfig(
        window_size=args.temporal_window,
        min_static_support=args.min_static_support,
        min_dynamic_support=args.min_dynamic_support,
        static_match_radius_m=args.static_match_radius_m,
        dynamic_match_radius_m=args.dynamic_match_radius_m,
    )

    if args.input_mode == "rpc":
        pipeline = build_rpc_pipeline(
            motion_cfg=motion_cfg,
            reliability_cfg=reliability_cfg,
            temporal_cfg=temporal_cfg,
        )
    else:
        if args.pose_root is not None:
            raise ValueError("--pose-root is currently supported only in RPC mode.")
        pipeline = build_raw_pipeline(
            cfar_backend=args.cfar_backend,
            motion_cfg=motion_cfg,
        )

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
        pose_root=args.pose_root,
        pose_dt_s=pose_cfg.frame_dt_s,
    )
    print("\nOutputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
