#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import matplotlib

# Headless rendering: no Qt/Tk/X11 required.
matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from tradition.core.config import (
    GridConfig,
    KRadarConfig,
    MotionConfig,
    PoseConfig,
    ReliabilityConfig,
)
from tradition.detection.rpc_reliability_filter import LocalPowerRPCFilter
from tradition.detection.rpc_target_detector import RPCPointTargetDetector
from tradition.experiment.dataset_runner import (
    _camera_map,
    _load_infos,
    _resolve_rpc_radar,
)
from tradition.io.pose_reader import RadarOccPoseReader
from tradition.io.rpc_radar_reader import KRadarRPCReader
from tradition.motion.doppler_classifier import EgoCompensatedDopplerClassifier
from tradition.motion.pose_ego_motion import PoseEgoMotionEstimator


from tradition.analysis.visualize_static_thresholds import (
    add_velocity_arguments,
    build_occupancy_persistence_classifier,
    build_velocity_unwrapper,
    ordered_scene_infos,
    save_velocity_diagnostics,
    velocity_partition_masks,
    velocity_source_summary,
)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render all selected K-Radar frames as a synchronized video: "
            "BEV tau=0.3 | BEV tau=0.5 | BEV tau=0.7 | RGB."
        )
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("data/annotations/kradar_dict_val_doppler8.pkl"),
    )
    parser.add_argument(
        "--radar-root",
        type=Path,
        default=Path("data/K-Radar_rpc"),
    )
    parser.add_argument(
        "--calib-root",
        type=Path,
        default=Path("data/K-Radar_calib"),
        help="Per-scene frame-difference calibration root.",
    )
    parser.add_argument(
        "--pose-root",
        type=Path,
        default=Path("data/K-RadarOcc"),
    )
    parser.add_argument(
        "--camera-root",
        type=Path,
        default=Path("data/K-Radar-RGB/K-Radar/K-Radar-RGB"),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="3",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=3,
        default=[0.3, 0.5, 0.7],
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start ordinal inside the selected scene.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames. Default: all remaining frames.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Output video FPS. K-Radar is commonly handled at 10 Hz here.",
    )
    parser.add_argument(
        "--camera-offset",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pose-dt-s",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--stationary-velocity-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
    )
    parser.add_argument(
        "--min-local-power-ratio",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--min-local-neighbors",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--all-point-size",
        type=float,
        default=7.0,
    )
    parser.add_argument(
        "--static-point-size",
        type=float,
        default=17.0,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work_dirs/static_threshold_video_scene3"),
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep intermediate PNG frames after MP4 encoding.",
    )
    add_velocity_arguments(parser)
    return parser.parse_args()


def resolve_repo_path(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def resolve_camera_dir(camera_root: Path, scene: str) -> Path:
    root = camera_root.expanduser().resolve()
    candidates = [
        root,
        root / scene,
        root / scene / "images_rb_switched",
        root / "K-Radar" / "K-Radar-RGB" / scene / "images_rb_switched",
    ]

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if any(
            p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
            for p in candidate.iterdir()
        ):
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot resolve RGB directory for scene={scene} from {root}"
    )


def estimate_ego_motion(
    pose_reader: RadarOccPoseReader,
    estimator: PoseEgoMotionEstimator,
    scene: str,
    token: str,
    dt_s: float,
):
    frame_index = pose_reader.frame_index(token)

    current_pose, current_path = pose_reader.read_index(scene, frame_index)

    previous = pose_reader.try_read_index(scene, frame_index - 1)
    following = pose_reader.try_read_index(scene, frame_index + 1)

    previous_pose = previous[0] if previous is not None else None
    next_pose = following[0] if following is not None else None

    motion = estimator.estimate(
        current_pose=current_pose,
        previous_pose=previous_pose,
        next_pose=next_pose,
        previous_dt_s=dt_s if previous_pose is not None else None,
        next_dt_s=(
            dt_s
            if previous_pose is None and next_pose is not None
            else None
        ),
    )

    return motion, current_path


def in_radarocc_roi(xyz: np.ndarray, grid: GridConfig) -> np.ndarray:
    finite = np.isfinite(xyz).all(axis=1)

    return (
        finite
        & (xyz[:, 0] >= grid.min_xyz[0])
        & (xyz[:, 0] < grid.max_xyz[0])
        & (xyz[:, 1] >= grid.min_xyz[1])
        & (xyz[:, 1] < grid.max_xyz[1])
        & (xyz[:, 2] >= grid.min_xyz[2])
        & (xyz[:, 2] < grid.max_xyz[2])
    )


def draw_bev(
    ax,
    xy: np.ndarray,
    residuals: np.ndarray,
    persistent_static_mask: np.ndarray,
    threshold: float,
    max_speed_mps: float,
    grid: GridConfig,
    all_point_size: float,
    static_point_size: float,
) -> tuple[int, int, float, int, int, int, int, int, int]:
    persistent = np.asarray(persistent_static_mask, dtype=bool)
    if persistent.shape != (len(xy),):
        raise ValueError("persistent_static_mask must align with xy")
    (
        persistent_static,
        persistent_dynamic,
        motion_static,
        _motion_dynamic,
        dynamic_mask,
        unknown_mask,
    ) = velocity_partition_masks(
        residuals,
        persistent,
        threshold_mps=threshold,
        max_speed_mps=max_speed_mps,
    )
    static_mask = persistent_static | motion_static

    total = int(len(xy))
    selected = int(np.count_nonzero(static_mask))
    ratio = selected / total if total else 0.0
    persistent_static_count = int(np.count_nonzero(persistent_static))
    persistent_dynamic_count = int(np.count_nonzero(persistent_dynamic))
    persistent_unresolved_count = int(np.count_nonzero(persistent & unknown_mask))
    motion_static_count = int(np.count_nonzero(motion_static))
    dynamic_count = int(np.count_nonzero(dynamic_mask))
    unresolved_count = int(np.count_nonzero(unknown_mask))
    display_x = -xy[:, 1]
    display_y = xy[:, 0]

    if total:
        ax.scatter(
            display_x, display_y, s=all_point_size, c="0.72",
            alpha=0.45, linewidths=0, rasterized=True,
            label="Reliable RPC in ROI",
        )
    if np.any(dynamic_mask):
        ax.scatter(
            display_x[dynamic_mask], display_y[dynamic_mask],
            s=static_point_size, c="tab:red", alpha=0.88,
            linewidths=0, rasterized=True,
            label=r"Resolved dynamic: $\tau_s<|v|<v_{max}$",
        )
    if np.any(unknown_mask):
        ax.scatter(
            display_x[unknown_mask], display_y[unknown_mask],
            s=static_point_size, c="tab:orange", marker="x",
            linewidths=0.7, rasterized=True,
            label="Velocity unresolved (either source)",
        )
    if np.any(motion_static):
        ax.scatter(
            display_x[motion_static], display_y[motion_static],
            s=static_point_size, c="tab:green", alpha=0.95,
            linewidths=0, rasterized=True,
            label=r"Non-persistent velocity-static: $|v|\leq\tau_s$",
        )
    if np.any(persistent_static):
        ax.scatter(
            display_x[persistent_static], display_y[persistent_static],
            s=static_point_size, c="tab:blue", alpha=0.95,
            linewidths=0, rasterized=True,
            label=r"Persistent and velocity-static: $|v|\leq\tau_s$",
        )

    ax.set_xlim(-grid.max_xyz[1], -grid.min_xyz[1])
    ax.set_ylim(grid.min_xyz[0], grid.max_xyz[0])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    ax.set_xlabel("← vehicle left    lateral [m]    vehicle right →")
    ax.set_ylabel("Forward x [m]")
    ax.set_title(
        rf"$\tau_s$ = {threshold:.2f} m/s, "
        rf"$v_{{max}}$ = {max_speed_mps:.1f} m/s"
        f"\nStatic {selected}/{total} ({100.0 * ratio:.1f}%)"
        f" | persistent {persistent_static_count} + other {motion_static_count}"
        f"\ndynamic {dynamic_count} (persistent→dynamic {persistent_dynamic_count})"
        f" | unresolved {unresolved_count}"
    )
    ax.scatter([0.0], [0.0], s=45, marker="^", c="black", zorder=5)
    ax.text(
        0.5, 0.985, "Forward ↑", transform=ax.transAxes,
        ha="center", va="top", fontsize=9,
    )
    return (
        selected, total, ratio, persistent_static_count,
        motion_static_count, dynamic_count, persistent_dynamic_count,
        unresolved_count, persistent_unresolved_count,
    )


def render_frame(
    save_path: Path,
    image_path: Path,
    token: str,
    frame_ordinal: int,
    xy: np.ndarray,
    residuals: np.ndarray,
    persistent_static_mask: np.ndarray,
    thresholds: list[float],
    grid: GridConfig,
    ego_velocity_radar: np.ndarray,
    reliable_count: int,
    all_point_size: float,
    static_point_size: float,
    dpi: int,
    max_speed_mps: float,
    velocity_mode: str = "off",
) -> None:
    image = mpimg.imread(image_path)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5.8),
        gridspec_kw={
            "width_ratios": [1.0, 1.0, 1.0, 1.45],
            "wspace": 0.20,
        },
    )

    for ax, threshold in zip(axes[:3], thresholds):
        draw_bev(
            ax=ax,
            xy=xy,
            residuals=residuals,
            persistent_static_mask=persistent_static_mask,
            threshold=float(threshold),
            max_speed_mps=max_speed_mps,
            grid=grid,
            all_point_size=all_point_size,
            static_point_size=static_point_size,
        )

    axes[0].legend(
        loc="upper right",
        fontsize=7,
        framealpha=0.9,
    )

    axes[3].imshow(image)
    axes[3].set_title(
        f"RGB reference\n{image_path.name}"
    )
    axes[3].axis("off")

    horizontal_speed = float(
        np.linalg.norm(np.asarray(ego_velocity_radar, dtype=np.float64)[:2])
    )

    fig.suptitle(
        f"Static threshold comparison | velocity mode: {velocity_mode}"
        f"   |   scene {token.split('_')[0]}"
        f"   |   token {token}"
        f"   |   frame {frame_ordinal}"
        f"   |   ego speed {horizontal_speed:.2f} m/s"
        f"   |   reliable RPC {reliable_count}"
        f"   |   ROI {len(xy)}",
        fontsize=12,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def encode_mp4(
    frames_dir: Path,
    output_path: Path,
    fps: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. "
            "Install it with: sudo apt install ffmpeg"
        )

    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    print()
    print("Encoding MP4 with ffmpeg...")
    subprocess.run(
        command,
        check=True,
    )


def main() -> None:
    args = parse_args()

    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.pose_dt_s <= 0:
        raise ValueError("--pose-dt-s must be > 0")

    thresholds = [float(value) for value in args.thresholds]
    if any(not np.isfinite(value) or value < 0.0 for value in thresholds):
        raise ValueError("Thresholds must be finite and non-negative.")
    if args.range_difference_max_speed_mps <= max(thresholds):
        raise ValueError(
            "--range-difference-max-speed-mps must exceed every threshold."
        )

    repo_root = args.repo_root.expanduser().resolve()

    annotation = resolve_repo_path(
        args.annotation,
        repo_root,
    )
    radar_root = resolve_repo_path(
        args.radar_root,
        repo_root,
    )
    calib_root = resolve_repo_path(
        args.calib_root,
        repo_root,
    )
    pose_root = resolve_repo_path(
        args.pose_root,
        repo_root,
    )
    camera_root = resolve_repo_path(
        args.camera_root,
        repo_root,
    )
    output_dir = resolve_repo_path(
        args.output_dir,
        repo_root,
    )

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale PNGs so frame numbering cannot mix with a previous run.
    for stale in frames_dir.glob("frame_*.png"):
        stale.unlink()

    infos_all = _load_infos(annotation)

    scene_infos = ordered_scene_infos(infos_all, args.scene)

    if not scene_infos:
        raise RuntimeError(
            f"No annotation frames found for scene={args.scene}"
        )

    stop = (
        len(scene_infos)
        if args.max_frames is None
        else min(
            len(scene_infos),
            args.start + args.max_frames,
        )
    )
    selected_infos = scene_infos[args.start:stop]

    if not selected_infos:
        raise RuntimeError(
            f"No frames selected: scene has {len(scene_infos)} frames, "
            f"start={args.start}, max_frames={args.max_frames}"
        )

    camera_dir = resolve_camera_dir(
        camera_root,
        str(args.scene),
    )

    camera_by_token = _camera_map(
        infos_all,
        str(args.scene),
        camera_dir,
        offset=args.camera_offset,
    )

    radar_cfg = KRadarConfig()
    grid_cfg = GridConfig()
    reliability_cfg = ReliabilityConfig(
        min_local_power_ratio=args.min_local_power_ratio,
        min_local_neighbors=args.min_local_neighbors,
    )

    reader = KRadarRPCReader()
    detector = RPCPointTargetDetector(
        radar_cfg=radar_cfg,
    )
    reliability_filter = LocalPowerRPCFilter(
        reliability_cfg,
    )

    pose_reader = RadarOccPoseReader(
        pose_root,
    )
    pose_estimator = PoseEgoMotionEstimator(
        radar_cfg=radar_cfg,
        pose_cfg=PoseConfig(
            frame_dt_s=args.pose_dt_s,
        ),
    )

    doppler_classifier = EgoCompensatedDopplerClassifier(
        radar_cfg=radar_cfg,
        motion_cfg=MotionConfig(
            static_residual_threshold_mps=0.0,
            dynamic_residual_threshold_mps=max(
                0.8,
                max(thresholds),
            ),
            stationary_velocity_sign=args.stationary_velocity_sign,
        ),
    )

    unwrapper = build_velocity_unwrapper(
        args, radar_cfg, doppler_classifier.motion_cfg,
    )
    persistence_classifier = build_occupancy_persistence_classifier(args)
    print(f"Velocity mode: {args.velocity_unwrapping}")
    print(f"Persistent velocity check: {args.persistent_velocity_check}")
    print("Static-threshold video")
    print(f"  scene       : {args.scene}")
    print(f"  frames      : {len(selected_infos)}")
    print(f"  scene total : {len(scene_infos)}")
    print(f"  start       : {args.start}")
    print(f"  thresholds  : {thresholds}")
    print(f"  persistence : {args.occupancy_persistence}")
    print(f"  occ cell    : {args.occupancy_cell_size_m:.2f} m")
    print(
        f"  occ support : {args.occupancy_min_support}/"
        f"{args.occupancy_history_size} historical frames"
    )
    print(f"  occ dilation: {args.occupancy_dilation_cells} cells")
    print(f"  cluster r   : {args.unwrap_cluster_radius_m:.2f} m")
    print(f"  patch extent: {args.unwrap_max_cluster_extent_m:.2f} m")
    print(
        f"  association : {args.range_difference_association_radius_m:.2f} m "
        "(track-speed adaptive)"
    )
    print(f"  history     : {args.range_difference_history} frames")
    print(f"  max error   : {args.range_difference_max_error_mps:.2f} m/s")
    print(f"  fps         : {args.fps}")
    print(f"  annotation  : {annotation}")
    print(f"  radar root  : {radar_root}")
    print(f"  calib root  : {calib_root}")
    print(f"  pose root   : {pose_root}")
    print(f"  camera dir  : {camera_dir}")
    print(f"  output dir  : {output_dir}")
    print()

    # Even when --start is nonzero, process preceding observations to build displacement history.
    processing_start = (
        0
        if unwrapper is not None or persistence_classifier is not None
        else args.start
    )
    for scene_index in range(processing_start, stop):
        info = scene_infos[scene_index]
        output_index = scene_index - args.start
        token = str(info["lidar_token"])

        radar_path = _resolve_rpc_radar(
            info,
            repo_root,
            radar_root,
            calib_root,
        )

        raw_detections = detector.detect(
            reader.read(radar_path)
        )
        reliable_detections = reliability_filter.filter(
            raw_detections
        )

        ego_motion, _ = estimate_ego_motion(
            pose_reader=pose_reader,
            estimator=pose_estimator,
            scene=str(args.scene),
            token=token,
            dt_s=args.pose_dt_s,
        )

        wrapped_residuals_all = doppler_classifier.residuals_with_velocity(
            reliable_detections, ego_motion.linear_velocity_radar_mps,
        )
        persistent_static_all = np.zeros(
            len(reliable_detections), dtype=bool
        )
        if persistence_classifier is not None:
            persistent_static_all, _ = persistence_classifier.update(
                reliable_detections, ego_motion, token,
            )

        if unwrapper is None:
            residuals_all = wrapped_residuals_all.copy()
            if args.persistent_velocity_check == "off":
                residuals_all[persistent_static_all] = 0.0
        else:
            temporal_residuals_all, _ = unwrapper.update(
                reliable_detections,
                ego_motion,
                token,
                excluded_mask=(
                    persistent_static_all
                    if args.persistent_velocity_check == "off"
                    else None
                ),
            )
            residuals_all = temporal_residuals_all

        analysis_summary = {}
        if unwrapper is not None:
            analysis_summary.update(
                velocity_source_summary(unwrapper, residuals_all)
            )
        if persistence_classifier is not None:
            analysis_summary["occupancy_persistence"] = dict(
                persistence_classifier.last_diagnostics
            )
        persistent_velocity_resolved = (
            persistent_static_all
            & np.isfinite(residuals_all)
            & (np.abs(residuals_all) < args.range_difference_max_speed_mps)
        )
        analysis_summary["persistent_velocity_check"] = {
            "enabled": args.persistent_velocity_check == "on",
            "resolved_count": int(np.count_nonzero(persistent_velocity_resolved)),
            "unresolved_count": int(
                np.count_nonzero(
                    persistent_static_all & ~persistent_velocity_resolved
                )
            ),
            "max_speed_mps": float(args.range_difference_max_speed_mps),
        }
        if output_index < 0:
            continue  # Warmup needs no RGB image and produces no video frame.
        camera_path = camera_by_token.get(token)
        if camera_path is None:
            raise FileNotFoundError(f"No RGB image mapped for token={token}")
        save_velocity_diagnostics(
            output_dir / "velocity_unwrapping" / f"{token}.json",
            unwrapper,
            analysis_summary,
        )

        xyz = np.asarray(
            [
                det.xyz_lidar_m
                for det in reliable_detections
            ],
            dtype=np.float64,
        ).reshape(-1, 3)

        roi_mask = in_radarocc_roi(
            xyz,
            grid_cfg,
        )

        xyz_roi = xyz[roi_mask]
        residuals_roi = residuals_all[roi_mask]
        persistent_static_roi = persistent_static_all[roi_mask]
        xy = xyz_roi[:, :2]

        save_path = (
            frames_dir
            / f"frame_{output_index:05d}.png"
        )

        render_frame(
            save_path=save_path,
            image_path=camera_path,
            token=token,
            frame_ordinal=args.start + output_index,
            xy=xy,
            residuals=residuals_roi,
            persistent_static_mask=persistent_static_roi,
            thresholds=thresholds,
            grid=grid_cfg,
            ego_velocity_radar=ego_motion.linear_velocity_radar_mps,
            reliable_count=len(reliable_detections),
            all_point_size=args.all_point_size,
            static_point_size=args.static_point_size,
            dpi=args.dpi,
            max_speed_mps=args.range_difference_max_speed_mps,
            velocity_mode=(
                f"{args.velocity_unwrapping}; "
                f"persistent-check={args.persistent_velocity_check}"
            ),
        )

        if (
            output_index == 0
            or (output_index + 1) % 25 == 0
            or output_index + 1 == len(selected_infos)
        ):
            counts = []
            persistent_to_dynamic = []
            for threshold in thresholds:
                (
                    persistent_velocity_static,
                    persistent_velocity_dynamic,
                    nonpersistent_velocity_static,
                    _nonpersistent_velocity_dynamic,
                    _dynamic,
                    _unresolved,
                ) = velocity_partition_masks(
                    residuals_roi,
                    persistent_static_roi,
                    threshold_mps=threshold,
                    max_speed_mps=args.range_difference_max_speed_mps,
                )
                counts.append(
                    int(
                        np.count_nonzero(
                            persistent_velocity_static
                            | nonpersistent_velocity_static
                        )
                    )
                )
                persistent_to_dynamic.append(
                    int(np.count_nonzero(persistent_velocity_dynamic))
                )

            print(
                f"[{output_index + 1:4d}/{len(selected_infos):4d}] "
                f"{token} | "
                f"reliable={len(reliable_detections):4d} | "
                f"ROI={len(xy):4d} | "
                f"grid_static={int(np.count_nonzero(persistent_static_roi))} | "
                f"unresolved={int(np.count_nonzero(~np.isfinite(residuals_roi)))} | "
                f"static={counts} | persistent_to_dynamic={persistent_to_dynamic}"
            )

    output_mp4 = (
        output_dir
        / (
            f"static_threshold_scene{args.scene}_"
            f"{thresholds[0]:.1f}_"
            f"{thresholds[1]:.1f}_"
            f"{thresholds[2]:.1f}.mp4"
        )
    )

    encode_mp4(
        frames_dir=frames_dir,
        output_path=output_mp4,
        fps=args.fps,
    )

    if not args.keep_frames:
        shutil.rmtree(
            frames_dir,
            ignore_errors=True,
        )

    print()
    print("Done")
    print(f"  video: {output_mp4}")
    print(
        f"  duration: approximately "
        f"{len(selected_infos) / args.fps:.1f} s"
    )


if __name__ == "__main__":
    main()
