#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# This script only saves a PNG; it never needs Qt/Tk/X11.
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


def add_velocity_arguments(parser):
    parser.add_argument(
        "--velocity-unwrapping", choices=("off", "range-kalman"),
        default="range-kalman",
        help="Use range-only KF history by default; off shows wrapped residuals.",
    )
    parser.add_argument("--unwrap-range-std-m", type=float, default=0.20)
    parser.add_argument("--unwrap-cluster-radius-m", type=float, default=1.5)


def build_velocity_unwrapper(args, radar_cfg, motion_cfg):
    if args.velocity_unwrapping == "off":
        return None
    from tradition.core.config import UnwrappingConfig
    from tradition.motion.range_kalman import RangeKalmanUnwrapper
    return RangeKalmanUnwrapper(
        radar_cfg=radar_cfg, motion_cfg=motion_cfg,
        config=UnwrappingConfig(
            frame_dt_s=args.pose_dt_s,
            range_std_m=args.unwrap_range_std_m,
            cluster_radius_m=args.unwrap_cluster_radius_m,
        ),
    )


def ordered_scene_infos(infos, scene):
    selected = sorted(
        (info for info in infos if str(info.get("scene_token")) == str(scene)),
        key=lambda info: RadarOccPoseReader.frame_index(str(info["lidar_token"])),
    )
    indices = [RadarOccPoseReader.frame_index(str(info["lidar_token"]))
               for info in selected]
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate pose indices in the selected scene.")
    return selected


def save_velocity_diagnostics(path, unwrapper):
    if unwrapper is None:
        return
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(unwrapper.last_diagnostics, allow_nan=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualitatively compare ego-motion-compensated static Doppler "
            "thresholds on one K-Radar frame. Output layout: "
            "BEV tau1 | BEV tau2 | BEV tau3 | RGB."
        )
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("data/annotations/kradar_dict_val_doppler8.pkl"),
        help="RadarOcc annotation PKL. Default: validation annotation.",
    )
    parser.add_argument(
        "--radar-root",
        type=Path,
        default=Path("data/K-Radar_rpc"),
        help="Enhanced K-Radar RPC root.",
    )
    parser.add_argument(
        "--pose-root",
        type=Path,
        default=Path("data/K-RadarOcc"),
        help="RadarOcc pose root.",
    )
    parser.add_argument(
        "--camera-root",
        type=Path,
        default=Path("data/K-Radar-RGB/K-Radar/K-Radar-RGB"),
        help=(
            "K-Radar RGB root. May be the common root or directly a "
            "scene/images_rb_switched directory."
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--scene", required=True, help="Scene token, e.g. 3.")
    parser.add_argument(
        "--token",
        required=True,
        help="Exact lidar_token, e.g. 3_00080.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="Static residual thresholds in m/s. Intended use: three values.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. If omitted, writes "
            "work_dirs/static_threshold_visualization/"
            "static_threshold_scene<scene>_<frame>.png."
        ),
    )
    parser.add_argument("--camera-offset", type=int, default=0)
    parser.add_argument("--pose-dt-s", type=float, default=0.10)
    parser.add_argument(
        "--stationary-velocity-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
    )
    parser.add_argument("--min-local-power-ratio", type=float, default=0.25)
    parser.add_argument("--min-local-neighbors", type=int, default=1)
    parser.add_argument(
        "--all-point-size",
        type=float,
        default=8.0,
        help="Marker size for all reliable RPC points.",
    )
    parser.add_argument(
        "--static-point-size",
        type=float,
        default=18.0,
        help="Marker size for points selected as static.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    add_velocity_arguments(parser)
    return parser.parse_args()


def resolve_camera_dir(camera_root: Path, scene: str) -> Path:
    root = camera_root.expanduser().resolve()
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

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
            path.is_file() and path.suffix.lower() in image_exts
            for path in candidate.iterdir()
        ):
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot resolve RGB directory for scene={scene} from {root}."
    )


def find_info(
    infos: list[dict],
    scene: str,
    token: str,
) -> dict:
    matches = [
        info
        for info in infos
        if str(info.get("scene_token")) == str(scene)
        and str(info.get("lidar_token")) == str(token)
    ]
    if not matches:
        raise RuntimeError(
            f"No annotation entry found for scene={scene}, token={token}."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple annotation entries found for scene={scene}, token={token}."
        )
    return matches[0]


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
    next_item = pose_reader.try_read_index(scene, frame_index + 1)

    previous_pose = previous[0] if previous is not None else None
    next_pose = next_item[0] if next_item is not None else None

    motion = estimator.estimate(
        current_pose=current_pose,
        previous_pose=previous_pose,
        next_pose=next_pose,
        previous_dt_s=dt_s if previous_pose is not None else None,
        next_dt_s=dt_s if previous_pose is None and next_pose is not None else None,
    )
    return motion, current_path


def output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        path = args.output.expanduser()
        if not path.is_absolute():
            path = (args.repo_root / path).resolve()
        else:
            path = path.resolve()
        return path

    frame_suffix = str(args.token).split("_")[-1]
    return (
        args.repo_root.expanduser().resolve()
        / "work_dirs"
        / "static_threshold_visualization"
        / f"static_threshold_scene{args.scene}_{frame_suffix}.png"
    )


def plot_bev(
    ax,
    xy: np.ndarray,
    residuals: np.ndarray,
    threshold: float,
    grid: GridConfig,
    all_point_size: float,
    static_point_size: float,
) -> tuple[int, int, float]:
    absolute = np.abs(residuals)
    static_mask = np.isfinite(absolute) & (absolute <= threshold)

    total = int(len(xy))
    selected = int(np.count_nonzero(static_mask))
    ratio = selected / total if total else 0.0

    # 原始车辆坐标：
    # x = forward
    # +y = vehicle left
    # -y = vehicle right
    #
    # 为了和 front RGB 对齐：
    # 图像纵轴 = +x forward
    # 图像左侧 = vehicle left (+y)
    # 图像右侧 = vehicle right (-y)
    #
    # 因此：
    # display horizontal = -y
    # display vertical   =  x
    display_x = -xy[:, 1]
    display_y = xy[:, 0]

    # 灰色：RadarOcc ROI 内所有 reliable RPC points
    if total:
        ax.scatter(
            display_x,
            display_y,
            s=all_point_size,
            c="0.72",
            alpha=0.55,
            linewidths=0,
            label="Reliable RPC points in RadarOcc ROI",
            rasterized=True,
        )

    unknown_mask = ~np.isfinite(absolute)
    if np.any(unknown_mask):
        ax.scatter(
            display_x[unknown_mask], display_y[unknown_mask],
            s=static_point_size, c="tab:orange", marker="x",
            linewidths=0.7, label="Unresolved velocity", rasterized=True,
        )

    # 蓝色：当前 threshold 下判定为 Static 的点
    if selected:
        ax.scatter(
            display_x[static_mask],
            display_y[static_mask],
            s=static_point_size,
            c="tab:blue",
            alpha=0.95,
            linewidths=0,
            label=r"Static: $|r|\leq\tau_s$",
            rasterized=True,
        )

    # 横轴现在是 -y：
    # 左边 = vehicle left
    # 右边 = vehicle right
    ax.set_xlim(
        -grid.max_xyz[1],
        -grid.min_xyz[1],
    )

    # 纵轴现在是 x：
    # 底部 = Ego
    # 顶部 = 前方 51.2 m
    ax.set_ylim(
        grid.min_xyz[0],
        grid.max_xyz[0],
    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)

    ax.set_xlabel("Lateral [m]   ← vehicle left | vehicle right →")
    ax.set_ylabel("Forward x [m]")

    ax.set_title(
        rf"$\tau_s$ = {threshold:.2f} m/s"
        f"\nStatic: {selected}/{total} ({100.0 * ratio:.1f}%)"
        f" | unresolved {int(np.count_nonzero(unknown_mask))}"
    )

    # Ego 位置
    ax.scatter(
        [0.0],
        [0.0],
        s=50,
        marker="^",
        c="black",
        zorder=5,
    )

    ax.text(
        0.0,
        1.1,
        "Ego",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    ax.text(
        0.5,
        0.985,
        "Forward ↑",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    return selected, total, ratio


def main() -> None:
    args = parse_args()

    if len(args.thresholds) != 3:
        raise ValueError(
            "This four-panel visualization expects exactly three thresholds, "
            "for example: --thresholds 0.3 0.5 0.7"
        )
    if any(not np.isfinite(value) or value < 0.0 for value in args.thresholds):
        raise ValueError("Thresholds must be non-negative.")

    repo_root = args.repo_root.expanduser().resolve()
    annotation = args.annotation.expanduser()
    radar_root = args.radar_root.expanduser()
    pose_root = args.pose_root.expanduser()
    camera_root = args.camera_root.expanduser()

    if not annotation.is_absolute():
        annotation = (repo_root / annotation).resolve()
    else:
        annotation = annotation.resolve()
    if not radar_root.is_absolute():
        radar_root = (repo_root / radar_root).resolve()
    else:
        radar_root = radar_root.resolve()
    if not pose_root.is_absolute():
        pose_root = (repo_root / pose_root).resolve()
    else:
        pose_root = pose_root.resolve()
    if not camera_root.is_absolute():
        camera_root = (repo_root / camera_root).resolve()
    else:
        camera_root = camera_root.resolve()

    infos = _load_infos(annotation)
    info = find_info(infos, str(args.scene), str(args.token))

    # Resolve exactly the same RPC representation used by the traditional runner.
    radar_path = _resolve_rpc_radar(info, repo_root, radar_root)

    radar_cfg = KRadarConfig()
    grid_cfg = GridConfig()
    reliability_cfg = ReliabilityConfig(
        min_local_power_ratio=args.min_local_power_ratio,
        min_local_neighbors=args.min_local_neighbors,
    )

    reader = KRadarRPCReader()
    detector = RPCPointTargetDetector(radar_cfg=radar_cfg)
    reliability_filter = LocalPowerRPCFilter(reliability_cfg)

    raw_detections = detector.detect(reader.read(radar_path))
    reliable_detections = reliability_filter.filter(raw_detections)

    # Pose-derived full ego velocity, identical in principle to the current
    # traditional pipeline's ego-motion-compensated Doppler residual.
    pose_reader = RadarOccPoseReader(pose_root)
    pose_estimator = PoseEgoMotionEstimator(
        radar_cfg=radar_cfg,
        pose_cfg=PoseConfig(frame_dt_s=args.pose_dt_s),
    )
    ego_motion, pose_path = estimate_ego_motion(
        pose_reader=pose_reader,
        estimator=pose_estimator,
        scene=str(args.scene),
        token=str(args.token),
        dt_s=args.pose_dt_s,
    )

    doppler_classifier = EgoCompensatedDopplerClassifier(
        radar_cfg=radar_cfg,
        motion_cfg=MotionConfig(
            static_residual_threshold_mps=0.0,
            # Irrelevant for residuals_with_velocity(), but must satisfy config.
            dynamic_residual_threshold_mps=max(0.8, max(args.thresholds)),
            stationary_velocity_sign=args.stationary_velocity_sign,
        ),
    )
    unwrapper = build_velocity_unwrapper(
        args, radar_cfg, doppler_classifier.motion_cfg,
    )
    if unwrapper is None:
        residuals_all = doppler_classifier.residuals_with_velocity(
            reliable_detections, ego_motion.linear_velocity_radar_mps,
        )
    else:
        # Replay all available preceding scene observations, in pose-index order.
        # Never use future detections and never initialize from just the target.
        target_index = pose_reader.frame_index(str(args.token))
        for history_info in ordered_scene_infos(infos, args.scene):
            history_token = str(history_info["lidar_token"])
            if pose_reader.frame_index(history_token) >= target_index:
                break
            history_path = _resolve_rpc_radar(history_info, repo_root, radar_root)
            history_detections = reliability_filter.filter(
                detector.detect(reader.read(history_path))
            )
            history_motion, _ = estimate_ego_motion(
                pose_reader, pose_estimator, str(args.scene),
                history_token, args.pose_dt_s,
            )
            unwrapper.update(history_detections, history_motion, history_token)
        residuals_all, _ = unwrapper.update(
            reliable_detections, ego_motion, str(args.token),
        )

    xyz = np.asarray(
        [det.xyz_lidar_m for det in reliable_detections],
        dtype=np.float64,
    ).reshape(-1, 3)

    # Keep exactly the points visible inside the RadarOcc evaluation ROI.
    finite_xyz = np.isfinite(xyz).all(axis=1)
    in_roi = (
        finite_xyz
        & (xyz[:, 0] >= grid_cfg.min_xyz[0])
        & (xyz[:, 0] < grid_cfg.max_xyz[0])
        & (xyz[:, 1] >= grid_cfg.min_xyz[1])
        & (xyz[:, 1] < grid_cfg.max_xyz[1])
        & (xyz[:, 2] >= grid_cfg.min_xyz[2])
        & (xyz[:, 2] < grid_cfg.max_xyz[2])
    )

    xyz_roi = xyz[in_roi]
    residuals = residuals_all[in_roi]
    xy = xyz_roi[:, :2]

    camera_dir = resolve_camera_dir(camera_root, str(args.scene))
    camera_mapping = _camera_map(
        infos,
        str(args.scene),
        camera_dir,
        offset=args.camera_offset,
    )
    camera_path = camera_mapping.get(str(args.token))
    if camera_path is None:
        raise FileNotFoundError(
            f"No RGB image mapped to token={args.token} in {camera_dir}."
        )

    image = mpimg.imread(camera_path)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(24, 6.5),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.35]},
    )

    summaries: list[tuple[float, int, int, float]] = []
    for ax, threshold in zip(axes[:3], args.thresholds):
        selected, total, ratio = plot_bev(
            ax=ax,
            xy=xy,
            residuals=residuals,
            threshold=float(threshold),
            grid=grid_cfg,
            all_point_size=args.all_point_size,
            static_point_size=args.static_point_size,
        )
        summaries.append((float(threshold), selected, total, ratio))

    # One legend is enough because all three BEVs use exactly the same encoding.
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[3].imshow(image)
    axes[3].set_title(f"RGB reference\n{camera_path.name}")
    axes[3].axis("off")

    velocity = ego_motion.linear_velocity_radar_mps
    speed = float(np.linalg.norm(velocity[:2]))
    fig.suptitle(
        f"Static threshold comparison | velocity mode: {args.velocity_unwrapping}"
        f"\nScene {args.scene} | token {args.token} | "
        f"ego speed = {speed:.2f} m/s | "
        f"reliable RPC: {len(reliable_detections)} | "
        f"RadarOcc-ROI points: {len(xyz_roi)}",
        fontsize=14,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))

    save_path = output_path(args)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    save_velocity_diagnostics(save_path.with_suffix(".velocity.json"), unwrapper)

    print("Static-threshold qualitative visualization")
    print(f"  velocity mode: {args.velocity_unwrapping}")
    print(f"  unresolved ROI: {int(np.count_nonzero(~np.isfinite(residuals)))}")
    print(f"  scene       : {args.scene}")
    print(f"  token       : {args.token}")
    print(f"  radar       : {radar_path}")
    print(f"  pose        : {pose_path}")
    print(f"  camera      : {camera_path}")
    print(
        "  ego vel radar [m/s]: "
        f"[{velocity[0]:.3f}, {velocity[1]:.3f}, {velocity[2]:.3f}]"
    )
    print(f"  raw RPC     : {len(raw_detections)}")
    print(f"  reliable RPC: {len(reliable_detections)}")
    print(f"  in ROI      : {len(xyz_roi)}")
    for threshold, selected, total, ratio in summaries:
        print(
            f"  tau={threshold:.2f}: "
            f"static {selected}/{total} ({100.0 * ratio:.1f}%)"
        )
    print(f"  output      : {save_path}")


if __name__ == "__main__":
    main()
