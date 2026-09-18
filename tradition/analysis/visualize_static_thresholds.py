#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import matplotlib

# This script only saves a PNG; it never needs Qt/Tk/X11.
matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

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
        "--velocity-unwrapping", choices=("off", "range-difference"),
        default="range-difference",
        help=(
            "Estimate target velocity from cross-frame world-position "
            "differences and solve integer k; off shows wrapped residuals."
        ),
    )
    parser.add_argument("--unwrap-cluster-radius-m", type=float, default=1.5)
    parser.add_argument("--range-difference-association-radius-m", type=float, default=1.0)
    parser.add_argument("--range-difference-history", type=int, default=3)
    parser.add_argument("--range-difference-max-speed-mps", type=float, default=40.0)
    parser.add_argument("--range-difference-max-error-mps", type=float, default=0.80)
    parser.add_argument("--fallback-static-window", type=int, default=5)
    parser.add_argument("--fallback-static-min-support", type=int, default=3)
    parser.add_argument("--fallback-static-match-radius-m", type=float, default=0.60)
    parser.add_argument("--fallback-wrapped-threshold-mps", type=float, default=0.50)
    parser.add_argument(
        "--disable-static-fallback", "--disable-static-prefilter",
        dest="disable_static_prefilter", action="store_true",
        help=(
            "Disable the world-stable point-wise static prefilter. The old "
            "--disable-static-fallback spelling remains as a compatibility alias."
        ),
    )


class RangeDifferenceUnwrapper:
    """Non-Kalman ambiguity resolution from cross-frame cluster displacement."""

    def __init__(
        self, radar_cfg, motion_cfg, frame_dt_s=0.10,
        cluster_radius_m=1.5, association_radius_m=1.0,
        history_size=3, min_cluster_points=2,
        max_speed_mps=40.0, max_doppler_error_mps=0.80,
    ):
        if frame_dt_s <= 0.0:
            raise ValueError("frame_dt_s must be positive")
        if cluster_radius_m <= 0.0 or association_radius_m <= 0.0:
            raise ValueError("cluster/association radii must be positive")
        if history_size < 2:
            raise ValueError("range-difference history must be >= 2")
        if max_speed_mps <= 0.0 or max_doppler_error_mps <= 0.0:
            raise ValueError("speed/error gates must be positive")
        self.radar_cfg = radar_cfg
        self.motion_cfg = motion_cfg
        self.frame_dt_s = float(frame_dt_s)
        self.cluster_radius_m = float(cluster_radius_m)
        self.association_radius_m = float(association_radius_m)
        self.history_size = int(history_size)
        self.min_cluster_points = int(min_cluster_points)
        self.max_speed_mps = float(max_speed_mps)
        self.max_doppler_error_mps = float(max_doppler_error_mps)
        self.max_gap_s = max(0.5, 2.5 * self.frame_dt_s)
        self.radar_to_lidar_rotation = np.asarray(
            PoseConfig().radar_to_lidar_rotation, dtype=np.float64
        )
        self.doppler = EgoCompensatedDopplerClassifier(radar_cfg, motion_cfg)
        self.reset()

    def reset(self):
        self.tracks = []
        self.next_id = 0
        self.last_time = None
        self.scene = None
        self.last_diagnostics = {}

    def _clusters(self, detections, excluded_mask=None):
        xyz = np.asarray(
            [det.xyz_lidar_m for det in detections], dtype=np.float64
        ).reshape(-1, 3)
        valid_mask = np.all(np.isfinite(xyz), axis=1)
        if excluded_mask is not None:
            excluded = np.asarray(excluded_mask, dtype=bool)
            if excluded.shape != (len(detections),):
                raise ValueError("excluded_mask must match detections")
            valid_mask &= ~excluded
        valid = np.flatnonzero(valid_mask)
        if not len(valid):
            return []
        tree = cKDTree(xyz[valid])
        unseen = set(range(len(valid)))
        clusters = []
        while unseen:
            seed = min(unseen)
            unseen.remove(seed)
            pending, component = [seed], []
            while pending:
                local_index = pending.pop()
                component.append(int(valid[local_index]))
                for neighbour in tree.query_ball_point(
                    xyz[valid[local_index]], self.cluster_radius_m
                ):
                    if neighbour in unseen:
                        unseen.remove(neighbour)
                        pending.append(neighbour)
            if len(component) >= self.min_cluster_points:
                clusters.append(np.asarray(sorted(component), dtype=np.int64))
        return clusters

    @staticmethod
    def _track_velocity_world(track):
        history = track["history"]
        if len(history) < 2:
            return None
        times = np.asarray([item[0] for item in history], dtype=np.float64)
        centers = np.asarray([item[1] for item in history], dtype=np.float64)
        centered = times - float(np.mean(times))
        denominator = float(centered @ centered)
        if denominator <= 1e-9:
            return None
        return (centered[:, None] * centers).sum(axis=0) / denominator

    def _predicted_center(self, track, time):
        previous_time, previous_center = track["history"][-1]
        velocity = self._track_velocity_world(track)
        if velocity is None:
            return previous_center
        return previous_center + velocity * (time - previous_time)

    def update(self, detections, ego_motion, token, excluded_mask=None):
        time = RadarOccPoseReader.frame_index(str(token)) * self.frame_dt_s
        scene = str(token).rsplit("_", 1)[0]
        if self.scene != scene or (
            self.last_time is not None and time <= self.last_time
        ):
            self.reset()
        self.scene, self.last_time = scene, time
        self.tracks = [
            track for track in self.tracks
            if time - track["history"][-1][0] <= self.max_gap_s
        ]

        wrapped = self.doppler.residuals_with_velocity(
            detections, ego_motion.linear_velocity_radar_mps
        )
        output = np.full(len(detections), np.nan, dtype=np.float64)
        track_ids = np.full(len(detections), -1, dtype=np.int64)
        ambiguity = [None] * len(detections)
        temporal_prediction = [None] * len(detections)
        failure_reason = ["no_cluster"] * len(detections)
        if excluded_mask is None:
            excluded = np.zeros(len(detections), dtype=bool)
        else:
            excluded = np.asarray(excluded_mask, dtype=bool)
            if excluded.shape != (len(detections),):
                raise ValueError("excluded_mask must match detections")
            for index in np.flatnonzero(excluded):
                # The static branch has already resolved this point. Preserve
                # its near-zero wrapped residual as the final residual and do
                # not send it through moving-object clustering.
                output[int(index)] = wrapped[int(index)]
                failure_reason[int(index)] = "static_preclassified"

        # Static background is resolved point-wise before this stage. Only the
        # remaining measurements may form moving-object clusters.
        clusters = self._clusters(detections, excluded_mask=excluded)
        pose = np.asarray(ego_motion.pose_lidar_to_world, dtype=np.float64)
        observations = []
        for indices in clusters:
            center_lidar = np.median(
                [detections[index].xyz_lidar_m for index in indices], axis=0
            )
            observations.append(
                pose[:3, :3] @ center_lidar + pose[:3, 3]
            )

        distances = np.full(
            (len(self.tracks), len(clusters)), np.inf, dtype=np.float64
        )
        for track_index, track in enumerate(self.tracks):
            dt = time - track["history"][-1][0]
            predicted_center = self._predicted_center(track, time)
            gate = self.association_radius_m + self.max_speed_mps * max(dt, 0.0)
            for cluster_index, center_world in enumerate(observations):
                distance = float(np.linalg.norm(center_world - predicted_center))
                if distance <= gate:
                    distances[track_index, cluster_index] = distance

        assignments = {}
        if distances.size:
            nearest_track = np.argmin(distances, axis=0)
            nearest_cluster = np.argmin(distances, axis=1)
            for cluster_index, track_index in enumerate(nearest_track):
                if (
                    np.isfinite(distances[track_index, cluster_index])
                    and nearest_cluster[track_index] == cluster_index
                ):
                    assignments[cluster_index] = self.tracks[track_index]

        for cluster_index, (indices, center_world) in enumerate(
            zip(clusters, observations)
        ):
            track = assignments.get(cluster_index)
            had_candidate = (
                distances.shape[0] > 0
                and np.any(np.isfinite(distances[:, cluster_index]))
            )
            if track is None:
                track = {"identifier": self.next_id, "history": []}
                self.next_id += 1
                self.tracks.append(track)
                reason = (
                    "association_ambiguous_or_unmatched"
                    if had_candidate else "new_track"
                )
            else:
                reason = "insufficient_history"

            track["history"].append((time, center_world.copy()))
            track["history"] = track["history"][-self.history_size:]
            track_ids[indices] = track["identifier"]
            for index in indices:
                failure_reason[index] = reason

            velocity_world = self._track_velocity_world(track)
            if velocity_world is None:
                continue
            speed = float(np.linalg.norm(velocity_world))
            if not np.isfinite(speed) or speed > self.max_speed_mps:
                for index in indices:
                    failure_reason[index] = "temporal_speed_gate_failed"
                continue

            velocity_lidar = pose[:3, :3].T @ velocity_world
            velocity_radar = self.radar_to_lidar_rotation.T @ velocity_lidar
            for index in indices:
                point_radar = np.asarray(
                    detections[index].xyz_radar_m, dtype=np.float64
                )
                distance = float(np.linalg.norm(point_radar))
                if distance <= 1e-9:
                    failure_reason[index] = "invalid_line_of_sight"
                    continue
                line_of_sight = point_radar / distance
                predicted = -self.motion_cfg.stationary_velocity_sign * float(
                    velocity_radar @ line_of_sight
                )
                temporal_prediction[index] = predicted
                value = wrapped[index]
                if not np.isfinite(value):
                    failure_reason[index] = "invalid_wrapped_residual"
                    continue
                period = self.radar_cfg.doppler_period_mps
                k = int(np.rint((predicted - value) / period))
                candidate = value + k * period
                if (
                    abs(candidate) > self.max_speed_mps
                    or abs(candidate - predicted) > self.max_doppler_error_mps
                ):
                    failure_reason[index] = "doppler_gate_failed"
                    continue
                output[index] = candidate
                ambiguity[index] = k
                failure_reason[index] = "resolved"

        self.last_diagnostics = {
            "method": "range_difference_position_history",
            "token": str(token),
            "doppler_period_mps": self.radar_cfg.doppler_period_mps,
            "track_ids": track_ids.tolist(),
            "ambiguity_k": ambiguity,
            "wrapped_residual_mps": [
                float(value) if np.isfinite(value) else None for value in wrapped
            ],
            "temporal_prediction_mps": temporal_prediction,
            "unwrapped_residual_mps": [
                float(value) if np.isfinite(value) else None for value in output
            ],
            "failure_reason": failure_reason,
            "resolved_count": int(np.count_nonzero(np.isfinite(output))),
            "unresolved_count": int(np.count_nonzero(~np.isfinite(output))),
            "static_preclassified_count": int(np.count_nonzero(excluded)),
            "range_difference_resolved_count": int(np.count_nonzero(
                np.isfinite(output) & ~excluded
            )),
        }
        return output, None


def build_velocity_unwrapper(args, radar_cfg, motion_cfg):
    if args.velocity_unwrapping == "off":
        return None
    return RangeDifferenceUnwrapper(
        radar_cfg=radar_cfg,
        motion_cfg=motion_cfg,
        frame_dt_s=args.pose_dt_s,
        cluster_radius_m=args.unwrap_cluster_radius_m,
        association_radius_m=args.range_difference_association_radius_m,
        history_size=args.range_difference_history,
        max_speed_mps=args.range_difference_max_speed_mps,
        max_doppler_error_mps=args.range_difference_max_error_mps,
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


def save_velocity_diagnostics(path, unwrapper, analysis_summary=None):
    if unwrapper is None:
        return
    import json
    payload = dict(unwrapper.last_diagnostics)
    if analysis_summary is not None:
        payload["visualization_static_prefilter"] = analysis_summary
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False),
        encoding="utf-8",
    )


class WorldStaticPreclassifier:
    """Confirm static points before any moving-object clustering.

    A point is a static candidate when its ego-compensated wrapped residual is
    near zero. It becomes confirmed only when similar candidates repeatedly
    occur at the same world location. Raw radar scatterers need not keep an ID.
    """

    def __init__(
        self, window_size=5, min_support=3, match_radius_m=0.60,
        wrapped_threshold_mps=0.50, enabled=True,
    ):
        if window_size < 2:
            raise ValueError("static prefilter window_size must be >= 2")
        if not 2 <= min_support <= window_size:
            raise ValueError(
                "static prefilter min_support must be in [2, window_size]"
            )
        if match_radius_m <= 0.0 or wrapped_threshold_mps < 0.0:
            raise ValueError(
                "static prefilter radii/thresholds must be non-negative"
            )
        self.window_size = int(window_size)
        self.min_support = int(min_support)
        self.match_radius_m = float(match_radius_m)
        self.wrapped_threshold_mps = float(wrapped_threshold_mps)
        self.enabled = bool(enabled)
        self.history = deque(maxlen=self.window_size - 1)

    def update(self, detections, ego_motion, wrapped_residuals):
        wrapped = np.asarray(wrapped_residuals, dtype=np.float64)
        if wrapped.shape != (len(detections),):
            raise ValueError("Wrapped residuals must match detections.")
        xyz_lidar = np.asarray(
            [det.xyz_lidar_m for det in detections], dtype=np.float64
        ).reshape(-1, 3)
        pose = np.asarray(ego_motion.pose_lidar_to_world, dtype=np.float64)
        world_xyz = xyz_lidar @ pose[:3, :3].T + pose[:3, 3]
        static_candidate = (
            np.isfinite(wrapped)
            & (np.abs(wrapped) <= self.wrapped_threshold_mps)
        )
        support = np.zeros(len(detections), dtype=np.int16)
        finite_current = np.all(np.isfinite(world_xyz), axis=1)
        current_candidate = finite_current & static_candidate
        support[current_candidate] = 1
        for previous_world in self.history:
            if not len(previous_world) or not np.any(current_candidate):
                continue
            distances, _ = cKDTree(previous_world).query(
                world_xyz[current_candidate], k=1
            )
            support[current_candidate] += (
                distances <= self.match_radius_m
            ).astype(np.int16)
        static_mask = (
            self.enabled
            & current_candidate
            & (support >= self.min_support)
        )
        # Store only static-Doppler candidates. Moving returns must not provide
        # future world-persistence support to guardrails or other background.
        self.history.append(world_xyz[current_candidate].copy())
        return static_mask, support


# Compatibility for external scripts that imported the previous name.
WorldStaticFallback = WorldStaticPreclassifier


def velocity_source_summary(
    unwrapper, temporal_residuals, static_mask, support,
):
    diagnostics = unwrapper.last_diagnostics
    temporal = np.asarray(temporal_residuals, dtype=np.float64)
    static = np.asarray(static_mask, dtype=bool)
    reasons = np.asarray(diagnostics["failure_reason"], dtype=object)
    reason_counts = {
        str(reason): int(count)
        for reason, count in zip(*np.unique(reasons, return_counts=True))
    }
    return {
        "range_difference_resolved_count": int(
            np.count_nonzero(np.isfinite(temporal) & ~static)
        ),
        "static_preclassified_count": int(np.count_nonzero(static)),
        "unresolved_after_both_branches_count": int(
            np.count_nonzero(~np.isfinite(temporal))
        ),
        "range_difference_failure_reasons": reason_counts,
        "world_support_histogram": {
            str(value): int(count)
            for value, count in zip(*np.unique(support, return_counts=True))
        },
    }


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
        "--calib-root",
        type=Path,
        default=Path("data/K-Radar_calib"),
        help="Per-scene frame-difference calibration root.",
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
    fallback_mask: np.ndarray,
    threshold: float,
    grid: GridConfig,
    all_point_size: float,
    static_point_size: float,
) -> tuple[int, int, float]:
    absolute = np.abs(residuals)
    static_mask = np.isfinite(absolute) & (absolute <= threshold)
    fallback_static_mask = static_mask & np.asarray(fallback_mask, dtype=bool)
    temporal_static_mask = static_mask & ~fallback_static_mask

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

    # 蓝色：距离差分估速成功解折叠后判定为 Static 的点
    if np.any(temporal_static_mask):
        ax.scatter(
            display_x[temporal_static_mask],
            display_y[temporal_static_mask],
            s=static_point_size,
            c="tab:blue",
            alpha=0.95,
            linewidths=0,
            label=r"Range-difference static: $|r|\leq\tau_s$",
            rasterized=True,
        )

    # 青色：距离差分未解出，但世界坐标稳定且 wrapped residual 接近零。
    if np.any(fallback_static_mask):
        ax.scatter(
            display_x[fallback_static_mask],
            display_y[fallback_static_mask],
            s=static_point_size,
            c="tab:cyan",
            alpha=0.95,
            linewidths=0,
            label="World-stable static prefilter",
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
        f" | prefilter {int(np.count_nonzero(fallback_static_mask))}"
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
    calib_root = args.calib_root.expanduser()
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
    if not calib_root.is_absolute():
        calib_root = (repo_root / calib_root).resolve()
    else:
        calib_root = calib_root.resolve()
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
    radar_path = _resolve_rpc_radar(
        info, repo_root, radar_root, calib_root
    )

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
    static_prefilter = WorldStaticPreclassifier(
        window_size=args.fallback_static_window,
        min_support=args.fallback_static_min_support,
        match_radius_m=args.fallback_static_match_radius_m,
        wrapped_threshold_mps=args.fallback_wrapped_threshold_mps,
        enabled=not args.disable_static_prefilter,
    )
    wrapped_residuals_all = doppler_classifier.residuals_with_velocity(
        reliable_detections, ego_motion.linear_velocity_radar_mps,
    )
    analysis_summary = None
    if unwrapper is None:
        residuals_all = wrapped_residuals_all
        fallback_mask_all = np.zeros(len(reliable_detections), dtype=bool)
        world_support_all = np.ones(len(reliable_detections), dtype=np.int16)
    else:
        target_index = pose_reader.frame_index(str(args.token))
        for history_info in ordered_scene_infos(infos, args.scene):
            history_token = str(history_info["lidar_token"])
            if pose_reader.frame_index(history_token) >= target_index:
                break
            history_path = _resolve_rpc_radar(
                history_info, repo_root, radar_root, calib_root
            )
            history_detections = reliability_filter.filter(
                detector.detect(reader.read(history_path))
            )
            history_motion, _ = estimate_ego_motion(
                pose_reader, pose_estimator, str(args.scene),
                history_token, args.pose_dt_s,
            )
            history_wrapped = doppler_classifier.residuals_with_velocity(
                history_detections, history_motion.linear_velocity_radar_mps,
            )
            history_static, _ = static_prefilter.update(
                history_detections, history_motion, history_wrapped,
            )
            unwrapper.update(
                history_detections, history_motion, history_token,
                excluded_mask=history_static,
            )
        static_mask_all, world_support_all = static_prefilter.update(
            reliable_detections, ego_motion, wrapped_residuals_all,
        )
        temporal_residuals_all, _ = unwrapper.update(
            reliable_detections, ego_motion, str(args.token),
            excluded_mask=static_mask_all,
        )
        residuals_all = temporal_residuals_all
        fallback_mask_all = static_mask_all
        analysis_summary = velocity_source_summary(
            unwrapper, temporal_residuals_all,
            static_mask_all, world_support_all,
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
    fallback_mask = fallback_mask_all[in_roi]
    world_support = world_support_all[in_roi]
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
            fallback_mask=fallback_mask,
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
    save_velocity_diagnostics(
        save_path.with_suffix(".velocity.json"), unwrapper, analysis_summary,
    )

    print("Static-threshold qualitative visualization")
    print(f"  velocity mode: {args.velocity_unwrapping}")
    print(f"  unresolved ROI: {int(np.count_nonzero(~np.isfinite(residuals)))}")
    print(f"  prefilter ROI : {int(np.count_nonzero(fallback_mask))}")
    if len(world_support):
        print(
            "  world support : "
            f"median={float(np.median(world_support)):.1f}, "
            f"max={int(np.max(world_support))}"
        )
    print(f"  scene       : {args.scene}")
    print(f"  token       : {args.token}")
    print(f"  radar       : {radar_path}")
    print(f"  calib root  : {calib_root}")
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
