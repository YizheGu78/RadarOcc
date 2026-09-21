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
    parser.add_argument(
        "--occupancy-persistence",
        choices=("on", "off"),
        default="on",
        help=(
            "Use pose-aligned world-grid persistence as the initial source "
            "label. The visualization-only persistent velocity check may "
            "override that label after temporal velocity estimation."
        ),
    )
    parser.add_argument(
        "--persistent-velocity-check",
        choices=("on", "off"),
        default="on",
        help=(
            "Visualization-only A/B switch. 'on' runs the same temporal "
            "velocity estimator for world-grid persistent points and allows "
            "resolved moving points to override the persistent-static display. "
            "'off' reproduces the previous persistence-bypasses-velocity view."
        ),
    )
    parser.add_argument("--occupancy-cell-size-m", type=float, default=0.40)
    parser.add_argument("--occupancy-history-size", type=int, default=5)
    parser.add_argument("--occupancy-min-support", type=int, default=3)
    parser.add_argument("--occupancy-dilation-cells", type=int, default=1)
    parser.add_argument("--unwrap-cluster-radius-m", type=float, default=0.8)
    parser.add_argument(
        "--unwrap-max-cluster-extent-m",
        type=float,
        default=4.0,
        help=(
            "Maximum XY diameter of a world-anchored local patch. "
            "Prevents long guardrails from becoming one cluster."
        ),
    )
    parser.add_argument("--range-difference-association-radius-m", type=float, default=1.0)
    parser.add_argument("--range-difference-history", type=int, default=5)
    parser.add_argument("--range-difference-max-speed-mps", type=float, default=40.0)
    parser.add_argument("--range-difference-max-error-mps", type=float, default=0.80)


class WorldOccupancyPersistenceClassifier:
    """Stationary evidence from repeated occupancy in fixed world XY cells.

    Only historical frames contribute to the support of a current point. The
    current frame is appended after classification, so a point needs genuine
    cross-frame persistence rather than receiving support from itself.
    """

    def __init__(
        self,
        cell_size_m=0.40,
        history_size=5,
        min_support=3,
        dilation_cells=1,
    ):
        if cell_size_m <= 0.0:
            raise ValueError("occupancy cell size must be positive")
        if history_size < 1:
            raise ValueError("occupancy history size must be >= 1")
        if min_support < 1 or min_support > history_size:
            raise ValueError(
                "occupancy min support must be within [1, history size]"
            )
        if dilation_cells < 0:
            raise ValueError("occupancy dilation cells must be >= 0")
        self.cell_size_m = float(cell_size_m)
        self.history_size = int(history_size)
        self.min_support = int(min_support)
        self.dilation_cells = int(dilation_cells)
        self.offsets = [
            (dx, dy)
            for dx in range(-self.dilation_cells, self.dilation_cells + 1)
            for dy in range(-self.dilation_cells, self.dilation_cells + 1)
        ]
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=self.history_size)
        self.last_frame_index = None
        self.scene = None
        self.last_diagnostics = {}

    def update(self, detections, ego_motion, token):
        token_text = str(token)
        scene = token_text.rsplit("_", 1)[0]
        frame_index = RadarOccPoseReader.frame_index(token_text)
        if self.scene != scene or (
            self.last_frame_index is not None
            and frame_index <= self.last_frame_index
        ):
            self.reset()
        self.scene = scene

        xyz_lidar = np.asarray(
            [det.xyz_lidar_m for det in detections], dtype=np.float64
        ).reshape(-1, 3)
        pose = np.asarray(ego_motion.pose_lidar_to_world, dtype=np.float64)
        xyz_world = (
            pose[:3, :3] @ xyz_lidar.T
        ).T + pose[:3, 3]
        valid = np.all(np.isfinite(xyz_world), axis=1)
        cells = np.zeros((len(detections), 2), dtype=np.int64)
        cells[valid] = np.floor(
            xyz_world[valid, :2] / self.cell_size_m
        ).astype(np.int64)

        support = np.zeros(len(detections), dtype=np.int16)
        for index in np.flatnonzero(valid):
            cell_x, cell_y = cells[index]
            for historic_cells in self.frames:
                if any(
                    (int(cell_x + dx), int(cell_y + dy)) in historic_cells
                    for dx, dy in self.offsets
                ):
                    support[index] += 1

        persistent = valid & (support >= self.min_support)
        current_cells = {
            (int(cell_x), int(cell_y))
            for cell_x, cell_y in cells[valid]
        }
        history_before_append = len(self.frames)
        self.frames.append(current_cells)
        self.last_frame_index = frame_index
        values, counts = np.unique(support, return_counts=True)
        self.last_diagnostics = {
            "method": "world_xy_occupancy_persistence",
            "token": token_text,
            "cell_size_m": self.cell_size_m,
            "history_size": self.history_size,
            "history_frames_available": history_before_append,
            "min_support": self.min_support,
            "dilation_cells": self.dilation_cells,
            "persistent_static_count": int(np.count_nonzero(persistent)),
            "motion_candidate_count": int(
                np.count_nonzero(valid & ~persistent)
            ),
            "support_histogram": {
                str(int(value)): int(count)
                for value, count in zip(values, counts)
            },
        }
        return persistent, support


def build_occupancy_persistence_classifier(args):
    if args.occupancy_persistence == "off":
        return None
    return WorldOccupancyPersistenceClassifier(
        cell_size_m=args.occupancy_cell_size_m,
        history_size=args.occupancy_history_size,
        min_support=args.occupancy_min_support,
        dilation_cells=args.occupancy_dilation_cells,
    )


class RangeDifferenceUnwrapper:
    """Non-Kalman ambiguity resolution from cross-frame cluster displacement."""

    def __init__(
        self, radar_cfg, motion_cfg, frame_dt_s=0.10,
        cluster_radius_m=0.8, max_cluster_extent_m=4.0,
        association_radius_m=1.0, history_size=5, min_cluster_points=2,
        max_speed_mps=40.0, max_doppler_error_mps=0.80,
    ):
        if frame_dt_s <= 0.0:
            raise ValueError("frame_dt_s must be positive")
        if (
            cluster_radius_m <= 0.0
            or max_cluster_extent_m <= 0.0
            or association_radius_m <= 0.0
        ):
            raise ValueError(
                "cluster radius, maximum extent, and association radius "
                "must be positive"
            )
        if history_size < 2:
            raise ValueError("range-difference history must be >= 2")
        if max_speed_mps <= 0.0 or max_doppler_error_mps <= 0.0:
            raise ValueError("speed/error gates must be positive")
        self.radar_cfg = radar_cfg
        self.motion_cfg = motion_cfg
        self.frame_dt_s = float(frame_dt_s)
        self.cluster_radius_m = float(cluster_radius_m)
        self.max_cluster_extent_m = float(max_cluster_extent_m)
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

    def _clusters(self, detections, pose, excluded_mask=None):
        """Build compact patches in a fixed world-coordinate grid.

        Radius connectivity is evaluated only inside one world-anchored patch.
        This removes single-linkage chaining along guardrails while keeping
        patch boundaries stable as the ego vehicle moves.
        """
        xyz_lidar = np.asarray(
            [det.xyz_lidar_m for det in detections], dtype=np.float64
        ).reshape(-1, 3)
        finite = np.all(np.isfinite(xyz_lidar), axis=1)
        if excluded_mask is None:
            excluded = np.zeros(len(detections), dtype=bool)
        else:
            excluded = np.asarray(excluded_mask, dtype=bool)
            if excluded.shape != (len(detections),):
                raise ValueError(
                    "excluded_mask must align with detections"
                )
        valid = np.flatnonzero(finite & ~excluded)
        if not len(valid):
            return []

        xyz_world = (
            pose[:3, :3] @ xyz_lidar.T
        ).T + pose[:3, 3]

        # A square cell with this side length has an XY diagonal no larger
        # than max_cluster_extent_m.
        patch_width = self.max_cluster_extent_m / np.sqrt(2.0)
        patch_keys = np.floor(
            xyz_world[valid, :2] / patch_width
        ).astype(np.int64)
        patches = {}
        for detection_index, key in zip(valid, patch_keys):
            patches.setdefault(tuple(key.tolist()), []).append(
                int(detection_index)
            )

        clusters = []
        for key in sorted(patches):
            member_indices = np.asarray(
                patches[key], dtype=np.int64
            )
            tree = cKDTree(xyz_world[member_indices])
            unseen = set(range(len(member_indices)))

            while unseen:
                seed = min(unseen)
                unseen.remove(seed)
                pending, component = [seed], []
                while pending:
                    local_index = pending.pop()
                    component.append(int(member_indices[local_index]))
                    for neighbour in tree.query_ball_point(
                        xyz_world[member_indices[local_index]],
                        self.cluster_radius_m,
                    ):
                        if neighbour in unseen:
                            unseen.remove(neighbour)
                            pending.append(neighbour)

                if len(component) >= self.min_cluster_points:
                    clusters.append(
                        np.asarray(sorted(component), dtype=np.int64)
                    )
        return clusters

    @staticmethod
    def _track_velocity_world(track):
        """Robust velocity from the median of all pairwise history slopes."""
        history = track["history"]
        if len(history) < 2:
            return None
        times = np.asarray([item[0] for item in history], dtype=np.float64)
        centers = np.asarray([item[1] for item in history], dtype=np.float64)
        slopes = []
        for first in range(len(history) - 1):
            for second in range(first + 1, len(history)):
                dt = float(times[second] - times[first])
                if dt > 1e-9:
                    slopes.append(
                        (centers[second] - centers[first]) / dt
                    )
        if not slopes:
            return None
        return np.median(np.asarray(slopes, dtype=np.float64), axis=0)

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
                raise ValueError(
                    "excluded_mask must align with detections"
                )
        output[excluded] = 0.0
        for index in np.flatnonzero(excluded):
            failure_reason[int(index)] = "occupancy_persistent_static"

        # Optional exclusions are retained for the old A/B view. With the
        # persistent velocity check enabled, all reliable points enter the
        # same displacement tracking and Doppler-unwrapping calculation.
        pose = np.asarray(ego_motion.pose_lidar_to_world, dtype=np.float64)
        clusters = self._clusters(
            detections, pose, excluded_mask=excluded,
        )
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
            dt = max(time - track["history"][-1][0], 0.0)
            velocity_world = self._track_velocity_world(track)
            predicted_center = self._predicted_center(track, time)
            if velocity_world is None:
                gate = self.association_radius_m
            else:
                track_speed = min(
                    float(np.linalg.norm(velocity_world)),
                    self.max_speed_mps,
                )
                gate = self.association_radius_m + track_speed * dt
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
            "cluster_radius_m": self.cluster_radius_m,
            "max_cluster_extent_m": self.max_cluster_extent_m,
            "association_radius_m": self.association_radius_m,
            "history_size": self.history_size,
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
            "occupancy_persistent_static_count": int(
                np.count_nonzero(excluded)
            ),
            "range_difference_resolved_count": int(
                np.count_nonzero(
                    np.asarray(failure_reason, dtype=object) == "resolved"
                )
            ),
            "resolved_count": int(np.count_nonzero(np.isfinite(output))),
            "unresolved_count": int(np.count_nonzero(~np.isfinite(output))),
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
        max_cluster_extent_m=args.unwrap_max_cluster_extent_m,
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
    if unwrapper is None and analysis_summary is None:
        return
    import json
    payload = (
        dict(unwrapper.last_diagnostics)
        if unwrapper is not None
        else {"method": "velocity_unwrapping_disabled"}
    )
    if analysis_summary is not None:
        payload["visualization_velocity_summary"] = analysis_summary
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False),
        encoding="utf-8",
    )


def velocity_source_summary(unwrapper, temporal_residuals):
    diagnostics = unwrapper.last_diagnostics
    temporal = np.asarray(temporal_residuals, dtype=np.float64)
    reasons = np.asarray(diagnostics["failure_reason"], dtype=object)
    reason_counts = {
        str(reason): int(count)
        for reason, count in zip(*np.unique(reasons, return_counts=True))
    }
    return {
        "occupancy_persistent_static_count": int(
            reason_counts.get("occupancy_persistent_static", 0)
        ),
        "range_difference_resolved_count": int(
            reason_counts.get("resolved", 0)
        ),
        "finite_velocity_or_static_count": int(
            np.count_nonzero(np.isfinite(temporal))
        ),
        "unresolved_count": int(
            np.count_nonzero(~np.isfinite(temporal))
        ),
        "range_difference_failure_reasons": reason_counts,
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


def velocity_partition_masks(
    residuals: np.ndarray,
    persistent_mask: np.ndarray,
    threshold_mps: float,
    max_speed_mps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition visualization points after applying velocity to both sources.

    Persistence is a temporal occupancy prior, not a semantic motion label.
    A persistent point is therefore moved to the dynamic display when its
    resolved absolute velocity lies in (threshold_mps, max_speed_mps).
    Unresolved points from either source remain explicitly unknown.
    """
    absolute = np.abs(np.asarray(residuals, dtype=np.float64))
    persistent = np.asarray(persistent_mask, dtype=bool)
    if absolute.shape != persistent.shape:
        raise ValueError("residuals and persistent_mask must align")
    if (
        not np.isfinite(threshold_mps)
        or not np.isfinite(max_speed_mps)
        or threshold_mps < 0.0
        or max_speed_mps <= threshold_mps
    ):
        raise ValueError("Require 0 <= threshold_mps < max_speed_mps.")

    resolved = np.isfinite(absolute) & (absolute < max_speed_mps)
    persistent_static = persistent & resolved & (absolute <= threshold_mps)
    persistent_dynamic = persistent & resolved & (absolute > threshold_mps)
    nonpersistent_static = ~persistent & resolved & (absolute <= threshold_mps)
    nonpersistent_dynamic = ~persistent & resolved & (absolute > threshold_mps)
    dynamic = persistent_dynamic | nonpersistent_dynamic
    unresolved = ~resolved
    return (
        persistent_static,
        persistent_dynamic,
        nonpersistent_static,
        nonpersistent_dynamic,
        dynamic,
        unresolved,
    )


def plot_bev(
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
            alpha=0.45, linewidths=0,
            label="Reliable RPC points in RadarOcc ROI",
            rasterized=True,
        )
    if np.any(dynamic_mask):
        ax.scatter(
            display_x[dynamic_mask], display_y[dynamic_mask],
            s=static_point_size, c="tab:red", alpha=0.88,
            linewidths=0,
            label=r"Resolved dynamic: $\tau_s<|v|<v_{max}$",
            rasterized=True,
        )
    if np.any(unknown_mask):
        ax.scatter(
            display_x[unknown_mask], display_y[unknown_mask],
            s=static_point_size, c="tab:orange", marker="x",
            linewidths=0.7, label="Velocity unresolved (either source)",
            rasterized=True,
        )
    if np.any(motion_static):
        ax.scatter(
            display_x[motion_static], display_y[motion_static],
            s=static_point_size, c="tab:green", alpha=0.95,
            linewidths=0,
            label=r"Non-persistent velocity-static: $|v|\leq\tau_s$",
            rasterized=True,
        )
    if np.any(persistent_static):
        ax.scatter(
            display_x[persistent_static], display_y[persistent_static],
            s=static_point_size, c="tab:blue", alpha=0.95,
            linewidths=0,
            label=r"Persistent and velocity-static: $|v|\leq\tau_s$",
            rasterized=True,
        )

    ax.set_xlim(-grid.max_xyz[1], -grid.min_xyz[1])
    ax.set_ylim(grid.min_xyz[0], grid.max_xyz[0])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    ax.set_xlabel("Lateral [m]   ← vehicle left | vehicle right →")
    ax.set_ylabel("Forward x [m]")
    ax.set_title(
        rf"$\tau_s$ = {threshold:.2f} m/s"
        f"\nStatic {selected}/{total} ({100.0 * ratio:.1f}%)"
        f" | persistent {persistent_static_count} + other {motion_static_count}"
        f"\ndynamic {dynamic_count} (persistent→dynamic {persistent_dynamic_count})"
        f" | unresolved {unresolved_count}"
    )
    ax.scatter([0.0], [0.0], s=50, marker="^", c="black", zorder=5)
    ax.text(0.0, 1.1, "Ego", ha="center", va="bottom", fontsize=8)
    ax.text(
        0.5, 0.985, "Forward ↑", transform=ax.transAxes,
        ha="center", va="top", fontsize=9,
    )
    return (
        selected, total, ratio, persistent_static_count,
        motion_static_count, dynamic_count, persistent_dynamic_count,
        unresolved_count, persistent_unresolved_count,
    )


def main() -> None:
    args = parse_args()

    if len(args.thresholds) != 3:
        raise ValueError(
            "This four-panel visualization expects exactly three thresholds, "
            "for example: --thresholds 0.3 0.5 0.7"
        )
    if any(not np.isfinite(value) or value < 0.0 for value in args.thresholds):
        raise ValueError("Thresholds must be non-negative.")
    if args.range_difference_max_speed_mps <= max(args.thresholds):
        raise ValueError(
            "--range-difference-max-speed-mps must exceed every threshold."
        )

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
    persistence_classifier = build_occupancy_persistence_classifier(args)
    wrapped_residuals_all = doppler_classifier.residuals_with_velocity(
        reliable_detections, ego_motion.linear_velocity_radar_mps,
    )

    # Warm up both temporal branches from the beginning of the scene.
    target_index = pose_reader.frame_index(str(args.token))
    if unwrapper is not None or persistence_classifier is not None:
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
            history_persistent = np.zeros(
                len(history_detections), dtype=bool
            )
            if persistence_classifier is not None:
                history_persistent, _ = persistence_classifier.update(
                    history_detections, history_motion, history_token,
                )
            if unwrapper is not None:
                unwrapper.update(
                    history_detections,
                    history_motion,
                    history_token,
                    excluded_mask=(
                        history_persistent
                        if args.persistent_velocity_check == "off"
                        else None
                    ),
                )

    persistent_static_all = np.zeros(
        len(reliable_detections), dtype=bool
    )
    if persistence_classifier is not None:
        persistent_static_all, _ = persistence_classifier.update(
            reliable_detections, ego_motion, str(args.token),
        )

    if unwrapper is None:
        residuals_all = wrapped_residuals_all.copy()
        if args.persistent_velocity_check == "off":
            residuals_all[persistent_static_all] = 0.0
    else:
        temporal_residuals_all, _ = unwrapper.update(
            reliable_detections,
            ego_motion,
            str(args.token),
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
            np.count_nonzero(persistent_static_all & ~persistent_velocity_resolved)
        ),
        "max_speed_mps": float(args.range_difference_max_speed_mps),
    }

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
    persistent_static = persistent_static_all[in_roi]
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

    summaries = []
    for ax, threshold in zip(axes[:3], args.thresholds):
        stats = plot_bev(
            ax=ax,
            xy=xy,
            residuals=residuals,
            persistent_static_mask=persistent_static,
            threshold=float(threshold),
            max_speed_mps=args.range_difference_max_speed_mps,
            grid=grid_cfg,
            all_point_size=args.all_point_size,
            static_point_size=args.static_point_size,
        )
        summaries.append((float(threshold), *stats))

    # One legend is enough because all three BEVs use exactly the same encoding.
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[3].imshow(image)
    axes[3].set_title(f"RGB reference\n{camera_path.name}")
    axes[3].axis("off")

    velocity = ego_motion.linear_velocity_radar_mps
    speed = float(np.linalg.norm(velocity[:2]))
    fig.suptitle(
        f"Static threshold comparison | velocity mode: {args.velocity_unwrapping}"
        f" | persistent velocity check: {args.persistent_velocity_check}"
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
    print(f"  persistent velocity check: {args.persistent_velocity_check}")
    print(
        "  occupancy persistent ROI: "
        f"{int(np.count_nonzero(persistent_static))}"
    )
    print(
        "  unresolved velocity ROI: "
        f"{int(np.count_nonzero(~np.isfinite(residuals)))}"
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
    for (
        threshold, selected, total, ratio, persistent_static_count,
        motion_static_count, dynamic_count, persistent_dynamic_count,
        unresolved_count, persistent_unresolved_count,
    ) in summaries:
        print(
            f"  tau={threshold:.2f}: static={selected}/{total} "
            f"({100.0 * ratio:.1f}%) "
            f"| persistent_static={persistent_static_count} "
            f"| nonpersistent_static={motion_static_count} "
            f"| dynamic={dynamic_count} "
            f"(persistent_to_dynamic={persistent_dynamic_count}) "
            f"| unresolved={unresolved_count} "
            f"(persistent_unresolved={persistent_unresolved_count})"
        )
    print(f"  output      : {save_path}")


if __name__ == "__main__":
    main()
