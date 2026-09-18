#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tradition.core.config import KRadarConfig, PoseConfig
from tradition.io.pose_reader import RadarOccPoseReader
from tradition.motion.pose_ego_motion import PoseEgoMotionEstimator


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RadarOcc pose-based ego speed with official "
            "K-Radar gt_rel odometry speed. Both speeds are converted "
            "to radar-origin horizontal speed, including yaw-rate "
            "lever-arm correction."
        )
    )

    parser.add_argument(
        "--scene",
        type=str,
        default="3",
        help="K-Radar scene number, e.g. 3.",
    )

    parser.add_argument(
        "--pose-root",
        type=Path,
        default=Path("data/K-RadarOcc"),
        help="RadarOcc pose root.",
    )

    parser.add_argument(
        "--odometry-root",
        type=Path,
        default=Path(
            "data/K-Radar-official-meta/resources/odometry"
        ),
        help="Official K-Radar resources/odometry directory.",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.10,
        help="Frame interval in seconds.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Default: "
            "work_dirs/pose_odometry_speed_scene<scene>.csv"
        ),
    )

    return parser.parse_args()


# ============================================================
# General helpers
# ============================================================

def resolve_path(
    path: Path,
    repo_root: Path,
) -> Path:
    path = path.expanduser()

    if path.is_absolute():
        return path.resolve()

    return (repo_root / path).resolve()


def wrap_angle(angle_rad: float) -> float:
    """
    Wrap angle to [-pi, pi).
    """
    return float(
        (angle_rad + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )


def yaw_from_rotation(
    rotation: np.ndarray,
) -> float:
    """
    Same yaw convention as the current PoseEgoMotionEstimator:

        yaw = atan2(R[1,0], R[0,0])
    """
    return float(
        np.arctan2(
            rotation[1, 0],
            rotation[0, 0],
        )
    )


# ============================================================
# Quaternion
# ============================================================

def quaternion_to_rotation_matrix(
    quaternion: np.ndarray,
) -> np.ndarray:
    """
    Convert:

        [qw, qx, qy, qz]

    to 3x3 rotation matrix.
    """

    qw, qx, qy, qz = map(
        float,
        quaternion,
    )

    norm = np.sqrt(
        qw * qw
        + qx * qx
        + qy * qy
        + qz * qz
    )

    if norm <= 1e-12:
        raise ValueError(
            "Invalid zero-norm quaternion."
        )

    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm

    rotation = np.array(
        [
            [
                1.0 - 2.0 * (qy*qy + qz*qz),
                2.0 * (qx*qy - qw*qz),
                2.0 * (qx*qz + qw*qy),
            ],
            [
                2.0 * (qx*qy + qw*qz),
                1.0 - 2.0 * (qx*qx + qz*qz),
                2.0 * (qy*qz - qw*qx),
            ],
            [
                2.0 * (qx*qz - qw*qy),
                2.0 * (qy*qz + qw*qx),
                1.0 - 2.0 * (qx*qx + qy*qy),
            ],
        ],
        dtype=np.float64,
    )

    return rotation


# ============================================================
# Official K-Radar odometry
# ============================================================

def load_official_gt_rel(
    odometry_root: Path,
    scene: str,
) -> tuple[np.ndarray, Path]:
    """
    Load:

        resources/odometry/gt_rel/03.txt

    Format per row:

        qw qx qy qz tx ty tz
    """

    path = (
        odometry_root
        / "gt_rel"
        / f"{int(scene):02d}.txt"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Official gt_rel not found: {path}"
        )

    data = np.loadtxt(
        path,
        dtype=np.float64,
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 7:
        raise ValueError(
            "Expected gt_rel shape [N, 7] "
            "[qw qx qy qz tx ty tz], "
            f"but got {data.shape}"
        )

    return data, path


def reconstruct_absolute_rotations(
    odom_rel: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct absolute orientation from relative quaternions.

    gt_rel[0] is normally identity.

    If:

        R_rel_i = R_(i-1)^T R_i

    then:

        R_i = R_(i-1) R_rel_i
    """

    num_frames = len(odom_rel)

    rotations = np.zeros(
        (num_frames, 3, 3),
        dtype=np.float64,
    )

    rotations[0] = np.eye(
        3,
        dtype=np.float64,
    )

    for i in range(1, num_frames):

        relative_rotation = (
            quaternion_to_rotation_matrix(
                odom_rel[i, :4]
            )
        )

        rotations[i] = (
            rotations[i - 1]
            @ relative_rotation
        )

    return rotations


def calculate_odometry_radar_speeds(
    odom_rel: np.ndarray,
    dt: float,
    radar_cfg: KRadarConfig,
    pose_cfg: PoseConfig,
) -> np.ndarray:
    """
    Convert official K-Radar gt_rel to radar-origin speed.

    We intentionally match the current PoseEgoMotionEstimator logic:

        1. translation velocity
        2. yaw rate
        3. omega x radar_lever_arm
        4. LiDAR -> radar frame
        5. horizontal speed sqrt(vx^2 + vy^2)


    Important frame convention
    --------------------------

    gt_rel[i] describes:

        frame i-1 -> frame i

    Its translation t_rel is expressed in frame i-1.

    For i > 0, the current traditional Pose estimator uses
    a BACKWARD difference and expresses velocity in CURRENT
    frame i.

    Therefore:

        v_current =
            R_rel^T * t_rel / dt


    For frame 0, the Pose estimator has no previous frame
    and uses a FORWARD fallback:

        frame 0 -> frame 1

    gt_rel[1] translation is already expressed in frame 0,
    so frame 0 uses:

        v_frame0 =
            t_rel[1] / dt
    """

    num_frames = len(odom_rel)

    speeds = np.full(
        num_frames,
        np.nan,
        dtype=np.float64,
    )

    if num_frames == 0:
        return speeds

    absolute_rotations = (
        reconstruct_absolute_rotations(
            odom_rel
        )
    )

    radar_lever_arm_lidar = np.asarray(
        radar_cfg.radar_to_lidar_translation_xyz_m,
        dtype=np.float64,
    )

    rotation_lidar_radar = np.asarray(
        pose_cfg.radar_to_lidar_rotation,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Frame 0:
    # match PoseEgoMotionEstimator forward fallback
    # --------------------------------------------------------

    if num_frames >= 2:

        relative_row = odom_rel[1]

        relative_rotation = (
            quaternion_to_rotation_matrix(
                relative_row[:4]
            )
        )

        translation_frame0 = np.asarray(
            relative_row[4:7],
            dtype=np.float64,
        )

        # gt_rel[1] translation is already in frame 0.
        velocity_lidar = (
            translation_frame0 / dt
        )

        yaw_0 = yaw_from_rotation(
            absolute_rotations[0]
        )

        yaw_1 = yaw_from_rotation(
            absolute_rotations[1]
        )

        yaw_delta = wrap_angle(
            yaw_1 - yaw_0
        )

        yaw_rate = (
            yaw_delta / dt
        )

        omega_lidar = np.array(
            [
                0.0,
                0.0,
                yaw_rate,
            ],
            dtype=np.float64,
        )

        radar_origin_velocity_lidar = (
            velocity_lidar
            + np.cross(
                omega_lidar,
                radar_lever_arm_lidar,
            )
        )

        velocity_radar = (
            rotation_lidar_radar.T
            @ radar_origin_velocity_lidar
        )

        speeds[0] = np.hypot(
            velocity_radar[0],
            velocity_radar[1],
        )

    # --------------------------------------------------------
    # Frame i > 0:
    # match PoseEgoMotionEstimator backward difference
    # --------------------------------------------------------

    for i in range(1, num_frames):

        row = odom_rel[i]

        quaternion = row[:4]

        translation_previous_frame = np.asarray(
            row[4:7],
            dtype=np.float64,
        )

        relative_rotation = (
            quaternion_to_rotation_matrix(
                quaternion
            )
        )

        # ----------------------------------------------------
        # Official relative translation is expressed in
        # PREVIOUS LiDAR frame.
        #
        # Current Pose estimator expresses velocity in CURRENT
        # LiDAR frame.
        #
        # Transform previous-frame vector into current frame.
        # ----------------------------------------------------

        translation_current_frame = (
            relative_rotation.T
            @ translation_previous_frame
        )

        velocity_lidar = (
            translation_current_frame / dt
        )

        # ----------------------------------------------------
        # Yaw rate
        # ----------------------------------------------------

        yaw_previous = yaw_from_rotation(
            absolute_rotations[i - 1]
        )

        yaw_current = yaw_from_rotation(
            absolute_rotations[i]
        )

        yaw_delta = wrap_angle(
            yaw_current
            - yaw_previous
        )

        yaw_rate = (
            yaw_delta / dt
        )

        # Same approximation as the current pose code:
        # use only yaw angular velocity.
        omega_lidar = np.array(
            [
                0.0,
                0.0,
                yaw_rate,
            ],
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # Radar lever-arm velocity:
        #
        #       v_radar = v_lidar + omega x r
        #
        # ----------------------------------------------------

        radar_origin_velocity_lidar = (
            velocity_lidar
            + np.cross(
                omega_lidar,
                radar_lever_arm_lidar,
            )
        )

        # ----------------------------------------------------
        # LiDAR frame -> Radar frame
        # ----------------------------------------------------

        velocity_radar = (
            rotation_lidar_radar.T
            @ radar_origin_velocity_lidar
        )

        # Horizontal radar-origin speed
        speeds[i] = np.hypot(
            velocity_radar[0],
            velocity_radar[1],
        )

    return speeds


# ============================================================
# RadarOcc pose speed
# ============================================================

def calculate_pose_speed(
    scene: str,
    frame_index: int,
    pose_reader: RadarOccPoseReader,
    estimator: PoseEgoMotionEstimator,
    dt: float,
) -> float:
    """
    Use the CURRENT traditional pipeline's PoseEgoMotionEstimator.

    frame > 0:
        previous -> current

    frame 0:
        current -> next fallback
    """

    current_item = (
        pose_reader.try_read_index(
            scene,
            frame_index,
        )
    )

    if current_item is None:
        return np.nan

    current_pose = current_item[0]

    previous_item = (
        pose_reader.try_read_index(
            scene,
            frame_index - 1,
        )
    )

    next_item = (
        pose_reader.try_read_index(
            scene,
            frame_index + 1,
        )
    )

    previous_pose = (
        previous_item[0]
        if previous_item is not None
        else None
    )

    next_pose = (
        next_item[0]
        if next_item is not None
        else None
    )

    if (
        previous_pose is None
        and next_pose is None
    ):
        return np.nan

    motion = estimator.estimate(
        current_pose=current_pose,
        previous_pose=previous_pose,
        next_pose=next_pose,
        previous_dt_s=(
            dt
            if previous_pose is not None
            else None
        ),
        next_dt_s=(
            dt
            if (
                previous_pose is None
                and next_pose is not None
            )
            else None
        ),
    )

    velocity_radar = np.asarray(
        motion.linear_velocity_radar_mps,
        dtype=np.float64,
    )

    return float(
        np.hypot(
            velocity_radar[0],
            velocity_radar[1],
        )
    )


def count_pose_files(
    pose_reader: RadarOccPoseReader,
    scene: str,
    max_search: int = 10000,
) -> int:
    """
    Count consecutive pose indices beginning from 0.
    """

    count = 0

    for frame_index in range(
        max_search
    ):
        item = (
            pose_reader.try_read_index(
                scene,
                frame_index,
            )
        )

        if item is None:
            break

        count += 1

    return count


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    if args.dt <= 0.0:
        raise ValueError(
            f"--dt must be > 0, got {args.dt}"
        )

    repo_root = Path.cwd().resolve()

    pose_root = resolve_path(
        args.pose_root,
        repo_root,
    )

    odometry_root = resolve_path(
        args.odometry_root,
        repo_root,
    )

    if args.output is None:
        output_path = (
            repo_root
            / "work_dirs"
            / (
                f"pose_odometry_speed_"
                f"scene{args.scene}.csv"
            )
        )
    else:
        output_path = resolve_path(
            args.output,
            repo_root,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "===================================================="
    )
    print(
        "Pose vs K-Radar Official Odometry Radar Speed"
    )
    print(
        "===================================================="
    )

    print(
        f"Scene         : {args.scene}"
    )

    print(
        f"Pose root     : {pose_root}"
    )

    print(
        f"Odometry root : {odometry_root}"
    )

    print(
        f"dt            : {args.dt:.3f} s"
    )

    print()

    # ========================================================
    # Official odometry
    # ========================================================

    odom_rel, odometry_path = (
        load_official_gt_rel(
            odometry_root=odometry_root,
            scene=args.scene,
        )
    )

    print(
        f"Official file : {odometry_path}"
    )

    print(
        f"Odometry rows : {len(odom_rel)}"
    )

    # ========================================================
    # Current Traditional pose estimator
    # ========================================================

    radar_cfg = KRadarConfig()

    pose_cfg = PoseConfig(
        frame_dt_s=args.dt,
    )

    pose_reader = RadarOccPoseReader(
        pose_root
    )

    estimator = PoseEgoMotionEstimator(
        radar_cfg=radar_cfg,
        pose_cfg=pose_cfg,
    )

    pose_count = count_pose_files(
        pose_reader=pose_reader,
        scene=args.scene,
    )

    print(
        f"Pose frames   : {pose_count}"
    )

    print()

    # ========================================================
    # Official odometry -> radar-origin speed
    # ========================================================

    odometry_speeds = (
        calculate_odometry_radar_speeds(
            odom_rel=odom_rel,
            dt=args.dt,
            radar_cfg=radar_cfg,
            pose_cfg=pose_cfg,
        )
    )

    # ========================================================
    # Compare same frame index
    # ========================================================

    num_frames = max(
        pose_count,
        len(odometry_speeds),
    )

    if (
        pose_count
        != len(odometry_speeds)
    ):
        print(
            "WARNING:"
        )
        print(
            "  Pose frame count and official "
            "odometry row count differ."
        )
        print(
            "  Same indices are still compared."
        )
        print(
            "  Missing entries are written as nan."
        )
        print()

    rows: list[
        tuple[float, float]
    ] = []

    for frame_index in range(
        num_frames
    ):

        # ----------------------------------------
        # Pose method
        # ----------------------------------------

        if frame_index < pose_count:

            pose_speed = (
                calculate_pose_speed(
                    scene=args.scene,
                    frame_index=frame_index,
                    pose_reader=pose_reader,
                    estimator=estimator,
                    dt=args.dt,
                )
            )

        else:

            pose_speed = np.nan

        # ----------------------------------------
        # Official odometry method
        # ----------------------------------------

        if (
            frame_index
            < len(odometry_speeds)
        ):

            odometry_speed = float(
                odometry_speeds[
                    frame_index
                ]
            )

        else:

            odometry_speed = np.nan

        rows.append(
            (
                pose_speed,
                odometry_speed,
            )
        )

        if frame_index < 20:

            pose_text = (
                f"{pose_speed:8.3f}"
                if np.isfinite(
                    pose_speed
                )
                else "     nan"
            )

            odom_text = (
                f"{odometry_speed:8.3f}"
                if np.isfinite(
                    odometry_speed
                )
                else "     nan"
            )

            print(
                f"frame {frame_index:04d} | "
                f"pose = {pose_text} m/s | "
                f"odometry = {odom_text} m/s"
            )

    # ========================================================
    # CSV
    # ========================================================

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        # Exactly two columns
        writer.writerow(
            [
                "pose_speed_mps",
                "odometry_speed_mps",
            ]
        )

        for (
            pose_speed,
            odometry_speed,
        ) in rows:

            pose_value = (
                f"{pose_speed:.6f}"
                if np.isfinite(
                    pose_speed
                )
                else "nan"
            )

            odometry_value = (
                f"{odometry_speed:.6f}"
                if np.isfinite(
                    odometry_speed
                )
                else "nan"
            )

            writer.writerow(
                [
                    pose_value,
                    odometry_value,
                ]
            )

    # ========================================================
    # Statistics
    # ========================================================

    pose_array = np.asarray(
        [
            row[0]
            for row in rows
        ],
        dtype=np.float64,
    )

    odometry_array = np.asarray(
        [
            row[1]
            for row in rows
        ],
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            pose_array
        )
        & np.isfinite(
            odometry_array
        )
    )

    print()
    print(
        "===================================================="
    )
    print(
        "Result"
    )
    print(
        "===================================================="
    )

    print(
        f"CSV saved to : {output_path}"
    )

    print(
        f"Total rows   : {len(rows)}"
    )

    print(
        f"Valid pairs  : "
        f"{np.count_nonzero(valid)}"
    )

    if np.any(valid):

        pose_valid = (
            pose_array[valid]
        )

        odom_valid = (
            odometry_array[valid]
        )

        difference = (
            pose_valid
            - odom_valid
        )

        abs_difference = (
            np.abs(
                difference
            )
        )

        print()

        print(
            "Statistics:"
        )

        print(
            "  Mean pose speed       : "
            f"{np.mean(pose_valid):.3f} m/s"
        )

        print(
            "  Mean odometry speed   : "
            f"{np.mean(odom_valid):.3f} m/s"
        )

        print(
            "  Mean signed difference: "
            f"{np.mean(difference):+.3f} m/s"
        )

        print(
            "  MAE                    : "
            f"{np.mean(abs_difference):.3f} m/s"
        )

        print(
            "  Median abs difference : "
            f"{np.median(abs_difference):.3f} m/s"
        )

        print(
            "  Max abs difference    : "
            f"{np.max(abs_difference):.3f} m/s"
        )

        if len(
            pose_valid
        ) >= 2:

            correlation = (
                np.corrcoef(
                    pose_valid,
                    odom_valid,
                )[0, 1]
            )

            print(
                "  Correlation            : "
                f"{correlation:.4f}"
            )

    print()
    print(
        "Definitions:"
    )

    print(
        "  pose_speed_mps:"
    )

    print(
        "    Current Traditional "
        "PoseEgoMotionEstimator radar-origin "
        "horizontal speed."
    )

    print(
        "  odometry_speed_mps:"
    )

    print(
        "    Official K-Radar gt_rel translation "
        "+ quaternion yaw rotation "
        "+ radar lever-arm correction."
    )

    print()

    print(
        "Both columns finally use:"
    )

    print(
        "  sqrt(v_radar_x^2 + v_radar_y^2)"
    )


if __name__ == "__main__":
    main()