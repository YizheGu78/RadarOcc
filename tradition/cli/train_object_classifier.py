from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from tradition.core.config import (
    GridConfig,
    KRadarConfig,
    MotionConfig,
    ObjectClusteringConfig,
    PoseConfig,
    ReliabilityConfig,
    TemporalConfig,
)
from tradition.core.types import MotionLabel, TemporalDetectionFrame
from tradition.detection.rpc_reliability_filter import LocalPowerRPCFilter
from tradition.detection.rpc_target_detector import RPCPointTargetDetector
from tradition.evaluation.radarocc_metrics import load_gt_sparse_xyz
from tradition.experiment.dataset_runner import _load_infos, _resolve_gt, _resolve_rpc_radar
from tradition.io.pose_reader import RadarOccPoseReader
from tradition.io.rpc_radar_reader import KRadarRPCReader
from tradition.motion.doppler_classifier import EgoCompensatedDopplerClassifier
from tradition.motion.pose_ego_motion import PoseEgoMotionEstimator
from tradition.motion.temporal_consistency import PoseAlignedTemporalClassifier
from tradition.semantics.object_classifier import DualBranchCandidateExtractor
from tradition.semantics.training import (
    ClusterLabellingConfig,
    RadarOccClusterLabeller,
    train_random_forest_objectness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the classical dual-branch cluster Random Forest from "
            "RadarOcc training GT. GT is used only here, never at inference."
        )
    )
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--gt-root", type=Path)
    parser.add_argument("--gt-order", choices=("xyz", "zyx"), default="xyz")
    parser.add_argument("--scene", action="append", help="Repeat to select scenes.")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--pose-dt-s", type=float, default=0.10)
    parser.add_argument("--static-residual-threshold-mps", type=float, default=0.50)
    parser.add_argument("--dynamic-residual-threshold-mps", type=float, default=0.80)
    parser.add_argument(
        "--stationary-velocity-sign", type=float, choices=(-1.0, 1.0), default=1.0
    )
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--min-static-support", type=int, default=2)
    parser.add_argument("--min-dynamic-support", type=int, default=2)
    parser.add_argument("--static-match-radius-m", type=float, default=0.60)
    parser.add_argument("--dynamic-match-radius-m", type=float, default=2.00)
    parser.add_argument("--min-local-power-ratio", type=float, default=0.25)
    parser.add_argument("--min-local-neighbors", type=int, default=1)
    parser.add_argument("--static-object-cell-size-m", type=float, default=0.40)
    parser.add_argument("--static-object-dilation-cells", type=int, default=1)
    parser.add_argument("--static-object-min-points", type=int, default=3)
    parser.add_argument("--static-object-min-cells", type=int, default=2)
    parser.add_argument("--dynamic-object-eps-xy-m", type=float, default=2.00)
    parser.add_argument("--dynamic-object-eps-z-m", type=float, default=1.00)
    parser.add_argument("--dynamic-object-min-points", type=int, default=2)
    parser.add_argument("--gt-match-radius-voxels", type=int, default=2)
    parser.add_argument("--positive-foreground-fraction", type=float, default=0.20)
    parser.add_argument("--negative-foreground-fraction", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=13)
    return parser.parse_args()


def _ego_motion(
    pose_reader: RadarOccPoseReader,
    estimator: PoseEgoMotionEstimator,
    scene: str,
    token: str,
):
    index = pose_reader.frame_index(token)
    current, _ = pose_reader.read_index(scene, index)
    previous = pose_reader.try_read_index(scene, index - 1)
    following = pose_reader.try_read_index(scene, index + 1) if previous is None else None
    return estimator.estimate(
        current_pose=current,
        previous_pose=previous[0] if previous is not None else None,
        next_pose=following[0] if following is not None else None,
    )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    radar_root = args.radar_root.expanduser().resolve()
    gt_root = args.gt_root.expanduser().resolve() if args.gt_root else None
    motion_cfg = MotionConfig(
        static_residual_threshold_mps=args.static_residual_threshold_mps,
        dynamic_residual_threshold_mps=args.dynamic_residual_threshold_mps,
        stationary_velocity_sign=args.stationary_velocity_sign,
    )
    temporal_cfg = TemporalConfig(
        window_size=args.temporal_window,
        min_static_support=args.min_static_support,
        min_dynamic_support=args.min_dynamic_support,
        static_match_radius_m=args.static_match_radius_m,
        dynamic_match_radius_m=args.dynamic_match_radius_m,
    )
    object_cfg = ObjectClusteringConfig(
        static_cell_size_m=args.static_object_cell_size_m,
        static_dilation_cells=args.static_object_dilation_cells,
        static_min_points=args.static_object_min_points,
        static_min_cells=args.static_object_min_cells,
        dynamic_eps_xy_m=args.dynamic_object_eps_xy_m,
        dynamic_eps_z_m=args.dynamic_object_eps_z_m,
        dynamic_min_points=args.dynamic_object_min_points,
    )

    reader = KRadarRPCReader()
    detector = RPCPointTargetDetector()
    reliability = LocalPowerRPCFilter(
        ReliabilityConfig(
            min_local_power_ratio=args.min_local_power_ratio,
            min_local_neighbors=args.min_local_neighbors,
        )
    )
    doppler = EgoCompensatedDopplerClassifier(motion_cfg=motion_cfg)
    temporal = PoseAlignedTemporalClassifier(temporal_cfg, motion_cfg)
    pose_reader = RadarOccPoseReader(args.pose_root)
    pose_estimator = PoseEgoMotionEstimator(
        radar_cfg=KRadarConfig(),
        pose_cfg=PoseConfig(frame_dt_s=args.pose_dt_s),
    )
    candidates = DualBranchCandidateExtractor(object_cfg)
    print(f"Object feature schema: {len(candidates.features.feature_names)} dimensions")
    labeller = RadarOccClusterLabeller(
        GridConfig(),
        ClusterLabellingConfig(
            match_radius_voxels=args.gt_match_radius_voxels,
            positive_foreground_fraction=args.positive_foreground_fraction,
            negative_foreground_fraction=args.negative_foreground_fraction,
        ),
    )

    infos = _load_infos(args.annotation.expanduser().resolve())
    if args.scene:
        wanted = {str(scene) for scene in args.scene}
        infos = [info for info in infos if str(info.get("scene_token")) in wanted]
    if args.max_frames is not None:
        infos = infos[: args.max_frames]
    if not infos:
        raise RuntimeError("No training frames selected.")

    x: list[np.ndarray] = []
    y: list[int] = []
    ignored = 0
    active_scene = None
    temporal.reset()
    for ordinal, info in enumerate(infos, start=1):
        scene = str(info.get("scene_token"))
        token = str(info["lidar_token"])
        if scene != active_scene:
            temporal.reset()
            active_scene = scene
        radar_path = _resolve_rpc_radar(info, repo_root, radar_root)
        gt_path = _resolve_gt(info, repo_root, gt_root)
        ego = _ego_motion(pose_reader, pose_estimator, scene, token)
        raw = detector.detect(reader.read(radar_path))
        reliable = reliability.filter(raw)
        residuals, evidence = doppler.evidence_with_velocity(
            reliable, ego.linear_velocity_radar_mps
        )
        result = temporal.update(
            TemporalDetectionFrame(
                token=token,
                pose_lidar_to_world=ego.pose_lidar_to_world,
                detections=reliable,
                doppler_residuals_mps=residuals,
                doppler_evidence=evidence,
            )
        )
        current_features = [
            replace(
                reliable[int(index)],
                radial_velocity_mps=float(residuals[int(index)]),
            )
            for index in result.current_indices
        ]
        combined = current_features + result.historic_detections
        motion = result.current_motion_labels + [MotionLabel.STATIC] * len(
            result.historic_detections
        )
        gt = load_gt_sparse_xyz(gt_path, coordinate_order=args.gt_order)
        frame_candidates = candidates.extract(combined, motion)
        for candidate in frame_candidates:
            target = labeller.label(candidate, combined, gt)
            if target is None:
                ignored += 1
                continue
            x.append(candidate.features)
            y.append(target)
        print(
            f"[{ordinal}/{len(infos)}] scene={scene} token={token} "
            f"raw={len(raw)} reliable={len(reliable)} "
            f"candidates={len(frame_candidates)} samples={len(y)} ignored={ignored}"
        )

    model_path = train_random_forest_objectness(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.int8),
        args.output_model,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        metadata={
            "annotation": str(args.annotation.expanduser().resolve()),
            "frame_count": len(infos),
            "ignored_ambiguous_candidates": ignored,
            "temporal_window": args.temporal_window,
            "positive_foreground_fraction": args.positive_foreground_fraction,
            "negative_foreground_fraction": args.negative_foreground_fraction,
        },
    )
    print(f"Saved object classifier: {model_path}")


if __name__ == "__main__":
    main()
