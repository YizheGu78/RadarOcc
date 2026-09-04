from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from tradition.core.config import (
    CFARConfig,
    EgoSpeedConfig,
    GridConfig,
    KRadarConfig,
    MappingConfig,
    MotionConfig,
    ObjectClusteringConfig,
    ReliabilityConfig,
    TemporalConfig,
)
from tradition.core.interfaces import (
    DetectionReliabilityFilter,
    EgoSpeedEstimator,
    MotionClassifier,
    OccupancyMapper,
    PoseAwareDopplerClassifier,
    PredictionWriter,
    RadarMeasurementReader,
    SemanticClassifier,
    TargetDetector,
    TemporalMotionClassifier,
    TemporalOccupancyMapper,
)
from tradition.core.types import (
    DopplerEvidence,
    EgoMotion,
    FramePrediction,
    MotionLabel,
    TemporalDetectionFrame,
)
from tradition.detection.cfar import NumpyCACFAR, OpenRadarCACFAR
from tradition.detection.rpc_target_detector import RPCPointTargetDetector
from tradition.detection.rpc_reliability_filter import LocalPowerRPCFilter
from tradition.detection.target_detector import ClassicalTargetDetector
from tradition.io.kradar_reader import KRadarTensorReader
from tradition.io.radarocc_writer import RadarOccPredictionWriter
from tradition.io.rpc_radar_reader import KRadarRPCReader
from tradition.mapping.occupancy_grid_3d import LogOddsOccupancyGrid3D
from tradition.mapping.temporal_occupancy_grid_3d import (
    TemporalLogOddsOccupancyGrid3D,
)
from tradition.motion.doppler_classifier import EgoCompensatedDopplerClassifier
from tradition.motion.ego_speed_estimator import RobustDopplerEgoSpeedEstimator
from tradition.motion.temporal_consistency import PoseAlignedTemporalClassifier
from tradition.semantics.classical_classifier import DopplerSemanticClassifier
from tradition.semantics.object_classifier import ObjectAwareSemanticClassifier


class TraditionalRadarPipeline:
    """Dependency-injected orchestration for a non-learning radar baseline."""

    def __init__(
        self,
        reader: RadarMeasurementReader,
        detector: TargetDetector,
        ego_speed_estimator: EgoSpeedEstimator,
        motion_classifier: MotionClassifier,
        semantic_classifier: SemanticClassifier,
        mapper: OccupancyMapper,
        writer: PredictionWriter,
        reliability_filter: DetectionReliabilityFilter | None = None,
        temporal_classifier: TemporalMotionClassifier | None = None,
    ) -> None:
        self.reader = reader
        self.detector = detector
        self.ego_speed_estimator = ego_speed_estimator
        self.motion_classifier = motion_classifier
        self.semantic_classifier = semantic_classifier
        self.mapper = mapper
        self.writer = writer
        self.reliability_filter = reliability_filter
        self.temporal_classifier = temporal_classifier

    def reset_sequence(self) -> None:
        if self.temporal_classifier is not None:
            self.temporal_classifier.reset()

    def predict_measurement(
        self,
        measurement: Any,
        ego_speed_mps: float | None = None,
    ) -> FramePrediction:
        detections = self.detector.detect(measurement)
        resolved_ego_speed_mps = (
            self.ego_speed_estimator.estimate(detections)
            if ego_speed_mps is None
            else float(ego_speed_mps)
        )
        motion_labels = self.motion_classifier.classify(
            detections, ego_speed_mps=resolved_ego_speed_mps
        )
        semantic_labels = self.semantic_classifier.classify(
            detections, motion_labels
        )
        self.mapper.reset()
        self.mapper.update(detections, semantic_labels)
        dense = self.mapper.labels()
        return FramePrediction(
            dense_labels_xyz=dense,
            detections=detections,
            motion_labels=motion_labels,
            semantic_labels=semantic_labels,
            metadata={
                "ego_speed_mps": resolved_ego_speed_mps,
                "ego_speed_source": (
                    "doppler_auto" if ego_speed_mps is None else "fixed"
                ),
                "ego_speed_estimator": type(self.ego_speed_estimator).__name__,
                "reader": type(self.reader).__name__,
                "detector": type(self.detector).__name__,
                "motion_classifier": type(self.motion_classifier).__name__,
                "semantic_classifier": type(self.semantic_classifier).__name__,
            },
        )

    def predict_tensor(
        self,
        radar_tensor_drea: np.ndarray,
        ego_speed_mps: float | None = None,
    ) -> FramePrediction:
        """Backward-compatible raw-tensor entry point."""
        return self.predict_measurement(
            radar_tensor_drea,
            ego_speed_mps=ego_speed_mps,
        )

    def predict_file(
        self,
        radar_path: str | Path,
        ego_speed_mps: float | None = None,
    ) -> FramePrediction:
        measurement = self.reader.read(radar_path)
        prediction = self.predict_measurement(
            measurement,
            ego_speed_mps=ego_speed_mps,
        )
        prediction.metadata["radar_path"] = str(radar_path)
        return prediction

    def predict_temporal_file(
        self,
        radar_path: str | Path,
        token: str,
        ego_motion: EgoMotion,
    ) -> FramePrediction:
        """Predict one RPC frame with pose motion and causal temporal evidence."""
        if self.reliability_filter is None or self.temporal_classifier is None:
            raise RuntimeError("This pipeline was not built for temporal RPC input.")
        if not isinstance(self.motion_classifier, PoseAwareDopplerClassifier):
            raise TypeError("Temporal RPC requires a pose-aware Doppler classifier.")
        if not isinstance(self.mapper, TemporalOccupancyMapper):
            raise TypeError("Temporal RPC requires a temporal occupancy mapper.")

        measurement = self.reader.read(radar_path)
        raw_detections = self.detector.detect(measurement)
        reliable_detections = self.reliability_filter.filter(raw_detections)
        residuals, evidence = self.motion_classifier.evidence_with_velocity(
            reliable_detections,
            ego_motion.linear_velocity_radar_mps,
        )
        temporal_frame = TemporalDetectionFrame(
            token=str(token),
            pose_lidar_to_world=ego_motion.pose_lidar_to_world,
            detections=reliable_detections,
            doppler_residuals_mps=residuals,
            doppler_evidence=evidence,
        )
        classification = self.temporal_classifier.update(temporal_frame)
        temporal_accepted_detections = [
            reliable_detections[int(index)]
            for index in classification.current_indices
        ]
        temporal_feature_detections = [
            replace(
                reliable_detections[int(index)],
                radial_velocity_mps=float(residuals[int(index)]),
            )
            for index in classification.current_indices
        ]
        temporal_motion_labels = classification.current_motion_labels
        historic_detections = classification.historic_detections
        combined_detections = temporal_feature_detections + historic_detections
        combined_motion_labels = temporal_motion_labels + [
            MotionLabel.STATIC
        ] * len(historic_detections)

        classifier_with_acceptance = getattr(
            self.semantic_classifier, "classify_with_acceptance", None
        )
        if classifier_with_acceptance is None:
            combined_semantic_labels = self.semantic_classifier.classify(
                combined_detections, combined_motion_labels
            )
            accepted_mask = np.ones(len(combined_detections), dtype=bool)
        else:
            combined_semantic_labels, accepted_mask = classifier_with_acceptance(
                combined_detections, combined_motion_labels
            )

        current_count = len(temporal_accepted_detections)
        current_mask = accepted_mask[:current_count]
        historic_mask = accepted_mask[current_count:]
        accepted_detections = [
            detection
            for detection, keep in zip(temporal_accepted_detections, current_mask)
            if keep
        ]
        motion_labels = [
            label
            for label, keep in zip(temporal_motion_labels, current_mask)
            if keep
        ]
        semantic_labels = [
            label
            for label, keep in zip(
                combined_semantic_labels[:current_count], current_mask
            )
            if keep
        ]
        accepted_historic_detections = [
            detection
            for detection, keep in zip(historic_detections, historic_mask)
            if keep
        ]
        historic_semantic_labels = [
            label
            for label, keep in zip(
                combined_semantic_labels[current_count:], historic_mask
            )
            if keep
        ]

        self.mapper.reset()
        self.mapper.update(accepted_detections, semantic_labels)
        self.mapper.update_historic_semantics(
            np.asarray(
                [item.xyz_lidar_m for item in accepted_historic_detections],
                dtype=np.float64,
            ).reshape(-1, 3),
            historic_semantic_labels,
        )
        dense = self.mapper.labels()
        finite_residuals = np.abs(residuals[np.isfinite(residuals)])
        return FramePrediction(
            dense_labels_xyz=dense,
            detections=accepted_detections,
            motion_labels=motion_labels,
            semantic_labels=semantic_labels,
            metadata={
                "radar_path": str(radar_path),
                "ego_speed_mps": float(
                    np.linalg.norm(ego_motion.linear_velocity_lidar_mps[:2])
                ),
                "ego_velocity_lidar_mps": (
                    ego_motion.linear_velocity_lidar_mps.tolist()
                ),
                "ego_velocity_radar_mps": (
                    ego_motion.linear_velocity_radar_mps.tolist()
                ),
                "yaw_rate_rps": ego_motion.yaw_rate_rps,
                "ego_speed_source": ego_motion.source,
                "raw_detection_count": len(raw_detections),
                "reliable_detection_count": len(reliable_detections),
                "temporal_accepted_detection_count": len(
                    temporal_accepted_detections
                ),
                "accepted_detection_count": len(accepted_detections),
                "historic_detection_count": len(accepted_historic_detections),
                "historic_background_count": sum(
                    label == 1 for label in historic_semantic_labels
                ),
                "historic_foreground_count": sum(
                    label == 2 for label in historic_semantic_labels
                ),
                "doppler_static_evidence_count": int(
                    np.count_nonzero(evidence == int(DopplerEvidence.STATIC))
                ),
                "doppler_uncertain_evidence_count": int(
                    np.count_nonzero(evidence == int(DopplerEvidence.UNCERTAIN))
                ),
                "doppler_dynamic_evidence_count": int(
                    np.count_nonzero(evidence == int(DopplerEvidence.DYNAMIC))
                ),
                "doppler_residual_median_mps": (
                    float(np.median(finite_residuals))
                    if finite_residuals.size
                    else float("nan")
                ),
                "doppler_residual_p90_mps": (
                    float(np.quantile(finite_residuals, 0.90))
                    if finite_residuals.size
                    else float("nan")
                ),
                "reader": type(self.reader).__name__,
                "detector": type(self.detector).__name__,
                "reliability_filter": type(self.reliability_filter).__name__,
                "motion_classifier": type(self.motion_classifier).__name__,
                "temporal_classifier": type(self.temporal_classifier).__name__,
                "semantic_classifier": type(self.semantic_classifier).__name__,
                "object_classifier": dict(
                    getattr(self.semantic_classifier, "last_diagnostics", {})
                ),
            },
        )

    def predict_and_write(
        self,
        radar_path: str | Path,
        output_root: str | Path,
        token: str,
        ego_speed_mps: float | None = None,
    ) -> Path:
        prediction = self.predict_file(radar_path, ego_speed_mps=ego_speed_mps)
        return self.writer.write(prediction, output_root, token)


def _common_components(
    grid_cfg: GridConfig,
    radar_cfg: KRadarConfig,
    ego_speed_cfg: EgoSpeedConfig,
    motion_cfg: MotionConfig,
    mapping_cfg: MappingConfig,
) -> tuple[
    EgoSpeedEstimator,
    MotionClassifier,
    SemanticClassifier,
    OccupancyMapper,
    PredictionWriter,
]:
    return (
        RobustDopplerEgoSpeedEstimator(
            radar_cfg=radar_cfg,
            motion_cfg=motion_cfg,
            estimator_cfg=ego_speed_cfg,
        ),
        EgoCompensatedDopplerClassifier(
            radar_cfg=radar_cfg,
            motion_cfg=motion_cfg,
        ),
        DopplerSemanticClassifier(),
        LogOddsOccupancyGrid3D(
            grid_cfg=grid_cfg,
            radar_cfg=radar_cfg,
            mapping_cfg=mapping_cfg,
        ),
        RadarOccPredictionWriter(),
    )


def build_raw_pipeline(
    cfar_backend: str = "numpy",
    grid_cfg: GridConfig | None = None,
    radar_cfg: KRadarConfig | None = None,
    cfar_cfg: CFARConfig | None = None,
    ego_speed_cfg: EgoSpeedConfig | None = None,
    motion_cfg: MotionConfig | None = None,
    mapping_cfg: MappingConfig | None = None,
) -> TraditionalRadarPipeline:
    """Build the genuine raw-4DRT CFAR baseline."""

    grid_cfg = grid_cfg or GridConfig()
    radar_cfg = radar_cfg or KRadarConfig()
    cfar_cfg = cfar_cfg or CFARConfig()
    ego_speed_cfg = ego_speed_cfg or EgoSpeedConfig()
    motion_cfg = motion_cfg or MotionConfig()
    mapping_cfg = mapping_cfg or MappingConfig()

    if cfar_backend == "numpy":
        backend = NumpyCACFAR(cfar_cfg.threshold_offset_db)
    elif cfar_backend == "openradar":
        backend = OpenRadarCACFAR(cfar_cfg.threshold_offset_db)
    else:
        raise ValueError("cfar_backend must be 'numpy' or 'openradar'.")

    components = _common_components(
        grid_cfg, radar_cfg, ego_speed_cfg, motion_cfg, mapping_cfg
    )
    (
        ego_speed_estimator,
        motion_classifier,
        semantic_classifier,
        mapper,
        writer,
    ) = components

    return TraditionalRadarPipeline(
        reader=KRadarTensorReader(radar_cfg),
        detector=ClassicalTargetDetector(
            cfar_backend=backend,
            radar_cfg=radar_cfg,
            cfar_cfg=cfar_cfg,
        ),
        ego_speed_estimator=ego_speed_estimator,
        motion_classifier=motion_classifier,
        semantic_classifier=semantic_classifier,
        mapper=mapper,
        writer=writer,
    )


def build_rpc_pipeline(
    grid_cfg: GridConfig | None = None,
    radar_cfg: KRadarConfig | None = None,
    ego_speed_cfg: EgoSpeedConfig | None = None,
    motion_cfg: MotionConfig | None = None,
    mapping_cfg: MappingConfig | None = None,
    reliability_cfg: ReliabilityConfig | None = None,
    temporal_cfg: TemporalConfig | None = None,
    object_cfg: ObjectClusteringConfig | None = None,
    object_model_path: str | Path | None = None,
) -> TraditionalRadarPipeline:
    """Build the default Enhanced K-Radar RPC point-cloud baseline.

    RPC/pc01p already contains Cartesian targets and physical Doppler velocity.
    This composition therefore skips CFAR, polar conversion and Doppler-bin
    conversion, then reuses the motion classifier, log-odds mapper and writer.
    """

    grid_cfg = grid_cfg or GridConfig()
    radar_cfg = radar_cfg or KRadarConfig()
    ego_speed_cfg = ego_speed_cfg or EgoSpeedConfig()
    motion_cfg = motion_cfg or MotionConfig()
    mapping_cfg = mapping_cfg or MappingConfig()
    reliability_cfg = reliability_cfg or ReliabilityConfig()
    temporal_cfg = temporal_cfg or TemporalConfig()
    object_cfg = object_cfg or ObjectClusteringConfig()

    ego_speed_estimator = RobustDopplerEgoSpeedEstimator(
        radar_cfg=radar_cfg,
        motion_cfg=motion_cfg,
        estimator_cfg=ego_speed_cfg,
    )
    motion_classifier = EgoCompensatedDopplerClassifier(
        radar_cfg=radar_cfg,
        motion_cfg=motion_cfg,
    )
    semantic_classifier: SemanticClassifier
    if object_model_path is None:
        semantic_classifier = DopplerSemanticClassifier()
    else:
        semantic_classifier = ObjectAwareSemanticClassifier.from_file(
            object_model_path,
            config=object_cfg,
        )
    mapper = TemporalLogOddsOccupancyGrid3D(
        grid_cfg=grid_cfg,
        radar_cfg=radar_cfg,
        mapping_cfg=mapping_cfg,
    )
    writer = RadarOccPredictionWriter()

    return TraditionalRadarPipeline(
        reader=KRadarRPCReader(),
        detector=RPCPointTargetDetector(radar_cfg=radar_cfg),
        ego_speed_estimator=ego_speed_estimator,
        motion_classifier=motion_classifier,
        semantic_classifier=semantic_classifier,
        mapper=mapper,
        writer=writer,
        reliability_filter=LocalPowerRPCFilter(reliability_cfg),
        temporal_classifier=PoseAlignedTemporalClassifier(
            temporal_cfg=temporal_cfg,
            motion_cfg=motion_cfg,
        ),
    )
