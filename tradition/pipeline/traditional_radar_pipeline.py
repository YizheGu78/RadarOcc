from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tradition.core.config import (
    CFARConfig,
    GridConfig,
    KRadarConfig,
    MappingConfig,
    MotionConfig,
    SparseDetectionConfig,
)
from tradition.core.interfaces import (
    MotionClassifier,
    OccupancyMapper,
    PredictionWriter,
    RadarTensorReader,
    TargetDetector,
)
from tradition.core.types import FramePrediction
from tradition.detection.cfar import NumpyCACFAR, OpenRadarCACFAR
from tradition.detection.sparse_target_detector import SparseCandidateTargetDetector
from tradition.detection.target_detector import ClassicalTargetDetector
from tradition.io.kradar_reader import KRadarTensorReader
from tradition.io.radarocc_writer import RadarOccPredictionWriter
from tradition.io.sparse_radar_reader import RadarOccSparseReader
from tradition.mapping.occupancy_grid_3d import LogOddsOccupancyGrid3D
from tradition.motion.doppler_classifier import EgoCompensatedDopplerClassifier


class TraditionalRadarPipeline:
    """Dependency-injected orchestration for a non-learning radar baseline."""

    def __init__(
        self,
        reader: RadarTensorReader,
        detector: TargetDetector,
        motion_classifier: MotionClassifier,
        mapper: OccupancyMapper,
        writer: PredictionWriter,
    ) -> None:
        self.reader = reader
        self.detector = detector
        self.motion_classifier = motion_classifier
        self.mapper = mapper
        self.writer = writer

    def predict_measurement(
        self,
        measurement: Any,
        ego_speed_mps: float = 0.0,
    ) -> FramePrediction:
        detections = self.detector.detect(measurement)
        motion_labels = self.motion_classifier.classify(
            detections, ego_speed_mps=ego_speed_mps
        )
        self.mapper.reset()
        self.mapper.update(detections, motion_labels)
        dense = self.mapper.labels()
        return FramePrediction(
            dense_labels_xyz=dense,
            detections=detections,
            motion_labels=motion_labels,
            metadata={
                "ego_speed_mps": float(ego_speed_mps),
                "reader": type(self.reader).__name__,
                "detector": type(self.detector).__name__,
            },
        )

    def predict_tensor(
        self,
        radar_tensor_drea: np.ndarray,
        ego_speed_mps: float = 0.0,
    ) -> FramePrediction:
        """Backward-compatible raw-tensor entry point."""
        return self.predict_measurement(
            radar_tensor_drea,
            ego_speed_mps=ego_speed_mps,
        )

    def predict_file(
        self,
        radar_path: str | Path,
        ego_speed_mps: float = 0.0,
    ) -> FramePrediction:
        measurement = self.reader.read(radar_path)
        prediction = self.predict_measurement(
            measurement,
            ego_speed_mps=ego_speed_mps,
        )
        prediction.metadata["radar_path"] = str(radar_path)
        return prediction

    def predict_and_write(
        self,
        radar_path: str | Path,
        output_root: str | Path,
        token: str,
        ego_speed_mps: float = 0.0,
    ) -> Path:
        prediction = self.predict_file(radar_path, ego_speed_mps=ego_speed_mps)
        return self.writer.write(prediction, output_root, token)


def _common_components(
    grid_cfg: GridConfig,
    radar_cfg: KRadarConfig,
    motion_cfg: MotionConfig,
    mapping_cfg: MappingConfig,
) -> tuple[MotionClassifier, OccupancyMapper, PredictionWriter]:
    return (
        EgoCompensatedDopplerClassifier(
            radar_cfg=radar_cfg,
            motion_cfg=motion_cfg,
        ),
        LogOddsOccupancyGrid3D(
            grid_cfg=grid_cfg,
            radar_cfg=radar_cfg,
            mapping_cfg=mapping_cfg,
        ),
        RadarOccPredictionWriter(),
    )


def build_default_pipeline(
    cfar_backend: str = "numpy",
    grid_cfg: GridConfig | None = None,
    radar_cfg: KRadarConfig | None = None,
    cfar_cfg: CFARConfig | None = None,
    motion_cfg: MotionConfig | None = None,
    mapping_cfg: MappingConfig | None = None,
) -> TraditionalRadarPipeline:
    """Build the genuine raw-4DRT CFAR baseline."""

    grid_cfg = grid_cfg or GridConfig()
    radar_cfg = radar_cfg or KRadarConfig()
    cfar_cfg = cfar_cfg or CFARConfig()
    motion_cfg = motion_cfg or MotionConfig()
    mapping_cfg = mapping_cfg or MappingConfig()

    if cfar_backend == "numpy":
        backend = NumpyCACFAR(cfar_cfg.threshold_offset_db)
    elif cfar_backend == "openradar":
        backend = OpenRadarCACFAR(cfar_cfg.threshold_offset_db)
    else:
        raise ValueError("cfar_backend must be 'numpy' or 'openradar'.")

    motion_classifier, mapper, writer = _common_components(
        grid_cfg,
        radar_cfg,
        motion_cfg,
        mapping_cfg,
    )

    return TraditionalRadarPipeline(
        reader=KRadarTensorReader(radar_cfg),
        detector=ClassicalTargetDetector(
            cfar_backend=backend,
            radar_cfg=radar_cfg,
            cfar_cfg=cfar_cfg,
        ),
        motion_classifier=motion_classifier,
        mapper=mapper,
        writer=writer,
    )


def build_sparse_pipeline(
    grid_cfg: GridConfig | None = None,
    radar_cfg: KRadarConfig | None = None,
    sparse_cfg: SparseDetectionConfig | None = None,
    motion_cfg: MotionConfig | None = None,
    mapping_cfg: MappingConfig | None = None,
) -> TraditionalRadarPipeline:
    """Build the practical EAsparse baseline.

    This starts after RadarOcc's mean-power Top-K sparsification and therefore
    does not claim to reproduce CFAR.
    """

    grid_cfg = grid_cfg or GridConfig()
    radar_cfg = radar_cfg or KRadarConfig()
    sparse_cfg = sparse_cfg or SparseDetectionConfig()
    motion_cfg = motion_cfg or MotionConfig()
    mapping_cfg = mapping_cfg or MappingConfig()

    motion_classifier, mapper, writer = _common_components(
        grid_cfg,
        radar_cfg,
        motion_cfg,
        mapping_cfg,
    )

    return TraditionalRadarPipeline(
        reader=RadarOccSparseReader(),
        detector=SparseCandidateTargetDetector(
            radar_cfg=radar_cfg,
            sparse_cfg=sparse_cfg,
        ),
        motion_classifier=motion_classifier,
        mapper=mapper,
        writer=writer,
    )
