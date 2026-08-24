from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .types import FramePrediction, MotionLabel, RadarDetection, SemanticLabel


class RadarMeasurementReader(ABC):
    """Load one radar representation without coupling the pipeline to storage."""

    @abstractmethod
    def read(self, path: str | Path) -> Any:
        """Return one representation-specific radar measurement."""


class CFARBackend(ABC):
    @abstractmethod
    def threshold(
        self,
        signal_db: np.ndarray,
        guard_len: int,
        noise_len: int,
    ) -> np.ndarray:
        """Return a threshold array with the same shape as signal_db."""


class TargetDetector(ABC):
    @abstractmethod
    def detect(self, measurement: Any) -> list[RadarDetection]:
        """Create a classical target list from a radar measurement."""


class MotionClassifier(ABC):
    @abstractmethod
    def classify(
        self,
        detections: Sequence[RadarDetection],
        ego_speed_mps: float,
    ) -> list[MotionLabel]:
        """Split target-list evidence into static and dynamic."""


class SemanticClassifier(ABC):
    @abstractmethod
    def classify(
        self,
        detections: Sequence[RadarDetection],
        motion_labels: Sequence[MotionLabel],
    ) -> list[SemanticLabel]:
        """Classify detections as semantic background or foreground."""


class OccupancyMapper(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Reset the map for independent single-frame evaluation."""

    @abstractmethod
    def update(
        self,
        detections: Sequence[RadarDetection],
        semantic_labels: Sequence[SemanticLabel],
    ) -> None:
        """Update inverse sensor-model evidence."""

    @abstractmethod
    def labels(self) -> np.ndarray:
        """Return a dense RadarOcc-style class grid [X,Y,Z]."""


class PredictionWriter(ABC):
    @abstractmethod
    def write(
        self,
        prediction: FramePrediction,
        output_root: str | Path,
        token: str,
    ) -> Path:
        """Write one prediction and return the sparse pred_c.npy path."""
