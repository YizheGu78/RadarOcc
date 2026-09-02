from __future__ import annotations

from typing import Sequence

from tradition.core.interfaces import SemanticClassifier
from tradition.core.types import MotionLabel, RadarDetection, SemanticLabel


class DopplerSemanticClassifier(SemanticClassifier):
    """Map compensated Doppler motion labels directly to RadarOcc classes."""

    def classify(
        self,
        detections: Sequence[RadarDetection],
        motion_labels: Sequence[MotionLabel],
    ) -> list[SemanticLabel]:
        if len(detections) != len(motion_labels):
            raise ValueError("detections and motion_labels must have equal length.")
        return [
            SemanticLabel.FOREGROUND
            if motion == MotionLabel.DYNAMIC
            else SemanticLabel.BACKGROUND
            for motion in motion_labels
        ]
