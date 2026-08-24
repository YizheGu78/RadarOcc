import numpy as np

from tradition.core.types import FramePrediction
from tradition.io.radarocc_writer import RadarOccPredictionWriter


def test_writer_uses_zyx_sparse_order(tmp_path):
    dense = np.zeros((128, 128, 14), dtype=np.uint8)
    dense[10, 20, 3] = 2
    prediction = FramePrediction(dense, [], [], [], {})
    path = RadarOccPredictionWriter().write(prediction, tmp_path, "abc")
    sparse = np.load(path, allow_pickle=False)
    np.testing.assert_array_equal(sparse, np.array([[3, 20, 10, 2]]))
