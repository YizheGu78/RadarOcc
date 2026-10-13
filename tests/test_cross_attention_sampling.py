"""CPU-only tests for the one-frame cross-attention sampling diagnostic."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/temporal_vis/record_cross_attention_sampling.py"
spec = importlib.util.spec_from_file_location("cross_sampling_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CrossSamplingAuditTests(unittest.TestCase):
    def test_zero_offsets_use_cuda_half_cell_rule(self):
        ref = np.array([[[0.5, 0.5, 0.5]]])
        offsets = np.zeros((1, 8, 1, 4, 3))
        logits = np.zeros((1, 8, 1, 4))
        stats, arrays = audit.analyze_samples(ref, offsets, logits, [[128, 128, 14]])
        np.testing.assert_allclose(arrays["base_feature"][0, 0], [63.5, 63.5, 6.5])
        np.testing.assert_allclose(arrays["sampling_feature_indices"][0], offsets[0] + [63.5, 63.5, 6.5])
        np.testing.assert_allclose(arrays["attention_weights"], 0.25)
        np.testing.assert_allclose(arrays["nominal_correction_feature_indices"][0, [0, 2]], 0.25)
        self.assertEqual(stats["sampled_queries"], 1)

    def test_quarter_cell_angle_compensation(self):
        ref = np.array([[[0.5, 0.5, 0.5]]])
        offsets = np.zeros((1, 8, 1, 4, 3))
        offsets[..., 0] = 0.25
        offsets[..., 2] = 0.25
        logits = np.zeros((1, 8, 1, 4))
        metrics, arrays = audit.analyze_samples(ref, offsets, logits, [[128, 128, 14]])
        np.testing.assert_allclose(arrays["weighted_centroid_feature"][0, [0, 2]],
                                   arrays["nominal_target_feature"][0, [0, 2]])
        self.assertAlmostEqual(
            metrics["weighted_centroid_minus_target_degrees"]["azimuth"]["mean"], 0.0)
        self.assertAlmostEqual(
            metrics["weighted_centroid_minus_target_degrees"]["elevation"]["mean"], 0.0)

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):
            audit.analyze_samples(np.zeros((1, 1, 3)),
                                  np.zeros((1, 8, 1, 4, 3)),
                                  np.zeros((1, 8, 1, 4)), [[64, 64, 10]])


if __name__ == "__main__":
    unittest.main()
