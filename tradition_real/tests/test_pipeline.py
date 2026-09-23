from dataclasses import replace
import unittest
import numpy as np
from tradition_real.config import Config
from tradition_real.pipeline import Pipeline
from tradition_real.gm2019.fusion import LogOddsFusion, logit
from tradition_real.adapters.octomap_grid import voxel_indices
from tradition_real.semantics.features import FEATURE_NAMES
from tradition_real.semantics.random_forest import training_target


def rpc_at(x=5.2, y=.2, z=.2, doppler=1.7):
    return np.array([[x, y, z, 9., doppler, np.sqrt(x*x+y*y+z*z),
                      np.arctan2(y, x), .1, 13, 53, 18]], np.float64)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(min_xyz=(0., -4., -2.), shape_xyz=(32, 20, 10), resolution=.4,
                          temporal_window=3, dbscan_min_samples=1)

    def test_gm_equation_and_unknown(self):
        fusion = LogOddsFusion((3,), .7, .35, .4)
        fusion.update(np.array([255, 0, 128]))
        np.testing.assert_allclose(fusion.probability, [.7, .35, .4])
        fusion.update(np.array([255, 128, 128]))
        self.assertAlmostEqual(fusion.log_odds[0], 2*logit(.7)-logit(.4))
        self.assertFalse(fusion.observed[2])
        self.assertAlmostEqual(fusion.probability[1], .35)

    def test_window_replay_not_double_counted_and_scene_reset(self):
        pipeline = Pipeline(self.cfg)
        cell = tuple(voxel_indices(rpc_at()[:, :3], self.cfg)[0][0])
        for n in range(6):
            result = pipeline.map_frame(rpc_at(), np.eye(4), np.zeros(3), '3', n)
            expected = 1/(1+np.exp(-min(n+1, 3)*logit(.7)))
            self.assertAlmostEqual(result.probability[cell], expected)
        result = pipeline.map_frame(rpc_at(), np.eye(4), np.zeros(3), '4', 0)
        self.assertAlmostEqual(result.probability[cell], .7)
        with self.assertRaisesRegex(ValueError, 'chronological'):
            pipeline.map_frame(rpc_at(), np.eye(4), np.zeros(3), '4', 0)

    def test_pose_alignment_and_height_layers(self):
        pipeline = Pipeline(self.cfg)
        pipeline.map_frame(rpc_at(6.2), np.eye(4), np.zeros(3), '3', 0)
        pose = np.eye(4)
        pose[0, 3] = 1.2
        result = pipeline.map_frame(rpc_at(5.), pose, np.zeros(3), '3', 1)
        cell = tuple(voxel_indices(rpc_at(5.)[:, :3], self.cfg)[0][0])
        self.assertAlmostEqual(result.probability[cell], 1/(1+np.exp(-2*logit(.7))))
        self.assertEqual(int(result.occupied.sum()), 1)
        self.assertEqual(np.count_nonzero(result.observed.sum(axis=(0, 1))), 1)
        feature = result.candidates[0]['features']
        self.assertAlmostEqual(feature[FEATURE_NAMES.index('doppler_raw_mean')], 1.7)
        self.assertEqual(feature[FEATURE_NAMES.index('unique_frame_count')], 2)

    def test_empty_frame_and_unknown_export(self):
        pipeline = Pipeline(self.cfg)
        result = pipeline.map_frame(np.empty((0, 11)), np.eye(4), np.zeros(3), '3', 0)
        dense, native = pipeline.predict(result, None)
        self.assertTrue(np.all(dense == 0))
        self.assertTrue(np.all(native == 255))
        self.assertFalse(result.observed.any())

    def test_semantics_cannot_change_occupancy(self):
        class Foreground:
            def foreground_probabilities(self, x):
                return np.ones(len(x))
        pipeline = Pipeline(self.cfg)
        result = pipeline.map_frame(rpc_at(), np.eye(4), np.zeros(3), '3', 0)
        dense, native = pipeline.predict(result, Foreground())
        np.testing.assert_array_equal(dense > 0, result.occupied)
        self.assertTrue(np.all(native[result.occupied] == 2))
        gt = np.zeros(self.cfg.shape_xyz, np.uint8)
        self.assertIsNone(training_target(result.candidates[0], gt)[0])
        gt[result.occupied] = 1
        self.assertEqual(training_target(result.candidates[0], gt)[0], 0)
        gt[result.occupied] = 2
        self.assertEqual(training_target(result.candidates[0], gt)[0], 1)

    def test_gap_resets_and_nonfinite_rejected(self):
        pipeline = Pipeline(self.cfg)
        pipeline.map_frame(rpc_at(), np.eye(4), np.zeros(3), '3', 0)
        result = pipeline.map_frame(rpc_at(), np.eye(4), np.zeros(3), '3', 4)
        self.assertAlmostEqual(result.probability[result.occupied][0], .7)
        bad = rpc_at(); bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            pipeline.map_frame(bad, np.eye(4), np.zeros(3), '3', 5)


if __name__ == '__main__':
    unittest.main()
