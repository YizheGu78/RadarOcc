"""CPU-only tests for region matching; no RadarOcc model or K-Radar data."""
import unittest
import numpy as np

from tools.temporal_vis.benchmark_region_matching import (
    aligned_box, label_regions, select_match, summarize)


class RegionMatchingTests(unittest.TestCase):
    def setUp(self):
        self.current = {'id': 2, 'center': np.array([20., -5., 0.]),
                        'box_size': np.array([6., 3., 3.])}
        self.y_gates = [0.5, 1.0, 1.5]
        self.center_gates = [4., 6., 8.]

    @staticmethod
    def history(idx, x, y, size):
        return {'id': idx, 'aligned_center': np.array([x, y, 0.]),
                'aligned_size': np.asarray(size, float)}

    def test_y_gate_by_history(self):
        h = self.history(5, 20., -4.2, [6., 3., 3.])
        self.assertIsNone(select_match(self.current, [h], 1, 1., 0.,
                                       self.y_gates, self.center_gates))
        match = select_match(self.current, [h], 2, 1., 0.,
                             self.y_gates, self.center_gates)
        self.assertEqual(match['id'], 5)

    def test_center_and_size_weights_choose_different_matches(self):
        close_large = self.history(0, 20.1, -5., [12., 5., 5.])
        far_equal = self.history(1, 23., -5., [6., 3., 3.])
        match1 = select_match(self.current, [close_large, far_equal], 1, 1., 0.,
                              self.y_gates, self.center_gates)
        match2 = select_match(self.current, [close_large, far_equal], 1, 0., 1.,
                              self.y_gates, self.center_gates)
        self.assertEqual(match1['id'], 0)
        self.assertEqual(match2['id'], 1)

    def test_transform_90_degree_rotation(self):
        pose = np.eye(4)
        pose[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        pose[:3, 3] = [10., -4., 2.]
        center, size = aligned_box([2., 3., 1.], [6., 2., 4.], pose)
        np.testing.assert_allclose(center, [7., -2., 3.])
        np.testing.assert_allclose(size, [2., 6., 4.])

    def test_gt_label_purity_and_duplicate_id(self):
        regions = [{'id': 1, 'accepted': True, 'support_point_ids': [0, 1, 2, 3, 4]},
                   {'id': 2, 'accepted': True, 'support_point_ids': [5, 6, 7]},
                   {'id': 3, 'accepted': False, 'support_point_ids': [0, 1, 2]}]
        labels = [0, 0, 0, 1, -1, 2, 2, 2]
        self.assertEqual(label_regions(regions, labels, ['CAR1', 'CAR2', 'DUP']),
                         {1: 'CAR1', 2: 'DUP'})
        self.assertEqual(label_regions(regions, labels, ['CAR1', 'CAR2', 'CAR1']), {})

    def test_unscorable_and_recall_denominator(self):
        base = dict(has_true_history=True, y_missed_true=False,
                    center_missed_true=False, predicted=True)
        rows = [dict(base, predicted_id='A', correct=True),
                dict(base, predicted_id=None, correct=False),
                dict(base, predicted_id='B', correct=False)]
        stats = summarize(rows)
        self.assertEqual(stats['has_true_history'], 3)
        self.assertEqual(stats['unscorable_prediction'], 1)
        self.assertAlmostEqual(stats['precision'], .5)
        self.assertAlmostEqual(stats['recall'], 1/3)


if __name__ == '__main__':
    unittest.main()
