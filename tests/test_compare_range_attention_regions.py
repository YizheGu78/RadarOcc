"""Regression checks for proposal diagnostics; no CUDA or RadarOcc model needed."""
import unittest

import numpy as np

from tools.temporal_vis.compare_range_attention_regions import gt_statistics, robust_descriptor_score


class RangeRegionDiagnosticsTests(unittest.TestCase):
    def test_sparse_response_survives_zero_mad_but_constants_do_not(self):
        features = np.zeros((20, 5))
        features[3] = [1, 2, 3, 4, 5]
        scores = robust_descriptor_score(features, num_ranges=2, k=10)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[3], 0)
        np.testing.assert_array_equal(scores[np.arange(20) != 3], 0)
        np.testing.assert_array_equal(
            robust_descriptor_score(np.full((20, 5), 4.), num_ranges=2, k=10), 0)

    @staticmethod
    def ground_truth():
        return dict(boxes=np.array([[40., 0., 0., 4., 4., 2., 0.]]),
                    ids=np.array(['truck']), duplicate_ids=np.array([False]),
                    radar_to_lidar=np.eye(4))

    def test_xy_detectability_excludes_returns_outside_analysis_height(self):
        xyz = np.array([[40., 0., 10.], [40.1, .1, 10.1], [39.9, -.1, 9.9]])
        result = dict(regions=[], analysis_mask=np.zeros(3, dtype=bool),
                      stats=dict(box_fraction_all_p1=0., n_support=0))
        stats = gt_statistics(result, xyz, self.ground_truth())
        self.assertEqual(stats['gt_detectable_objects'], 0)
        self.assertIsNone(stats['gt_object_coverage'])

    def test_unlabeled_proposals_keep_purity_evidence_and_rejection_reason(self):
        xyz = np.array([[40., 0., 1.]]*3 + [[50., 0., 1.]]*7)
        result = dict(regions=[dict(id=0, accepted=True, support_point_ids=np.arange(10))],
                      analysis_mask=np.ones(10, dtype=bool),
                      stats=dict(box_fraction_all_p1=1., n_support=10))
        strict = gt_statistics(result, xyz, self.ground_truth())
        self.assertEqual(strict['gt_rejected_low_purity'], 1)
        self.assertEqual(strict['gt_detectable_objects'], 1)
        self.assertEqual(strict['gt_detected_objects'], 0)
        self.assertIsNone(strict['mean_dominant_fraction'])
        self.assertAlmostEqual(strict['mean_dominant_fraction_all'], .3)
        loose = gt_statistics(result, xyz, self.ground_truth(), purity=.2)
        self.assertEqual(loose['gt_detected_objects'], 1)
        self.assertEqual(loose['gt_object_coverage'], 1.)
        # Detectability uses the same requested minimum GT point count.
        small = dict(regions=[dict(id=0, accepted=True, support_point_ids=np.arange(2))],
                     analysis_mask=np.ones(2, dtype=bool),
                     stats=dict(box_fraction_all_p1=1., n_support=2))
        fewer = gt_statistics(small, xyz[:2], self.ground_truth(), min_gt_points=1)
        self.assertEqual(fewer['gt_detectable_objects'], 1)
        self.assertEqual(fewer['gt_detected_objects'], 1)


if __name__ == '__main__':
    unittest.main()
