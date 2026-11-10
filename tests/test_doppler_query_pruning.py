"""Analytical physics, budget, causal-pose and GT-coverage checks; CPU only."""
import unittest
import numpy as np

from tools.temporal_vis.audit_doppler_queries import causal_velocity, evaluate_mask
from tools.temporal_vis.doppler_query_pruning import (
    circular_peaks, fft_period, query_masks, spatial_budget, static_consistency, wrap_velocity)


def features(bins):
    bins = np.asarray(bins)
    return np.column_stack((np.ones((len(bins), 3)), np.repeat(bins[:, None], 3, axis=1),
                            np.ones((len(bins), 2))))


class DopplerQueryTests(unittest.TestCase):
    def test_velocity_seam_is_not_averaged_to_false_zero(self):
        table = np.arange(-2, 2, .25)
        f = features([0])
        f[0, :3] = [1, 1, 0]
        f[0, 3:6] = [0, 15, 1]  # -2 and +1.75 are adjacent circular bins.
        p = circular_peaks(f, table)
        self.assertEqual(fft_period(table), 4.)  # Not max(table)-min(table).
        self.assertTrue(p['reliable'][0])
        self.assertTrue(p['seam_crossing'][0])
        self.assertAlmostEqual(p['linear_mean_mps'][0], -.125)
        self.assertGreater(abs(p['folded_mps'][0]), 1.8)
        self.assertAlmostEqual(p['circular_std_mps'][0], .125)

    def test_opposite_peaks_and_invalid_bins_abstain(self):
        f = features([0, 0, 0])
        f[0, :3], f[0, 3:6] = [1, 1, 0], [4, 12, 4]  # Equal peaks at -1/+1.
        f[1, 3], f[2, :3] = 16, 0
        p = circular_peaks(f, np.arange(-2, 2, .25))
        self.assertFalse(p['reliable'].any())

    def test_folded_static_prediction_uses_los_and_both_signs(self):
        table = np.arange(-2, 2, .25)
        xyz = np.array([[10., 0, 0], [6., 8, 0]])
        for sign in [1, -1]:
            observed = sign*wrap_velocity(np.array([-15., -9.]), 4)
            bins = np.rint((observed+2)/.25).astype(int)
            p = circular_peaks(features(bins), table)
            c = static_consistency(xyz, p, [15, 0, 0], sign)
            np.testing.assert_allclose(c['residual_mps'], 0, atol=1e-12)
            self.assertTrue(c['reliable'].all())

    def test_alias_guard_preserves_moving_alternatives_to_zero(self):
        table = np.arange(-2, 2, .25)
        coords = np.column_stack((np.zeros(8, int), np.arange(8), np.zeros(8, int)))
        xyz = np.tile([10., 0, 0], (8, 1))
        p = circular_peaks(features([8]*8), table)
        c = static_consistency(xyz, p, [0, 0, 0], 1, max_target_radial_mps=30)
        m = query_masks(coords, c, .2, .25)
        self.assertEqual(int(m['doppler_heuristic'].sum()), 2)
        self.assertEqual(int(m['spatial_same_budget'].sum()), 2)
        self.assertTrue(m['alias_ambiguous_static'].all())
        self.assertTrue(m['alias_guarded'].all())
        # This changes only an explicit assumed speed bound, not measured Doppler.
        c = static_consistency(xyz, p, [0, 0, 0], 1, max_target_radial_mps=.5)
        self.assertEqual(int(query_masks(coords, c, .2, .25)['alias_guarded'].sum()), 2)

    def test_no_ego_or_unreliable_doppler_keeps_all_queries(self):
        coords = np.array([[0, 0, 0], [0, 1, 0]])
        xyz = np.array([[10., 0, 0], [10., 1, 0]])
        p = circular_peaks(features([8, 8]), np.arange(-2, 2, .25))
        c = static_consistency(xyz, p, None, 1)
        masks = query_masks(coords, c, .2, 0)
        self.assertTrue(masks['doppler_heuristic'].all())
        self.assertTrue(masks['alias_guarded'].all())
        p['reliable'][:] = False
        c = static_consistency(xyz, p, [0, 0, 0], 1)
        self.assertTrue(query_masks(coords, c, .2, 0)['doppler_heuristic'].all())

    def test_spatial_control_has_exact_reproducible_budget(self):
        coords = np.column_stack((np.repeat([0, 1, 2], [5, 7, 8]), np.arange(20), np.arange(20)%3))
        eligible = np.arange(20)%2 == 0
        for budget in [0, 1, 7, 10]:
            mask = spatial_budget(coords, budget, eligible)
            self.assertEqual(int(mask.sum()), budget)
            self.assertFalse((mask & ~eligible).any())
            np.testing.assert_array_equal(mask, spatial_budget(coords, budget, eligible))

    def test_causal_pose_never_uses_next_frame(self):
        rows = []
        for order, x in [(79, 0), (80, 1), (81, 1000)]:
            pose = np.eye(4)
            pose[0, 3] = x
            rows.append(dict(order_in_scene=order, timestamp_sec=order*.1,
                             ego_pose_valid=True, ego_pose=pose))
        v, details = causal_velocity(rows, 1, np.eye(4))
        np.testing.assert_allclose(v, [10, 0, 0])
        self.assertEqual(details['previous_order'], 79)
        rows[0]['ego_pose_valid'] = False
        self.assertIsNone(causal_velocity(rows, 1, np.eye(4))[0])

    def test_gt_empty_boxes_and_union_denominators(self):
        boxes = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=bool)
        counts, before, after = evaluate_mask(np.array([0, 0, 1, 1], dtype=bool), boxes)
        self.assertEqual(counts['n_gt_union'], 3)
        self.assertEqual(counts['n_gt_union_kept'], 1)
        self.assertEqual(counts['n_gt_boxes_empty_before'], 1)
        self.assertEqual(counts['n_gt_boxes_nonempty'], 2)
        self.assertEqual(counts['n_gt_boxes_all_pruned'], 1)
        np.testing.assert_array_equal(before, [2, 2, 0])
        np.testing.assert_array_equal(after, [0, 1, 0])


if __name__ == '__main__':
    unittest.main()
