"""CPU-only synthetic sanity checks. No K-Radar data / model checkpoints."""
import unittest

import numpy as np

from tools.temporal_vis.audit_region_quality import (
    aabb_iou, decision, fraction_for_target, lidar_box_corners_radar,
    polygon_area, summarize, xy_polygon_iou)


class RegionQualityAuditTests(unittest.TestCase):
    def test_oriented_xy_gt_iou_exact_axis_aligned_case(self):
        corners = lidar_box_corners_radar([5, 0, 0, 2, 2, 2, 0], np.eye(4))
        self.assertAlmostEqual(xy_polygon_iou(
            np.array([4., -1., -1.]), np.array([6., 1., 3.]), corners), 1.)
        self.assertAlmostEqual(xy_polygon_iou(
            np.array([10., -1., -1.]), np.array([12., 1., 3.]), corners), 0.)

    def test_polygon_clip_rotated_and_aabb(self):
        c = lidar_box_corners_radar([0, 0, 0, 2, 2, 2, np.pi / 4], np.eye(4))
        result = xy_polygon_iou(np.array([-2., -2.]), np.array([2., 2.]), c)
        self.assertTrue(.2 < result < .3)
        self.assertAlmostEqual(aabb_iou(
            [0, 0], [2, 2], [1, 1], [3, 3]), 1 / 7)
        self.assertAlmostEqual(polygon_area([[0, 0], [1, 0], [0, 1]]), .5)

    def test_support_purity_must_not_be_confused_with_box_purity(self):
        tags = np.array(["car", "car", "car", "other", None, "other",
                         None, None, "other", None], dtype=object)
        support = [0, 1, 2, 3]
        box = list(range(10))
        self.assertEqual(fraction_for_target(support, tags, "car"), (.75, 3))
        self.assertEqual(fraction_for_target(box, tags, "car"), (.3, 3))
        self.assertEqual(fraction_for_target(box, tags, None), (None, 0))

    def test_correct_recall_does_not_certify_geometry_or_unknown_links(self):
        def row(**kwargs):
            default = dict(
                scorable=True, has_true_history=True,
                has_true_history_strict=True, predicted=True, same_id_loose=False,
                same_id_strict=False, unknown_history=False,
                confirmed_3d_target_points_both=False,
                current_box_purity_xyz=.5, history_box_purity_xyz=.4,
                matched_pair_xy_aabb_iou=.25, current_gt_xy_iou=.1,
                history_gt_xy_iou=.1)
            default.update(kwargs)
            return default
        data = [row(same_id_loose=True, same_id_strict=True,
                    confirmed_3d_target_points_both=True),
                row(unknown_history=True),
                row(scorable=False, predicted=False, has_true_history=False,
                    has_true_history_strict=False)]
        result = summarize(data)
        self.assertEqual(result["correct_loose"], 1)
        self.assertEqual(result["eligible_loose"], 2)
        self.assertEqual(result["unknown_history"], 1)
        self.assertAlmostEqual(result["recall_loose"], .5)
        self.assertAlmostEqual(result["precision_known"], 1.)
        self.assertAlmostEqual(result["precision_conservative"], .5)
        self.assertAlmostEqual(result["matched_min_box_xyz_purity_median"], .4)
        self.assertIn("INCONCLUSIVE", decision(result))


if __name__ == "__main__":
    unittest.main()
