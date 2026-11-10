"""Power comparison regressions: overlap, inclusion space, and unequal box sizes."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.temporal_vis.region_power_statistics import (
    all_frames_box_table, frame_power_statistics, power_statistics, write_power_statistics)


class RegionPowerTests(unittest.TestCase):
    def test_overlap_is_counted_once_in_union_but_in_each_individual_box(self):
        xyz = np.array([[0., 0., .5], [1., 0., .5], [2., 0., .5], [3., 0., .5]])
        power = np.array([0., 10., 20., 90.])
        regions = [dict(id=0, accepted=True, box_min=[-.5, -1, 0], box_max=[1.5, 1, 1],
                        support_point_ids=[1]),
                   dict(id=1, accepted=True, box_min=[.5, -1, 0], box_max=[3.5, 1, 1],
                        support_point_ids=[1, 3])]
        results = dict(P0_power=dict(analysis_mask=np.ones(4, dtype=bool), regions=regions))
        # Two overlapping GT footprints; the overlapping return must be retained.
        gt = dict(boxes=np.array([[.5, 0, 0, 2, 2, 1, 0], [1., 0, 0, 1, 2, 1, 0]]),
                  ids=['a', 'b'], radar_to_lidar=np.eye(4))
        boxes, frames = frame_power_statistics(results, xyz, power, gt)
        rows = {(r['method'], r['point_set']): r for r in frames}
        self.assertEqual(rows['GT', 'box']['n_points'], 2)
        self.assertEqual(rows['GT', 'box']['mean_power'], 5.)
        self.assertEqual(rows['P0_power', 'box']['n_points'], 4)
        self.assertEqual(rows['P0_power', 'box']['mean_power'], 30.)
        self.assertEqual(rows['P0_power', 'support']['n_points'], 2)
        self.assertEqual(rows['P0_power', 'support']['mean_power'], 50.)
        self.assertEqual(rows['P0_power', 'box_outside_gt']['mean_power'], 55.)
        with tempfile.TemporaryDirectory() as temp:
            summary = write_power_statistics(temp, boxes, frames)
            row = next(r for r in summary if r['method'] == 'P0_power' and r['point_set'] == 'box')
            self.assertEqual(row['mean_power'], 30.)
            self.assertAlmostEqual(row['macro_box_mean_power'], (5. + 40.) / 2)
            with (Path(temp) / 'box_power.csv').open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(len(saved), len(boxes))

    def test_xy_xyz_both_apply_the_same_analysis_roi(self):
        xyz = np.array([[0., 0., .5], [0., 0., 3.], [0., 0., 20.]])
        power = np.array([2., 8., 1000.])
        result = dict(analysis_mask=np.array([True, True, False]),
                      regions=[dict(id=0, accepted=True, box_min=[-1, -1, 0],
                                    box_max=[1, 1, 1], support_point_ids=[0])])
        gt = dict(boxes=np.array([[0., 0., 0., 2., 2., 1., 0.]]), ids=['a'],
                  radar_to_lidar=np.eye(4))
        for space, expected in [('xy', 5.), ('xyz', 2.)]:
            _, frames = frame_power_statistics(dict(P0_power=result), xyz, power, gt, space)
            for row in frames:
                if row['point_set'] == 'box':
                    self.assertEqual(row['mean_power'], expected)

    def test_individual_box_extrema_subsets_and_repeated_gt_ids_stay_separate(self):
        xyz = np.array([[0., 0., .5], [1., 0., .5], [2., 0., .5], [3., 0., .5]])
        power = np.array([0., 10., 20., 90.])
        result = dict(analysis_mask=np.ones(4, dtype=bool),
                      regions=[dict(id=7, accepted=True, box_min=[.5, -1, 0],
                                    box_max=[3.5, 1, 1], support_point_ids=[1, 3])])
        gt = dict(boxes=np.array([[.5, 0, 0, 2, 2, 1, 0], [1., 0, 0, 1, 2, 1, 0]]),
                  ids=['repeat', 'repeat'], radar_to_lidar=np.eye(4))
        boxes, _ = frame_power_statistics(dict(P0_power=result), xyz, power, gt)
        contextual = [dict(scene='3', order=80, **r) for r in boxes]
        # Same GT object ID in the next frame remains a separate table row.
        contextual.append(dict(scene='3', order=85, **boxes[0]))
        table = all_frames_box_table(contextual)
        self.assertEqual(len(table), 4)  # Two GT + one proposal + next-frame GT.
        self.assertEqual(len({r['box_key'] for r in table}), 4)
        gt_rows = [r for r in table if r['method'] == 'GT' and r['order'] == 80]
        self.assertEqual([r['mean_power'] for r in gt_rows], [5., 10.])
        self.assertEqual([r['min_power'] for r in gt_rows], [0., 10.])
        self.assertEqual([r['max_power'] for r in gt_rows], [10., 10.])
        proposal = next(r for r in table if r['method'] == 'P0_power')
        self.assertEqual(proposal['min_power'], 10.)
        self.assertEqual(proposal['max_power'], 90.)
        self.assertEqual(proposal['mean_power'], 40.)
        self.assertEqual(proposal['support_mean_power'], 50.)
        self.assertEqual(proposal['inside_gt_n_points'], 1)
        self.assertEqual(proposal['outside_gt_n_points'], 2)
        self.assertEqual(proposal['outside_gt_min_power'], 20.)
        self.assertEqual(proposal['outside_gt_max_power'], 90.)
        self.assertEqual(proposal['outside_gt_mean_power'], 55.)
        self.assertEqual(proposal['gt_relation'], 'overlaps_gt')

    def test_empty_box_extrema_are_missing_not_zero(self):
        empty = power_statistics([])
        self.assertEqual(empty['n_points'], 0)
        for key in ('min_power', 'max_power', 'mean_power', 'median_power', 'p05_power', 'p95_power'):
            self.assertIsNone(empty[key])
        self.assertEqual(power_statistics([0.])['min_power'], 0.)


if __name__ == '__main__':
    unittest.main()
