"""CPU-only tests for region matching; no RadarOcc model or K-Radar data."""
import unittest
import csv
import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

from tools.temporal_vis.benchmark_region_matching import (
    aligned_box, gt_memberships, label_regions, select_match, summarize)


class RegionMatchingTests(unittest.TestCase):
    def test_xy_gt_membership_ignores_token_height_but_keeps_yaw(self):
        boxes = np.array([[12.54, -.3, -1.7, 4, 2, 2, np.pi / 2]])
        extrinsic = np.eye(4)
        extrinsic[:3, 3] = [2.54, -.3, -.7]
        xyz = np.array([[10, 0, 0], [10, 0, 100], [10, 1.5, -100], [11.5, 0, 0]])
        np.testing.assert_array_equal(gt_memberships(xyz, boxes, extrinsic), [0, -1, -1, -1])
        np.testing.assert_array_equal(gt_memberships(xyz, boxes, extrinsic, 'xy'), [0, 0, 0, -1])

    def test_xy_gt_membership_uses_calibrated_projected_corners_with_tilt(self):
        extrinsic = np.eye(4)
        angle = np.pi / 4
        c, s = np.cos(angle), np.sin(angle)
        extrinsic[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        boxes = [[0, 0, 0, 2, 2, 4, 0]]
        xyz = [[-2, 0, 100], [-2, 0, -100], [2, 0, 0]]
        np.testing.assert_array_equal(gt_memberships(xyz, boxes, extrinsic, 'xy'), [0, 0, -1])

    def test_xy_gt_overlaps_empty_boxes_and_tied_ids_are_unscorable(self):
        points = [[0, 0, 100], [9, 0, 0]]
        boxes = [[0, 0, 0, 2, 2, 2, 0], [0, 0, 10, 2, 2, 2, 0]]
        np.testing.assert_array_equal(gt_memberships(points, boxes, np.eye(4), 'xy'), [-2, -1])
        np.testing.assert_array_equal(gt_memberships(points, [], np.eye(4), 'xy'), [-1, -1])
        regions = [dict(id=0, accepted=True, support_point_ids=[0, 1, 2])]
        result, rows = label_regions(regions, [0, 1, -1], ['A', 'B'], purity=0,
                                    min_gt_points=1, return_diagnostics=True, reject_ties=True)
        self.assertEqual(result, {})
        self.assertEqual(rows[0]['reason'], 'ambiguous_dominant')
        result = label_regions(regions, [0, -1, -1], ['A'], purity=0,
                               min_gt_points=1, reject_ties=True)
        self.assertEqual(result, {0: 'A'})
        self.assertEqual(label_regions(regions, [0, -1, -1], ['A', 'A'], purity=0,
                                       min_gt_points=1, reject_ties=True), {})

    def test_gt_label_diagnostics_explain_each_exclusion(self):
        indices = [list(range(0, 5)), list(range(5, 9)), list(range(9, 15)),
                   list(range(15, 18)), list(range(18, 21)), []]
        regions = [dict(id=i, accepted=True, support_point_ids=ids) for i, ids in enumerate(indices)]
        labels = np.array([0, 0, 0, 1, -1, 0, 1, -1, -2,
                           0, 0, 0, 1, 1, -1, 2, 2, 2, -1, -1, -2])
        assigned, diagnostics = label_regions(regions, labels, ['A', 'B', 'DUP', 'DUP'],
                                              return_diagnostics=True)
        self.assertEqual(assigned, {0: 'A'})
        self.assertEqual([row['reason'] for row in diagnostics],
                         ['labeled', 'insufficient_gt_points', 'low_purity',
                          'duplicate_id_only', 'no_gt_overlap', 'no_support'])
        self.assertEqual(diagnostics[3]['duplicate_id_points'], 3)
        self.assertEqual(diagnostics[2]['dominant_fraction'], .5)

    def test_unknown_current_ids_do_not_contaminate_gt_metrics(self):
        known = dict(current_scorable=True, has_true_history=True, y_missed_true=False,
                     center_missed_true=False, predicted=True, predicted_id='A', correct=True)
        unknown = dict(current_scorable=False, has_true_history=None, y_missed_true=None,
                       center_missed_true=None, predicted=True, predicted_id='B', correct=None)
        stats = summarize([known, unknown])
        self.assertEqual(stats['all_queries'], 2)
        self.assertEqual(stats['all_predictions'], 2)
        self.assertEqual(stats['current_labeled'], 1)
        self.assertEqual(stats['current_unlabeled'], 1)
        self.assertEqual(stats['scorable_prediction'], 1)
        self.assertEqual(stats['precision'], 1.)
        self.assertEqual(stats['recall'], 1.)

    def run_cli_fixture(self, directory, constant_power=False, gt_overlap=False, merge_space='xyz',
                        no_xy_size_limit=False, gt_space='xyz', match_space='xyz',
                        gt_z=None, gt_purity=.6, min_gt_points=3):
        root = Path(__file__).resolve().parents[1]
        count = 175 * 250
        power = np.zeros((8, count))
        az = np.tile(np.arange(250) % 64, 175)
        el = np.full(count, 3)
        for radial in range(48, 55):
            ids = radial * 250 + np.arange(25)
            az[ids] = 30 + np.arange(25) % 4
            el[ids] = 3 + np.arange(25) % 2
            power[2, ids] = 10 - abs(radial - 51) * .1
        if constant_power:
            power[:] = 1
        radar = directory / 'radar.npz'
        np.savez(radar, power_val=power, azimuth_ind=az, elevation_ind=el)
        bins = directory / 'bins.npz'
        np.savez(bins, range_m=np.arange(175) * .462890625,
                 azimuth_rad=np.linspace(-.6, .6, 64), elevation_rad=np.linspace(-.15, .15, 8))
        bbox_dir = directory / 'bbox_root/3/bbox'
        bbox_dir.mkdir(parents=True)
        infos = []
        for i, order in enumerate(range(87, 91)):
            # All GT boxes are far from radar returns: proposals exist, GT labels do not.
            # Processed boxes use a BOTTOM center; cover radar-to-LiDAR Z offset.
            box = [25, 0, -5, 20, 10, 10, 0] if gt_overlap else [1000, 1000, 0, 2, 2, 2, 0]
            if gt_z is not None:
                box[2] = gt_z
            np.save(bbox_dir / f'bbox{i}.npy', [box])
            np.save(bbox_dir / f'bbox_token{i}.npy', [42])
            infos.append(dict(scene_id='3', order_in_scene=order, occ_frame_idx=i,
                              radar_path=str(radar), ego_pose_valid=True, ego_pose=np.eye(4)))
        annotation = directory / 'frames.pkl'
        with annotation.open('wb') as stream:
            pickle.dump(dict(infos=infos), stream)
        output = directory / 'output'
        command = [sys.executable, str(root / 'tools/temporal_vis/benchmark_region_matching.py'),
                   '--ann-file', str(annotation), '--bins', str(bins),
                   '--bbox-root', str(directory / 'bbox_root'), '--scenes', '3',
                   '--start-order', '90', '--end-order', '90', '--seed-percentile', '95',
                   '--merge-space', merge_space,
                   '--gt-space', gt_space, '--match-space', match_space,
                   '--gt-purity', str(gt_purity), '--min-gt-points', str(min_gt_points),
                   '--output-dir', str(output)]
        if no_xy_size_limit:
            command.append('--no-xy-size-limit')
        run = subprocess.run(command, cwd=root, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        return output, run.stdout

    def test_cli_exports_matches_when_every_current_gt_label_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            output, stdout = self.run_cli_fixture(Path(directory))
            with (output / 'matches.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertGreater(len(rows), 0)
            self.assertEqual({int(r['history_order']) for r in rows}, {87, 88, 89})
            self.assertTrue(all(r['current_scorable'] == 'False' for r in rows))
            self.assertTrue(all(r['predicted'] == 'True' for r in rows))
            self.assertTrue(all(r['correct'] == '' and r['has_true_history'] == '' for r in rows))
            report = json.loads((output / 'report.json').read_text())
            for metric in report['metrics']:
                self.assertEqual(metric['current_labeled'], 0)
                self.assertIsNone(metric['precision'])
                self.assertIsNone(metric['recall'])
                self.assertGreater(metric['all_queries'], 0)
            with (output / 'frame_diagnostics.csv').open() as stream:
                frames = list(csv.DictReader(stream))
            self.assertEqual(len(frames), 4)
            self.assertTrue(all(int(r['accepted']) > 0 and int(r['labeled']) == 0 for r in frames))
            with (output / 'region_gt_diagnostics.csv').open() as stream:
                diagnostics = list(csv.DictReader(stream))
            self.assertEqual({r['reason'] for r in diagnostics}, {'no_gt_overlap'})
            self.assertIn('zero GT-labeled current regions', stdout)

    def test_cli_zero_proposals_has_full_header_and_explicit_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            output, stdout = self.run_cli_fixture(Path(directory), constant_power=True)
            with (output / 'matches.csv').open() as stream:
                reader = csv.DictReader(stream)
                self.assertIn('current_scorable', reader.fieldnames)
                self.assertIn('candidates_after_center', reader.fieldnames)
                self.assertEqual(list(reader), [])
            self.assertIn('no region/history queries', stdout)
            with (output / 'frame_diagnostics.csv').open() as stream:
                self.assertTrue(all(r['accepted'] == '0' for r in csv.DictReader(stream)))

    def test_cli_known_same_id_matches_keep_valid_gt_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _ = self.run_cli_fixture(Path(directory), gt_overlap=True)
            with (output / 'matches.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            scored = [r for r in rows if r['current_scorable'] == 'True']
            self.assertTrue(scored)
            self.assertTrue(all(r['correct'] == 'True' and r['has_true_history'] == 'True' for r in scored))
            report = json.loads((output / 'report.json').read_text())
            self.assertTrue(all(r['precision'] == 1. and r['recall'] == 1. for r in report['metrics']))

    def test_cli_xy_proposal_mode_preserves_gt_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _ = self.run_cli_fixture(Path(directory), gt_overlap=True, merge_space='xy')
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['config']['merge_space'], 'xy')
            self.assertTrue(all(r['precision'] == 1. and r['recall'] == 1. for r in report['metrics']))

    def test_cli_uncapped_xy_records_experiment_and_keeps_gt_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _ = self.run_cli_fixture(Path(directory), gt_overlap=True, merge_space='xy',
                                             no_xy_size_limit=True)
            report = json.loads((output / 'report.json').read_text())
            self.assertTrue(report['config']['no_xy_size_limit'])
            self.assertTrue(all(r['precision'] == 1. and r['recall'] == 1. for r in report['metrics']))

    def test_cli_xy_scoring_recovers_height_only_misses_without_changing_links(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _ = self.run_cli_fixture(Path(directory), gt_overlap=True, gt_z=100,
                                             merge_space='xy')
            with (output / 'matches.csv').open() as stream:
                original = list(csv.DictReader(stream))
            original_report = json.loads((output / 'report.json').read_text())
            self.assertTrue(all(r['current_labeled'] == 0 for r in original_report['metrics']))
        with tempfile.TemporaryDirectory() as directory:
            output, _ = self.run_cli_fixture(Path(directory), gt_overlap=True, gt_z=100,
                merge_space='xy', gt_space='xy', gt_purity=0, min_gt_points=1)
            with (output / 'matches.csv').open() as stream:
                projected = list(csv.DictReader(stream))
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['config']['gt_space'], 'xy')
            self.assertEqual(report['config']['match_space'], 'xyz')
            self.assertTrue(all(r['precision'] == 1. and r['recall'] == 1. for r in report['metrics']))
            fields = ('current_region', 'matched_region', 'predicted', 'candidates_after_center')
            self.assertEqual([[r[f] for f in fields] for r in original],
                             [[r[f] for f in fields] for r in projected])

    def test_cli_full_xy_mode_is_recorded_and_has_valid_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _ = self.run_cli_fixture(Path(directory), gt_overlap=True, gt_z=100,
                merge_space='xy', gt_space='xy', match_space='xy', gt_purity=0, min_gt_points=1)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['config']['match_space'], 'xy')
            self.assertTrue(all(r['precision'] == 1. and r['recall'] == 1. for r in report['metrics']))
            with (output / 'frame_diagnostics.csv').open() as stream:
                frames = list(csv.DictReader(stream))
            self.assertTrue(all(r['gt_space'] == 'xy' for r in frames))

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

    def test_xy_match_ignores_height_in_center_gate_and_size_cost(self):
        high = self.history(0, 20.1, -5., [6., 3., 300.])
        high['aligned_center'][2] = 100.
        far = self.history(1, 23., -5., [7., 3., 3.])
        self.assertIsNone(select_match(self.current, [high], 1, 1., 0.,
                                       self.y_gates, self.center_gates))
        match = select_match(self.current, [high, far], 1, 0., 1.,
                             self.y_gates, self.center_gates, match_space='xy')
        self.assertEqual(match['id'], 0)
        high['aligned_center'][1] = -4.
        self.assertIsNone(select_match(self.current, [high], 1, 1., 0.,
                                       self.y_gates, self.center_gates, match_space='xy'))

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

