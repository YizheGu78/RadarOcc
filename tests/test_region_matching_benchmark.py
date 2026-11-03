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
    aligned_box, label_regions, select_match, summarize)


class RegionMatchingTests(unittest.TestCase):
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

    def run_cli_fixture(self, directory, constant_power=False, gt_overlap=False, merge_space='xyz'):
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
                   '--output-dir', str(output)]
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

