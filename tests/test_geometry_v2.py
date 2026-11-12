"""Independent physical-coordinate checks, not shared positive-sign assumptions."""
import argparse
import inspect
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from scipy.io import loadmat

from temporal.core.ego_alignment import (RadarBinGeometry, KRADAR_COORDINATES,
    LEGACY_COORDINATES, geometry_from_args, align_to_current, radar_to_lidar_default)
from temporal.core.learned_topk import PairRanker, save_ranker, load_ranker
from temporal.fusion.box_z_p0 import BoxZP0Fusion, physical_grid, masked_interpolate, scatter_field
from tools.temporal_vis.audit_dynamic_objects import memberships, transform

ROOT = Path(__file__).resolve().parents[1]
BINS = ROOT/'data/K-Radar-official-meta/resources/info_arr.mat'


class PhysicalCoordinateTests(unittest.TestCase):
    def test_actual_tables_match_official_scalar_conversion(self):
        if not BINS.exists():
            self.skipTest('K-Radar bin resource unavailable')
        source = loadmat(BINS)
        actual = RadarBinGeometry.from_file(BINS, coordinate_convention=KRADAR_COORDINATES)
        coords = np.array([[37, 13, 5], [81, 77, 29], [126, 92, 17]])
        reference = []
        for r_id, a_id, e_id in coords:
            r = float(source['arrRange'].ravel()[r_id])
            az = -math.radians(float(source['arrAzimuth'].ravel()[a_id]))
            el = -math.radians(float(source['arrElevation'].ravel()[e_id]))
            reference.append([r*math.cos(el)*math.cos(az), r*math.cos(el)*math.sin(az), r*math.sin(el)])
        np.testing.assert_allclose(actual.to_xyz(coords), reference, atol=1e-12, rtol=0)
        self.assertEqual(actual.coordinate_convention, KRADAR_COORDINATES)

    def test_legacy_unchanged_and_explicit_v2_conflict_rejected(self):
        legacy = RadarBinGeometry([10], [.3], [.2])
        fixed = RadarBinGeometry([10], [.3], [.2], coordinate_convention=KRADAR_COORDINATES)
        np.testing.assert_allclose(fixed.to_xyz([[0, 0, 0]]), legacy.to_xyz([[0, 0, 0]])*[1, -1, -1])
        with self.assertRaises(ValueError):
            RadarBinGeometry([10], [.3], [.2], coordinate_convention=KRADAR_COORDINATES, azimuth_sign=1)

    def test_gt_stays_in_lidar_frame(self):
        geom = RadarBinGeometry([20], [.3], [.15], coordinate_convention=KRADAR_COORDINATES)
        p = geom.to_xyz([[0, 0, 0]])
        e = radar_to_lidar_default()
        center = transform(p, e)[0]
        box = np.r_[center-[0, 0, .5], [1., 1., 1.], .4][None]
        np.testing.assert_array_equal(memberships(transform(p, e), box), [0])
        np.testing.assert_array_equal(memberships(transform(p*[1, -1, -1], e), box), [-1])

    def test_world_pose_alignment_with_native_radar_basis(self):
        poses = np.tile(np.eye(4), (1, 4, 1, 1))
        for i in range(4):
            a = .07*i
            poses[0, i, :3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
            poses[0, i, :3, 3] = [i, .4*i, .1*i]
        e = radar_to_lidar_default()
        world = np.array([[20., 6., 2.], [30., -4., 1.]])
        physical = np.stack([transform(world, np.linalg.inv(p@e)) for p in poses[0]])[None]
        native = physical*[1, -1, -1]
        aligned, _ = align_to_current(native*[1, -1, -1], poses)
        np.testing.assert_allclose(aligned, np.broadcast_to(physical[:, -1:], physical.shape), atol=1e-12)
        wrong, _ = align_to_current(native, poses)
        self.assertGreater(np.max(np.abs(wrong-native[:, -1:])), .5)

    def test_box_z_forward_inverse_bin_roundtrip_and_samples(self):
        bins = [np.array([3., 8., 12., 18.]), np.array([-.4, -.1, .2, .5]), np.array([-.3, 0., .25])]
        module = BoxZP0Fusion(bins, coordinate_convention=KRADAR_COORDINATES)
        coords = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 0]])
        xyz = module.xyz(coords)
        grid, valid = physical_grid(xyz, module.bins, -1, -1)
        expected = 2*(coords.float()+.5)/torch.tensor([4., 4., 3.])-1
        torch.testing.assert_close(grid, expected.flip(-1), atol=2e-6, rtol=0)
        self.assertTrue(bool(valid.all()))
        source = torch.arange(24.).reshape(3, 8)+1
        field, obs = scatter_field(source, coords, (4, 4, 3))
        samples, good = masked_interpolate(field, obs, xyz, module.bins, azimuth_sign=-1, elevation_sign=-1)
        self.assertTrue(bool(good.all()))
        torch.testing.assert_close(samples, source, atol=2e-5, rtol=0)

    def test_legacy_ranker_rejected_by_v2_even_with_same_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new = [Path(directory)/name for name in ('old.pth', 'new.pth')]
            model = PairRanker()
            save_ranker(old, model)
            save_ranker(new, model, coordinate_convention=KRADAR_COORDINATES)
            with self.assertRaises(ValueError):
                load_ranker(old, expected_coordinate_convention=KRADAR_COORDINATES)
            load_ranker(new, expected_coordinate_convention=KRADAR_COORDINATES)
            load_ranker(old)  # Existing callers remain runnable.

    def test_all_fixed_series_configs_share_protocol_and_scratch(self):
        from mmcv import Config
        from temporal.experiments_v2.registry import experiments, WORK, SERIES, PROJECTION_VERSION, RANKER
        from temporal.temporal_plugin import TemporalPlugin

        def assert_no_merge_directives(value):
            if isinstance(value, dict):
                self.assertNotIn('_delete_', value, 'Unconsumed MMCV merge directive reaches constructor')
                for child in value.values():
                    assert_no_merge_directives(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    assert_no_merge_directives(child)

        for row in experiments().values():
            if row['kind'] != 'occupancy-training':
                continue
            cfg = Config.fromfile(str(ROOT/row['config']))
            self.assertIsNone(cfg.load_from)
            self.assertIsNone(cfg.resume_from)
            self.assertEqual(cfg.runner.max_epochs, 15)
            self.assertEqual(cfg.work_dir, row['work_dir'])
            self.assertEqual(cfg.work_dir, WORK+'/'+row['method'])
            self.assertEqual(cfg.projection_version, PROJECTION_VERSION)
            self.assertEqual(cfg.experiment_series, SERIES)
            self.assertEqual(cfg.checkpoint_config.meta.projection_version, PROJECTION_VERSION)
            self.assertEqual(cfg.data.train.ann_file, 'data/annotations/kradar_dict_train_official_temporal_doppler8_20pct.pkl')
            self.assertEqual(cfg.optimizer.lr, .0003)
            self.assertEqual(cfg.data.train.current_only, row['method'] == 'current_only')
            model = cfg.model
            assert_no_merge_directives(model)
            if model.get('temporal_cfg'):
                options = dict(model.temporal_cfg)
                options.pop('type')
                inspect.signature(TemporalPlugin).bind(**options)
                self.assertEqual(model.temporal_cfg.alignment.coordinate_convention, KRADAR_COORDINATES)
                selection = model.temporal_cfg.get('attention', {}).get('selection')
                if selection and selection['method'] == 'learned':
                    self.assertEqual(selection['checkpoint'], RANKER)
            elif 'p0_temporal_cfg' in model:
                self.assertEqual(model.p0_temporal_cfg.coordinate_convention, KRADAR_COORDINATES)
            elif 'box_z_cfg' in model:
                self.assertEqual(model.box_z_cfg.coordinate_convention, KRADAR_COORDINATES)
                self.assertTrue(model.box_z_cfg.diagnostic_path.startswith(WORK+'/'))

    def test_fixed_registry_dependencies_and_smoke_commands(self):
        from temporal.experiments_v2.registry import experiments, WORK
        registry = experiments()
        self.assertEqual(len([r for r in registry.values() if r['kind']=='occupancy-training']), 16)
        def visit(identifier, stack):
            self.assertNotIn(identifier, stack, 'Dependency cycle')
            for dependency in registry[identifier]['depends']:
                self.assertIn(dependency, registry)
                visit(dependency, stack+[identifier])
        for identifier, spec in registry.items():
            visit(identifier, [])
            if spec['kind']=='occupancy-training':
                self.assertIn('V13', spec['depends'])
                self.assertIn('model-smoke', spec['smoke_command'])
                self.assertEqual(spec['smoke_command'][-1], identifier)
                self.assertNotEqual(spec['work_dir'], 'work_dirs/geometry_v2/'+spec['method'])
        self.assertIn(WORK+'/ranker', registry['P03']['command'])

    def test_old_projection_checkpoints_rejected_including_single_frame(self):
        from tools.temporal.run_geometry_v2 import check_checkpoint
        from temporal.experiments_v2.registry import PROJECTION_VERSION
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pth'
            torch.save(dict(meta=dict(config="radar_coordinate_convention = 'kradar_physical_v2'\n")), path)
            for current_only in (False, True):
                with self.assertRaisesRegex(ValueError, 'Old projection checkpoint'):
                    check_checkpoint(path, current_only)
            config=f"projection_version = '{PROJECTION_VERSION}'\nradar_coordinate_convention = 'kradar_physical_v2'\n"
            torch.save(dict(meta=dict(config=config)), path)
            check_checkpoint(path, False)
            check_checkpoint(path, True)

    def test_table_regeneration_keeps_old_metrics_out_of_current_series(self):
        from tools.temporal.run_geometry_v2 import document
        from temporal.experiments_v2.registry import SERIES
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            old=dict(id='T01', status='旧模型已训练', SSC_mean='0.18796', checkpoint='old.pth')
            with (output/'experiments.csv').open('w', newline='') as stream:
                writer=csv.DictWriter(stream, fieldnames=list(old));writer.writeheader();writer.writerow(old)
            document(output)
            with (output/'experiments.csv').open() as stream:
                rows={row['id']:row for row in csv.DictReader(stream)}
            self.assertEqual(rows['T01']['series'], SERIES)
            self.assertEqual(rows['T01']['SSC_mean'], '')
            self.assertEqual(json.loads(rows['T01']['historical_record'])['SSC_mean'], '0.18796')
            document(output)
            with (output/'experiments.csv').open() as stream:
                again={row['id']:row for row in csv.DictReader(stream)}
            self.assertEqual(again['T01']['historical_record'], rows['T01']['historical_record'])

    def test_whole_model_smoke_does_not_claim_success_without_cuda(self):
        from unittest.mock import patch
        from tools.temporal.run_geometry_v2 import model_smoke
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'NOT run'):
                model_smoke('T00')

    def test_knn_v2_config_builds_real_temporal_module(self):
        from mmcv import Config
        from temporal.builder import build_temporal_plugin
        cfg = Config.fromfile(str(ROOT/'temporal/experiments_v2/configs/knn.py'))
        module = build_temporal_plugin(cfg.model.temporal_cfg)
        self.assertEqual(module.geometry.coordinate_convention, KRADAR_COORDINATES)
        self.assertIsNotNone(module.fusion)
        self.assertIsNone(module.fusion.selector)

    def test_paired_fusion_bypass_preserves_weights_and_restores_forward(self):
        from mmcv import Config
        from temporal.builder import build_temporal_plugin
        from tools.temporal.run_geometry_v2 import fusion_bypass
        cfg = Config.fromfile(str(ROOT/'temporal/experiments_v2/configs/knn.py'))
        module = build_temporal_plugin(cfg.model.temporal_cfg).eval()
        with torch.no_grad():
            module.fusion.output.bias.fill_(.2)
        features = torch.ones(1, 4, 3, 8)
        coords = torch.zeros(1, 4, 3, 3, dtype=torch.long)
        args = dict(poses=np.tile(np.eye(4), (1, 4, 1, 1)),
                    frame_offsets=np.arange(-3, 1)[None], valid_mask=np.ones((1, 4), bool))
        state = {key: value.clone() for key, value in module.state_dict().items()}
        original, _ = module(features, coords, **args)
        self.assertFalse(torch.equal(original, features[:, -1]))
        with self.assertRaisesRegex(RuntimeError, 'evaluation failure'):
            with fusion_bypass(module, enabled=True):
                bypassed, debug = module(features, coords, **args)
                self.assertTrue(torch.equal(bypassed, features[:, -1]))
                self.assertTrue(debug['fusion_bypassed'])
                self.assertEqual(set(state), set(module.state_dict()))
                raise RuntimeError('evaluation failure')
        restored, _ = module(features, coords, **args)
        torch.testing.assert_close(restored, original, rtol=0, atol=0)
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
