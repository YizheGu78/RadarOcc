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
            from temporal.experiments_v2.registry import BOX_RUN_SUFFIX
            suffix = BOX_RUN_SUFFIX if row['method'].startswith('box_z') else ''
            self.assertEqual(cfg.work_dir, WORK+'/'+row['method']+suffix)
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

    def test_c_gates_are_explicit_in_cache_and_model_configs(self):
        from mmcv import Config
        from temporal.experiments_v2.registry import experiments, BOX_CACHE
        registry = experiments()
        for identifier in ('P05', 'P06'):
            command = registry[identifier]['command']
            self.assertEqual(command[command.index('--y-gates')+1:command.index('--y-gates')+4], ['2', '2', '1.5'])
            self.assertEqual(command[command.index('--center-gates')+1:command.index('--center-gates')+4], ['8', '10', '12'])
        for identifier in ('T06', 'T07', 'T08', 'T09'):
            cfg = Config.fromfile(str(ROOT/registry[identifier]['config']))
            for split, cache_split in [('train', 'train20'), ('val', 'val20'), ('test', 'val20')]:
                self.assertEqual(cfg.data[split].expected_y_gates, [2, 2, 1.5])
                self.assertEqual(cfg.data[split].expected_center_gates, [8, 10, 12])
                self.assertEqual(cfg.data[split].box_cache, BOX_CACHE+'/'+cache_split)
            self.assertFalse(cfg.model.box_z_cfg.freeze_original)
            self.assertTrue(cfg.work_dir.endswith('_c_20261116'))

    def test_c_dataset_rejects_old_pair_cache(self):
        import hashlib
        from unittest.mock import patch
        from temporal.experiments_v2.plugin import BoxZTemporalDatasetV2
        from projects.occ_plugin.datasets.temporal_kradar_dataset import TemporalKRadarDataset
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ann = path/'ann'; ann.write_text('fixture')
            bins = path/'bins'; bins.write_text('fixture bins')
            manifest = dict(complete=True, frames=1, policy=dict(
                coordinate_convention=KRADAR_COORDINATES, gt_used=False,
                format='box_z_pairs_v1', weights=[1, 0], y_gates=[2, 2, 1.5],
                center_gates=[8, 10, 12], ann_sha256=hashlib.sha256(ann.read_bytes()).hexdigest(),
                bins_sha256=hashlib.sha256(bins.read_bytes()).hexdigest()))
            def initialize(instance, **kwargs):
                instance.ann_file = str(ann)
                instance.data_infos = [{}]
                instance.frame_nums = 4
                instance.current_only = False
            with patch.object(TemporalKRadarDataset, '__init__', initialize):
                (path/'manifest.json').write_text(json.dumps(manifest))
                BoxZTemporalDatasetV2(path, box_bins_path=bins,
                    expected_y_gates=[2, 2, 1.5], expected_center_gates=[8, 10, 12])
                manifest['policy']['y_gates'] = [4, 5, 3]
                manifest['policy']['center_gates'] = [4, 6, 8]
                (path/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'matching gates'):
                    BoxZTemporalDatasetV2(path, box_bins_path=bins,
                        expected_y_gates=[2, 2, 1.5], expected_center_gates=[8, 10, 12])

    def test_online_box_pairs_use_loaded_radar_and_ignore_missing_history(self):
        from unittest.mock import patch
        from temporal.experiments_v2.plugin import BoxZTemporalDatasetV2
        from projects.occ_plugin.datasets.temporal_kradar_dataset import TemporalKRadarDataset
        with tempfile.TemporaryDirectory() as directory:
            bins = Path(directory)/'bins.npz'
            np.savez(bins, range_m=np.arange(1., 176.), azimuth_rad=[-.1, 0., .1],
                     elevation_rad=[-.1, .1])
            infos = []
            for order in range(4):
                pose = np.eye(4); pose[0, 3] = .2*order
                infos.append(dict(scene_id='11', order_in_scene=order,
                                  ego_pose=pose, ego_pose_valid=True))
            radar = dict(power_val=np.ones((8, 175*250)),
                         azimuth_ind=np.zeros(175*250, int),
                         elevation_ind=np.zeros(175*250, int))
            payload = dict(temporal_sparse_radar=[radar]*4,
                           temporal_valid_mask=np.ones(4, bool))
            region = dict(id=0, accepted=True, center=np.array([10., 0., 0.]),
                          box_size=np.array([6., 3., 3.]))
            def initialize(instance, **kwargs):
                instance.frame_nums = 4
                instance.current_only = False
                instance.data_infos = infos
                instance.sequence_indices = [list(range(4))]
            with patch.object(TemporalKRadarDataset, '__init__', initialize), \
                 patch.object(TemporalKRadarDataset, '_attach_temporal_payload',
                              lambda *args: dict(payload)):
                data = BoxZTemporalDatasetV2(box_bins_path=bins, box_pair_mode='online',
                    box_cache=Path(directory)/'nonexistent',
                    expected_y_gates=[2, 2, 1.5], expected_center_gates=[8, 10, 12])
                with patch('numpy.load', side_effect=AssertionError('Online pairing must not load cache or GT')), \
                     patch('tools.temporal_vis.p1_reflection_regions.extract_regions',
                           return_value={'regions': [region]}) as extract:
                    result = data._attach_temporal_payload({}, 0)['box_pairs']
                    self.assertEqual(extract.call_count, 4)
                    self.assertTrue(result['matched'].all())
                    np.testing.assert_allclose(result['history'][0, :, 0], [9.8, 9.6, 9.4])
                    np.testing.assert_allclose(result['inverse'][:, 0, 3], [.2, .4, .6])
                    data.sequence_indices = [[0, 0, 0, 0]]
                    payload['temporal_valid_mask'] = np.array([False, False, False, True])
                    extract.reset_mock()
                    result = data._attach_temporal_payload({}, 0)['box_pairs']
                    self.assertEqual(extract.call_count, 1)
                    self.assertFalse(result['matched'].any())
                    np.testing.assert_array_equal(result['inverse'], np.repeat(np.eye(4)[None], 3, 0))
                self.assertIsNone(data.box_manifest)
                self.assertEqual(list(Path(directory).iterdir()), [bins])

    def test_readwrite_pairs_reuse_across_restarts_and_reject_stale_inputs(self):
        from unittest.mock import patch
        from temporal.experiments_v2.plugin import BoxZTemporalDatasetV2
        from projects.occ_plugin.datasets.temporal_kradar_dataset import TemporalKRadarDataset
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            bins, ann, radar = directory/'bins.npz', directory/'ann.pkl', directory/'radar.npz'
            np.savez(bins, range_m=np.arange(1., 176.), azimuth_rad=[-.1, .1],
                     elevation_rad=[-.1, .1])
            ann.write_bytes(b'annotation')
            radar.write_bytes(b'radar')
            infos = [dict(scene_id='11', order_in_scene=i, sparse_radar_path=str(radar))
                     for i in range(5)]
            def initialize(instance, **kwargs):
                instance.frame_nums, instance.current_only = 4, False
                instance.ann_file = str(ann)
                instance.data_infos = infos
                instance.sequence_indices = [list(range(4)), list(range(1, 5))]
            payload = dict(temporal_valid_mask=np.ones(4, bool))
            pairs = dict(current=np.ones((1, 6), np.float32),
                         history=np.ones((1, 3, 6), np.float32),
                         matched=np.ones((1, 3), bool), inverse=np.repeat(np.eye(4)[None], 3, 0))
            options = dict(box_bins_path=bins, box_pair_mode='readwrite',
                           box_cache=directory/'cache', expected_y_gates=[2, 2, 1.5],
                           expected_center_gates=[8, 10, 12])
            with patch.object(TemporalKRadarDataset, '__init__', initialize), \
                 patch.object(TemporalKRadarDataset, '_attach_temporal_payload',
                              lambda *args: dict(payload)):
                data = BoxZTemporalDatasetV2(**options)
                with patch.object(data, '_online_box_pairs', return_value=pairs) as compute:
                    data._attach_temporal_payload({}, 0)
                    data._attach_temporal_payload({}, 0)
                    self.assertEqual(compute.call_count, 1)
                restarted = BoxZTemporalDatasetV2(**options)
                with patch.object(restarted, '_online_box_pairs', side_effect=AssertionError('cache miss')):
                    actual = restarted._attach_temporal_payload({}, 0)['box_pairs']
                    for name in pairs:
                        np.testing.assert_array_equal(actual[name], pairs[name])
                # A partially populated cache remains usable; a new frame is computed lazily.
                empty = dict(current=np.empty((0, 6), np.float32),
                             history=np.empty((0, 3, 6), np.float32),
                             matched=np.empty((0, 3), bool), inverse=pairs['inverse'])
                with patch.object(restarted, '_online_box_pairs', return_value=empty) as compute:
                    restarted._attach_temporal_payload({}, 1)
                    result = restarted._attach_temporal_payload({}, 1)['box_pairs']
                    self.assertEqual(compute.call_count, 1)
                    self.assertEqual(result['current'].shape, (0, 6))
                with self.assertRaisesRegex(ValueError, 'policy mismatch'):
                    BoxZTemporalDatasetV2(**dict(options, expected_center_gates=[4, 6, 8]))
                radar.write_bytes(b'changed radar')
                with self.assertRaisesRegex(ValueError, 'input/history changed'):
                    restarted._attach_temporal_payload({}, 0)
                ann.write_bytes(b'new annotation')
                with self.assertRaisesRegex(ValueError, 'policy mismatch'):
                    BoxZTemporalDatasetV2(**options)
                self.assertFalse(list((directory/'cache').glob('.pending-*')))

    def test_online_box_pairs_reject_invalid_protocol(self):
        from unittest.mock import patch
        from temporal.experiments_v2.plugin import BoxZTemporalDatasetV2
        from projects.occ_plugin.datasets.temporal_kradar_dataset import TemporalKRadarDataset
        def initialize(instance, **kwargs):
            instance.frame_nums = 4
            instance.current_only = False
        with self.assertRaisesRegex(ValueError, 'box_pair_mode'):
            BoxZTemporalDatasetV2(box_pair_mode='typo')
        with patch.object(TemporalKRadarDataset, '__init__', initialize):
            with self.assertRaisesRegex(ValueError, 'positive gates'):
                BoxZTemporalDatasetV2(box_pair_mode='online', expected_y_gates=[2, -2, 1.5])
            with self.assertRaisesRegex(ValueError, 'box_cache'):
                BoxZTemporalDatasetV2(box_pair_mode='cache')

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


class FusionBatchEvaluationTests(unittest.TestCase):
    def test_best_checkpoint_selection_never_uses_latest_or_guesses(self):
        from tools.temporal.ablate_all_fusion import choose_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'latest.pth').touch()
            self.assertIsNone(choose_checkpoint(root))
            first = root/'best_SSC_mean_epoch_3.pth'
            first.touch()
            self.assertEqual(choose_checkpoint(root), first)
            (root/'best_SSC_mean_epoch_7.pth').touch()
            with self.assertRaisesRegex(ValueError, 'Multiple best'):
                choose_checkpoint(root)
            self.assertEqual(choose_checkpoint(root, first), first)

    def test_all_temporal_methods_have_a_supported_boundary(self):
        from mmcv import Config
        from tools.temporal.ablate_all_fusion import model_signature
        from temporal.experiments_v2.registry import experiments
        signatures = []
        for spec in experiments().values():
            if spec['kind'] != 'occupancy-training':
                continue
            cfg = Config.fromfile(str(ROOT/spec['config']))
            signature = model_signature(cfg.model)
            if spec['id'] != 'T00':
                self.assertFalse(cfg.data.val.current_only)
                self.assertTrue(signature['mode'] or signature['box'] or signature['port'] or signature['unaligned'])
            signatures.append(json.dumps(signature, sort_keys=True))
        self.assertEqual(len(signatures), 16)
        self.assertEqual(len(set(signatures)), 16)

    def test_plain_and_imported_plugin_bypass_keeps_weights_and_restores_on_error(self):
        from types import SimpleNamespace
        from temporal.centerformer.fusion import CenterFormerP0
        from temporal.rodnet.fusion import RODNetP0
        from temporal.fusion.unaligned_memory import UnalignedMemoryTemporal
        from tools.temporal.run_geometry_v2 import model_fusion_bypass
        if not BINS.exists():
            self.skipTest('K-Radar bin resource unavailable')
        modules = [CenterFormerP0(bins_path=str(BINS), channels=8, heads=2),
                   RODNetP0(bins_path=str(BINS), channels=8),
                   UnalignedMemoryTemporal(embed_dims=8, num_heads=2, num_layers=1, memory_tokens=2)]
        features = torch.arange(1*4*3*8).reshape(1, 4, 3, 8).float()
        coords = torch.zeros(1, 4, 3, 3)
        for module in modules:
            state = {key: value.clone() for key, value in module.state_dict().items()}
            original_forward = module.forward
            with self.assertRaisesRegex(RuntimeError, 'deliberate failure'):
                with model_fusion_bypass(SimpleNamespace(temporal_plugin=module), True):
                    result, debug = module(features=features, coords=coords)
                    self.assertTrue(torch.equal(result, features[:, -1]))
                    self.assertTrue(debug['fusion_bypassed'])
                    raise RuntimeError('deliberate failure')
            self.assertEqual(module.forward, original_forward)
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_box_bypass_uses_last_time_slot_and_preserves_module(self):
        from types import SimpleNamespace
        from tools.temporal.run_geometry_v2 import model_fusion_bypass
        bins = [np.array([1., 2.]), np.array([-.1, .1]), np.array([-.1, .1])]
        module = BoxZP0Fusion(bins, dim=8, heads=2)
        module.diagnostics = dict(fused_queries=5)
        original_forward = module.forward
        state = {k: v.clone() for k, v in module.state_dict().items()}
        features = torch.arange(4*3*8).reshape(4, 3, 8).float()
        with model_fusion_bypass(SimpleNamespace(box_z_fusion=module), True):
            output = module(features, torch.zeros(4, 3, 3), {})
            self.assertTrue(torch.equal(output, features[-1]))
            self.assertTrue(module.diagnostics['fusion_bypassed'])
        self.assertEqual(module.forward, original_forward)
        self.assertEqual(module.diagnostics, dict(fused_queries=5))
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_near_power_mask_remains_active_when_attention_is_bypassed(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from mmcv import Config
        from temporal.builder import build_temporal_plugin
        from tools.temporal.run_geometry_v2 import model_fusion_bypass
        if not BINS.exists():
            self.skipTest('K-Radar bin resource unavailable')
        cfg = Config.fromfile(str(ROOT/'temporal/experiments_v2/configs/knn_mask37.py'))
        module = build_temporal_plugin(cfg.model.temporal_cfg).eval()
        module.geometry = RadarBinGeometry([1., 10.], [0.], [0.])
        features = torch.ones(1, 4, 2, 8)
        coords = torch.zeros(1, 4, 2, 3, dtype=torch.long)
        coords[:, :, 1, 0] = 1
        kwargs = dict(poses=np.tile(np.eye(4), (1, 4, 1, 1)),
                      frame_offsets=np.arange(-3, 1)[None], valid_mask=np.ones((1, 4), bool))
        with patch.object(module.fusion, 'forward', side_effect=lambda x, *a, **kw: (x[:, -1]+2, {})):
            on, _ = module(features, coords, **kwargs)
            with model_fusion_bypass(SimpleNamespace(temporal_plugin=module), True):
                off, _ = module(features, coords, **kwargs)
            self.assertFalse(torch.equal(on, off))
        self.assertTrue(torch.equal(off[0, 0, [0, 1, 2, 6, 7]], torch.zeros(5)))
        self.assertTrue(torch.equal(off[0, 1], torch.ones(8)))
        self.assertTrue(torch.equal(off[..., 3:6], features[:, -1, :, 3:6]))
        self.assertTrue(torch.equal(features, torch.ones_like(features)))

    def test_checkpoint_preflight_rejects_wrong_method_version_and_val_split(self):
        import copy
        from mmcv import Config
        from temporal.experiments_v2.registry import experiments
        from tools.temporal.ablate_all_fusion import inspect_checkpoint, validation_contract
        spec = experiments()['T10']
        cfg = Config.fromfile(str(ROOT/spec['config']))
        contract = validation_contract(cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'best.pth'
            def save(config):
                torch.save(dict(meta=dict(config=config.pretty_text, epoch=3),
                                state_dict={'weight': torch.ones(1)}), path)
            save(cfg)
            self.assertEqual(inspect_checkpoint(path, spec, contract)['epoch'], 3)
            with self.assertRaisesRegex(ValueError, 'method disagrees'):
                inspect_checkpoint(path, experiments()['T11'], contract)
            old = copy.deepcopy(cfg)
            old.projection_version = 'old'
            save(old)
            with self.assertRaisesRegex(ValueError, 'Old projection'):
                inspect_checkpoint(path, spec, contract)
            changed = copy.deepcopy(cfg)
            changed.data.val.ann_file = 'data/annotations/different.pkl'
            save(changed)
            with self.assertRaisesRegex(ValueError, 'common val20'):
                inspect_checkpoint(path, spec, contract)

    def test_summary_contains_baseline_delta_and_missing_runs_without_fake_metrics(self):
        from tools.temporal.ablate_all_fusion import collect
        metrics = dict(SSC_mean=.3, SSC_Foreground=.2, SSC_Background=.4,
                       **{'SC_non-empty': .5}, SSC1_mean=.4, SSC2_mean=.5)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            items = []
            for identifier, arms in [('T00', ['reference']), ('T10', ['on', 'off'])]:
                target = output/identifier
                target.mkdir()
                for arm in arms:
                    values = dict(metrics, SSC_mean=.2 if arm == 'off' else .3)
                    (target/f'{arm}_metrics.json').write_text(json.dumps(dict(metrics=values,
                        fusion_diagnostics=dict(batches_using_history=int(arm == 'on'), queries_with_history=4))))
                items.append(dict(id=identifier, method=identifier, status='completed', output=str(target)))
            items.append(dict(id='T11', method='rodnet_tdc', status='missing', reason='No best'))
            collect(dict(experiments=items, validation=dict(annotation='val20')), output)
            with (output/'summary.csv').open() as stream:
                rows = {row['id']: row for row in csv.DictReader(stream)}
            self.assertAlmostEqual(float(rows['T10']['on_minus_off_pp']), 10.)
            self.assertAlmostEqual(float(rows['T10']['on_minus_T00_pp']), 0.)
            self.assertEqual(rows['T11']['on_SSC_mean'], '')
            self.assertEqual(rows['T11']['status'], 'missing')

    def test_paired_evaluator_loads_once_and_resumes_only_the_missing_arm(self):
        from unittest.mock import patch
        from mmcv import Config
        from tools.temporal.run_geometry_v2 import knn_ablation
        cfg = Config.fromfile(str(ROOT/'temporal/experiments_v2/configs/centerformer.py'))
        if not (ROOT/cfg.data.val.ann_file).exists():
            self.skipTest('val20 annotation unavailable')

        class ToyPlugin(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(2.))

            def forward(self, features, coords):
                return features[:, -1]+self.bias, dict(used_history=True, queries_with_history=2)

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.temporal_plugin = ToyPlugin()

            def cuda(self):
                return self

            def forward(self, features, coords):
                return self.temporal_plugin(features, coords)[0]

        class ToyDataset:
            CLASSES = ('free', 'background', 'foreground')

            def __len__(self):
                return 2

            def evaluate(self, aggregate):
                score = aggregate['score']
                return {key: score for key in ('SSC_mean', 'SSC_Foreground', 'SSC_Background',
                                               'SC_non-empty', 'SSC1_mean', 'SSC2_mean')}

        visits = []
        def evaluate(model, loader):
            visits.append(id(model))
            values = [model(torch.ones(1, 4, 2, 8), torch.zeros(1, 4, 2, 3)) for _ in loader]
            return dict(score=float(torch.stack(values).mean())/10)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/'best.pth'
            torch.save(dict(meta=dict(config=cfg.pretty_text, epoch=3), state_dict=ToyModel().state_dict()), checkpoint)
            output = root/'paired'
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.set_device'), \
                 patch('mmdet.apis.set_random_seed'), \
                 patch('mmcv.parallel.MMDataParallel', side_effect=lambda model, **kw: model), \
                 patch('mmdet3d.models.build_model', side_effect=lambda *a, **kw: ToyModel()) as build, \
                 patch('projects.occ_plugin.datasets.builder.custom_build_dataset', return_value=ToyDataset()), \
                 patch('projects.occ_plugin.datasets.builder.build_dataloader', return_value=[0, 1]), \
                 patch('projects.occ_plugin.occupancy.apis.test.custom_single_gpu_test', side_effect=evaluate):
                result = knn_ablation(checkpoint, output, identifier='T10', resume=True)
                self.assertEqual(build.call_count, 1)
                self.assertEqual(visits[0], visits[1])
                self.assertAlmostEqual(result['on']['SSC_mean'], .3)
                self.assertAlmostEqual(result['off']['SSC_mean'], .1)
                on_report = json.loads((output/'on_metrics.json').read_text())
                self.assertEqual(on_report['fusion_diagnostics']['queries_with_history'], 4)
                self.assertEqual(json.loads((output/'off_metrics.json').read_text())['fusion_diagnostics']['batches_using_history'], 0)
                (output/'off_metrics.json').unlink()
                knn_ablation(checkpoint, output, identifier='T10', resume=True)
                self.assertEqual(len(visits), 3)  # Re-evaluate off only.
            with patch('torch.cuda.is_available', return_value=False):
                again = knn_ablation(checkpoint, output, identifier='T10', resume=True)
                self.assertEqual(result, again)
            changed = torch.load(checkpoint, weights_only=False)
            changed['state_dict']['temporal_plugin.bias'].fill_(7.)
            torch.save(changed, checkpoint)
            with self.assertRaisesRegex(ValueError, 'Checkpoint/data/code changed'):
                knn_ablation(checkpoint, output, identifier='T10', resume=True)


if __name__ == '__main__':
    unittest.main()
