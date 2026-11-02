"""Proposal geometry and end-to-end plotting, without real data or model packages."""
import json
import itertools
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.temporal_vis.p1_reflection_regions import RegionConfig, extract_regions, grow_regions, npz_payload
from tools.temporal_vis.vis_p1_reflection_regions import project_points, project_box_edges, read_p1, load_scene


def cloud(centers):
    rng = np.random.default_rng(7)
    background = rng.uniform([0, -8, -3], [20, 8, 4], size=(1000, 3))
    xyz = [background]
    powers = [rng.uniform(.001, .1, len(background))]
    for center in centers:
        offsets = np.array(list(itertools.product((-.5, 0, .5), repeat=3)))
        xyz.append(np.repeat(offsets + center, 2, axis=0))
        powers.append(np.repeat(10 - np.linalg.norm(offsets, axis=1), 2))
    return np.concatenate(xyz), np.concatenate(powers)


def config(**kwargs):
    return RegionConfig(x_limits=(0, 20), y_limits=(-8, 8), z_limits=(-3, 5),
                        xy_resolution=.5, z_resolution=.5, range_band_m=50, **kwargs)


class ReflectionRegionTests(unittest.TestCase):
    def test_auto_scene_lookup_finds_train_instead_of_assuming_val(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / 'data/annotations'
            annotations.mkdir(parents=True)
            for split, scene in [('train', '03'), ('val', 15)]:
                path = annotations / f'kradar_dict_{split}_official_temporal_doppler8.pkl'
                with path.open('wb') as stream:
                    pickle.dump(dict(infos=[dict(scene_id=scene, order_in_scene=80)]), stream)
            path, rows = load_scene(root, None, '3')
            self.assertIn('train', path.name)
            self.assertEqual(rows[0]['order_in_scene'], 80)
            wrong = annotations / 'kradar_dict_val_official_temporal_doppler8.pkl'
            with self.assertRaisesRegex(ValueError, 'Scene 3 is absent') as caught:
                load_scene(root, wrong, '3')
            self.assertIn('15 (n=1, orders=80..80)', str(caught.exception))

    def test_auto_scene_lookup_refuses_to_merge_split_local_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / 'data/annotations'
            annotations.mkdir(parents=True)
            for split in ('train', 'test'):
                with (annotations / f'kradar_dict_{split}_official_temporal_doppler8.pkl').open('wb') as stream:
                    pickle.dump(dict(infos=[dict(scene_id=3, order_in_scene=0)]), stream)
            with self.assertRaisesRegex(ValueError, 'multiple splits'):
                load_scene(root, None, '3')

    def test_two_peaks_remain_separate_despite_weak_bridge(self):
        xyz, power = cloud([[6, 0, 0], [11, 0, 0]])
        bridge = np.c_[np.arange(6.5, 11, .5), np.zeros((9, 2))]
        xyz = np.r_[xyz, bridge]
        power = np.r_[power, np.ones(len(bridge))]
        result = extract_regions(xyz, power, config())
        accepted = [r for r in result['regions'] if r['accepted']]
        self.assertEqual(len(accepted), 2)
        self.assertTrue(all(np.linalg.norm(r['center'] - c) < 1 for r, c in
                            zip(sorted(accepted, key=lambda r: r['center'][0]), [[6, 0, 0], [11, 0, 0]])))

    def test_overlapping_bev_regions_can_have_different_heights(self):
        xyz, power = cloud([[8, 0, -1], [8, 0, 3]])
        result = extract_regions(xyz, power, config())
        accepted = [r for r in result['regions'] if r['accepted']]
        self.assertEqual(len(accepted), 2)
        self.assertGreater(np.ptp([r['center'][2] for r in accepted]), 3)

    def test_boxes_include_weak_original_points_and_preserve_ids(self):
        xyz, power = cloud([[8, 0, 0]])
        xyz = np.r_[xyz, [[8.1, .1, .1]]]
        power = np.r_[power, [.00001]]
        xyz_before, power_before = xyz.copy(), power.copy()
        result = extract_regions(xyz, power, config())
        region = next(r for r in result['regions'] if r['accepted'])
        self.assertIn(len(xyz) - 1, region['box_point_ids'])
        self.assertNotIn(len(xyz) - 1, region['support_point_ids'])
        np.testing.assert_allclose(region['box_max'] - region['box_min'], [6, 3, 3])
        np.testing.assert_array_equal(xyz, xyz_before)
        np.testing.assert_array_equal(power, power_before)
        payload = npz_payload(result, xyz, power, np.zeros((len(xyz), 3), int))
        np.testing.assert_array_equal(payload['p1_npz_indices'], np.arange(len(xyz)))
        self.assertTrue(all(v.dtype != object for v in payload.values()))
        self.assertEqual(payload['box_point_offsets'][-1], len(payload['box_point_indices']))

    def test_constant_power_does_not_force_percentile_seeds(self):
        xyz, _ = cloud([[8, 0, 0]])
        result = extract_regions(xyz, np.ones(len(xyz)), config())
        self.assertEqual(result['stats']['n_regions'], 0)
        self.assertEqual(result['stats']['n_seed_points'], 0)

    def test_invalid_points_and_empty_roi_are_handled(self):
        xyz = np.array([[np.nan, 0, 0], [100, 0, 0], [1, 0, 0]])
        power = np.array([10, 10, np.inf])
        result = extract_regions(xyz, power, config())
        self.assertEqual(result['stats']['n_analysis'], 0)
        self.assertEqual(len(result['regions']), 0)
        self.assertTrue((result['box_region_ids'] == -1).all())

    def test_growth_cannot_jump_empty_cells_and_is_bounded(self):
        cfg = config(grow_half_size=(1, 1, 1))
        support = np.zeros((10, 1, 1), bool)
        support[:4] = True
        support[6:] = True
        owner = grow_regions(support, [(0, 0, 0)], cfg)
        self.assertTrue((owner[6:] == 0).all())
        self.assertEqual(owner[3, 0, 0], 0)

    def test_deterministic_output_and_explicit_seed_cap(self):
        xyz, power = cloud([[5, 0, 0], [12, 0, 0]])
        first = extract_regions(xyz, power, config(max_seed_count=1))
        second = extract_regions(xyz, power, config(max_seed_count=1))
        np.testing.assert_array_equal(first['support_region_ids'], second['support_region_ids'])
        self.assertEqual(first['stats']['seeds_used'], 1)
        self.assertGreaterEqual(first['stats']['seeds_dropped_by_cap'], 1)

    def test_invalid_parameters_rejected(self):
        for kwargs in (dict(xy_resolution=0), dict(seed_percentile=70), dict(box_size=(1, -1, 1)),
                       dict(seed_mad=np.nan), dict(max_seed_count=0)):
            with self.assertRaises(ValueError):
                RegionConfig(**kwargs).validate()

    def test_projection_rejects_invalid_matrix_and_clips_near_plane(self):
        projection = np.array([[100, 0, 40, 0], [0, 100, 30, 0], [0, 0, 1, 0]], float)
        uv, depth = project_points(np.array([[0, 0, 5], [0, 0, -5]]), projection)
        np.testing.assert_allclose(uv[0], [40, 30])
        self.assertTrue(np.isnan(uv[1]).all())
        self.assertLess(depth[1], 0)
        edges = project_box_edges([-1, -1, -1], [1, 1, 2], projection)
        self.assertTrue(edges)
        self.assertTrue(all(np.isfinite(edge).all() for edge in edges))
        with self.assertRaises(ValueError):
            project_points(np.zeros((0, 3)), np.zeros((3, 4)))

    def test_loader_rejects_raw_spectrum_instead_of_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.npz'
            np.savez(path, power_val=np.zeros((64, 43750)))
            with self.assertRaisesRegex(ValueError, 'Doppler8'):
                read_p1(path)

    def test_cli_real_p1_shape_rgb_mapping_exports_and_gif(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            count = 175 * 250
            r = np.repeat(np.arange(175), 250)
            a = np.tile(np.arange(250) % 64, 175)
            e = np.full(count, 3)
            power = np.tile(np.linspace(.001, .02, count), (8, 1))
            for radial in range(48, 55):
                ids = radial * 250 + np.arange(25)
                a[ids] = 30 + np.arange(25) % 4
                e[ids] = 3 + np.arange(25) % 2
                power[2, ids] = 10 - abs(radial - 51) * .1
            radar = directory / 'radar.npz'
            np.savez(radar, power_val=power, range_ind=r, azimuth_ind=a, elevation_ind=e)
            bins = directory / 'bins.npz'
            np.savez(bins, range_m=np.arange(175) * .462890625,
                     azimuth_rad=np.linspace(-.6, .6, 64), elevation_rad=np.linspace(-.15, .15, 8))
            infos = [dict(scene_id='3', order_in_scene=80 + i, radar_frame_id=500 + i,
                          radar_path=str(radar), lidar_token=f'3_{i}',
                          gt_path='/must/not/be/read.npy') for i in range(2)]
            annotation = directory / 'frames.pkl'
            with annotation.open('wb') as stream:
                pickle.dump(dict(infos=infos), stream)
            camera = directory / 'rgb'
            camera.mkdir()
            Image.new('RGB', (320, 180), 'navy').save(camera / 'camera_2.png')
            Image.new('RGB', (320, 180), 'green').save(camera / 'camera_10.png')
            output = directory / 'output'
            command = [sys.executable, str(ROOT / 'tools/temporal_vis/vis_p1_reflection_regions.py'),
                '--ann-file', str(annotation), '--bins', str(bins), '--camera-dir', str(camera),
                '--start-order', '80', '--end-order', '81', '--stride', '1',
                '--output-dir', str(output), '--gif']
            run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(len(manifest['frames']), 2)
            self.assertTrue(manifest['frames'][1]['rgb']['path'].endswith('camera_10.png'))
            self.assertFalse(manifest['gt_used_for_proposals'])
            self.assertTrue((output / 'index.html').is_file())
            self.assertTrue((output / 'regions.gif').is_file())
            self.assertTrue((output / 'frame_summary.csv').is_file())
            with np.load(output / 'scene3_order00080.npz', allow_pickle=False) as arrays:
                self.assertEqual(arrays['xyz'].shape, (43750, 3))
                self.assertGreater(len(arrays['region_ids']), 0)
            with Image.open(output / 'scene3_order00080.png') as figure:
                self.assertGreater(figure.width, 1000)
            refused = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('Output directory is not empty', refused.stderr)


if __name__ == '__main__':
    unittest.main()
