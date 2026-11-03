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
from tools.temporal_vis.vis_p1_reflection_regions import (
    project_points, project_box_edges, project_cuboid_edges, read_p1, load_scene,
    load_gt, gt_diagnostics)


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
    def test_uncapped_xy_accepts_regions_larger_than_12_by_4(self):
        cases = [([[x, y, 0] for x in (4, 7, 10, 13, 16) for y in (-1, 1)], 0, 12),
                 ([[8, -3, 0], [8, 0, 0], [8, 3, 0]], 1, 4)]
        for centers, axis, cap in cases:
            with self.subTest(axis=axis):
                xyz, power = cloud(centers)
                limited = extract_regions(xyz, power, config(merge_space='xy'))
                unlimited = extract_regions(xyz, power, config(merge_space='xy', xy_size_limit=False))
                self.assertGreater(limited['stats']['n_regions'], 1)
                self.assertEqual(unlimited['stats']['n_regions'], 1)
                region = unlimited['regions'][0]
                self.assertTrue(region['accepted'])
                self.assertGreater(region['box_size'][axis], cap)
                self.assertEqual(unlimited['stats']['merge_links_rejected_size'], 0)
                self.assertEqual(region['support_outside_box_count'], 0)
                np.testing.assert_array_equal(unlimited['fine_support_region_ids'], limited['fine_support_region_ids'])
                json.dumps(unlimited['config'], allow_nan=False)

    def test_uncapped_xy_disables_size_dependent_chain_flag_but_retains_shape_filter(self):
        xyz, power = cloud([[x, 0, 0] for x in (3, 5.5, 8, 10.5, 13, 15.5, 18)])
        result = extract_regions(xyz, power, config(merge_space='xy', xy_size_limit=False))
        self.assertEqual(result['stats']['n_regions'], 1)
        status = result['regions'][0]['status']
        self.assertNotIn('oversized_support', status)
        self.assertNotIn('long_structure_candidate', status)
        self.assertIn('elongated_support', status)

    def test_uncapped_xy_still_excludes_rejected_bridge_boxes(self):
        xyz, power = cloud([[4, 0, 0], [16, 0, 0]])
        xyz, power = np.r_[xyz, [[10, 0, 0]]], np.r_[power, [20.]]
        result = extract_regions(xyz, power, config(merge_space='xy', xy_size_limit=False, min_points=20))
        self.assertEqual(result['stats']['n_filtered_before_merge'], 1)
        self.assertEqual(result['stats']['n_regions'], 2)
        self.assertEqual(result['stats']['merge_candidate_links'], 0)
        self.assertEqual([r['source_seed_ids'] for r in result['regions']], [[1], [2]])

    def test_xy_filters_small_boxes_before_merging_and_cannot_rescue_them(self):
        xyz, power = cloud([[7, 0, 0], [9.5, 0, 0]])
        result = extract_regions(xyz, power, config(merge_space='xy', min_points=80))
        self.assertEqual(result['stats']['n_fine_regions'], 2)
        self.assertEqual(result['stats']['n_fine_accepted'], 0)
        self.assertEqual(result['stats']['n_merge_input_regions'], 0)
        self.assertEqual(result['stats']['n_filtered_before_merge'], 2)
        self.assertEqual(result['stats']['n_seed_regions_merged'], 0)
        self.assertEqual(result['regions'], [])
        self.assertTrue((result['support_region_ids'] == -1).all())

    def test_xy_rejected_box_cannot_bridge_two_accepted_boxes(self):
        xyz, power = cloud([[4, 0, 0], [16, 0, 0]])
        xyz, power = np.r_[xyz, [[10, 0, 0]]], np.r_[power, [20.]]
        result = extract_regions(xyz, power, config(merge_space='xy', min_points=20,
                                                   max_box_size=(20, 4, 4)))
        rejected = [r for r in result['fine_regions'] if not r['accepted']]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]['id'], 0)
        # This rejected middle box touches BOTH good boxes in XY. If admitted
        # as a bridge, the generous whole-union bound would allow all to merge.
        for good in (r for r in result['fine_regions'] if r['accepted']):
            gap = np.maximum(np.maximum(good['box_min'][:2] - rejected[0]['box_max'][:2],
                                        rejected[0]['box_min'][:2] - good['box_max'][:2]), 0)
            self.assertTrue((gap <= .5 + 1e-8).all())
        self.assertEqual(result['stats']['n_regions'], 2)
        self.assertEqual(result['stats']['n_merge_input_regions'], 2)
        self.assertEqual(result['stats']['n_filtered_before_merge'], 1)
        self.assertEqual(result['stats']['n_seed_regions_merged'], 0)
        self.assertEqual(result['stats']['merge_candidate_links'], 0)
        self.assertEqual([r['source_seed_ids'] for r in result['regions']], [[1], [2]])
        self.assertTrue((result['support_region_ids'][rejected[0]['support_point_ids']] == -1).all())
        arrays = npz_payload(result, xyz, power, np.zeros((len(xyz), 3), int))
        np.testing.assert_array_equal(arrays['source_seed_indices'], [1, 2])

    def test_xy_merges_passing_boxes_and_keeps_noncontiguous_seed_provenance(self):
        xyz, power = cloud([[6, 0, 0], [11, 0, 0]])
        xyz, power = np.r_[xyz, [[18, 0, 0]]], np.r_[power, [20.]]
        result = extract_regions(xyz, power, config(merge_space='xy', min_points=20))
        self.assertEqual(result['stats']['n_regions'], 1)
        self.assertTrue(result['regions'][0]['accepted'])
        self.assertEqual(result['regions'][0]['source_seed_ids'], [1, 2])
        self.assertEqual(result['stats']['n_filtered_before_merge'], 1)
        self.assertEqual(result['stats']['n_seed_regions_merged'], 1)

    def test_xy_merges_same_footprint_despite_large_height_separation(self):
        xyz, power = cloud([[8, 0, -1], [8, 0, 3]])
        cfg = config(merge_space='xy')
        result = extract_regions(xyz, power, cfg)
        self.assertEqual(result['stats']['n_fine_regions'], 2)
        self.assertEqual(result['stats']['n_regions'], 1)
        region = result['regions'][0]
        self.assertTrue(region['accepted'])
        self.assertEqual(region['source_seed_ids'], [0, 1])
        self.assertGreater(region['box_size'][2], cfg.max_box_size[2])
        self.assertEqual(region['support_outside_box_count'], 0)

    def test_xy_merges_box_footprints_without_support_cell_contact(self):
        xyz, power = cloud([[6, 0, 0], [11, 0, 0]])
        xyz_before, power_before = xyz.copy(), power.copy()
        original = extract_regions(xyz, power, config())
        merged = extract_regions(xyz, power, config(merge_space='xy'))
        self.assertEqual(original['stats']['n_regions'], 2)
        self.assertEqual(merged['stats']['n_regions'], 1)
        np.testing.assert_array_equal(xyz, xyz_before)
        np.testing.assert_array_equal(power, power_before)
        np.testing.assert_array_equal(merged['fine_support_region_ids'], original['fine_support_region_ids'])
        self.assertEqual(merged['stats']['n_recovered_support'], 0)
        again = extract_regions(xyz, power, config(merge_space='xy'))
        np.testing.assert_array_equal(merged['support_region_ids'], again['support_region_ids'])

    def test_xy_uses_edge_gap_and_keeps_distant_boxes_separate(self):
        xyz, power = cloud([[6, 0, 0], [12.4, 0, 0]])
        overlap_only = extract_regions(xyz, power, config(merge_space='xy', xy_box_gap=(0, 0)))
        small_gap = extract_regions(xyz, power, config(merge_space='xy'))
        self.assertEqual(overlap_only['stats']['n_regions'], 2)
        self.assertEqual(small_gap['stats']['n_regions'], 1)
        xyz, power = cloud([[4, 0, 0], [13, 0, 0]])
        distant = extract_regions(xyz, power, config(merge_space='xy'))
        self.assertEqual(distant['stats']['n_regions'], 2)

    def test_xy_whole_union_width_and_chain_length_are_bounded(self):
        xyz, power = cloud([[8, 0, 0], [8, 3.2, 0]])
        result = extract_regions(xyz, power, config(merge_space='xy'))
        self.assertEqual(result['stats']['n_regions'], 2)
        self.assertGreater(result['stats']['merge_links_rejected_size'], 0)
        xyz, power = cloud([[x, 0, 0] for x in np.arange(3, 18.1, 2.5)])
        cfg = config(merge_space='xy')
        result = extract_regions(xyz, power, cfg)
        self.assertGreater(result['stats']['n_regions'], 1)
        for r in result['regions']:
            self.assertTrue((r['box_size'][:2] <= np.asarray(cfg.max_box_size)[:2] + 1e-8).all())
            self.assertIn('long_structure_candidate', r['status'])

    def test_xy_grouping_does_not_flatten_z_or_change_3d_gt_scoring(self):
        xyz, power = cloud([[10, 0, 0], [10, 0, 3]])
        result = extract_regions(xyz, power, config(merge_space='xy'))
        self.assertEqual(result['stats']['n_regions'], 1)
        with tempfile.TemporaryDirectory() as directory:
            gt = gt_diagnostics(result, xyz, self.make_gt_fixture(directory))
            self.assertEqual(gt['region_labels'], {})
            self.assertLess(gt['region_diagnostics'][0]['dominant_fraction'], .6)
            self.assertTrue((gt['point_labels'][-54:] == -1).all())
        self.assertGreater(np.ptp(xyz[:, 2]), 3)

    def make_gt_fixture(self, directory, ids=(42,), boxes=None):
        root = Path(directory)
        folder = root / '3/bbox'
        folder.mkdir(parents=True, exist_ok=True)
        if boxes is None:
            boxes = [[12.54, -.3, -1.7, 4, 2, 2, np.pi / 2]]
        np.save(folder / 'bbox7.npy', np.asarray(boxes).reshape(-1, 7))
        np.save(folder / 'bbox_token7.npy', ids)
        extrinsic = np.eye(4)
        extrinsic[:3, 3] = [2.54, -.3, -.7]
        return load_gt(dict(occ_frame_idx=7), '3', root, extrinsic)

    def test_gt_preserves_yaw_bottom_height_and_sensor_extrinsic(self):
        with tempfile.TemporaryDirectory() as directory:
            gt = self.make_gt_fixture(directory)
            np.testing.assert_allclose(gt['corners'][0].min(axis=0), [9, -2, -1], atol=1e-8)
            np.testing.assert_allclose(gt['corners'][0].max(axis=0), [11, 2, 1], atol=1e-8)
            self.assertEqual(gt['source']['occ_frame_idx'], 7)
            np.testing.assert_array_equal(gt['ids'], ['42'])
            # Nontrivial GT yaw is retained, not reset to lock the image view.
            self.assertAlmostEqual(gt['boxes'][0, 6], np.pi / 2)

    def test_gt_evidence_uses_3d_support_not_every_point_in_fusion_box(self):
        with tempfile.TemporaryDirectory() as directory:
            gt = self.make_gt_fixture(directory)
            xyz = np.array([[10, 0, 0], [10, 1, 0], [10, -1, 0], [10, 0, 3],
                            [10, 0, .1], [10, 0, 4]])
            result = dict(regions=[dict(id=7, accepted=True, support_point_ids=np.arange(4))])
            gt = gt_diagnostics(result, xyz, gt)
            np.testing.assert_array_equal(gt['point_labels'], [0, 0, 0, -1, 0, -1])
            evidence = gt['region_diagnostics'][7]
            self.assertEqual(evidence['dominant_points'], 3)
            self.assertEqual(evidence['support_points'], 4)
            self.assertEqual(evidence['dominant_fraction'], .75)
            self.assertEqual(gt['region_labels'], {7: '42'})

    def test_gt_duplicate_ids_are_displayed_but_not_used_as_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            boxes = [[12.54, -.3, -1.7, 4, 2, 2, 0], [40, 0, -1, 4, 2, 2, 0]]
            gt = self.make_gt_fixture(directory, ids=(42, 42), boxes=boxes)
            self.assertTrue(gt['duplicate_ids'].all())
            gt = gt_diagnostics(dict(regions=[dict(id=0, accepted=True, support_point_ids=np.arange(3))]),
                                np.array([[10, 0, 0]] * 3), gt)
            self.assertEqual(gt['region_labels'], {})
            self.assertEqual(gt['region_diagnostics'][0]['reason'], 'duplicate_id_only')

    def test_gt_missing_frame_mapping_and_mismatched_tokens_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'occ_frame_idx'):
                load_gt({}, '3', Path(directory), np.eye(4))
            with self.assertRaisesRegex(ValueError, 'count mismatch'):
                self.make_gt_fixture(directory, ids=(42, 43))

    def test_empty_gt_and_oriented_cuboid_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = self.make_gt_fixture(directory, ids=(), boxes=[])
            self.assertEqual(empty['corners'].shape, (0, 8, 3))
            gt = self.make_gt_fixture(directory)
            projection = np.array([[0, 80, 50, 0], [0, 0, 40, 0], [1, 0, 0, 0]], float)
            edges = project_cuboid_edges(gt['corners'][0], projection)
            self.assertEqual(len(edges), 12)
            self.assertTrue(all(np.isfinite(edge).all() for edge in edges))

    def test_nearby_peaks_merge_before_small_support_filtering(self):
        xyz, power = cloud([[7, 0, 0], [9.5, 0, 0]])
        result = extract_regions(xyz, power, config(min_points=80))
        self.assertEqual(result['stats']['n_fine_regions'], 2)
        self.assertEqual(result['stats']['n_fine_accepted'], 0)
        self.assertEqual(result['stats']['n_regions'], 1)
        region = result['regions'][0]
        self.assertTrue(region['accepted'])
        self.assertEqual(region['source_seed_ids'], [0, 1])
        assigned = np.flatnonzero(result['fine_support_region_ids'] >= 0)
        self.assertTrue((result['support_region_ids'][assigned] == 0).all())

    def test_adaptive_box_encloses_all_support_of_a_longer_body(self):
        xyz, power = cloud([[5, 0, 0], [8, 0, 0], [11, 0, 0]])
        cfg = config(merge_gap=(2.1, 1, 1))
        result = extract_regions(xyz, power, cfg)
        self.assertEqual(len(result['regions']), 1)
        region = result['regions'][0]
        self.assertTrue(region['accepted'])
        self.assertGreater(region['box_size'][0], cfg.box_size[0])
        self.assertTrue((region['box_size'] <= cfg.max_box_size).all())
        self.assertEqual(region['support_outside_box_count'], 0)
        self.assertTrue(set(region['support_point_ids']).issubset(region['box_point_ids']))

    def test_unseeded_fragment_can_attach_across_small_empty_gap(self):
        xyz, power = cloud([[8, 0, 0]])
        offsets = np.array(list(itertools.product((-.5, 0, .5), repeat=3)))
        xyz = np.r_[xyz, xyz[-54:], offsets + [10.5, 0, 0]]
        power = np.r_[power, power[-54:], np.ones(len(offsets))]
        result = extract_regions(xyz, power, config())
        satellite = np.arange(len(xyz) - len(offsets), len(xyz))
        self.assertTrue((result['fine_support_region_ids'][satellite] == -1).all())
        self.assertTrue((result['support_region_ids'][satellite] >= 0).all())
        self.assertGreaterEqual(result['stats']['n_recovered_support'], len(offsets))
        self.assertEqual(result['stats']['n_fine_regions'], 1)
        self.assertEqual(result['stats']['n_regions'], 1)

    def test_real_spatial_gap_keeps_two_bodies_separate(self):
        xyz, power = cloud([[6, 0, 0], [11, 0, 0]])
        result = extract_regions(xyz, power, config())
        self.assertEqual(result['stats']['n_regions'], 2)
        self.assertEqual(result['stats']['n_seed_regions_merged'], 0)

    def test_oversized_fine_support_is_rejected_without_clipping(self):
        xyz, power = cloud([])
        body = np.array(list(itertools.product(np.arange(6, 13.1, .5), (-.5, 0, .5), (-.5, 0, .5))))
        xyz, power = np.r_[xyz, body], np.r_[power, np.full(len(body), 10.)]
        result = extract_regions(xyz, power, config(grow_half_size=(8, 2, 2), max_box_size=(6, 3, 3)))
        self.assertEqual(len(result['regions']), 1)
        region = result['regions'][0]
        self.assertFalse(region['accepted'])
        self.assertIn('oversized_support', region['status'])
        self.assertGreater(region['box_size'][0], 6)
        self.assertTrue(set(region['support_point_ids']).issubset(region['box_point_ids']))

    def test_zero_padding_still_includes_boundary_support_points(self):
        xyz, power = cloud([[5, 0, 0], [8, 0, 0], [11, 0, 0]])
        result = extract_regions(xyz, power, config(merge_gap=(2.1, 1, 1), box_padding=(0, 0, 0)))
        self.assertEqual(len(result['regions']), 1)
        region = result['regions'][0]
        self.assertTrue(set(region['support_point_ids']).issubset(region['box_point_ids']))

    def test_whole_union_size_rejects_lateral_merger(self):
        xyz, power = cloud([[8, 0, 0], [8, 3.2, 0]])
        result = extract_regions(xyz, power, config(merge_gap=(1.5, 2.6, 1)))
        self.assertEqual(result['stats']['n_regions'], 2)
        self.assertGreater(result['stats']['merge_links_rejected_size'], 0)

    def test_long_chain_is_bounded_and_marked_low_confidence(self):
        centers = [[x, 0, 0] for x in np.arange(3, 18.1, 2.5)]
        xyz, power = cloud(centers)
        cfg = config()
        result = extract_regions(xyz, power, cfg)
        self.assertGreater(result['stats']['n_regions'], 1)
        self.assertGreater(result['stats']['merge_links_rejected_size'], 0)
        self.assertEqual(result['stats']['n_accepted'], 0)
        for region in result['regions']:
            self.assertIn('long_structure_candidate', region['status'])
            self.assertTrue((region['box_size'] <= cfg.max_box_size).all())

    def test_no_merge_reproduces_original_seed_membership_and_fixed_boxes(self):
        xyz, power = cloud([[7, 0, 0], [9.5, 0, 0]])
        result = extract_regions(xyz, power, config(merge_regions=False))
        self.assertEqual(result['stats']['n_regions'], 2)
        np.testing.assert_array_equal(result['support_region_ids'], result['fine_support_region_ids'])
        self.assertEqual(result['stats']['n_recovered_support'], 0)
        for region in result['regions']:
            np.testing.assert_allclose(region['box_size'], [6, 3, 3])

    def test_merged_npz_retains_sources_and_original_point_indices(self):
        xyz, power = cloud([[7, 0, 0], [9.5, 0, 0]])
        result = extract_regions(xyz, power, config())
        again = extract_regions(xyz, power, config())
        np.testing.assert_array_equal(result['support_region_ids'], again['support_region_ids'])
        arrays = npz_payload(result, xyz, power, np.zeros((len(xyz), 3), int))
        self.assertTrue(all(value.dtype != object for value in arrays.values()))
        np.testing.assert_array_equal(arrays['source_seed_offsets'], [0, 2])
        np.testing.assert_array_equal(arrays['source_seed_indices'], [0, 1])
        np.testing.assert_array_equal(arrays['p1_npz_indices'], np.arange(len(xyz)))
        self.assertEqual(arrays['fine_center'].shape, (2, 3))
        self.assertEqual(arrays['center'].shape, (1, 3))

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
                       dict(seed_mad=np.nan), dict(max_seed_count=0), dict(merge_gap=(0, 1, 1)),
                       dict(max_box_size=(5, 3, 3)), dict(box_padding=(-1, 0, 0)),
                       dict(merge_min_contact_cells=0), dict(merge_space='bad'),
                       dict(xy_box_gap=(-1, 0)), dict(xy_box_gap=(0,)),
                       dict(xy_size_limit=False),
                       dict(merge_space='xy', merge_regions=False, xy_size_limit=False)):
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
                          radar_path=str(radar), lidar_token=f'3_{i}', occ_frame_idx=i,
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
            self.assertEqual(manifest['schema_version'], 3)
            self.assertEqual(manifest['config']['algorithm']['seed_percentile'], 97.)
            self.assertFalse(manifest['gt_overlays'])
            self.assertTrue(manifest['config']['algorithm']['merge_regions'])
            frame = json.loads((output / 'scene3_order00080.json').read_text())
            self.assertIn('fine_regions', frame)
            self.assertIn('source_seed_ids', frame['regions'][0])
            self.assertTrue((output / 'index.html').is_file())
            self.assertTrue((output / 'regions.gif').is_file())
            self.assertTrue((output / 'frame_summary.csv').is_file())
            with np.load(output / 'scene3_order00080.npz', allow_pickle=False) as arrays:
                self.assertEqual(arrays['xyz'].shape, (43750, 3))
                self.assertGreater(len(arrays['region_ids']), 0)
                self.assertIn('fine_support_region_ids', arrays)
                self.assertIn('source_seed_offsets', arrays)
            with Image.open(output / 'scene3_order00080.png') as figure:
                self.assertGreater(figure.width, 1000)
            refused = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('Output directory is not empty', refused.stderr)
            bbox_root = directory / 'gt'
            folder = bbox_root / '3/bbox'
            folder.mkdir(parents=True)
            for i in range(2):
                np.save(folder / f'bbox{i}.npy', [[25, 0, -5, 20, 10, 10, .2]])
                np.save(folder / f'bbox_token{i}.npy', [42])
            gt_output = directory / 'gt_output'
            gt_command = command.copy()
            gt_command[gt_command.index('--output-dir') + 1] = str(gt_output)
            gt_command += ['--show-gt', '--bbox-root', str(bbox_root)]
            run = subprocess.run(gt_command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            gt_manifest = json.loads((gt_output / 'manifest.json').read_text())
            self.assertTrue(gt_manifest['gt_overlays'])
            self.assertFalse(gt_manifest['gt_used_for_proposals'])
            self.assertEqual(gt_manifest['view']['image_rotation_degrees'], 0)
            self.assertTrue(gt_manifest['view']['gt_yaw_preserved'])
            self.assertIn('image-orientation:none', (gt_output / 'index.html').read_text())
            base_frame = json.loads((output / 'scene3_order00080.json').read_text())
            gt_frame = json.loads((gt_output / 'scene3_order00080.json').read_text())
            self.assertEqual(gt_frame['regions'], base_frame['regions'])
            self.assertEqual(gt_frame['stats'], base_frame['stats'])
            self.assertGreater(gt_frame['gt']['inside_gt_points'], 0)
            with np.load(output / 'scene3_order00080.npz', allow_pickle=False) as before, \
                    np.load(gt_output / 'scene3_order00080.npz', allow_pickle=False) as after:
                for key in before.files:
                    np.testing.assert_array_equal(before[key], after[key])
                self.assertEqual(after['gt_corners_radar'].shape, (1, 8, 3))
                self.assertTrue(all(after[key].dtype != object for key in after.files))
            xy_output = directory / 'xy_output'
            xy_command = [sys.executable, str(ROOT / 'tools/temporal_vis/vis_p1_reflection_regions.py'),
                '--ann-file', str(annotation), '--bins', str(bins), '--no-rgb',
                '--start-order', '80', '--end-order', '80', '--merge-space', 'xy',
                '--no-xy-size-limit',
                '--xy-box-gap-m', '.5', '.5', '--show-gt', '--bbox-root', str(bbox_root),
                '--output-dir', str(xy_output)]
            run = subprocess.run(xy_command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            xy_manifest = json.loads((xy_output / 'manifest.json').read_text())
            self.assertEqual(xy_manifest['config']['algorithm']['merge_space'], 'xy')
            self.assertFalse(xy_manifest['config']['algorithm']['xy_size_limit'])
            self.assertEqual(xy_manifest['config']['xy_merge_order'], 'filter_then_merge')
            xy_frame = json.loads((xy_output / 'scene3_order00080.json').read_text())
            good_ids = {r['id'] for r in xy_frame['fine_regions'] if r['accepted']}
            for region in xy_frame['regions']:
                self.assertTrue(set(region['source_seed_ids']).issubset(good_ids))
            with np.load(xy_output / 'scene3_order00080.npz', allow_pickle=False) as arrays:
                self.assertEqual(arrays['xyz'].shape, (43750, 3))
                self.assertIn('gt_corners_radar', arrays)


if __name__ == '__main__':
    unittest.main()

