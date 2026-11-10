import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from temporal.core.query_budget import TemporalQueryBudget, causal_ego_velocity, spread_by_power
from temporal.core.learned_topk import BoxTopKSelector, PairRanker, save_ranker
from temporal.core.sparse_topk import select_query_subset
from temporal.fusion.local_attention import LocalTemporalAttention
from temporal.temporal_plugin import TemporalPlugin

torch.set_num_threads(2)


class QueryBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bins = self.root/'doppler.npy'
        np.save(self.bins, (np.arange(64)-32)*.06)

    def test_exact_range_counts_reserves_and_permutation(self):
        rng = np.random.default_rng(8)
        coords = np.column_stack((np.repeat(np.arange(3), 250), np.tile(np.arange(250), 3), np.zeros(750)))
        features = rng.random((750, 8))+1
        # 100 motion, 75 uncertain, 75 static observations per range.
        for r in range(3):
            start = r*250
            features[start:start+100, 3:6] = 40
            features[start+100:start+175, 3:6] = [0, 21, 42]
            features[start+100:start+175, :3] = 1
            features[start+175:start+250, 3:6] = 32
        xyz = np.tile([10., 0, 0], (750, 1))
        for budget in (64, 32):
            selector = TemporalQueryBudget(budget, doppler_bins=self.bins)
            keep, detail = selector.select_frame(features, coords, xyz, [0, 0, 0])
            for r in range(3):
                selected = keep & (coords[:, 0] == r)
                self.assertEqual(int(selected.sum()), budget)
                self.assertEqual(int((selected & (detail['state'] == 2)).sum()), budget//2)
                self.assertEqual(int((selected & (detail['state'] == 0)).sum()), budget//4)
                self.assertEqual(int((selected & (detail['state'] == 1)).sum()), budget//4)
            permutation = rng.permutation(750)
            shuffled, _ = selector.select_frame(features[permutation], coords[permutation], xyz[permutation], [0, 0, 0])
            restored = np.zeros(750, bool)
            restored[permutation] = shuffled
            np.testing.assert_array_equal(restored, keep)

    def test_no_ego_spreads_unknowns_and_small_ranges(self):
        selector = TemporalQueryBudget(32, doppler_bins=self.bins)
        coords = np.array([[0, a, 0] for a in range(250)]+[[1, a, 0] for a in range(10)])
        features = np.ones((260, 8))
        features[:, 3:6] = 32
        keep, detail = selector.select_frame(features, coords, np.tile([10, 0, 0], (260, 1)))
        self.assertEqual(int(keep[:250].sum()), 32)
        self.assertEqual(int(keep[250:].sum()), 10)
        self.assertTrue((detail['state'] == 0).all())
        occupied = np.unique((coords[:250][keep[:250], 1]/250*8).astype(int))
        self.assertEqual(len(occupied), 8)
        eligible = np.zeros(260, bool)
        eligible[:4] = True
        keep, _ = selector.select_frame(features, coords, np.tile([10, 0, 0], (260, 1)), eligible=eligible)
        np.testing.assert_array_equal(keep, eligible)

    def test_partial_spatial_round_covers_both_angle_extremes(self):
        coords = np.array([[0, a, 0] for a in range(80)])
        features = np.ones((80, 8))
        chosen = spread_by_power(coords, features, np.arange(80), 4, (8, 1))
        np.testing.assert_array_equal(coords[chosen, 1], [10, 30, 50, 70])

    def test_causal_velocity_uses_latest_past_pose_with_lever_arm(self):
        poses = np.tile(np.eye(4), (4, 1, 1))
        poses[:, 0, 3] = [0, 1, 2, 3]
        velocity, reason = causal_ego_velocity(poses, [0, .1, .2, .3], [1, 1, 1, 1])
        np.testing.assert_allclose(velocity, [10, 0, 0])
        self.assertEqual(reason, 'ok')
        velocity, reason = causal_ego_velocity(poses, [0, .1, .2, .3], [0, 0, 0, 1])
        self.assertIsNone(velocity)
        self.assertEqual(reason, 'no_past_pose')

    def test_sparse_ranker_matches_full_and_only_scores_selected_queries(self):
        path = self.root/'ranker.pth'
        ranker = PairRanker()
        with torch.no_grad():
            ranker.network[-1].weight.normal_(std=.1)
        save_ranker(path, ranker)
        selector = BoxTopKSelector(checkpoint=path, query_chunk_size=3)
        raw = torch.rand(2, 4, 11, 8)
        xyz = np.zeros((2, 4, 11, 3))
        xyz[..., 0] = np.arange(11)*.1
        valid = np.ones((2, 4), bool)
        times = np.tile([0., .1, .2, .3], (2, 1))
        full, mask = selector(raw, xyz, valid, times)
        queries = [np.array([0, 4, 10]), np.array([1, 8])]
        scored = []
        handle = selector.ranker.register_forward_pre_hook(lambda module, args: scored.append(len(args[0])))
        sparse, good = select_query_subset(selector, raw, xyz, valid, times, None, queries)
        handle.remove()
        self.assertEqual(sum(scored), 5*3)
        for b, q in enumerate(queries):
            np.testing.assert_array_equal(sparse[b][good[b]], full[b, q][mask[b, q]])
            np.testing.assert_array_equal(good[b], mask[b, q])

    def test_attention_compacts_queries_scatter_identity_and_gradients(self):
        torch.manual_seed(4)
        module = LocalTemporalAttention(selection=dict(method='distance'), chunk_size=3)
        with torch.no_grad():
            module.output.weight.normal_(std=.04)
        reference = copy.deepcopy(module)
        raw = (torch.rand(1, 4, 12, 8)+1).requires_grad_()
        xyz = np.zeros((1, 4, 12, 3))
        xyz[..., 0] = np.arange(12)*.1
        valid, offsets = np.ones((1, 4), bool), [[-3, -2, -1, 0]]
        qids = np.array([0, 5, 11])
        attended = []
        handle = module.q.register_forward_pre_hook(lambda module, args: attended.append(len(args[0])))
        out, debug = module(raw, xyz, offsets, valid, query_indices=[qids])
        handle.remove()
        full, _ = reference(raw, xyz, offsets, valid)
        torch.testing.assert_close(out[:, qids], full[:, qids])
        rest = np.setdiff1d(np.arange(12), qids)
        self.assertTrue(torch.equal(out[:, rest], raw[:, -1, rest]))
        self.assertTrue(torch.equal(out[..., 3:6], raw[:, -1, :, 3:6]))
        self.assertEqual(sum(attended), 3)
        out.sum().backward()
        self.assertGreater(float(raw.grad[:, :-1].abs().sum()), 0)
        self.assertGreater(float(module.output.weight.grad.abs().sum()), 0)
        identity, _ = module(raw, xyz, offsets, valid, query_indices=[np.array([], dtype=int)])
        self.assertTrue(torch.equal(identity, raw[:, -1]))

    def test_sparse_reuses_full_read_only_cache_without_changing_namespace(self):
        path = self.root/'ranker.pth'
        save_ranker(path, PairRanker())
        raw = torch.rand(1, 4, 10, 8)
        xyz, valid = np.zeros((1, 4, 10, 3)), np.ones((1, 4), bool)
        times = [[0., .1, .2, .3]]
        build = BoxTopKSelector(checkpoint=path, cache_dir=self.root/'cache')
        expected, mask = build(raw, xyz, valid, times)
        read = BoxTopKSelector(checkpoint=path, cache_dir=self.root/'cache', cache_mode='read')
        with patch.object(read, '_select_online', side_effect=AssertionError('unexpected online full search')):
            selected, good = select_query_subset(read, raw, xyz, valid, times, None, [np.array([1, 4])])
        np.testing.assert_array_equal(selected[0], expected[0, [1, 4]])
        self.assertEqual(read.cache.hits, 1)
        with self.assertRaisesRegex(ValueError, 'read mode'):
            select_query_subset(build, raw, xyz, valid, times, None, [np.array([1])])

    def test_plugin_end_to_end_sparse_budget_and_frozen_ranker(self):
        bins = self.root/'geometry.npz'
        np.savez(bins, range_m=[10., 20.], azimuth_rad=np.linspace(-.4, .4, 10), elevation_rad=[0.])
        ranker = self.root/'ranker.pth'
        save_ranker(ranker, PairRanker())
        plugin = TemporalPlugin(mode='ego_attention', alignment=dict(bins_path=str(bins)),
            attention=dict(selection=dict(checkpoint=str(ranker))),
            query_budget=dict(per_range=3, doppler_bins=self.bins))
        coords = np.array([[r, a, 0] for r in range(2) for a in range(10)])
        coords = torch.tensor(np.tile(coords, (1, 4, 1, 1)))
        raw = torch.rand(1, 4, 20, 8)+1
        raw[..., 3:6] = 32
        out, debug = plugin(raw, coords, poses=torch.eye(4).repeat(1, 4, 1, 1),
            frame_offsets=torch.tensor([[-3, -2, -1, 0]]), valid_mask=torch.ones(1, 4, dtype=torch.bool),
            timestamps=torch.tensor([[0., .1, .2, .3]], dtype=torch.float64))
        self.assertEqual(debug['temporal_queries'], 6)
        self.assertTrue(torch.equal(out, raw[:, -1]))
        plugin.train()
        self.assertFalse(plugin.fusion.selector.ranker.training)
        self.assertTrue(all(not p.requires_grad for p in plugin.fusion.selector.ranker.parameters()))


if __name__ == '__main__':
    unittest.main()
