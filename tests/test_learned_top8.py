import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from temporal.core.box_candidates import box_candidates, temporal_box_candidates
from temporal.core.learned_topk import (DEFAULT_POLICY, FORMAT, BoxTopKSelector,
    PairRanker, load_ranker, masked_topk, normalize_descriptors, ranking_loss, save_ranker)
from temporal.fusion.local_attention import LocalTemporalAttention
from temporal.temporal_plugin import TemporalPlugin
from tools.temporal.train_top8_ranker import PairCache, ensure_disjoint, evaluate

torch.set_num_threads(2)


class BoxCandidateTests(unittest.TestCase):
    def test_matches_bruteforce_with_closed_boundary_and_ties(self):
        rng = np.random.default_rng(2)
        history = np.vstack((rng.normal(size=(93, 3)), [[4., 1.5, 1.], [-4., -1.5, -1.]]))
        queries = np.vstack(([[0., 0., 0.]], rng.normal(size=(8, 3))))
        half = np.array([4., 1.5, 1.])
        ids, mask = box_candidates(queries, history, half, k=64, query_chunk=3)
        for i, query in enumerate(queries):
            ix = np.flatnonzero((np.abs(history-query) <= half+1e-9).all(1))
            distance = ((history[ix]-query)**2).sum(1)
            expected = ix[np.lexsort((ix, distance))[:64]]
            np.testing.assert_array_equal(ids[i, mask[i]], expected)
        ids, valid = box_candidates([[0., 0., 0.]], [[1., 0., 0.], [-1., 0., 0.]], half)
        np.testing.assert_array_equal(ids[0, :2], [0, 1])
        self.assertEqual(int(valid.sum()), 2)

    def test_per_frame_widths_and_masks(self):
        xyz = np.zeros((1, 4, 2, 3))
        xyz[:, :-1, :, 0] = 5.
        frames = np.ones((1, 4), bool)
        ids, valid = temporal_box_candidates(xyz, frames)
        self.assertTrue(valid[..., :64].any())    # t-3: +/-8
        self.assertTrue(valid[..., 64:128].any()) # t-2: +/-6
        self.assertFalse(valid[..., 128:].any())  # t-1: +/-4
        frames[:, 0] = False
        points = np.ones((1, 4, 2), bool)
        points[:, -1, 0] = False
        _, valid = temporal_box_candidates(xyz, frames, point_valid_mask=points)
        self.assertFalse(valid[..., :64].any())
        self.assertFalse(valid[:, 0].any())

    def test_empty_and_bad_inputs(self):
        ids, valid = box_candidates(np.zeros((2, 3)), np.zeros((0, 3)), [4., 1.5, 1.])
        self.assertEqual(ids.shape, (2, 64))
        self.assertFalse(valid.any())
        with self.assertRaises(ValueError):
            box_candidates(np.zeros((1, 3)), np.zeros((1, 3)), [0., 1., 1.])


class RankerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)

    def test_raw_dynamic_range_is_finite_and_stateless(self):
        raw = torch.tensor([[1e34, 0., -1e20, 1., 31., 63., 1e32, 2.99e34]])
        result = normalize_descriptors(raw)
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result[:, 3:6], raw[:, 3:6]/64)
        torch.testing.assert_close(result, normalize_descriptors(raw.repeat(7, 1))[:1])

    def test_zero_initialization_reproduces_distance_and_padding(self):
        model = PairRanker()
        rel = torch.zeros(2, 64, 3)
        rel[..., 0] = torch.arange(64)
        score = model(torch.ones(2, 8), torch.ones(2, 64, 8), rel, -torch.ones(2))
        valid = torch.zeros(2, 64, dtype=torch.bool)
        valid[0, :3] = True
        order, good = masked_topk(score, valid)
        self.assertEqual(order[0, :3].tolist(), [0, 1, 2])
        self.assertEqual(int(good[0].sum()), 3)
        self.assertFalse(good[1].any())

    def test_all_candidates_supervised_and_no_positive_skipped(self):
        score = torch.zeros(3, 64, requires_grad=True)
        positive = torch.zeros_like(score, dtype=torch.bool)
        positive[0, 63] = True
        valid = torch.ones_like(positive)
        valid[2] = False
        ranking_loss(score, positive, valid).backward()
        self.assertLess(float(score.grad[0, 63]), 0.)
        self.assertGreater(float(score.grad[0, 0]), 0.)
        self.assertEqual(float(score.grad[1:].abs().sum()), 0.)
        self.assertEqual(float(ranking_loss(score, torch.zeros_like(positive), valid).detach()), 0.)

    def test_training_can_promote_farther_same_descriptor_candidate(self):
        model = PairRanker(hidden_dims=16)
        q = torch.ones(8, 8)
        h = torch.ones(8, 64, 8)*100
        h[:, 1] = 1
        rel = torch.zeros(8, 64, 3)
        rel[:, :, 0] = torch.arange(1, 65)
        dt = -torch.ones(8)
        positive = torch.zeros(8, 64, dtype=torch.bool)
        positive[:, 1] = True
        valid = torch.ones_like(positive)
        before = ranking_loss(model(q, h, rel, dt), positive, valid).item()
        optimizer = torch.optim.Adam(model.parameters(), lr=.02)
        for _ in range(80):
            optimizer.zero_grad()
            loss = ranking_loss(model(q, h, rel, dt), positive, valid)
            loss.backward()
            optimizer.step()
        scores = model(q, h, rel, dt)
        self.assertLess(ranking_loss(scores, positive, valid).item(), before*.4)
        self.assertTrue((scores.argmax(-1) == 1).all())

    def test_checkpoint_roundtrip_and_policy_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ranker.pth'
            original = PairRanker(16)
            save_ranker(path, original, epoch=1)
            loaded, payload = load_ranker(path, DEFAULT_POLICY)
            self.assertEqual(payload['metadata']['epoch'], 1)
            for key, value in original.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key])
            with self.assertRaises(ValueError):
                load_ranker(path, dict(DEFAULT_POLICY, time_unit='frames'))
            with self.assertRaises(FileNotFoundError):
                load_ranker(Path(tmp)/'missing.pth')

    def test_selector_matches_distance_when_initialized_and_requires_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ranker.pth'
            save_ranker(path, PairRanker())
            selector = BoxTopKSelector(checkpoint=path, query_chunk_size=3)
            baseline = BoxTopKSelector(method='distance')
            features = torch.rand(1, 4, 12, 8)
            xyz = np.zeros((1, 4, 12, 3))
            xyz[..., 0] = np.arange(12)*.2
            frames = np.ones((1, 4), bool)
            with self.assertRaises(ValueError):
                selector(features, xyz, frames)
            selected, mask = selector(features, xyz, frames, [[0., .1, .2, .3]])
            expected, good = baseline(features, xyz, frames)
            np.testing.assert_array_equal(selected, expected)
            np.testing.assert_array_equal(mask, good)
            frames[:, 0] = False
            selector(features, xyz, frames, [[np.nan, .1, .2, .3]])

    def test_frozen_selector_and_attention_input_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ranker.pth'
            save_ranker(path, PairRanker())
            module = LocalTemporalAttention(selection=dict(checkpoint=str(path)), chunk_size=4)
            module.train()
            self.assertFalse(module.selector.ranker.training)
            self.assertTrue(all(not p.requires_grad for p in module.selector.parameters()))
            features = (torch.rand(1, 4, 12, 8)+1).requires_grad_()
            xyz = np.zeros((1, 4, 12, 3))
            xyz[..., 0] = np.arange(12)*.2
            offsets = np.array([[-3, -2, -1, 0]])
            frames = np.ones((1, 4), bool)
            timestamps = [[0., .1, .2, .3]]
            output, _ = module(features, xyz, offsets, frames, timestamps=timestamps)
            self.assertTrue(torch.equal(output, features[:, -1]))
            with torch.no_grad():
                module.output.weight.fill_(.03)
            before = copy.deepcopy(module.selector.state_dict())
            optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=.01)
            output, _ = module(features, xyz, offsets, frames, timestamps=timestamps)
            output.sum().backward()
            self.assertGreater(float(features.grad[:, :-1].abs().sum()), 0.)
            self.assertGreater(float(module.q.weight.grad.abs().sum()), 0.)
            self.assertTrue(all(p.grad is None for p in module.selector.parameters()))
            optimizer.step()
            for key, value in before.items():
                self.assertTrue(torch.equal(value, module.selector.state_dict()[key]))
            self.assertTrue(torch.equal(output[..., 3:6], features[:, -1, :, 3:6]))

    def test_plugin_passes_real_timestamps_to_frozen_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ranker.pth'
            bins = Path(tmp)/'bins.npz'
            save_ranker(path, PairRanker())
            np.savez(bins, range_m=np.arange(16), azimuth_rad=[0.], elevation_rad=[0.])
            plugin = TemporalPlugin(mode='ego_attention', alignment=dict(bins_path=str(bins)),
                attention=dict(selection=dict(checkpoint=str(path))))
            plugin.train()
            coords = torch.zeros(1, 4, 3, 3)
            coords[..., 0] = torch.tensor([4, 5, 6])
            poses = torch.eye(4).repeat(1, 4, 1, 1)
            features = torch.rand(1, 4, 3, 8)
            valid = torch.ones(1, 4, dtype=torch.bool)
            offsets = torch.tensor([[-3, -2, -1, 0]])
            times = torch.tensor([[0., .1, .2, .3]], dtype=torch.float64)
            out, debug = plugin(features, coords, poses, offsets, valid_mask=valid, timestamps=times)
            self.assertTrue(torch.equal(out, features[:, -1]))
            self.assertTrue(debug['used_history'])
            self.assertFalse(plugin.fusion.selector.ranker.training)
            with self.assertRaisesRegex(ValueError, 'timestamps'):
                plugin(features, coords, poses, offsets, valid_mask=valid)
            valid[:, :-1] = False
            times[:, :-1] = torch.nan
            out, debug = plugin(features, coords, poses, offsets, valid_mask=valid, timestamps=times)
            self.assertTrue(torch.equal(out, features[:, -1]))
            self.assertFalse(debug['used_history'])


class CacheTests(unittest.TestCase):
    def test_train_val_overlap_including_history_is_rejected(self):
        class Cache:
            pass
        train, val = Cache(), Cache()
        train.manifest = dict(split='train', frames=['3/80', '3/79'],
                              bins_sha256='bins', extrinsic_sha256='extrinsic')
        val.manifest = dict(train.manifest, split='val', frames=['3/79', '3/81'])
        with self.assertRaisesRegex(ValueError, 'leakage'):
            ensure_disjoint(train, val)
        val.manifest['frames'] = ['4/80']
        ensure_disjoint(train, val)
        val.manifest['split'] = 'test'
        with self.assertRaises(ValueError):
            ensure_disjoint(train, val)

    def test_evaluation_keeps_no_positive_queries_and_uses_actual_selected_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            positive = np.zeros((3, 64), bool)
            positive[0, 0], positive[1, 20] = True, True
            valid = np.zeros_like(positive)
            valid[0, :2] = True
            valid[1:, :] = True
            np.savez(root/'pairs.npz', query=np.ones((3, 8), np.float32),
                history=np.ones((3, 64, 8), np.float32),
                relative_xyz=np.broadcast_to(np.arange(64)[None, :, None], (3, 64, 3)).astype(np.float32),
                delta_t=-np.ones(3, np.float32), valid=valid, loss_valid=valid,
                positive=positive, eligible=np.ones(3, bool), covered=np.array([True, True, False]),
                slot=np.ones(3, np.int64))
            (root/'manifest.json').write_text(json.dumps(dict(format=FORMAT, policy=DEFAULT_POLICY,
                shards=[dict(file='pairs.npz', count=3)])))
            dataset = PairCache(root)
            rows = evaluate(PairRanker(), torch.utils.data.DataLoader(dataset, batch_size=2), 'cpu')
            small = next(row for row in rows if row['method'] == 'learned_top8')
            pool = next(row for row in rows if row['method'] == 'pool_top64')
            self.assertEqual(small['eligible'], 3)
            self.assertEqual(small['recall'], 1/3)
            self.assertEqual(small['topk_miss'], 1)
            self.assertEqual(small['selected'], 18)
            self.assertEqual(pool['recall'], 2/3)


if __name__ == '__main__':
    unittest.main()
