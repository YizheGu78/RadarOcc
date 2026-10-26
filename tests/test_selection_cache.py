import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from temporal.core.learned_topk import BoxTopKSelector, PairRanker, save_ranker
from temporal.fusion.local_attention import LocalTemporalAttention

torch.set_num_threads(2)


class SelectionCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root/'ranker.pth'
        ranker = PairRanker()
        with torch.no_grad():
            ranker.network[-1].weight.normal_(0, .02)
        save_ranker(self.checkpoint, ranker)
        self.features = torch.rand(1, 4, 17, 8)+1
        self.xyz = np.zeros((1, 4, 17, 3))
        self.xyz[..., 0] = np.arange(17)*.3
        self.frames = np.array([[False, True, True, True]])
        self.times = np.array([[np.nan, .1, .2, .3]])
        self.points = np.ones((1, 4, 17), bool)
        self.points[:, -1, 0] = False

    def tearDown(self):
        self.tmp.cleanup()

    def selector(self, mode='readwrite', checkpoint=None):
        return BoxTopKSelector(checkpoint=checkpoint or self.checkpoint,
            cache_dir=self.root/'cache', cache_mode=mode, query_chunk_size=5)

    def run_selector(self, selector, **updates):
        args = dict(features=self.features, xyz=self.xyz, valid_mask=self.frames,
            timestamps=self.times, point_valid_mask=self.points)
        args.update(updates)
        return selector(**args)

    def test_reload_skips_search_and_ranker_for_all_queries(self):
        writer = self.selector()
        expected = self.run_selector(writer)
        reader = self.selector('read')
        with patch.object(reader, '_select_online', side_effect=AssertionError('online executed')):
            actual = self.run_selector(reader)
        for x, y in zip(expected, actual):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(reader.cache.hits, 1)
        self.assertFalse(actual[1][:, 0].any())
        self.assertFalse(actual[1][..., :8].any())
        self.assertEqual(actual[0].shape, (1, 17, 24))

    def test_changed_order_features_geometry_time_and_mask_miss(self):
        self.run_selector(self.selector())
        reader = self.selector('read')
        changed_xyz = self.xyz.copy()
        changed_xyz[..., 0] += .01
        for update in (dict(features=self.features.flip(2)), dict(xyz=changed_xyz),
                       dict(timestamps=np.array([[np.nan, .1, .2, .4]])),
                       dict(point_valid_mask=np.zeros_like(self.points))):
            with self.subTest(update=list(update)), self.assertRaises(FileNotFoundError):
                self.run_selector(reader, **update)

    def test_changed_checkpoint_and_restored_model_weights_miss(self):
        writer = self.selector()
        self.run_selector(writer)
        state = copy.deepcopy(writer.state_dict())
        state['ranker.network.4.bias'] += .1
        writer.load_state_dict(state)
        writer.cache.mode = 'read'
        with self.assertRaises(FileNotFoundError):
            self.run_selector(writer)
        ranker = PairRanker()
        with torch.no_grad():
            ranker.network[-1].bias.fill_(2.)
        other = self.root/'other.pth'
        save_ranker(other, ranker)
        with self.assertRaises(FileNotFoundError):
            self.run_selector(self.selector('read', other))

    def test_attention_outputs_and_gradients_equal_with_nonzero_fusion(self):
        online = LocalTemporalAttention(selection=dict(checkpoint=str(self.checkpoint),
            query_chunk_size=5), chunk_size=6)
        with torch.no_grad():
            online.output.weight.fill_(.03)
        cached = copy.deepcopy(online)
        cached.selector = self.selector()
        offsets = np.array([[-3, -2, -1, 0]])
        args = dict(timestamps=self.times, point_valid_mask=self.points)
        x = self.features.clone().requires_grad_()
        y = self.features.clone().requires_grad_()
        first, _ = online(x, self.xyz, offsets, self.frames, **args)
        # Populate, then exercise actual disk hit with online selection disabled.
        self.run_selector(cached.selector)
        with patch.object(cached.selector, '_select_online', side_effect=AssertionError('online')):
            second, _ = cached(y, self.xyz, offsets, self.frames, **args)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        first.sum().backward()
        second.sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        self.assertGreater(float(y.grad[:, :-1].abs().sum()), 0)
        for (name, parameter), (_, other) in zip(online.named_parameters(), cached.named_parameters()):
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0, msg=name)
        self.assertTrue(all(p.grad is None for p in cached.selector.parameters()))

    def test_corrupt_entry_fails_instead_of_silently_recomputing(self):
        self.run_selector(self.selector())
        path = next((self.root/'cache').glob('*/*.npz'))
        path.write_bytes(b'broken cache')
        with self.assertRaises(ValueError):
            self.run_selector(self.selector('read'))

    def test_read_mode_requires_prebuilt_directory(self):
        with self.assertRaises(FileNotFoundError):
            self.selector('read')

    def test_full_p1_query_count_storage_and_reload(self):
        n = 43750
        features = torch.ones(1, 4, n, 8)
        xyz = np.zeros((1, 4, n, 3))
        indices = np.zeros((1, n, 24), np.int64)
        valid = np.ones_like(indices, bool)
        writer = self.selector()
        with patch.object(writer, '_select_online', return_value=(indices, valid)):
            writer(features, xyz, np.ones((1, 4), bool), [[0., .1, .2, .3]])
        reader = self.selector('read')
        with patch.object(reader, '_select_online', side_effect=AssertionError('online')):
            actual = reader(features, xyz, np.ones((1, 4), bool), [[0., .1, .2, .3]])
        np.testing.assert_array_equal(actual[0], indices)
        np.testing.assert_array_equal(actual[1], valid)
        size = next((self.root/'cache').glob('*/*.npz')).stat().st_size
        self.assertLess(size, 4_400_000)


if __name__ == '__main__':
    unittest.main()
