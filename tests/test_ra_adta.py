import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from temporal import build_temporal_plugin
from temporal.core.ego_alignment import RadarBinGeometry
from temporal.core.range_anisotropic_geometry import RangeAnisotropicGeometry, normalized_displacement
from temporal.fusion.local_attention import LocalTemporalAttention
from temporal.fusion.ra_adta import RangeAwareDeformableTemporalAttention
from temporal.fusion.sparse_deformable_sampler import (anisotropic_candidates,
    anisotropic_knn_candidates, sparse_sample)


def physical_bins():
    return RadarBinGeometry(np.arange(256) * 0.462890625,
        np.deg2rad(np.arange(-53, 54)), np.deg2rad(np.arange(-18, 19)))


class GeometryTest(unittest.TestCase):
    def test_physical_scales_and_local_basis(self):
        geometry = RangeAnisotropicGeometry(physical_bins())
        ranges = np.array([0, 10, 20, 40, 60])
        xyz = np.stack((ranges, np.zeros_like(ranges), np.zeros_like(ranges)), axis=-1)
        basis, scales = geometry.query_geometry(xyz)
        np.testing.assert_allclose(basis.swapaxes(-1, -2) @ basis,
                                   np.broadcast_to(np.eye(3), basis.shape), atol=1e-14)
        np.testing.assert_allclose(scales[:, 0], 3 * 0.462890625)
        np.testing.assert_allclose(scales[:, 1], np.maximum(3 * ranges * np.pi / 180, 0.25))
        np.testing.assert_allclose(scales[:, 1], scales[:, 2])
        q = np.array([10., 8., 3.])
        basis, scales = geometry.query_geometry(q)
        delta = np.array([0.2, -0.3, 0.4])
        point = q + basis @ (scales * delta)
        np.testing.assert_allclose(normalized_displacement(point, q, basis, scales), delta)

    def test_intervals_from_actual_table_and_signs(self):
        bins = RadarBinGeometry([0, 1, 3], np.deg2rad([-2, 0, 4]),
                                np.deg2rad([-3, 0, 3]), azimuth_sign=-1)
        geometry = RangeAnisotropicGeometry(bins, min_angular_scale_m=0.01)
        _, scale = geometry.query_geometry(np.array([3., 0., 0.]))
        np.testing.assert_allclose(scale, [6., 3 * 3 * np.deg2rad(3), 3 * 3 * np.deg2rad(3)])
        with self.assertRaises(ValueError):
            RangeAnisotropicGeometry(RadarBinGeometry([0, 1], [0], [0, 1]))

    def test_dynamic_radius_and_spatial_coverage(self):
        # Many nearer points must not hide support >2 m away at long range.
        xyz = np.zeros((1, 2, 100, 3))
        xyz[..., 0] = 60
        xyz[0, 0, :, 0] += np.linspace(-0.01, 0.01, 100)
        xyz[0, 0, -1] = [60, 2.7, 0]
        geometry = RangeAnisotropicGeometry(physical_bins())
        ids, mask, local = anisotropic_candidates(xyz, [[True, True]], geometry)
        self.assertIn(99, ids[0, 0, 0][mask[0, 0, 0]])
        self.assertGreater(local[0, 0, 0, :, 1].max(), 0.8)
        xyz[..., 0] -= 50
        ids, mask, _ = anisotropic_candidates(xyz, [[True, True]], geometry)
        self.assertNotIn(99, ids[0, 0, 0][mask[0, 0, 0]])

    def test_filtered_history_ids_and_search_chunk_equivalence(self):
        rng = np.random.default_rng(12)
        xyz = rng.normal(scale=0.3, size=(2, 4, 31, 3))
        xyz[..., 0] += 20
        points = rng.random((2, 4, 31)) > 0.3
        frames = np.ones((2, 4), dtype=bool)
        frames[0, 1] = False
        geometry = RangeAnisotropicGeometry(physical_bins())
        first = anisotropic_candidates(xyz, frames, geometry, candidate_pool_size=5,
                                        search_chunk_size=1, point_valid_mask=points)
        full = anisotropic_candidates(xyz, frames, geometry, candidate_pool_size=5,
                                       search_chunk_size=31, point_valid_mask=points)
        for left, right in zip(first, full):
            np.testing.assert_allclose(left, right)
        ids, good, _ = first
        self.assertFalse(good[0, :, 1].any())
        for batch in range(2):
            self.assertFalse(good[batch, ~points[batch, -1]].any())
            for frame in range(3):
                selected = ids[batch, :, frame][good[batch, :, frame]]
                self.assertTrue(points[batch, frame, selected % 31].all())
                self.assertTrue((selected // 31 == frame).all())

    def test_anisotropic_knn8_chunk_masks_and_azimuth_wrap(self):
        geometry = RangeAnisotropicGeometry(physical_bins())
        rng = np.random.default_rng(12)
        xyz = rng.normal(scale=0.3, size=(2, 4, 31, 3))
        xyz[..., 0] += 20
        points = rng.random((2, 4, 31)) > 0.3
        frames = np.ones((2, 4), dtype=bool)
        frames[0, 1] = False
        short = anisotropic_knn_candidates(
            xyz, frames, geometry, candidate_pool_size=8,
            search_chunk_size=1, point_valid_mask=points)
        full = anisotropic_knn_candidates(
            xyz, frames, geometry, candidate_pool_size=8,
            search_chunk_size=31, point_valid_mask=points)
        for left, right in zip(short, full):
            np.testing.assert_allclose(left, right)
        ids, mask, local = short
        self.assertEqual(ids.shape, (2, 31, 3, 8))
        self.assertFalse(mask[0, :, 1].any())
        self.assertTrue(np.isfinite(local).all())
        for batch in range(2):
            self.assertFalse(mask[batch, ~points[batch, -1]].any())
            for frame in range(3):
                picked = ids[batch, :, frame][mask[batch, :, frame]]
                self.assertTrue(points[batch, frame, picked % 31].all())
                self.assertTrue((picked // 31 == frame).all())

        # Across -pi/pi these points are close, not opposite directions.
        seam = np.array([[[[-20, -0.04, 0]], [[-20, 0.04, 0]]]])
        _, valid, _ = anisotropic_knn_candidates(seam, [[True, True]], geometry)
        self.assertTrue(valid[0, 0, 0, 0])

        # Nearest in spherical coordinates is not necessarily a valid match.
        far = np.zeros((1, 2, 2, 3))
        far[0, 0, :, 0], far[0, 1, :, 0] = 1000, 10
        _, valid, _ = anisotropic_knn_candidates(far, [[True, True]], geometry)
        self.assertFalse(valid.any())


class SamplerTest(unittest.TestCase):
    def test_continuous_offset_gradient_and_empty_samples(self):
        key = torch.tensor([0., 1., 4.], dtype=torch.double).reshape(1, 1, 3, 1, 1)
        local = torch.tensor([[-0.4, 0, 0], [0.2, 0, 0], [0.7, 0, 0]],
                             dtype=torch.double).reshape(1, 1, 3, 3)
        offset = torch.tensor([0.1, 0.05, 0.02], dtype=torch.double).reshape(1, 1, 1, 1, 3)
        offset.requires_grad_()
        mask = torch.ones(1, 1, 3, dtype=torch.bool)
        sample = lambda delta: sparse_sample(key, key, local, delta, mask)[0]
        self.assertTrue(torch.autograd.gradcheck(sample, (offset,)))
        sample(offset).sum().backward()
        self.assertGreater(offset.grad.abs().sum(), 0)
        sampled, _, valid = sparse_sample(key, key, local, offset, ~mask)
        self.assertTrue(torch.equal(sampled, torch.zeros_like(sampled)))
        self.assertFalse(valid.any())
        sampled, _, valid = sparse_sample(key, key, local, offset + 100, mask)
        self.assertTrue(torch.isfinite(sampled).all())
        self.assertFalse(valid.any())


class RAADTATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.features = torch.rand(2, 4, 9, 8) + 1
        self.xyz = np.zeros((2, 4, 9, 3))
        self.xyz[..., 0] = 20 + np.arange(9) * 0.1
        self.xyz[:, :-1, :, 1] = np.linspace(-0.4, 0.4, 9)
        self.offsets = np.tile([-3, -2, -1, 0], (2, 1))
        self.mask = np.ones((2, 4), dtype=bool)

    def module(self, **kwargs):
        return RangeAwareDeformableTemporalAttention(physical_bins(), **kwargs)

    def test_identity_then_all_learned_paths_receive_gradients(self):
        module = self.module(chunk_size=4)
        out, debug = module(self.features, self.xyz, self.offsets, self.mask)
        self.assertTrue(torch.equal(out, self.features[:, -1]))
        self.assertTrue(debug['used_history'])
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        out.square().mean().backward()
        self.assertGreater(module.output.weight.grad.abs().sum(), 0)
        # A zero residual projection gates upstream gradients on the first step.
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        out, _ = module(self.features, self.xyz, self.offsets, self.mask)
        out.square().mean().backward()
        for name in ('offset.0.weight', 'offset.2.weight', 'q.weight',
                     'k_proj.weight', 'v.weight', 'position.0.weight'):
            grad = dict(module.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(grad.abs().sum(), 0, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
        self.assertTrue(torch.equal(out[..., 3:6], self.features[:, -1, :, 3:6]))

    def test_checkpoint_and_chunk_output_and_gradient_equivalence(self):
        module = self.module(chunk_size=4)
        with torch.no_grad():
            module.output.weight.normal_(std=0.03)
        other = copy.deepcopy(module)
        other.chunk_size, other.checkpoint_chunks = 100, False
        out, _ = module(self.features, self.xyz, self.offsets, self.mask)
        full, _ = other(self.features, self.xyz, self.offsets, self.mask)
        torch.testing.assert_close(out, full)
        out.sum().backward()
        full.sum().backward()
        for (name, p), (_, other_p) in zip(module.named_parameters(), other.named_parameters()):
            torch.testing.assert_close(p.grad, other_p.grad, atol=2e-5, rtol=1e-4, msg=name)

    def test_fallback_partial_masks_extreme_power_and_no_mutation(self):
        module = self.module(chunk_size=3)
        with torch.no_grad():
            module.output.bias.fill_(0.2)
        self.features[..., 7] = 2.98e34
        source = self.features.clone()
        self.mask[0, :-1] = False
        self.xyz[1, -1, -1] = [100, 100, 100]
        point_mask = np.ones((2, 4, 9), dtype=bool)
        point_mask[1, -1, 0] = False
        point_mask[1, 0] = False
        out, _ = module(self.features, self.xyz, self.offsets, self.mask, point_mask)
        self.assertTrue(torch.equal(out[0], source[0, -1]))
        self.assertTrue(torch.equal(out[1, [0, -1]], source[1, -1, [0, -1]]))
        self.assertTrue(torch.equal(out[..., 3:6], source[:, -1, :, 3:6]))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.equal(self.features, source))
        for slot in (slice(None), slice(-1, None)):
            mask = self.mask[:, slot].copy()
            mask[:, :-1] = False
            fallback, debug = module(self.features[:, slot], self.xyz[:, slot],
                                      self.offsets[:, slot], mask)
            self.assertTrue(torch.equal(fallback, source[:, -1]))
            self.assertFalse(debug['used_history'])
        self.mask[:] = True
        self.xyz[:, :-1] += 1000
        fallback, debug = module(self.features, self.xyz, self.offsets, self.mask)
        self.assertTrue(torch.equal(fallback, source[:, -1]))
        self.assertFalse(debug['used_history'])

    def test_anisotropic_knn8_forward_and_backward(self):
        module = self.module(search_mode='anisotropic_knn',
                             candidate_pool_size=8, chunk_size=4)
        out, debug = module(self.features, self.xyz, self.offsets, self.mask)
        self.assertTrue(torch.equal(out, self.features[:, -1]))
        self.assertEqual(debug['candidate_pool_size'], 8)
        self.assertTrue(debug['used_history'])
        out.square().mean().backward()
        self.assertGreater(module.output.weight.grad.abs().sum(), 0)

    def test_anisotropic_knn8_optimized_chunk_equivalence(self):
        # Larger search/attention chunks must preserve the selected neighbours,
        # forward values and the gradients of the deformable attention.
        baseline = self.module(search_mode='anisotropic_knn',
                               candidate_pool_size=8, chunk_size=3,
                               search_chunk_size=1)
        with torch.no_grad():
            baseline.output.weight.normal_(std=0.03)
        optimized = copy.deepcopy(baseline)
        optimized.chunk_size = 9
        optimized.search_chunk_size = 32

        reference, reference_debug = baseline(
            self.features, self.xyz, self.offsets, self.mask)
        actual, actual_debug = optimized(
            self.features, self.xyz, self.offsets, self.mask)
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
        self.assertEqual(actual_debug, reference_debug)

        reference.sum().backward()
        actual.sum().backward()
        for (name, a), (_, b) in zip(baseline.named_parameters(),
                                     optimized.named_parameters()):
            self.assertIsNotNone(a.grad, name)
            self.assertIsNotNone(b.grad, name)
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=1e-4,
                                       msg=name)

    def test_bulk_candidate_upload_matches_previous_chunk_upload(self):
        geometry = RangeAnisotropicGeometry(physical_bins())
        ids_np, valid_np, local_np = anisotropic_knn_candidates(
            self.xyz, self.mask, geometry, candidate_pool_size=8,
            search_chunk_size=3)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for array in (ids_np, valid_np, local_np):
            bulk = torch.as_tensor(array, device=device)
            for batch in range(array.shape[0]):
                for start in range(0, array.shape[1], 4):
                    end = start + 4
                    original_chunk = torch.as_tensor(
                        array[batch, start:end], device=device)
                    torch.testing.assert_close(bulk[batch, start:end],
                                               original_chunk, rtol=0, atol=0)

    def test_plugin_alignment_masking_and_existing_knn_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bins.npz'
            np.savez(path, range_m=np.arange(30), azimuth_rad=[0, 0.1], elevation_rad=[0, 0.1])
            cfg = dict(type='TemporalPlugin', position='P1', alignment=dict(bins_path=str(path)))
            plugin = build_temporal_plugin(dict(cfg, mode='ra_adta', attention=dict(
                alpha_range=0.1, min_angular_scale_m=0.1, interpolation_cutoff=0.2)))
            coords = torch.zeros(1, 2, 1, 3)
            coords[:, 0, :, 0], coords[:, 1, :, 0] = 15, 4
            poses = torch.eye(4).repeat(1, 2, 1, 1)
            poses[:, 1, 0, 3] = 11
            features = torch.rand(1, 2, 1, 8) + 1
            with torch.no_grad():
                plugin.fusion.output.bias.fill_(0.2)
            args = dict(features=features, coords=coords, poses=poses,
                        frame_offsets=torch.tensor([[-1, 0]]), valid_mask=torch.ones(1, 2, dtype=torch.bool))
            out, debug = plugin(**args)
            self.assertTrue(debug['used_history'])
            self.assertFalse(torch.equal(out, features[:, -1]))
            args['poses'] = torch.eye(4).repeat(1, 2, 1, 1)
            out, debug = plugin(**args)
            self.assertFalse(debug['used_history'])
            self.assertTrue(torch.equal(out, features[:, -1]))
            args['poses'] = poses
            plugin.power_mask_enabled, plugin.power_mask_radius = True, 5.0
            out, debug = plugin(**args)
            self.assertFalse(debug['used_history'])
            self.assertTrue(torch.equal(out[..., [0, 1, 2, 6, 7]], torch.zeros(1, 1, 5)))
            self.assertTrue(torch.equal(out[..., 3:6], features[:, -1, :, 3:6]))
            knn = build_temporal_plugin(dict(cfg, mode='ego_attention'))
            self.assertIsInstance(knn.fusion, LocalTemporalAttention)
            knn.fusion.load_state_dict(LocalTemporalAttention().state_dict(), strict=True)

    def test_invalid_config(self):
        for kwargs in (dict(num_heads=3), dict(candidate_pool_size=0),
                       dict(offset_bound=float('nan')), dict(alpha_range=0),
                       dict(temperature=0), dict(chunk_size=0)):
            with self.assertRaises(ValueError):
                self.module(**kwargs)


@unittest.skipUnless(importlib.util.find_spec('mmcv'), 'MMCV is needed to resolve training configs')
class ConfigTest(unittest.TestCase):
    def test_architecture_selection_and_fair_20pct_configs(self):
        from mmcv import Config
        root = Path(__file__).resolve().parents[1] / 'projects/configs/baselines'
        prefix = 'RadarOcc_Small_5060_true_fp32_temporal_t4_'
        knn = Config.fromfile(str(root / (prefix + 'ego_attention_p1_20pct.py')))
        ra = Config.fromfile(str(root / (prefix + 'ra_adta_p1_20pct.py')))
        self.assertEqual(knn.model.temporal_cfg.mode, 'ego_attention')
        self.assertEqual(ra.model.temporal_cfg.mode, 'ra_adta')
        self.assertEqual(ra.model.temporal_cfg.attention.search_mode, 'anisotropic_knn')
        self.assertEqual(ra.model.temporal_cfg.attention.candidate_pool_size, 8)
        self.assertEqual(ra.model.temporal_cfg.attention.chunk_size, 512)
        self.assertEqual(ra.model.temporal_cfg.attention.search_chunk_size, 1024)
        self.assertNotIn('radius_m', ra.model.temporal_cfg.attention)
        self.assertNotIn('neighbors_per_frame', ra.model.temporal_cfg.attention)
        for key in ('data', 'runner', 'evaluation', 'optimizer', 'load_from', 'resume_from'):
            self.assertEqual(knn[key], ra[key], key)
        self.assertEqual(ra.runner.max_epochs, 15)
        self.assertEqual(ra.evaluation.save_best, 'SSC_mean')
        self.assertIsNone(ra.load_from)
        self.assertTrue(ra.find_unused_parameters)
        self.assertFalse(ra.model.occ_encoder_neck.checkpoint_use_reentrant)
        self.assertIn('_20pct.pkl', ra.data.val.ann_file)
        self.assertNotIn('_20pct.pkl', ra.data.test.ann_file)
        knn_model, ra_model = copy.deepcopy(knn.model), copy.deepcopy(ra.model)
        knn_model.pop('temporal_cfg')
        ra_model.pop('temporal_cfg')
        self.assertEqual(knn_model, ra_model)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bins.npz'
            bins = physical_bins().bins
            np.savez(path, range_m=bins[0], azimuth_rad=bins[1], elevation_rad=bins[2])
            for config in (knn, ra):
                config.model.temporal_cfg.alignment.bins_path = str(path)
                build_temporal_plugin(config.model.temporal_cfg)


if __name__ == '__main__':
    unittest.main()
