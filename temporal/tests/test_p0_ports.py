import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from temporal.centerformer.attention import ms_deform_attn_core_pytorch, TemporalMSDeformAttn
from temporal.centerformer.fusion import CenterFormerP0
from temporal.p0_common import sample_volume, masked_xy_pool
from temporal.rodnet.fusion import RODNetP0
from temporal.rodnet.tdc import CausalTDC


class TDCOperatorTest(unittest.TestCase):
    def test_observation_normalization_and_observed_zero(self):
        layer = CausalTDC(1, 1, frames=2, kernel_size=1, checkpoint_chunks=False)
        with torch.no_grad():
            layer.weight.zero_(); layer.weight[:, :, 0] = 1
        x = torch.zeros(1, 1, 2, 4, 4)
        x[:, :, 0, 1, 1] = 1
        mask = (x != 0).float()
        offsets = torch.zeros(1, 4, 1, 4, 4)
        offsets[:, :2] = .5
        dense = layer(x, offsets)
        normalized, history, _ = layer(x, offsets, mask, return_support=True)
        self.assertAlmostEqual(dense[0, 0, 1, 1].item(), .25, places=6)
        self.assertAlmostEqual(normalized[0, 0, 1, 1].item(), 1., places=6)
        self.assertTrue(bool(history[0, 0, 1, 1]))
        # An observed zero is different from an unobserved corner.
        mask[:, :, 0, 1, 2] = 1
        normalized = layer(x, offsets, mask)
        self.assertAlmostEqual(normalized[0, 0, 1, 1].item(), .5, places=6)
        empty, history, observed = layer(x, offsets, mask*0, return_support=True)
        self.assertTrue(torch.equal(empty, torch.zeros_like(empty)))
        self.assertFalse(bool(history.any() or observed.any()))

    def test_zero_offset_matches_conv3d_and_gradients(self):
        torch.manual_seed(4)
        layer = CausalTDC(2, 3, frames=3, chunk_size=7).double()
        x = torch.randn(1, 2, 3, 4, 5, dtype=torch.double, requires_grad=True)
        actual = layer(x)
        reference = F.conv3d(x, layer.weight, padding=(0, 1, 1))[:, :, 0]
        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)
        actual.square().mean().backward()
        self.assertGreater(layer.conv_offset.weight.grad.abs().sum().item(), 0)
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_learned_spatial_shift_and_support(self):
        layer = CausalTDC(1, 1, frames=2, kernel_size=1, checkpoint_chunks=False)
        with torch.no_grad():
            layer.weight.zero_(); layer.weight[:, :, 0] = 1
        x = torch.zeros(1, 1, 2, 3, 5)
        x[0, 0, 0, 1, 1] = 2
        off = torch.zeros(1, 4, 1, 3, 5)
        off[:, 1] = -2  # Historical dx=-2, samples source x=1 at output x=3.
        actual = layer(x, off)
        self.assertAlmostEqual(actual[0, 0, 1, 3].item(), 2., places=5)
        masks = (x != 0).float()
        _, support, _ = layer(x, off, masks, return_support=True)
        self.assertGreater(support[0, 0, 1, 3].item(), .99)
        self.assertEqual(support[0, 0, 1, 1].item(), 0.)


class CenterCoreTest(unittest.TestCase):
    def test_height_sampling_offset_receives_gradient(self):
        attention = TemporalMSDeformAttn(channels=1, levels=4, heads=1, points=1)
        with torch.no_grad():
            attention.sampling_offsets.bias.zero_()
            attention.value_proj.weight.fill_(1); attention.value_proj.bias.zero_()
            attention.output_proj.weight.fill_(1); attention.output_proj.bias.zero_()
        field = torch.ones(1, 1, 2, 2, 2); field[:, :, 1] = 3
        maps = [torch.zeros_like(field) for _ in range(3)]+[field]
        masks = [torch.zeros_like(field) for _ in range(3)]+[torch.ones_like(field)]
        out, support, _ = attention(torch.zeros(1, 1, 1), torch.tensor([[[.5, .5, .5]]]),
            attention.project_values(maps, masks), [(2, 2, 2)]*4, masks, torch.ones(1, 4, dtype=torch.bool))
        self.assertAlmostEqual(out.item(), 2., places=6)
        self.assertTrue(bool(support.item()))
        out.sum().backward()
        z_gradient = attention.sampling_offsets.bias.grad.reshape(1, 4, 1, 3)[0, -1, 0, 2]
        self.assertGreater(z_gradient.abs().item(), 0)

    def test_sparse_samples_normalized_and_empty_candidates_excluded(self):
        attention = TemporalMSDeformAttn(channels=1, levels=4, heads=1, points=1)
        with torch.no_grad():
            attention.sampling_offsets.bias.zero_()
            attention.value_proj.weight.fill_(1); attention.value_proj.bias.zero_()
            attention.output_proj.weight.fill_(1); attention.output_proj.bias.zero_()
        field = torch.zeros(1, 1, 4, 4, 4); field[:, :, 1, 1, 1] = 1
        maps = [torch.zeros_like(field) for _ in range(3)]+[field]
        masks = [(x != 0).float() for x in maps]
        positions = torch.tensor([[[.5, .5, .5]]])
        args = (torch.zeros(1, 1, 1), positions, attention.project_values(maps, masks),
                [(4, 4, 4)]*4, masks, torch.ones(1, 4, dtype=torch.bool))
        out, support, _ = attention(*args)
        # One observed corner: trilinear interpolation would yield 0.125.
        # Three empty source levels must also receive no softmax mass.
        self.assertAlmostEqual(out.item(), 1., places=6)
        self.assertTrue(bool(support.item()))
        masks[-1][:, :, 1, 1, 2] = 1
        args = (args[0], positions, attention.project_values(maps, masks), *args[3:])
        out, _, _ = attention(*args)
        self.assertAlmostEqual(out.item(), .5, places=6)

    def test_volume_and_pool_keep_observed_mean_and_height(self):
        field = torch.zeros(1, 1, 2, 4, 4)
        mask = torch.zeros_like(field)
        field[:, :, 0, 1, 1] = 2; field[:, :, 1, 1, 1] = 10
        mask[:, :, :, 1, 1] = 1
        pos = torch.tensor([[[.5, .5, .25], [.5, .5, .75], [.5, .5, 2.]]])
        values, valid = sample_volume(field, pos, mask)
        torch.testing.assert_close(values[0, :, 0], torch.tensor([2., 10., 0.]))
        self.assertEqual(valid.tolist(), [[True, True, False]])
        pooled, observed = masked_xy_pool(field, mask, 2)
        self.assertEqual(pooled.shape[-3], 2)
        torch.testing.assert_close(pooled[0, 0, :, 0, 0], torch.tensor([2., 10.]))
        self.assertEqual(observed[0, 0, :, 0, 0].tolist(), [1., 1.])

    def test_reference_sampling_and_gradients(self):
        # Bilinear sampling at centre of four cells must produce their mean.
        value = torch.tensor([1., 2., 3., 4.]).reshape(1, 4, 1, 1).requires_grad_()
        locations = torch.tensor([.5, .5]).reshape(1, 1, 1, 1, 1, 2).requires_grad_()
        weights = torch.ones(1, 1, 1, 1, 1)
        out = ms_deform_attn_core_pytorch(value, [(2, 2)], locations, weights)
        self.assertAlmostEqual(out.item(), 2.5)
        out.sum().backward()
        self.assertGreater(locations.grad.abs().sum().item(), 0)

    def test_matches_upstream_reference(self):
        path = Path(__file__).parents[1]/'centerformer/upstream/ms_deform_attn_func.py.txt'
        # Execute only the independent reference function, excluding CUDA imports.
        fn = path.read_text().split('def ms_deform_attn_core_pytorch', 1)[1]
        namespace = {'torch': torch, 'F': F}
        exec('def ms_deform_attn_core_pytorch'+fn, namespace)
        torch.manual_seed(9)
        value = torch.randn(2, 10, 2, 3)
        locations = torch.rand(2, 5, 2, 2, 4, 2)
        weights = torch.randn(2, 5, 2, 8).softmax(-1).reshape(2, 5, 2, 2, 4)
        expected = namespace['ms_deform_attn_core_pytorch'](value, [(2, 3), (2, 2)], locations, weights)
        torch.testing.assert_close(ms_deform_attn_core_pytorch(value, [(2, 3), (2, 2)], locations, weights), expected)


class P0BoundaryTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        self.tmp = tempfile.TemporaryDirectory()
        self.bins = str(Path(self.tmp.name)/'bins.npz')
        np.savez(self.bins, range_m=np.arange(1., 9.),
                 azimuth_rad=np.array([-.2, 0., .2]), elevation_rad=np.array([0.]))
        self.coords = torch.zeros(1, 4, 5, 3)
        self.coords[..., 0] = torch.arange(1, 6)
        self.coords[..., 1] = 1
        self.features = torch.rand(1, 4, 5, 8)+1
        self.poses = torch.eye(4).repeat(1, 4, 1, 1)
        self.offsets = torch.tensor([[-3, -2, -1, 0]])
        self.mask = torch.ones(1, 4, dtype=torch.bool)

    def tearDown(self):
        self.tmp.cleanup()

    def model(self, kind, chunk=3):
        return kind(bins_path=self.bins, channels=8, chunk_size=chunk,
                    bev_bounds=(-2., -4., 10., 4.), cell_size=1.)

    def run_model(self, model, features=None, coords=None, poses=None, mask=None):
        return model(self.features if features is None else features,
                     self.coords if coords is None else coords,
                     self.poses if poses is None else poses, self.offsets,
                     valid_mask=self.mask if mask is None else mask)

    def test_identity_gradients_and_no_doppler_dependency(self):
        for kind in (CenterFormerP0, RODNetP0):
            with self.subTest(kind=kind.__name__):
                model = self.model(kind)
                features = self.features.clone(); features[..., 7] = 1e34
                out, _ = self.run_model(model, features)
                self.assertTrue(torch.equal(out, features[:, -1]))
                (out[..., :3].sum()+out[..., 6:].sum()/1e34).backward()
                self.assertGreater(model.output.weight.grad.abs().sum().item(), 0)
                with torch.no_grad():
                    model.output.weight.normal_(0., .02)
                model.zero_grad()
                out, _ = self.run_model(model)
                out.square().mean().backward()
                offset_layer = model.attention.sampling_offsets if kind is CenterFormerP0 else model.tdc.conv_offset
                self.assertGreater(offset_layer.weight.grad.abs().sum().item(), 0)
                modified = self.features.clone(); modified[:, :-1] *= 3
                changed, _ = self.run_model(model, modified)
                self.assertGreater((changed-out).abs().max().item(), 1e-8)
                doppler = self.features.clone(); doppler[..., 3:6] = 987
                same, _ = self.run_model(model, doppler)
                torch.testing.assert_close(same[..., [0,1,2,6,7]], out[..., [0,1,2,6,7]])
                self.assertTrue(torch.equal(same[..., 3:6], doppler[:, -1, :, 3:6]))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_masks_and_chunk_equivalence(self):
        for kind in (CenterFormerP0, RODNetP0):
            model = self.model(kind); other = self.model(kind, 10000)
            with torch.no_grad():
                model.output.bias.fill_(.2); model.output.weight.normal_(0., .1)
            other.load_state_dict(model.state_dict())
            out, _ = self.run_model(model)
            full, _ = self.run_model(other)
            torch.testing.assert_close(out, full)
            mask = self.mask.clone(); mask[:, :-1] = False
            fallback, info = self.run_model(model, mask=mask)
            self.assertTrue(torch.equal(fallback, self.features[:, -1]))
            self.assertFalse(info['used_history'])
            model.zero_grad()
            fallback.sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            mask[:, -2] = True
            partial, _ = self.run_model(model, mask=mask)
            modified = self.features.clone(); modified[:, :2] = 1e20
            ignored, _ = self.run_model(model, modified, mask=mask)
            torch.testing.assert_close(partial, ignored)
            poses = self.poses.clone(); poses[:, :-1, 0, 3] = 100
            fallback, _ = self.run_model(model, poses=poses)
            self.assertTrue(torch.equal(fallback, self.features[:, -1]))

    def test_pose_alignment_and_input_immutability(self):
        for kind in (CenterFormerP0, RODNetP0):
            model = self.model(kind)
            with torch.no_grad():
                model.output.weight.normal_(0., .1)
            original = self.features.clone()
            coords = self.coords.clone(); coords[:, :-1, :, 0] += 1
            poses = self.poses.clone(); poses[:, :-1, 0, 3] = -1
            base, _ = self.run_model(model)
            self.assertGreater((base-self.features[:, -1]).abs().max().item(), 1e-6)
            wrong, _ = self.run_model(model, coords=coords)
            self.assertGreater((wrong-base).abs().max().item(), 1e-6)
            aligned, _ = self.run_model(model, coords=coords, poses=poses)
            torch.testing.assert_close(base, aligned, rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.equal(self.features, original))

    def test_same_xy_different_heights_do_not_mix(self):
        # Two points at identical XY=(4,0), but Z=-1.5,+1.5 metres.
        angle = np.arctan2(1.5, 4.)
        np.savez(self.bins, range_m=[np.hypot(1.5, 4.)],
                 azimuth_rad=[0.], elevation_rad=[-angle, angle])
        coords = torch.zeros(1, 4, 2, 3); coords[..., 1, 2] = 1
        features = torch.ones(1, 4, 2, 8)*2
        features[:, :-1, 0] = 1e3
        features[:, :-1, 1] = 1e12
        changed = features.clone(); changed[:, :-1, 1] = 1e30
        for kind in (CenterFormerP0, RODNetP0):
            model = self.model(kind)
            with torch.no_grad():
                model.output.weight.normal_(0., .1)
                if kind is CenterFormerP0:
                    model.attention.sampling_offsets.bias.zero_()
            out, _ = self.run_model(model, features, coords)
            altered, _ = self.run_model(model, changed, coords)
            torch.testing.assert_close(out[:, 0], altered[:, 0], rtol=1e-6, atol=1e-6)
            self.assertGreater((out[:, 1]-altered[:, 1]).abs().max().item(), 1e-6)

    def test_train_and_eval_use_same_inputs(self):
        for kind in (CenterFormerP0, RODNetP0):
            model = self.model(kind)
            with torch.no_grad(): model.output.weight.normal_(0., .1)
            train, _ = self.run_model(model)
            model.eval()
            evaluate, _ = self.run_model(model)
            torch.testing.assert_close(train, evaluate)


if __name__ == '__main__':
    unittest.main()
