import json
import math
from pathlib import Path
import tempfile
import unittest
import torch
from temporal.fusion.box_z_p0 import (BoxZP0Fusion, scatter_field, masked_interpolate,
                                      map_coordinates, physical_grid)
from tools.temporal.summarize_box_z_ablation import summarize


class BoxZTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.bins = [torch.tensor([1., 2., 3., 4.]), torch.tensor([-.2, 0., .2]),
                     torch.tensor([-.2, 0., .2])]
        self.coords = torch.cartesian_prod(*[torch.arange(len(b)) for b in self.bins])
        n = len(self.coords)
        raw = torch.arange(n).float()[:, None]/10+torch.arange(8).float()[None]+1
        self.raw = raw[None].repeat(4, 1, 1)
        self.coords4 = self.coords[None].repeat(4, 1, 1)
        self.pairs = dict(current=torch.tensor([[2., 0., 0., 8., 4., 4.]]),
            history=torch.tensor([[[2., 0., 0., 8., 4., 4.]]*3]),
            matched=torch.ones(1, 3, dtype=torch.bool), inverse=torch.eye(4)[None].repeat(3, 1, 1))

    def module(self, mode):
        torch.manual_seed(42)
        return BoxZP0Fusion(self.bins, mode=mode, dim=8, heads=2, chunk_size=11)

    def test_mapping_formulas_and_z0_height_aggregation_locations(self):
        p = torch.tensor([[10., 2., 2.6]])
        ct = torch.tensor([[8., 1., 2., 4., 2., 3.]])
        ch = torch.tensor([[20., 3., 1., 8., 4., 2.]])
        z1 = map_coordinates(p, ct, ch, 'Z1')
        z2 = map_coordinates(p, ct, ch, 'Z2')
        self.assertTrue(torch.allclose(z1, torch.tensor([[24., 5., 2.6]])))
        self.assertTrue(torch.allclose(z2, torch.tensor([[24., 5., 1.4]])))
        z3 = map_coordinates(p, ct, ch, 'Z3', torch.zeros(1), torch.zeros(1))
        self.assertTrue(torch.equal(z2, z3))
        z0 = map_coordinates(p, ct, ch, 'Z0', samples=8)
        self.assertEqual(z0.shape, (1, 8, 3))
        self.assertTrue(torch.allclose(z0[0, :, 2], torch.arange(8)*.25+.125))

    def test_actual_bin_centres_interpolation_and_coordinate_gradient(self):
        field, obs = scatter_field(torch.arange(len(self.coords)).float()[:, None], self.coords, (4, 3, 3))
        module = self.module('Z1')
        xyz = module.xyz(self.coords)
        values, valid = masked_interpolate(field, obs, xyz, self.bins)
        self.assertTrue(torch.allclose(values[:, 0][valid], torch.arange(len(self.coords)).float()[valid], atol=1e-4))
        point = torch.tensor([[2.4, .05, .04]], requires_grad=True)
        value, good = masked_interpolate(field, obs, point, self.bins)
        self.assertTrue(good.item()); value.sum().backward()
        self.assertTrue(torch.isfinite(point.grad).all()); self.assertGreater(float(point.grad.abs().sum()), 0)

    def test_empty_cells_are_not_observations_and_duplicate_returns_average(self):
        coords = torch.tensor([[1, 1, 1], [1, 1, 1]])
        field, obs = scatter_field(torch.tensor([[2.], [4.]]), coords, (4, 3, 3))
        value, good = masked_interpolate(field, obs, torch.tensor([[2., 0., 0.], [4., 0., 0.], [10., 0., 0.]]), self.bins)
        self.assertAlmostEqual(float(value[0]), 3.)
        self.assertEqual(good.tolist(), [True, False, False])

    def test_nonuniform_physical_range_uses_fractional_bin_index(self):
        bins = [torch.tensor([1., 2., 4.]), self.bins[1], self.bins[2]]
        coords = torch.cartesian_prod(torch.arange(3), torch.arange(3), torch.arange(3))
        values = bins[0][coords[:, 0], None]
        field, obs = scatter_field(values, coords, (3, 3, 3))
        point = torch.tensor([[3., 0., 0.]], requires_grad=True)
        sampled, valid = masked_interpolate(field, obs, point, bins)
        self.assertTrue(valid.item()); self.assertAlmostEqual(float(sampled), 3., places=5)
        sampled.sum().backward()
        self.assertAlmostEqual(float(point.grad[0, 0]), 1., places=5)

    def test_initial_identity_all_modes_and_shared_initialization(self):
        models = [self.module(mode) for mode in ('Z0', 'Z1', 'Z2', 'Z3')]
        for m in models:
            out = m(self.raw, self.coords4, self.pairs)
            self.assertTrue(torch.equal(out, self.raw[-1]))
        for key, tensor in models[0].state_dict().items():
            for m in models[1:]:
                self.assertTrue(torch.equal(tensor, m.state_dict()[key]), key)

    def test_z3_gradients_through_sampling_and_ids_preserved(self):
        m = self.module('Z3')
        with torch.no_grad():
            m.output.weight.fill_(.2)
        out = m(self.raw, self.coords4, self.pairs)
        out[:, [0, 1, 2, 6, 7]].sum().backward()
        self.assertIsNotNone(m.z_linear.weight.grad)
        self.assertTrue(torch.isfinite(m.z_linear.weight.grad).all())
        self.assertGreater(float(m.z_linear.weight.grad.abs().sum()), 0.)
        self.assertTrue(torch.equal(out[:, 3:6], self.raw[-1, :, 3:6]))

    def test_all_invalid_is_exact_identity_and_finite_large_powers(self):
        m = self.module('Z3')
        self.pairs['inverse'][:, 0, 3] = 1000
        with torch.no_grad():
            m.output.weight.fill_(1)
        raw = self.raw.clone(); raw[:, :, :3] = 1e34
        out = m(raw, self.coords4, self.pairs)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.equal(out, raw[-1]))
        self.assertEqual(m.diagnostics['sample_valid'], 0)
        self.assertGreater(m.diagnostics['sample_attempted'], 0)
        out[:, 0].mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_empty_cache_no_history_identity_backward(self):
        m = self.module('Z3')
        pairs = dict(current=torch.empty(0, 6), history=torch.empty(0, 3, 6),
                     matched=torch.empty(0, 3, dtype=torch.bool), inverse=torch.eye(4)[None].repeat(3, 1, 1))
        out = m(self.raw, self.coords4, pairs)
        self.assertTrue(torch.equal(out, self.raw[-1])); out.sum().backward()
        self.assertEqual(float(m.losses['loss_z_scale']), 0.)

    def test_regularizers_are_per_matched_pair_not_query_or_invalid_pair(self):
        m = self.module('Z3')
        with torch.no_grad():
            m.z_linear.bias.copy_(torch.tensor([.3, .4]))
        self.pairs['matched'] = torch.tensor([[True, False, False]])
        m(self.raw, self.coords4, self.pairs)
        expected_scale = .001*(math.log(2)*math.tanh(.3))**2
        expected_shift = .001*(.5*math.tanh(.4))**2
        self.assertAlmostEqual(float(m.losses['loss_z_scale']), expected_scale, places=10)
        self.assertAlmostEqual(float(m.losses['loss_z_shift']), expected_shift, places=10)
        m(self.raw.repeat(1, 2, 1), self.coords4.repeat(1, 2, 1), self.pairs)
        self.assertAlmostEqual(float(m.losses['loss_z_scale']), expected_scale, places=10)

    def test_summary_keeps_all_epochs_and_uses_project_metrics(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); (root/'Z1').mkdir()
            (root/'single_frame_metrics.json').write_text(json.dumps({'SSC_mean': .3}))
            records = [dict(mode='val',epoch=i,SSC_mean=v,SSC_free=.8,SSC_Background=.4,SSC_Foreground=.2)
                       for i,v in [(1,.3),(2,.35),(3,.32)]]
            (root/'Z1/a.log.json').write_text('\n'.join(json.dumps(r) for r in records))
            summarize(root, 2)
            result = json.loads((root/'comparison.json').read_text())
            self.assertEqual(result[1]['epoch'], 2)
            self.assertEqual(result[1]['last_epoch'], 3)
            self.assertEqual(len((root/'epoch_iou.csv').read_text().splitlines()), 4)


if __name__ == '__main__':
    unittest.main()
