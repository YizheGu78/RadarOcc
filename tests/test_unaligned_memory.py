import unittest
import torch
from temporal.fusion.unaligned_memory import UnalignedMemoryTemporal


class UnalignedMemoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)
        self.x = torch.rand(2, 4, 9, 8) + 1
        self.coords = torch.rand(2, 4, 9, 3) * 100
        self.offsets = torch.tensor([[-3, -2, -1, 0]]).expand(2, -1)
        self.valid = torch.ones(2, 4, dtype=torch.bool)

    def module(self, **kwargs):
        return UnalignedMemoryTemporal(embed_dims=16, num_heads=4,
                                       memory_tokens=4, chunk_size=3, **kwargs)

    def run_model(self, model, x=None, coords=None, **kwargs):
        return model(self.x if x is None else x, self.coords if coords is None else coords,
                     frame_offsets=self.offsets, valid_mask=self.valid, **kwargs)[0]

    def activate(self, model):
        with torch.no_grad():
            model.output.weight.normal_(std=0.1)

    def test_identity_and_ids(self):
        model = self.module()
        self.assertEqual(len(model.layers), 2)
        self.assertTrue(torch.equal(self.run_model(model), self.x[:, -1]))
        self.activate(model)
        out = self.run_model(model)
        self.assertFalse(torch.equal(out, self.x[:, -1]))
        self.assertTrue(torch.equal(out[..., 3:6], self.x[:, -1, :, 3:6]))
        self.assertTrue(torch.isfinite(out).all())

    def test_history_time_position_and_pose_independence(self):
        model = self.module().eval()
        self.activate(model)
        ref = self.run_model(model)
        x = self.x.clone(); x[:, :-1] *= 4
        self.assertGreater((ref - self.run_model(model, x=x)).abs().max().item(), 1e-5)
        coords = self.coords.clone(); coords[:, :-1] += 80
        self.assertGreater((ref - self.run_model(model, coords=coords)).abs().max().item(), 1e-5)
        self.offsets = self.offsets * 3
        self.assertGreater((ref - self.run_model(model)).abs().max().item(), 1e-5)
        self.offsets = self.offsets // 3
        # Malformed pose metadata must never be read or converted.
        self.assertTrue(torch.equal(ref, self.run_model(model, poses=object())))

    def test_invalid_history_and_single_frame(self):
        model = self.module().eval(); self.activate(model)
        self.valid[0, :-1] = False
        self.valid[1, 0] = False
        ref = self.run_model(model)
        x = self.x.clone(); coords = self.coords.clone()
        x[~self.valid] = float('nan'); coords[~self.valid] = float('nan')
        actual = self.run_model(model, x=x, coords=coords)
        torch.testing.assert_close(ref, actual)
        self.assertTrue(torch.equal(actual[0], self.x[0, -1]))
        self.valid[:, :-1] = False
        self.assertTrue(torch.equal(self.run_model(model), self.x[:, -1]))
        out, _ = model(self.x[:, -1:], self.coords[:, -1:])
        self.assertTrue(torch.equal(out, self.x[:, -1]))

    def test_training_reaches_both_layers_after_identity_warmup(self):
        model = self.module()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # Relative scaling avoids squaring the raw radar power.
        for step in range(2):
            optimizer.zero_grad()
            out = self.run_model(model)
            loss = ((out / self.x[:, -1] - 1.2) ** 2).mean()
            loss.backward()
            self.assertGreater(model.output.weight.grad.abs().sum().item(), 0)
            if step:
                for layer in model.layers:
                    self.assertGreater(layer.attention.in_proj_weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.compressor.attention.in_proj_weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.latents.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()

    def test_chunk_equivalence_large_power_and_checkpoint(self):
        model = self.module().eval(); self.activate(model)
        self.x[..., 7] = 2.98e34
        out = self.run_model(model)
        other = self.module().eval(); other.chunk_size = 100
        other.load_state_dict(model.state_dict())
        torch.testing.assert_close(out, self.run_model(other))
        self.assertTrue(torch.isfinite(out).all())

    def test_invalid_shapes_and_future_frames(self):
        model = self.module()
        self.offsets = -self.offsets
        with self.assertRaises(ValueError):
            self.run_model(model)
        with self.assertRaises(ValueError):
            UnalignedMemoryTemporal(num_layers=0)


if __name__ == '__main__':
    unittest.main()
