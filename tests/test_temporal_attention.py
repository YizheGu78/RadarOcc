import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from temporal.fusion.local_attention import LocalTemporalAttention, local_neighbors
from temporal.temporal_plugin import TemporalPlugin


class AttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.features = torch.rand(1, 4, 7, 8) + 1
        self.xyz = np.zeros((1, 4, 7, 3))
        self.xyz[..., 0] = np.arange(7)
        self.offsets = np.array([[-3, -2, -1, 0]])
        self.mask = np.ones((1, 4), dtype=bool)

    def test_identity_and_training(self):
        module = LocalTemporalAttention(chunk_size=3)
        out, debug = module(self.features, self.xyz, self.offsets, self.mask)
        self.assertTrue(torch.equal(out, self.features[:, -1]))
        self.assertTrue(debug['used_history'])
        out.sum().backward()
        self.assertGreater(module.output.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            module.output.weight.add_(0.03)
        out, _ = module(self.features, self.xyz, self.offsets, self.mask)
        self.assertFalse(torch.equal(out, self.features[:, -1]))
        self.assertTrue(torch.equal(out[..., 3:6], self.features[:, -1, :, 3:6]))
        out.sum().backward()
        self.assertGreater(module.q.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()))

    def test_masks_and_radius(self):
        module = LocalTemporalAttention()
        with torch.no_grad():
            module.output.bias.fill_(0.2)
        self.mask[:, :-1] = False
        out, debug = module(self.features, self.xyz, self.offsets, self.mask)
        self.assertTrue(torch.equal(out, self.features[:, -1]))
        self.assertFalse(debug['used_history'])
        self.mask[:] = True
        self.xyz[:, :-1] += 100
        out, _ = module(self.features, self.xyz, self.offsets, self.mask)
        self.assertTrue(torch.equal(out, self.features[:, -1]))
        out, _ = module(self.features[:, -1:], self.xyz[:, -1:], self.offsets[:, -1:], self.mask[:, -1:])
        self.assertTrue(torch.equal(out, self.features[:, -1]))

    def test_locality_and_invalid_history(self):
        self.mask[:, 0] = False
        ids, mask = local_neighbors(self.xyz, self.mask, k=2, radius_m=0.1)
        self.assertFalse(mask[..., :2].any())
        self.assertTrue(mask[..., 2].all())
        self.assertTrue(np.array_equal(ids[0, :, 2], np.arange(7)+7))

    def test_partial_empty_and_chunk_equivalence(self):
        module = LocalTemporalAttention(chunk_size=3)
        other = LocalTemporalAttention(chunk_size=100)
        with torch.no_grad():
            module.output.bias.fill_(0.2)
        other.load_state_dict(module.state_dict())
        self.xyz[:, -1, -1] = [100, 100, 100]
        self.features[..., 7] = 2.98e34
        out, _ = module(self.features, self.xyz, self.offsets, self.mask)
        full, _ = other(self.features, self.xyz, self.offsets, self.mask)
        torch.testing.assert_close(out, full)
        self.assertTrue(torch.equal(out[:, -1], self.features[:, -1, -1]))
        self.assertTrue(torch.isfinite(out).all())

    def test_plugin_pose_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bins.npz'
            np.savez(path, range_m=np.arange(10), azimuth_rad=[0], elevation_rad=[0])
            plugin = TemporalPlugin(mode='ego_attention', alignment=dict(bins_path=str(path)),
                                    attention=dict(radius_m=0.1))
            coords = torch.zeros(1, 2, 1, 3)
            coords[:, 0, :, 0], coords[:, 1, :, 0] = 5, 4
            poses = torch.eye(4).repeat(1, 2, 1, 1)
            poses[:, 1, 0, 3] = 1
            features = self.features[:, :2, :1].clone()
            with torch.no_grad():
                plugin.fusion.output.bias.fill_(0.2)
            out, debug = plugin(features, coords, poses, torch.tensor([[-1, 0]]),
                                valid_mask=torch.ones(1, 2, dtype=torch.bool))
            self.assertTrue(debug['used_history'])
            self.assertFalse(torch.equal(out, features[:, -1]))


if __name__ == '__main__':
    unittest.main()
