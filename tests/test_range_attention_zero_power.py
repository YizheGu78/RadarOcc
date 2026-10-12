"""Exercise actual range attention classes without requiring CUDA/spconv/MMCV."""
import ast
from pathlib import Path
import unittest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
source = ast.parse((ROOT/'projects/occ_plugin/occupancy/voxel_encoder/sparse_lidar_enc.py').read_text())
classes = [node for node in source.body if isinstance(node, ast.ClassDef)
           and node.name in ('MultiLayerRangeAttention', 'SelfAttentionBlock')]
namespace = dict(torch=torch, nn=nn, F=F)
exec(compile(ast.Module(body=classes, type_ignores=[]), '<actual-range-attention>', 'exec'), namespace)
RangeAttention = namespace['MultiLayerRangeAttention']


class ZeroPowerEncodingTest(unittest.TestCase):
    def test_positive_inputs_unchanged(self):
        torch.manual_seed(5)
        old = RangeAttention(num_layers=2, num_ranges=2, K=3)
        new = RangeAttention(num_layers=2, num_ranges=2, K=3, allow_zero_power=True)
        new.load_state_dict(old.state_dict(), strict=True)
        power = torch.rand(6,8)*100 + 1
        power[:,3:6] = 4
        idx = torch.zeros(6, dtype=torch.long)
        self.assertTrue(torch.equal(old(power,idx,idx), new(power,idx,idx)))

    def test_zero_power_forward_and_backward(self):
        torch.manual_seed(7)
        old = RangeAttention(num_layers=4, num_ranges=2, K=3)
        new = RangeAttention(num_layers=4, num_ranges=2, K=3, allow_zero_power=True)
        new.load_state_dict(old.state_dict(), strict=True)
        power = torch.rand(6,8)*100 + 1
        power[:,3:6] = 3
        power[:3,[0,1,2,6,7]] = 0  # Entire first range has been masked.
        power.requires_grad_()
        idx = torch.zeros(6, dtype=torch.long)
        self.assertFalse(torch.isfinite(old(power,idx,idx)).all())
        out = new(power,idx,idx)
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        self.assertTrue(torch.isfinite(power.grad).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in new.parameters()))
        self.assertTrue((power.grad[0,[0,1,2,6,7]] == 0).all())

    def test_invalid_power_is_not_silently_replaced(self):
        module=RangeAttention(allow_zero_power=True)
        for value in [-1,float('inf'),float('nan')]:
            with self.assertRaises(ValueError):module.power_log10(torch.tensor([value]))


if __name__=='__main__':unittest.main()
