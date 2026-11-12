"""Exercise the installed Small source contracts without CUDA-only imports.

CPU fixtures replace learned modules / sparse convolution, not geometry code.
The optional GPU test runs the real sparse encoder when CUDA is available.
"""
from itertools import product
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
import hashlib
import io
import subprocess
import tempfile
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tools.temporal.audit_projection_chain import (source_function, DETECTOR, ENCODER,
    ROOT, CUDA, validate_extension_build)

TRANSFORMER = ROOT / 'projects/occ_plugin/occupancy/deformable_transformer'


def trilinear_oracle(volume, locations):
    """Independent eight-neighbour sum; input [B,C,A,R,E], locations [R,A,E]."""
    batch, channels, a, r, e = volume.shape
    points = locations * locations.new_tensor([r, a, e]) - .5
    result = volume.new_zeros(batch, locations.shape[1], channels)
    for b in range(batch):
        for q, point in enumerate(points[b]):
            low = torch.floor(point).long()
            frac = point - low
            for dr, da, de in product((0, 1), repeat=3):
                rr, aa, ee = low + low.new_tensor([dr, da, de])
                if 0 <= rr < r and 0 <= aa < a and 0 <= ee < e:
                    weight = ((frac[0] if dr else 1-frac[0]) *
                              (frac[1] if da else 1-frac[1]) *
                              (frac[2] if de else 1-frac[2]))
                    result[b, q] += weight * volume[b, :, aa, rr, ee]
    return result


def encoder_fixture(query, value=None, spherical=None, spatial=(2, 2, 2)):
    calls = []
    def layer(q, *args, **kwargs):
        calls.append(kwargs)
        return q
    forward = source_function(TRANSFORMER/'encoder_3D.py', 'VoxFormerEncoder3D', 'forward')
    reference = source_function(TRANSFORMER/'encoder_3D.py', 'VoxFormerEncoder3D', 'get_reference_points')
    obj = SimpleNamespace(return_intermediate=False, get_reference_points=reference, layers=[layer])
    out = forward(obj, query, value, value, bev_pos=torch.zeros_like(query),
                  spatial_shapes=list(spatial), sph_ref_3d=spherical)
    return out, calls[0]


class ProjectionChainTests(unittest.TestCase):
    def test_stale_compiled_extension_is_rejected_before_kernel_execution(self):
        extension=SimpleNamespace(__file__='/tmp/extension.so')
        with self.assertRaisesRegex(RuntimeError,'Rebuild from repository root'):
            validate_extension_build(extension)
        extension.trilinear_gradient_version='trilinear_grad_v2'
        extension.kernel_source_sha256='stale'
        with self.assertRaisesRegex(RuntimeError,'Stale deform3dattn'):
            validate_extension_build(extension)
        extension.kernel_source_sha256=hashlib.sha256(CUDA.read_bytes()).hexdigest()
        validate_extension_build(extension)

    def test_subprocess_failure_keeps_original_error_in_log(self):
        from tools.temporal.run_geometry_v2 import run
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'child.log'
            stdout,stderr=io.StringIO(),io.StringIO()
            with redirect_stdout(stdout),redirect_stderr(stderr):
                with self.assertRaises(subprocess.CalledProcessError):
                    run(['python','-c',"print('probe stage',flush=True); raise RuntimeError('original CUDA error')"],log_path=path)
            text=path.read_text()
            self.assertIn('probe stage',text)
            self.assertIn('Traceback',text)
            self.assertIn('RuntimeError: original CUDA error',text)
            self.assertIn(str(path),stderr.getvalue())

    def test_physical_radar_angles_invert_original_bins(self):
        inverse = source_function(DETECTOR, 'RadarOcc_small', 'cartesian_to_spherical')
        raw = torch.tensor([[r, a, e] for r in (1, 60, 174)
                            for a in (0, 31, 53, 87, 106) for e in (5, 18, 32)], dtype=torch.float64)
        distance = raw[:, 0] * .462890625
        az = torch.deg2rad(raw[:, 1]-53)
        el = torch.deg2rad(raw[:, 2]-18)
        xyz = torch.stack([distance*torch.cos(el)*torch.cos(az),
                           -distance*torch.cos(el)*torch.sin(az), -distance*torch.sin(el)], -1)
        rr, aa, ee = inverse(None, *xyz.unbind(-1))
        torch.testing.assert_close(torch.stack([rr, aa, ee], -1),
                                   torch.stack([distance, az, el], -1))

    def test_actual_cartesian_references_follow_encoder_centres(self):
        inverse = source_function(DETECTOR, 'RadarOcc_small', 'cartesian_to_spherical')
        refs = source_function(DETECTOR, 'RadarOcc_small', 'get_reference_points_spherical')
        obj = SimpleNamespace(cartesian_to_spherical=MethodType(inverse, SimpleNamespace()))
        got = refs(obj, spherical_shape=[128, 128, 14], scale=2, dtype=torch.float64)
        indices = torch.cartesian_prod(torch.arange(128), torch.arange(128), torch.arange(14))
        lidar = (indices.double()+.5)*.4 + torch.tensor([0., -25.6, -2.6], dtype=torch.float64)
        xyz = lidar - torch.tensor([2.54, -.30, -.70], dtype=torch.float64)
        rr = torch.linalg.vector_norm(xyz, dim=-1)
        raw_r = rr / .462890625
        raw_a = -torch.rad2deg(torch.atan2(xyz[:, 1], xyz[:, 0])) + 53
        raw_e = -torch.rad2deg(torch.asin(xyz[:, 2]/rr)) + 18
        feature = torch.stack([raw_r/2, (2*raw_a+149)/4, (2*raw_e-9)/4], -1)
        torch.testing.assert_close(got*got.new_tensor([128, 128, 14])-.5, feature,
                                   rtol=0, atol=2e-6)

    def test_projected_residual_fractional_boundary_batch_and_gradient(self):
        project = source_function(DETECTOR, 'RadarOcc_small', 'project_spherical_residual', extra=dict(F=F))
        torch.manual_seed(12)
        source = torch.randn(2, 3, 7, 11, 5, dtype=torch.float64, requires_grad=True)
        points = torch.tensor([[8., 2., 3.], [4.2, 1.3, 2.5],
                               [10.1, 6.2, 4.2], [-.2, .1, 1.], [18., 2., 3.]], dtype=torch.float64)
        locations = ((points+.5)/points.new_tensor([11, 7, 5])).requires_grad_()
        got = project(source, locations, (5, 1, 1)).flatten(2).transpose(1, 2)
        expected = trilinear_oracle(source, locations[None].expand(2, -1, -1))
        torch.testing.assert_close(got, expected, rtol=0, atol=2e-14)
        grad_got = torch.autograd.grad(got.square().sum(), (source, locations), retain_graph=True)
        grad_expected = torch.autograd.grad(expected.square().sum(), (source, locations))
        for actual, oracle in zip(grad_got, grad_expected):
            torch.testing.assert_close(actual, oracle, rtol=1e-10, atol=1e-10)
        self.assertTrue(torch.equal(got[:, -1], torch.zeros_like(got[:, -1])))

    def test_encoder_cross_references_have_value_batch_self_queue_unchanged(self):
        for batch in (1, 2):
            query = torch.randn(5, batch, 3)
            refs = torch.rand(5, 3)
            out, call = encoder_fixture(query, torch.randn(batch, 8, 3), refs)
            self.assertEqual(out.shape, (batch, 5, 3))
            torch.testing.assert_close(call['sph_ref_3d'], refs[None, :, None].expand(batch, -1, -1, -1))
            _, self_call = encoder_fixture(torch.randn(8, batch, 3))
            self.assertEqual(self_call['ref_2d'].shape, (batch*2, 8, 1, 3))

    def test_cross_attention_actual_forward_and_duplicate_batch_rejected(self):
        forward = source_function(TRANSFORMER/'deformable_cross_attention_3D_custom.py',
                                  'DeformCrossAttention3DCustom', 'forward', extra={
            'multi_scale_deformable_attn_pytorch': lambda value, shapes, locations, weights:
                trilinear_oracle(value[:, :, 0].transpose(1, 2).reshape(2, 3, 7, 11, 5),
                                 locations[:, :, 0, 0, 0]) * weights[:, :, 0, 0]})
        torch.manual_seed(19)
        source = torch.randn(2, 3, 7, 11, 5)
        locations = torch.rand(2, 4, 1, 3)
        obj = SimpleNamespace(batch_first=True, num_bev_queue=1, num_heads=1,
            num_levels=1, num_points=1, im2col_step=1, value_proj=nn.Identity(),
            sampling_offsets=lambda q: q.new_zeros(*q.shape[:2], 3),
            attention_weights=lambda q: q.new_zeros(*q.shape[:2], 1),
            output_proj=nn.Identity(), dropout=nn.Identity())
        query = torch.zeros(2, 4, 3)
        value = source.flatten(2).transpose(1, 2)
        out = forward(obj, query, value=value, spatial_shapes=[7, 11, 5], reference_points=locations)
        torch.testing.assert_close(out, trilinear_oracle(source, locations[:, :, 0]))
        with self.assertRaisesRegex(ValueError, 'references must be'):
            forward(obj, query, value=value, spatial_shapes=[7, 11, 5],
                    reference_points=locations.repeat_interleave(2, dim=0))

    def test_sparse_bounds_crop_follows_full_range_attention(self):
        calls = {}
        def range_attention(features, e, a):
            calls['range_input_count'] = len(features)
            return features + 100
        def sparse_tensor(features, indices, shape, batch):
            calls.update(features=features, indices=indices, shape=shape, batch=batch)
            return SimpleNamespace(dense=lambda: torch.zeros(1, 11, 2, 3, 4))
        forward = source_function(ENCODER, 'RadarEncV8small', 'forward',
                                  extra=dict(spconv=SimpleNamespace(SparseConvTensor=sparse_tensor)))
        obj = SimpleNamespace(record_time=False, sparse_shape_xyz=[512, 512, 56],
            range_attn=range_attention, **{key: nn.Identity() for key in ('conv_input', 'conv1', 'conv2', 'conv_out')})
        coords = torch.tensor([[0, e, 60, 58] for e in range(37)])
        features = torch.arange(37.)[:, None].expand(-1, 8).clone()
        out = forward(obj, features, coords, 1)
        self.assertEqual(calls['range_input_count'], 37)
        self.assertEqual(calls['indices'][:, 1].tolist(), list(range(1, 56, 2)))
        torch.testing.assert_close(calls['features'][:, 3:], features[5:33]+100)
        torch.testing.assert_close(calls['features'][:, :3], calls['indices'][:, 1:].float())
        self.assertEqual(out['x'].shape, (1, 11, 4, 3, 2))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable: real sparse encoder probe skipped')
    def test_real_sparse_encoder_forward_backward_cuda(self):
        # Import the real registered implementation only on a configured GPU host.
        from projects.occ_plugin.occupancy.voxel_encoder.sparse_lidar_enc import RadarEncV8small
        module = RadarEncV8small(input_channel=11, norm_cfg=dict(type='BN1d'),
            base_channel=32, out_channel=144, sparse_shape_xyz=[512, 512, 56], top_k=1).cuda()
        features = torch.rand(175, 8, device='cuda') * 100 + 1
        features[:, 3:6] = 3
        features.requires_grad_()
        coords = torch.stack([torch.zeros(175), torch.arange(175) % 37,
                              torch.arange(175), torch.full((175,), 53)], -1).int().cuda()
        out = module(features, coords, 1)['x']
        self.assertEqual(out.shape, (1, 144, 128, 128, 14))
        self.assertTrue(bool(torch.isfinite(out).all()))
        out.square().mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(bool(torch.isfinite(features.grad).all()))


if __name__ == '__main__':
    unittest.main()
