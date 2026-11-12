#!/usr/bin/env python3
"""Audit the active Small projection without editing the model or training.

Executes selected source methods on CPU and tests sampling against the checked-in
CUDA kernel's coordinate contract. Child networks are replaced by labelled
identity/zero fixtures only for the explicit tensor-layout experiment. This is
not a full-network/checkpoint validation. Optional CUDA probes use tiny tensors.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace, MethodType

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat, whosmat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from temporal.core.ego_alignment import RadarBinGeometry, radar_to_lidar_default, align_to_current
from temporal.fusion.local_attention import local_neighbors
from tools.temporal_vis.vis_alignment import read_infos, scene_infos
from tools.temporal_vis.compare_coordinate_neighbor_selection import frame_data, setup_plotting, write_json
from tools.temporal_vis.audit_dynamic_objects import transform
from tools.temporal.train_top8_ranker import ranker_memberships
from tools.temporal_vis.vis_p1_reflection_regions import load_gt

DETECTOR = ROOT/'projects/occ_plugin/occupancy/detectors/radarocc_self_small.py'
ENCODER = ROOT/'projects/occ_plugin/occupancy/voxel_encoder/sparse_lidar_enc.py'
CUDA = ROOT/'third_party/VoxFormer/deform_attn_3d/csrc/ms_deform_attn_cuda_kernel.cuh'
S = np.array([1., -1., -1.])
SHAPE = (128, 128, 14)  # [A,R,E] feature and [X,Y,Z] occupancy, respectively.
ORIGIN = np.array([0., -25.6, -2.6])
EXT = radar_to_lidar_default()


class CPUCalls(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
            if node.args or node.keywords:
                raise ValueError('Unexpected cuda(...) arguments in source fixture')
            return node.func.value
        return node


def source_function(path, cls, name, cpu_calls=False, extra=None):
    tree = ast.parse(path.read_text())
    parent = tree if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = copy.deepcopy(next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    if cpu_calls:
        node = CPUCalls().visit(node)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    env = dict(torch=torch, np=np, copy=copy, F=F)
    env.update(extra or {})
    exec(compile(module, str(path), 'exec'), env)
    return env[name]


def statistics(error):
    e = np.asarray(error)
    return dict(median=float(np.median(e)), p95=float(np.quantile(e, .95)), maximum=float(e.max()))


def inverse_raw(xyz, bins):
    """Physical radar XYZ -> original table indices; independent of model formula."""
    xyz = np.asarray(xyz, dtype=np.float64)
    r = np.linalg.norm(xyz, axis=-1)
    az = -np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0]))
    el = -np.rad2deg(np.arcsin(np.clip(xyz[..., 2]/np.maximum(r, 1e-30), -1, 1)))
    return np.stack([(r-bins[0][0])/(bins[0][1]-bins[0][0]),
        (az-bins[1][0])/(bins[1][1]-bins[1][0]),
        (el-bins[2][0])/(bins[2][1]-bins[2][0])], -1)


def nominal_feature_indices(raw):
    # The two 3x3 stride-2, pad-1 convolutions have total jump 4 and centre offset 0.
    return np.stack([raw[..., 0]/2, (2*raw[..., 1]+149)/4, (2*raw[..., 2]-9)/4], -1)


def sampler_reference(radar_xyz, bins):
    # CUDA locations [W,H,D] = [R,A,E] for flattened [A,R,E] storage.
    return (nominal_feature_indices(inverse_raw(radar_xyz, bins))+.5)/np.array([128,128,14])


def sample_kernel_contract(volume, locations):
    """[B,C,A,R,E], normalized [R,A,E] -> [B,Q,C], align_corners=False."""
    source = volume.permute(0,1,4,2,3).contiguous()  # grid_sample [B,C,E,A,R]
    grid = (locations*2-1)[:, :, None, None, :]
    return F.grid_sample(source, grid, align_corners=False, padding_mode='zeros')[:, :, :, 0, 0].transpose(1,2)


def validate_extension_build(extension):
    expected = hashlib.sha256(CUDA.read_bytes()).hexdigest()
    if (getattr(extension, 'trilinear_gradient_version', None) != 'trilinear_grad_v2' or
            getattr(extension, 'kernel_source_sha256', None) != expected):
        raise RuntimeError(
            'Stale deform3dattn_custom_cn .so: CUDA gradient source and compiled extension disagree. '
            'Rebuild from repository root: '
            '(cd third_party/VoxFormer/deform_attn_3d && FORCE_CUDA=1 MAX_JOBS=2 '
            'python setup.py build_ext --inplace). Then restart Python and rerun V13. '
            f'Loaded extension: {extension.__file__}')


def probe_cuda_sampler(tiny, gpu):
    """Actual extension forward/backward; stage is retained if a check fails."""
    gpu['stage'] = 'extension_import'
    sys.path.insert(0,str(ROOT/'third_party/VoxFormer/deform_attn_3d'))
    import deform3dattn_custom_cn as extension
    gpu['extension_file'] = extension.__file__
    gpu['stage'] = 'extension_source_fingerprint'
    validate_extension_build(extension)
    gpu['kernel_source_sha256'] = extension.kernel_source_sha256
    _,_,a,r,e = tiny.shape
    value=tiny.permute(0,2,3,4,1).reshape(1,-1,1,3).cuda().contiguous()
    points=torch.tensor([[[2.,8.,3.],[1.3,4.2,2.5],[6.2,10.1,4.2],[-.2,.1,1.]]])
    cpu_locations=(points[...,[1,0,2]]+.5)/torch.tensor([r,a,e])
    locations=cpu_locations.cuda()[:,:,None,None,None,:].contiguous()
    shapes=torch.tensor([[a,r,e]],dtype=torch.long,device='cuda')
    starts=torch.tensor([0],dtype=torch.long,device='cuda')
    weight=torch.ones((1,points.shape[1],1,1,1),device='cuda')
    gpu['stage'] = 'cuda_forward_vs_cpu'
    got=extension.ms_deform_attn_forward(value,shapes,starts,locations,weight,im2col_step=1).cpu()
    expected=sample_kernel_contract(tiny,cpu_locations)
    gpu.update(actual=got.tolist(),expected=expected.tolist())
    torch.testing.assert_close(got,expected,rtol=0,atol=3e-5)
    cpu_values=tiny.clone().requires_grad_()
    cpu_locations=cpu_locations.clone().requires_grad_()
    sample_kernel_contract(cpu_values,cpu_locations).sum().backward()
    gv=torch.zeros_like(value);gl=torch.zeros_like(locations);ga=torch.zeros_like(weight)
    gpu['stage'] = 'cuda_backward'
    extension.ms_deform_attn_backward(value,shapes,starts,locations,weight,
        torch.ones_like(got,device='cuda'),gv,gl,ga,im2col_step=1)
    gpu['stage'] = 'value_gradient_vs_cpu'
    expected_gv=cpu_values.grad.permute(0,2,3,4,1).reshape_as(gv.cpu())
    torch.testing.assert_close(gv.cpu(),expected_gv,rtol=0,atol=3e-5)
    gpu['stage'] = 'sampling_location_gradient_vs_cpu'
    gpu.update(actual_location_gradient=gl.cpu()[:,:,0,0,0].tolist(),
               expected_location_gradient=cpu_locations.grad.tolist())
    torch.testing.assert_close(gl.cpu()[:,:,0,0,0],cpu_locations.grad,rtol=0,atol=3e-4)
    gpu['stage'] = 'attention_weight_gradient_vs_cpu'
    torch.testing.assert_close(ga.cpu()[:,:,0,0,0],expected.sum(-1),rtol=0,atol=3e-5)
    gpu.update(compiled_kernel='pass',stage='complete',device=torch.cuda.get_device_name(0),
               forward_and_backward_vs_CPU='pass')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/projection_chain_audit/20261112_fixed')
    parser.add_argument('--check-cuda', action='store_true', help='Also require the tiny compiled-kernel probe')
    parser.add_argument('--require-pass', action='store_true', help='Return 1 when current-chain findings fail')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    def check(name, passed, evidence, status=None):
        checks.append(dict(name=name, status=status or ('pass' if passed else 'fail'), evidence=evidence))

    bins_path = ROOT/'data/K-Radar-official-meta/resources/info_arr.mat'
    table = loadmat(bins_path)
    bins = [np.asarray(table[k], float).ravel() for k in ['arrRange','arrAzimuth','arrElevation']]
    geometry = RadarBinGeometry.from_file(bins_path, coordinate_convention='kradar_physical_v2')
    methods = {name:source_function(DETECTOR,'RadarOcc_small',name) for name in
               ['cartesian_to_spherical','get_reference_points_spherical','_prepare_p1_frame',
                'project_spherical_residual']}
    obj = SimpleNamespace()
    captured_queries = {}
    def record_cartesian(self, x, y, z):
        captured_queries['radar_xyz'] = torch.stack([x,y,z],-1).detach().cpu().numpy()
        return methods['cartesian_to_spherical'](self,x,y,z)
    obj.cartesian_to_spherical = MethodType(record_cartesian, obj)
    ref = methods['get_reference_points_spherical'](obj, device='cpu', scale=2,
                 spherical_shape=list(SHAPE), dtype=torch.float32).numpy()
    voxel_indices = np.indices(SHAPE).reshape(3,-1).T
    lidar_xyz = (voxel_indices+.5)*.4+ORIGIN
    radar_xyz = transform(lidar_xyz, np.linalg.inv(EXT))
    raw_expected = inverse_raw(radar_xyz, bins)
    expected = nominal_feature_indices(raw_expected)  # [R,A,E]
    decoded = ref*np.array([128,128,14])-.5  # Actual CUDA's interpretation.
    valid = (radar_xyz[:,0]>0)&np.all((raw_expected>=0)&(raw_expected<=[174,106,36]),axis=1)
    # Check the actual angle method independently of the reference axis order.
    _, actual_az, actual_el = methods['cartesian_to_spherical'](
        obj, *torch.from_numpy(radar_xyz).unbind(-1))
    expected_angles = np.stack([bins[1][0]+raw_expected[:,1],
                                bins[2][0]+raw_expected[:,2]], -1)
    angular_delta = np.abs(torch.stack([actual_az,actual_el],-1).numpy()*180/np.pi-expected_angles)
    check('inverse_angle_sign', bool(angular_delta[valid].max()<1e-8),
          dict(valid_query_count=int(valid.sum()), error_degrees=statistics(angular_delta[valid]),
               source=f'{DETECTOR.relative_to(ROOT)}:100', expected='negative atan2/asin for original K-Radar indices'))
    check('sampler_range_azimuth_order', bool(np.max(np.abs(decoded[valid]-expected[valid]))<1e-3),
          dict(storage='[B,C,A,R,E]', cuda_location_order='[R,A,E]', source_reference_order='[R,A,E]',
               max_error_in_feature_cells=float(np.abs(decoded[valid]-expected[valid]).max())))
    check('range_spacing_and_nominal_centres', bool(np.max(np.abs(decoded-expected))<2e-4),
          dict(max_error_cells=float(np.abs(decoded-expected).max()), tested=len(decoded),
          note='Actual detector references vs original angle tables and sparse convolution nominal centres.'))
    # All nonzero raw bins, not just selected/high-power examples.
    raw_grid = np.stack(np.meshgrid(np.arange(1,len(bins[0])),np.arange(107),np.arange(37),indexing='ij'),-1).reshape(-1,3)
    roundtrip = inverse_raw(geometry.to_xyz(raw_grid), bins)
    check('raw_bin_physical_XYZ_roundtrip', bool(np.max(np.abs(roundtrip-raw_grid))<1e-9),
          dict(tested=len(raw_grid), max_error_raw_bins=float(np.max(np.abs(roundtrip-raw_grid)))))
    fixed_ref = sampler_reference(radar_xyz, bins)
    # Decode through the CUDA centre convention and sparse-encoder affine map,
    # then independently reconstruct physical XYZ with the forward trig formula.
    recovered_feature=fixed_ref*np.array([128,128,14])-.5
    recovered_raw=np.stack([recovered_feature[:,0]*2,
        (recovered_feature[:,1]*4-149)/2,(recovered_feature[:,2]*4+9)/2],-1)
    rr=recovered_raw[:,0]*(bins[0][1]-bins[0][0])+bins[0][0]
    aa=-np.deg2rad(recovered_raw[:,1]*(bins[1][1]-bins[1][0])+bins[1][0])
    ee=-np.deg2rad(recovered_raw[:,2]*(bins[2][1]-bins[2][0])+bins[2][0])
    recovered_xyz=np.stack([rr*np.cos(ee)*np.cos(aa),rr*np.cos(ee)*np.sin(aa),rr*np.sin(ee)],-1)
    check('independent_corrected_reference', bool(np.max(np.abs(recovered_xyz-radar_xyz))<1e-10),
          dict(tested=len(lidar_xyz), max_roundtrip_error_m=float(np.max(np.abs(recovered_xyz-radar_xyz))),
               note='Independent forward/inverse oracle; actual detector references are checked separately above.'))

    infos = scene_infos(read_infos(ROOT/'data/annotations/kradar_dict_test_official_temporal_doppler8.pkl'),3)
    seq = [f for f in infos if 77<=int(f['order_in_scene'])<=80]
    assert [int(f['order_in_scene']) for f in seq]==[77,78,79,80]
    current=seq[-1]
    with np.load(current['sparse_radar_path']) as raw:
        payload={k:torch.from_numpy(raw[k].copy())[None] for k in raw.files}
    features, coords, batch = methods['_prepare_p1_frame'](None,payload)
    frames=[frame_data(f,geometry) for f in seq]
    check('P1_descriptor_index_association', bool(np.array_equal(features.numpy(),frames[-1][1]) and
          np.array_equal(coords[:,[2,3,1]].numpy(),frames[-1][2])),
          dict(tokens=len(features),channels=features.shape[1],source='actual _prepare_p1_frame on CPU'))
    sparse_e = coords[:,1].numpy()*2-9
    invalid = (sparse_e<0)|(sparse_e>=56)
    sparse_capture={}
    def range_identity(values,*args):
        sparse_capture['range_input_count']=len(values)
        return values
    def capture_sparse(values,indices,shape,batch):
        sparse_capture.update(values=values,indices=indices,shape=shape)
        return SimpleNamespace(dense=lambda:torch.zeros(1,11,1,1,1))
    encoder_source=source_function(ENCODER,'RadarEncV8small','forward',
        extra=dict(spconv=SimpleNamespace(SparseConvTensor=capture_sparse)))
    sparse_fixture=SimpleNamespace(record_time=False,sparse_shape_xyz=[512,512,56],range_attn=range_identity,
        **{name:torch.nn.Identity() for name in ['conv_input','conv1','conv2','conv_out']})
    encoder_source(sparse_fixture,features,coords,batch)
    indices_after=sparse_capture['indices']; bounds=torch.tensor([1,56,512,512])
    filtered_correctly=torch.equal(sparse_capture['values'][:,3:],features[~invalid])
    check('sparse_encoder_index_bounds', bool(((indices_after>=0)&(indices_after<bounds)).all()) and
          len(indices_after)==int((~invalid).sum()) and filtered_correctly,
          dict(cropped_tokens=int(invalid.sum()),total=len(invalid), fraction=float(invalid.mean()),
               passed_to_sparse_encoder=len(indices_after),range_attention_input_count=sparse_capture['range_input_count'],
               formula='E_sparse=2*E_raw-9, allowed 0..55',
               invalid_raw_E=sorted(set(coords[invalid,1].numpy().tolist())),
               note='Executed actual encoder forward with identity Range Attention / sparse-convolution fixtures. Invalid coordinates and matching features are cropped together AFTER Range Attention. No clamping.'))

    # Execute the detector's actual post-sparse path. Only child networks are
    # replaced with labelled fixtures, and local AST .cuda() calls are omitted.
    a,r,e=torch.meshgrid(torch.arange(128),torch.arange(128),torch.arange(14),indexing='ij')
    field=torch.stack([a,r,e]).float()[None]
    captured={}
    # Execute both original Transformer/Encoder wrappers as well: only the
    # learned attention/FFN layer is identity, so its output layout is measured.
    encoder_path=ROOT/'projects/occ_plugin/occupancy/deformable_transformer/encoder_3D.py'
    transformer_path=ROOT/'projects/occ_plugin/occupancy/deformable_transformer/transformer_3D.py'
    encoder_forward=source_function(encoder_path,'VoxFormerEncoder3D','forward')
    reference_function=source_function(encoder_path,'VoxFormerEncoder3D','get_reference_points')
    layer_calls=[]
    def identity_layer(query,*args,**kwargs):
        layer_calls.append(kwargs)
        return query
    encoder_fixture=SimpleNamespace(return_intermediate=False,
        get_reference_points=reference_function,layers=[identity_layer])
    def encoder_call(*args,**kwargs):return encoder_forward(encoder_fixture,*args,**kwargs)
    diffuse=source_function(transformer_path,'PerceptionTransformer3D','diffuse_vox_features')
    transformer_fixture=SimpleNamespace(encoder=encoder_call)
    def identity_self(*args,**kwargs):return diffuse(transformer_fixture,*args,**kwargs)
    cart_forward=source_function(transformer_path,'PerceptionTransformer3D','get_cart_features')
    def cross(query,spherical,*args,**kwargs):
        captured['spherical_input']=spherical.detach().clone()
        return cart_forward(transformer_fixture,query,spherical,*args,**kwargs)
    fixture=SimpleNamespace(record_time=False,temporal_plugin=None,embed_dims=3,
        _prepare_p1_frame=lambda _: (features,coords,1),
        pts_middle_encoder=lambda *args:dict(x=field,pts_feats=[]),
        positional_encoding=lambda mask: torch.zeros(1,3,*mask.shape[1:]),
        self_transformer=SimpleNamespace(diffuse_vox_features=identity_self),
        transformer=SimpleNamespace(get_cart_features=cross),get_ref_3d=lambda:(None,np.zeros((1,3),np.float32)),
        cart_voxel_emb=SimpleNamespace(weight=torch.zeros(np.prod(SHAPE),3)))
    fixture.get_reference_points_spherical=MethodType(methods['get_reference_points_spherical'],obj)
    fixture.project_spherical_residual=methods['project_spherical_residual']
    extractor=source_function(DETECTOR,'RadarOcc_small','extract_pts_feat',cpu_calls=True)
    fixture_output,_=extractor(fixture,None)
    layout=captured['spherical_input']
    bad_values=int((layout!=field).sum())
    check('self_attention_output_to_grid', bad_values==0,
          dict(mismatched_scalar_values=bad_values,total_values=field.numel(),
               original_contract='[B,N,C] returned by encoder',
               expected_conversion='transpose(1,2), then reshape(B,C,A,R,E)',
               example_A_R_E=[30,70,5],expected_channels=field[0,:,30,70,5].tolist(),
               observed_channels=layout[0,:,30,70,5].tolist(),
               fixture='Original Detector + PerceptionTransformer3D + VoxFormerEncoder3D methods; identity attention/FFN layer and zero cross-attention; local cuda calls removed'))
    # Separate tensor/projection correctness from FP32 reference round-off.
    # Near zero-padding boundaries, a 1e-5-cell location difference is amplified
    # by a position-coded value of ~128. Angle/reference precision is checked above.
    physical_skip=sample_kernel_contract(field,torch.from_numpy(fixed_ref).float()[None])
    physical_skip=physical_skip.transpose(1,2).reshape_as(fixture_output)
    expected_skip=sample_kernel_contract(field,torch.from_numpy(ref).float()[None])
    expected_skip=expected_skip.transpose(1,2).reshape_as(fixture_output)
    skip_error=float((fixture_output-expected_skip).abs().max())
    check('spherical_skip_projected_to_cartesian', skip_error<3e-5,
          dict(max_tensor_projection_error=skip_error,
               max_FP32_vs_independent_reference_value_difference=float((fixture_output-physical_skip).abs().max()),
               zero_cross_output_leaves_spherical_grid=bool(torch.equal(fixture_output,layout)),
               note='Actual detector skip vs independently computed physical references on a position-coded field; both residual and cross output now use Cartesian XYZ.'))
    cross_calls=[c for c in layer_calls if c['sph_ref_3d'] is not None]
    actual_reference_shape=list(cross_calls[-1]['sph_ref_3d'].shape)
    check('cross_attention_reference_batch', actual_reference_shape==[1,int(np.prod(SHAPE)),1,3],
          dict(value_batch=1,reference_shape=actual_reference_shape,
               note='Actual PerceptionTransformer3D.get_cart_features + VoxFormerEncoder3D.forward executed; learned attention layer replaced by identity.'))

    # Independent non-square lattice exposes H/W swaps masked by 128==128.
    A,R,E=7,11,5
    ia,ir,ie=torch.meshgrid(torch.arange(A),torch.arange(R),torch.arange(E),indexing='ij')
    tiny=torch.stack([ia,ir,ie]).float()[None]
    desired=torch.tensor([[[2.,8.,3.]]])
    good_loc=(desired[...,[1,0,2]]+.5)/torch.tensor([R,A,E])
    bad_loc=(desired+.5)/torch.tensor([A,R,E])
    good_sample=sample_kernel_contract(tiny,good_loc)
    bad_sample=sample_kernel_contract(tiny,bad_loc)
    check('nonsquare_zero_offset_sampling_reference', bool(torch.allclose(good_sample,desired,atol=2e-6)),
          dict(expected=desired.tolist(),correct_order_sample=good_sample.tolist(),
               swapped_order_sample=bad_sample.tolist(),shape_A_R_E=[A,R,E]))

    # Run the actual loader's assignment logic, including duplicate-voxel voting.
    loader_path=ROOT/'projects/occ_plugin/datasets/pipelines/loading.py'
    vote=source_function(loader_path,None,'nb_process_label')
    import os
    loader_tree=ast.parse(loader_path.read_text())
    loader_node=copy.deepcopy(next(n for n in loader_tree.body if isinstance(n,ast.ClassDef) and n.name=='LoadOccupancy'))
    loader_node.decorator_list=[]
    env=dict(np=np,torch=torch,os=os,copy=copy,nb_process_label=vote)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[loader_node],type_ignores=[])),str(loader_path),'exec'),env)
    loader=env['LoadOccupancy'](grid_size=list(SHAPE),pc_range=[*ORIGIN,51.2,25.6,3.],simply_label=True)
    loaded=loader(dict(occ_path=current['occ_path']))['gt_occ']
    gt_raw=np.load(current['occ_path']); expected_gt=np.zeros(SHAPE,np.uint8)
    from collections import defaultdict,Counter
    groups=defaultdict(list)
    for x,y,z,label in gt_raw.astype(int):
        groups[x,y,z].append(255 if label in (0,255) else (2 if label>=3 else label))
    for key,labels in groups.items():expected_gt[key]=sorted(Counter(labels).items(),key=lambda item:(-item[1],item[0]))[0][0]
    check('GT_XYZ_indices_to_dense_target', bool(np.array_equal(loaded,expected_gt)),
          dict(rows=len(gt_raw),dense_shape=list(loaded.shape),matching_voxels=int((loaded==expected_gt).sum()),
               total_voxels=loaded.size,note='Actual loader methods; GT file columns are X,Y,Z. Its unused auxiliary ZYX conversion does not change the dense target.'))
    # The actual evaluation method should give a diagonal confusion matrix when
    # handed a perfect predictor in the loader's XYZ ordering.
    fast_hist=source_function(DETECTOR,None,'fast_hist')
    evaluation=source_function(DETECTOR,'RadarOcc_small','evaluation_semantic',extra=dict(fast_hist=fast_hist))
    oracle=F.one_hot(torch.from_numpy(loaded.astype(np.int64)).clamp_max(2),num_classes=3).permute(3,0,1,2)[None].float()
    target=torch.from_numpy(loaded.astype(np.int64))[None]
    sc,sc1,sc2=evaluation(SimpleNamespace(empty_idx=0),oracle,target,'SC')
    ssc,ssc1,ssc2=evaluation(SimpleNamespace(empty_idx=0),oracle,target,'SSC')
    all_hists=[sc,sc1,sc2,ssc,ssc1,ssc2]
    check('prediction_GT_and_range_evaluation_axes', all(int(h.sum()-np.trace(h))==0 for h in all_hists),
          dict(oracle_SSC_confusion=ssc,oracle_SC_confusion=sc,
               range1_XY_m=[[0,25.6],[-12.8,12.8]],range2_XY_m=[[0,12.8],[-6.4,6.4]],
               note='Actual evaluation_semantic with a perfect XYZ-indexed prediction; this does not test a learned model.'))
    # dense_heads/__init__.py registers this implementation even in FP32 mode.
    head_path=ROOT/'projects/occ_plugin/occupancy/dense_heads/occ_head_fp16.py'
    coarse=source_function(head_path,'OccHead','forward_coarse_voxel',extra=dict(F=F))
    coarse_fixture=SimpleNamespace(occ_convs=[torch.nn.Identity() for _ in range(4)],soft_weights=True,
        voxel_soft_weights=lambda x:torch.zeros(x.shape[0],4,*x.shape[2:]),
        occ_pred_conv=torch.nn.Identity())
    coarse_result=coarse(coarse_fixture,[tiny.clone() for _ in range(4)])['occ'][0]
    check('occupancy_head_preserves_XYZ_order',bool(torch.equal(coarse_result,tiny)),
          dict(shape=[A,R,E],note='Actual forward_coarse_voxel; identity convolutions and fixed uniform scale weights on a non-square position-coded grid.'))
    query_centres = transform(captured_queries['radar_xyz'],EXT)
    check('occupancy_query_voxel_centres', bool(np.max(np.abs(query_centres-lidar_xyz))<1e-5),
          dict(shape_X_Y_Z=list(SHAPE),voxel_size_m=.4,
               minimum_centre=query_centres.min(0),maximum_centre=query_centres.max(0),
               max_error_m=float(np.abs(query_centres-lidar_xyz).max()),
               gt_generation='Original Z origin -5.0, crop six 0.4m Z layers -> -2.6; no Y/Z sign change'))

    # An independent original MAT -> NPZ spot check, with no geometric assumptions.
    raw_mat=ROOT/'data/K-Radar/11/radar_tensor_8doppler/tesseract_00034.mat'
    raw_npz=ROOT/'data/RadarOcc_8doppler/11/radar_tensor_8doppler/EAsparse_00034.npz'
    if raw_mat.is_file() and raw_npz.is_file():
        names=whosmat(raw_mat);key='arrDREA'
        assert key in [n[0] for n in names],names
        original=loadmat(raw_mat,variable_names=[key])[key]
        with np.load(raw_npz) as sparse:
            pick=np.arange(0,len(sparse['range_ind']),47)
            ri=sparse['range_ind'][pick].astype(int);ai=sparse['azimuth_ind'][pick].astype(int);ei=sparse['elevation_ind'][pick].astype(int)
            desc=sparse['power_val'][:,pick]
            di=desc[3:6].astype(int)
            observed=original[di,ri[None],ei[None],ai[None]]
            values_match=np.isclose(observed,desc[:3],rtol=2e-6,atol=0)
            check('raw_MAT_to_NPZ_unflipped_indices',bool(values_match.all()),dict(raw_shape=list(original.shape),
                tested_token_columns=len(pick),tested_top3_values=values_match.size,
                matches=int(values_match.sum()),MAT=str(raw_mat.relative_to(ROOT)),NPZ=str(raw_npz.relative_to(ROOT))))
        del original
    else:
        check('raw_MAT_to_NPZ_unflipped_indices',False,dict(note='Raw MAT or corresponding NPZ absent'),status='not_run')

    # Distinguish a global basis change from incorrectly mixing an ego transform.
    native=np.stack([f[0]*S for f in frames]);poses=np.stack([f['ego_pose'] for f in seq])
    old,transforms=align_to_current(native[None],poses[None],EXT)
    new,_=align_to_current((native*S)[None],poses[None],EXT)
    knn_old,mask_old=local_neighbors(old,np.ones((1,4),bool))
    knn_new,mask_new=local_neighbors(new*S,np.ones((1,4),bool))
    history=[]
    for slot in [2,1,0]:
        delta=new[0,slot]-old[0,slot]*S
        Rm=transforms[0,slot,:3,:3];tm=transforms[0,slot,:3,3];Sm=np.diag(S)
        algebra=native[slot]@(Rm@Sm-Sm@Rm).T+tm-Sm@tm
        np.testing.assert_allclose(delta,algebra,rtol=0,atol=1e-12)
        sl=slice(slot*8,(slot+1)*8)
        k1=np.sort(np.where(mask_old[0,:,sl],knn_old[0,:,sl],-1),axis=1)
        k2=np.sort(np.where(mask_new[0,:,sl],knn_new[0,:,sl],-1),axis=1)
        history.append(dict(gap=3-slot,position_change_m=statistics(np.linalg.norm(delta,axis=1)),
            changed_K8_query_sets=int(np.any(k1!=k2,axis=1).sum()),total_queries=len(delta)))
    p=np.array([30.,5.,1.]);synthetic=[]
    for label,yaw,t in [('pure_forward_translation',0,[-3,0,0]),('lateral_translation',0,[-3,.3,0]),('yaw_5_degrees',5,[-3,0,0])]:
        angle=np.deg2rad(yaw);c,s=np.cos(angle),np.sin(angle)
        Rm=np.array([[c,-s,0],[s,c,0],[0,0,1]]);tm=np.array(t)
        delta=(Rm@np.diag(S)-np.diag(S)@Rm)@p+tm-np.diag(S)@tm
        synthetic.append(dict(case=label,delta_xyz_m=delta,distance_m=float(np.linalg.norm(delta))))
    check('ego_basis_commutator_identity',True,dict(equation='new-old=(R S-S R)q+(I-S)t',
          actual_scene3_order80=history,synthetic_point_native_xyz=p,synthetic=synthetic))
    gt=load_gt(current,3,ROOT/'data/K-RadarOcc/train',np.eye(4))
    lab_old=ranker_memberships(transform(native[-1],EXT),gt['boxes'],gt['ids'])
    lab_new=ranker_memberships(transform(native[-1]*S,EXT),gt['boxes'],gt['ids'])
    check('absolute_XYZ_change_vs_GT',True,dict(before_inside_unique_GT=int((lab_old>=0).sum()),
          after_inside_unique_GT=int((lab_new>=0).sum()),
          note='These are absolute membership counts for unchanged GT; not an ego-warp displacement measurement.'))
    # The same high-power echo highlighted in the Scene 3 visualization.
    echo_raw=np.array([[60,58,15]])
    echo_lidar=transform(geometry.to_xyz(echo_raw),EXT)[0]
    echo_voxel=np.floor((echo_lidar-ORIGIN)/.4).astype(int)
    echo_flat=np.ravel_multi_index(echo_voxel,SHAPE)
    echo_example=dict(raw_R_A_E=echo_raw[0],lidar_xyz_m=echo_lidar,
        nearest_cart_voxel_X_Y_Z=echo_voxel,voxel_centre_lidar_m=lidar_xyz[echo_flat],
        actual_sampler_R_A_E=decoded[echo_flat],expected_sampler_R_A_E=expected[echo_flat],
        note='References before learned offsets, near the exact echo used in the previous GT overlay')

    gpu=dict(available=torch.cuda.is_available(),compiled_kernel='not_run')
    if args.check_cuda and gpu['available']:
        try:
            probe_cuda_sampler(tiny, gpu)
        except Exception as error:
            gpu.update(compiled_kernel='fail',error_type=type(error).__name__,
                       error=str(error),traceback=traceback.format_exc())
            write_json(output/'cuda_failure.json',gpu)
            traceback.print_exc()
            print('CUDA审计失败详情：'+str(output/'cuda_failure.json'),file=sys.stderr,flush=True)
    elif args.check_cuda:
        gpu['reason']='No CUDA device is visible in this execution environment'
    from temporal.experiments_v2.registry import PROJECTION_VERSION
    report=dict(scope='Active RadarOcc_small / RadarEncV8small / scale2 / 128x128x14',checks=checks,gpu=gpu,
        projection_version=PROJECTION_VERSION,
        scene3_high_power_echo_projection=echo_example,
        counts={status:sum(c['status']==status for c in checks) for status in ['pass','fail','review','not_run']},
        full_network_validation='not_run; this is source+geometry+layout auditing, not weighted-network accuracy validation',
        audit_modifies_model=False,source_variant='Small projection correction; old blocks preserved as comments',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
          [DETECTOR,ENCODER,CUDA,loader_path,head_path,ROOT/'projects/occ_plugin/occupancy/deformable_transformer/transformer_3D.py',
           ROOT/'projects/occ_plugin/occupancy/deformable_transformer/encoder_3D.py',
           ROOT/'projects/occ_plugin/occupancy/deformable_transformer/deformable_cross_attention_3D_custom.py',
           ROOT/'projects/occ_plugin/occupancy/deformable_transformer/deformable_self_attention_3D_custom.py']})
    compatibility_path=output/'checkpoint_compatibility.json'
    if compatibility_path.exists():
        report['checkpoint_compatibility']=json.loads(compatibility_path.read_text())
    write_json(output/'report.json',report)
    setup_plotting()
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for col,title in enumerate(['Range 特征索引','Azimuth 特征索引','Elevation 特征索引']):
        visible=valid.reshape(SHAPE)[:,:,6]
        arrays=[np.where(visible,decoded[:,col].reshape(SHAPE)[:,:,6],np.nan),
                np.where(visible,expected[:,col].reshape(SHAPE)[:,:,6],np.nan)]
        vmin=min(np.nanmin(v) for v in arrays);vmax=max(np.nanmax(v) for v in arrays)
        for row,values in enumerate(arrays):
            artist=axes[row,col].imshow(values.T,origin='lower',extent=[0,51.2,-25.6,25.6],
                aspect='equal',vmin=vmin,vmax=vmax,cmap='viridis')
            axes[row,col].set(title=('当前代码传入采样核' if row==0 else '独立物理坐标计算')+'：'+title,
                xlabel='LiDAR X (m)',ylabel='LiDAR Y (m)')
        fig.colorbar(artist,ax=list(axes[:,col]),shrink=.75)
    fig.suptitle('同一 Cartesian 体素查询，究竟定位到哪个球坐标特征格？\n'
                 '共同物理视野内 Z=0m 切片；上下同一颜色范围。比较参考位置，不是学习到的 attention offset。',fontsize=15)
    fig.savefig(output/'reference_locations.png',dpi=150);plt.close(fig)
    # Detailed prose is kept separate, so raw numbers and checks remain usable.
    cpu_pass=not any(c['status'] in ('fail','review') for c in checks)
    verdict='CPU几何与张量契约检查全部通过' if cpu_pass else '仍有未通过的检查，请查看逐项数据'
    lines=['# RadarOcc Small 投影链修正后审计','',
       f'**{verdict}。这不是带权重整网或CUDA验证。**','',
       '检查范围为当前Small模型实际路径，不包括另一个Full/Strict模型；源码SHA256见report.json。',
       '本报告执行修正后的源码方法。涉及学习层或稀疏卷积的CPU实验使用identity/zero夹具；实际CUDA采样核仅在--check-cuda且有GPU时运行。',
       '修正前的报告保留在[20261112/REPORT.md](../20261112/REPORT.md)，其中fail/review描述的是旧源码，不是当前修正后的状态。','',
       '## 逐段结果','', '| 环节 | 结果 | 证据 |','|---|---|---|']
    for c in checks:
        lines.append(f'| {c["name"]} | {c["status"]} | 详见report.json对应条目 |')
    lines += ['', '## 已应用到当前Small源码的修正','',
       '1. 物理XYZ查原始角度bin改为负atan2/asin，对应官方雷达坐标约定；LiDAR GT不反号。',
       '2. 球坐标特征保持[B,C,A,R,E]，采样参考位置改为CUDA接受的[R,A,E]。',
       '3. Self-Attention的[B,N,C]先transpose(1,2)，再恢复[B,C,A,R,E]，保留通道和空间对应。',
       f'4. 保留原[56,512,512]稀疏网格。本帧{int(invalid.sum()):,}/{len(invalid):,}个越界点在Range Attention之后成对裁剪坐标/特征，剩下{int((~invalid).sum()):,}点进入spconv；不把越界高度clamp到边界。',
       '   这是一项明确的裁剪选择：保留E_raw=5..32，舍去0..4与33..36。Range Attention仍看到全部输入；不是保留37个高度bin的扩展方案。',
       '5. 球坐标residual用相同物理参考位置、align_corners=False、越界补零，先插值到Cartesian XYZ再相加。保留残差分支，但修正其位置含义；效果需训练验证。',
       '6. Cross-Attention的reference与value均为B批，并在注意力入口检查形状；Self-Attention的两份队列保持原结构。',
       '', '被替换的旧实现均保留为注释。新增方法不增加训练参数；权重尺寸兼容不表示旧checkpoint等价于修正后训练。','',
       '## 已确认正确的部分','',
       '- P1当前描述符与R/A/E索引关联一致；实际GT加载后的[X,Y,Z]顺序正确。',
       '- 距离间距保持0.462890625 m/bin；参考位置与卷积名义中心及CUDA半格规则一致。',
       '- GT原Z范围[-5,3]裁掉下面6层，每层0.4m，变成[-2.6,3]，与Cartesian查询中心一致。',
       '- 独立参考几何检查原始bin往返；实际Small参考位置另与全部229,376个Cartesian体素的独立物理计算比较。','',
       '![实际采样参考位置与独立计算](reference_locations.png)','',
       '## 为什么此前历史位置只差十几厘米','',
       '直接反号改变绝对位置，雷达原点下变化为(0,-2Y,-2Z)。与固定GT比较、生成ranker标签和逆向投影都会受影响。',
       '历史图把两套结果先放回同一物理坐标系，再比较变换混用的额外误差：',
       '`new-old=(R S-S R)q+(I-S)t`，S=diag(1,-1,-1)。',
       '纯X平移与S可交换，这项误差严格为0；横移、升降、yaw/pitch会使其非零。小位移也可能改变离散TopK集合，集合改变不等于同目标匹配错误。','',
       '| 历史 | 位置变化中位数 | K8候选集合改变的query |','|---|---:|---:|']
    for row in history:
        lines.append(f'| t−{row["gap"]} | {row["position_change_m"]["median"]*100:.2f} cm | {row["changed_K8_query_sets"]:,} / 43,750 |')
    lines += ['', '| 合成变换（同一q） | 额外误差 |','|---|---:|']
    for row in synthetic:lines.append(f'| {row["case"]} | {row["distance_m"]:.4f} m |')
    lines += ['', '## 上次高power星号对应的位置','',
       f'原始bin为{echo_raw[0].tolist()}，对应物理LiDAR XYZ={np.round(echo_lidar,3).tolist()} m。',
       f'取最近Cartesian体素{echo_voxel.tolist()}，中心{np.round(lidar_xyz[echo_flat],3).tolist()} m。',
       '下面均为卷积后的特征格索引、CUDA的位置顺序[R,A,E]，不包含学习到的offset：','',
       '| 维度 | 当前实际参考 | 独立物理参考 |','|---|---:|---:|']
    for dim,old_i,new_i in zip(['Range','Azimuth','Elevation'],decoded[echo_flat],expected[echo_flat]):
        lines.append(f'| {dim} | {old_i:.4f} | {new_i:.4f} |')
    lines += ['', '## 验证边界与运行方法','',
       f'CUDA：{gpu}。未执行完整稀疏卷积、带权重整网或整套验证集IoU。不能对实测标定精度或模型预测作绝对保证。',
       '修正直接作用于当前Small源码，单帧与使用该模型的时序配置都会使用新投影。GT、已有checkpoint、历史日志不改写。',
       '请重新训练单帧对照和时序模型；旧模型上的开关融合只能诊断旧模型，不能替代修正后训练。','',
       '```bash','source use_radarocc5060.sh',
       'python -m pytest -q tests/test_projection_chain.py tests/test_geometry_v2.py',
       'python tools/temporal/audit_projection_chain.py --require-pass',
       '# GPU机器上补实际采样核前向/反向检查（使用独立输出目录）',
       'python tools/temporal/audit_projection_chain.py --require-pass --check-cuda --output-dir outputs/projection_chain_audit/20261112_fixed_cuda',
       '```','',
       '官方坐标定义：[K-Radar反号代码](https://github.com/kaist-avelab/K-Radar/blob/f90c28993d1757ce0236632b4d912c92a14976b2/tools/radar_film/get_pw_dist.py#L165)。',
       '论文意图：[RadarOcc §4.4/4.5](https://arxiv.org/html/2405.14014)。论文没有完整列出实际张量排列，不能用论文示意代替实现检查。']
    if 'checkpoint_compatibility' in report:
        compatibility=report['checkpoint_compatibility']
        lines += ['', '## 本地权重结构兼容性检查', '',
            f'`{compatibility["checkpoint"]}` 的{compatibility["tensor_keys"]}个权重项严格加载结果：{compatibility["strict_load"]}。',
            '仅在构造时将未使用的旧插值常量留在CPU；没有运行带权重整网前向。',
            '记录见[checkpoint_compatibility.json](checkpoint_compatibility.json)。这证明参数结构兼容，不证明旧权重在新几何下的精度。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>投影链审计</title>'
       '<style>body{font:17px/1.8 system-ui;max-width:1200px;margin:30px auto;padding:20px}img{width:100%}</style>'
       f'<h1>Small投影链修正：{verdict}</h1><p>旧实现已留在源码注释中。查看 '
       '<a href="REPORT.md">完整中文报告</a> 和 <a href="report.json">逐项数据</a>。</p>'
       '<p>下图比较实际参考位置与物理坐标应定位到的球坐标特征格；不是网络预测结果。</p>'
       '<img src="reference_locations.png"><p>已修项目、明确的E轴裁剪选择、CPU/CUDA验证范围和复现命令均在报告中。</p>')
    print(json.dumps(dict(counts=report['counts'],gpu=gpu,output=str(output),
          findings=[c['name'] for c in checks if c['status'] in ('fail','review')]),ensure_ascii=False,indent=2))
    if args.check_cuda and gpu['compiled_kernel']!='pass':return 2
    return 1 if args.require_pass and not cpu_pass else 0


if __name__=='__main__':
    sys.exit(main())
