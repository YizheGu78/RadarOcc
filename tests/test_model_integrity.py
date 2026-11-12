"""Real learned Small modules on CPU; no spconv/CUDA or identity substitutes.

The reduced synthetic volume tests composition and gradients, not real-data IoU.
"""
import copy
import gc
import importlib
from itertools import product
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from mmcv import Config

ROOT = Path(__file__).resolve().parents[1]
importlib.import_module('temporal.experiments_v2.plugin')
from mmdet3d.models import build_model
from projects.occ_plugin.utils.position_embedding import LearnedPositionalEncoding3D
from projects.occ_plugin.utils.semkitti import CE_ssc_loss, geo_scal_loss, sem_scal_loss
from projects.occ_plugin.occupancy.dense_heads.lovasz_softmax import lovasz_softmax
from projects.occ_plugin.occupancy.deformable_transformer.multi_scale_deformable_attn_3D_custom_function import multi_scale_deformable_attn_3d_pytorch
from projects.occ_plugin.occupancy.deformable_transformer.deformable_self_attention_3D_custom import DeformSelfAttention3DCustom
from projects.occ_plugin.occupancy.voxel_encoder.sparse_lidar_enc import MultiLayerRangeAttention
from projects.occ_plugin.datasets.builder import custom_build_dataset
from projects.occ_plugin.datasets.pipelines.loading import LoadOccupancy
from temporal import build_temporal_plugin
from temporal.core.learned_topk import PairRanker, save_ranker

PROBES = {}


def config(name):
    return Config.fromfile(str(ROOT/f'temporal/experiments_v2/configs/{name}.py'))


def build(name='current_only', cfg=None):
    cfg = cfg or config(name)
    return build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))


def sample_oracle(volume, locations):
    """Eight neighbours independently, volume [A,R,E,C], positions [R,A,E]."""
    a, r, e, channels = volume.shape
    rows = []
    for location in locations:
        index = location*location.new_tensor([r,a,e]) - .5
        low = index.floor().long()
        fraction = index-low
        out = volume.new_zeros(channels)
        for step in product((0,1), repeat=3):
            rr,aa,ee = (low + low.new_tensor(step)).tolist()
            if 0 <= rr < r and 0 <= aa < a and 0 <= ee < e:
                weight = torch.stack([fraction[d] if step[d] else 1-fraction[d]
                                      for d in range(3)]).prod()
                out = out + volume[aa,rr,ee]*weight
        rows.append(out)
    return torch.stack(rows)


class ModelIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.model = build()
        cls.model.init_weights()  # Actual parent initialization, not attention alone.

    def test_position_encoding_cells_and_gradients(self):
        encoding = LearnedPositionalEncoding3D(num_feats=2, row_num_embed=8,
                                               col_num_embed=8, z_num_embed=8)
        with torch.no_grad():
            for offset, table in enumerate((encoding.col_embed, encoding.row_embed, encoding.z_embed)):
                table.weight.copy_(torch.arange(16).reshape(8,2)+100*offset)
        out = encoding(torch.zeros(2,3,5,2))
        self.assertEqual(out.shape, (2,6,3,5,2))
        for y,x,z in product(range(3),range(5),range(2)):
            expected = torch.cat((encoding.col_embed.weight[x],
                                  encoding.row_embed.weight[y], encoding.z_embed.weight[z]))
            torch.testing.assert_close(out[1,:,y,x,z], expected)
            torch.testing.assert_close(out.flatten(2)[0,:,(y*5+x)*2+z], expected)
        out.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in encoding.parameters()))
        PROBES['positional_encoding'] = dict(shape=list(out.shape), checked_cells=30,
                                             flatten_matches_feature_order=True)

    def test_parent_initialization_and_portable_constants(self):
        rows=[]
        for name,module in self.model.named_modules():
            if hasattr(module,'sampling_offsets') and hasattr(module,'attention_weights'):
                for layer in (module.sampling_offsets, module.attention_weights):
                    self.assertEqual(int(torch.count_nonzero(layer.weight)),0)
                self.assertGreater(int(torch.count_nonzero(module.sampling_offsets.bias)),0)
                rows.append(name)
        self.assertEqual(len(rows),2)
        keys=self.model.state_dict().keys()
        for name in ('azimuth_indices','elevation_indices','range_indices'):
            self.assertEqual(getattr(self.model,name).device.type,'cpu')
            self.assertNotIn(name,keys)
        self.model.load_state_dict(self.model.state_dict(),strict=True)
        PROBES['initialization'] = dict(attention_layers=rows, zero_offset_logit_weights=True,
            constants_nonpersistent=True, cpu_constructor_without_patching=True)

    def test_3d_fallback_against_independent_oracle_and_gradients(self):
        torch.manual_seed(7)
        shapes=torch.tensor([[3,5,2],[2,3,2]])
        sizes=[int(s.prod()) for s in shapes]
        value=torch.randn(2,sum(sizes),2,3,dtype=torch.float64,requires_grad=True)
        locations=(torch.rand(2,4,2,2,3,3,dtype=torch.float64)*1.2-.1).requires_grad_()
        weights=torch.rand(*locations.shape[:-1],dtype=torch.float64,requires_grad=True)
        out=multi_scale_deformable_attn_3d_pytorch(value,shapes,locations,weights)
        batches=[]
        for b in range(2):
            heads=[]
            for head in range(2):
                total=torch.zeros(4,3,dtype=value.dtype)
                for level,(part,shape) in enumerate(zip(value[b].split(sizes),shapes)):
                    volume=part[:,head].reshape(*shape.tolist(),3)
                    sampled=sample_oracle(volume,locations[b,:,head,level].reshape(-1,3))
                    total=total+(sampled.reshape(4,3,3)*weights[b,:,head,level,:,None]).sum(1)
                heads.append(total)
            batches.append(torch.cat(heads,-1))
        expected=torch.stack(batches)
        torch.testing.assert_close(out,expected,rtol=1e-12,atol=1e-12)
        ga=torch.autograd.grad(out.square().sum(),(value,locations,weights),retain_graph=True)
        gb=torch.autograd.grad(expected.square().sum(),(value,locations,weights))
        for actual,reference in zip(ga,gb):
            torch.testing.assert_close(actual,reference,rtol=1e-10,atol=1e-10)
        PROBES['cpu_3d_sampler'] = dict(batches=2, heads=2, levels=2,
            forward_max_error=float((out-expected).abs().max()),
            gradient_max_errors=[float((a-b).abs().max()) for a,b in zip(ga,gb)])

    def test_self_attention_batch_independence_and_residual_switch(self):
        torch.manual_seed(17)
        module=DeformSelfAttention3DCustom(embed_dims=12,num_heads=3,num_levels=1,
                                          num_points=2,dropout=0,batch_first=True)
        # Nonzero learned weights expose queue conditioning bugs hidden at init.
        nn.init.normal_(module.sampling_offsets.weight,std=.02)
        nn.init.normal_(module.attention_weights.weight,std=.1)
        query=torch.randn(2,12,12)
        axes=[(torch.arange(n)+.5)/n for n in (2,3,2)]
        grid=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1)[...,[1,0,2]].reshape(1,12,1,3)
        refs=grid.expand(4,-1,-1,-1).clone()
        together=module(query,reference_points=refs,spatial_shapes=[2,3,2])
        separately=torch.cat([module(query[b:b+1],reference_points=refs[:2],
                                    spatial_shapes=[2,3,2]) for b in range(2)])
        torch.testing.assert_close(together,separately,rtol=1e-5,atol=1e-6)
        without=module(query,reference_points=refs,spatial_shapes=[2,3,2],residual=False)
        torch.testing.assert_close(together-without,query)
        PROBES['attention_batch_independence'] = dict(batch=2,
            max_error=float((together-separately).abs().max()), active_whole_model_batch=1)

    def test_losses_edge_cases_and_real_head(self):
        cases={'all_ignore':torch.full((1,2,3,2),255),
               'all_free':torch.zeros(1,2,3,2,dtype=torch.long),
               'all_background':torch.ones(1,2,3,2,dtype=torch.long),
               'all_foreground':torch.full((1,2,3,2),2),
               'mixed':torch.arange(12).remainder(3).reshape(1,2,3,2),
               'one_valid':torch.full((1,2,3,2),255)}
        cases['one_valid'][0,0,0,0]=1
        record={}
        for name,target in cases.items():
            logits=torch.randn(1,3,2,3,2,requires_grad=True)
            values=self.model.pts_bbox_head.loss_voxel(logits,target,'probe')
            self.assertEqual(len(values),4)
            for value in values.values():
                self.assertEqual(value.ndim,0)
                self.assertTrue(torch.isfinite(value))
            sum(values.values()).backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            if name=='all_ignore':
                self.assertEqual(float(logits.grad.abs().sum()),0)
                self.assertTrue(all(float(v)==0 for v in values.values()))
            else:
                self.assertGreater(float(logits.grad.abs().sum()),0)
            record[name]={key:float(value.detach()) for key,value in values.items()}
        PROBES['loss_edges']=record

    def test_ignore_labels_excluded_from_all_metric_ranges(self):
        target=torch.zeros(1,128,128,14,dtype=torch.long)
        target[0,0,50,0]=255
        target[0,1,51,0]=2
        copy_target=target.clone()
        logits=torch.nn.functional.one_hot(target.clamp_max(2),3).permute(0,4,1,2,3).float()
        for kind in ('SC','SSC'):
            whole,r1,r2=self.model.evaluation_semantic(logits,target,kind)
            self.assertEqual(int(whole.sum()),128*128*14-1)
            self.assertEqual(int(r1.sum()),64*64*14-1)
            self.assertEqual(int(r2.sum()),32*32*14-1)
            self.assertEqual(int(whole.trace()),int(whole.sum()))
            self.assertEqual(int(r2.trace()),int(r2.sum()))
        self.assertTrue(torch.equal(target,copy_target))
        PROBES['metric_ignore_contract']=dict(global_and_near_ranges_exclude_255=True,
                                              input_gt_unchanged=True)

    def test_occupancy_loader_preserves_ignored_sparse_labels(self):
        raw=np.array([[0,0,0,0],[1,0,0,1],[2,0,0,3],[0,1,0,255]],dtype=np.int64)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'gt.npy'
            np.save(path,raw)
            loader=LoadOccupancy(grid_size=[3,5,2],pc_range=[0,0,0,3,5,2])
            target=loader(dict(occ_path=str(path)))['gt_occ']
            self.assertEqual(int(target[0,0,0]),255)
            self.assertEqual(int(target[1,0,0]),1)
            self.assertEqual(int(target[2,0,0]),2)
            self.assertEqual(int(target[0,1,0]),255)
            self.assertTrue(np.array_equal(np.load(path),raw))
        PROBES['gt_loader_ignore_contract']=dict(raw_0_and_255_preserved_as_ignore=True,
                                                raw_gt_file_unchanged=True)

    def test_real_learned_post_sparse_forward_backward_optimizer_step(self):
        """Learned modules stay real; sparse output is explicitly synthetic."""
        model=self.model
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(23)
        spatial=(3,5,2)
        cart=(16,12,8)
        field=torch.randn(1,model.embed_dims,*spatial,requires_grad=True)
        sphere_pos=model.positional_encoding(torch.zeros(1,*spatial))
        sphere=model.self_transformer.diffuse_vox_features(field.flatten(2),*spatial[:2],
            ref_3d=np.zeros((np.prod(spatial),3),np.float32),spatial_shapes=list(spatial),bev_pos=sphere_pos)
        sphere=sphere.transpose(1,2).reshape_as(field)
        refs=torch.stack(torch.meshgrid(*[torch.linspace(.15,.85,n) for n in cart],
                                       indexing='ij'),-1).reshape(-1,3)
        query=model.cart_voxel_emb.weight[:np.prod(cart)].T.unsqueeze(0)
        cart_pos=model.positional_encoding(torch.zeros(1,*cart))
        mapped=model.transformer.get_cart_features(query,sphere,*cart[:2],
            ref_3d=refs.numpy(),sph_ref_3d=refs,spatial_shapes=list(spatial),
            cart_spatial_shape=list(cart),bev_pos=cart_pos,spherical_pos=sphere_pos)
        mapped=mapped.transpose(1,2).reshape(1,model.embed_dims,*cart)
        mapped=mapped+model.project_spherical_residual(sphere,refs,cart)
        levels=model.occ_encoder(mapped)
        target=torch.randint(0,3,(1,*levels[0].shape[2:]))
        # Execute the original forward_train loss normalization and _parse_losses.
        # Only the CUDA-dependent extract_feat boundary is supplied by this graph.
        had_extract='extract_feat' in model.__dict__
        previous=model.__dict__.get('extract_feat')
        model.extract_feat=lambda **kwargs:(levels,None,None,None)
        try:
            losses=model.forward_train(sparse_radar={},gt_occ=target,img_metas=[{}])
        finally:
            if had_extract:model.extract_feat=previous
            else:del model.extract_feat
        loss,logs=model._parse_losses(losses)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss),4,places=5)
        loss.backward()
        grads={name:p.grad for name,p in model.named_parameters() if p.grad is not None}
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads.values()))
        self.assertTrue(torch.isfinite(field.grad).all())
        self.assertGreater(float(field.grad.abs().sum()),0)
        groups=('self_transformer','transformer','positional_encoding','occ_encoder_backbone',
                'occ_encoder_neck','pts_bbox_head')
        for prefix in groups:
            self.assertGreater(sum(float(g.abs().sum()) for n,g in grads.items() if n.startswith(prefix)),0,prefix)
        parameter=model.pts_bbox_head.occ_pred_conv[-1].weight
        before=parameter.detach().clone()
        # Cross query controls offsets/logits only. Their zero initial matrices
        # intentionally delay embedding gradients until the first update.
        self.assertEqual(float(model.cart_voxel_emb.weight.grad.abs().sum()),0)
        offset=model.transformer.encoder.layers[0].attentions[0].sampling_offsets.weight
        self.assertGreater(float(offset.grad.abs().sum()),0)
        torch.optim.SGD(model.parameters(),lr=.001).step()
        self.assertFalse(torch.equal(before,parameter))
        model.eval()
        with torch.no_grad():
            prediction=model.pts_bbox_head(voxel_feats=[v.detach() for v in levels],img_metas=[{}])
        logits=prediction['output_voxels'][0]
        self.assertTrue(torch.isfinite(logits).all())
        PROBES['learned_subgraph']=dict(synthetic_sparse_field=list(field.shape),
            synthetic_interior_references=True, cartesian_shape=list(cart),
            output_shape=list(logits.shape), raw_losses={k:float(v) for k,v in losses.items() if k.startswith('raw_')},
            normalized_total=float(loss.detach()), parameter_tensors_with_gradients=len(grads),
            nonzero_gradient_groups=list(groups), finite_gradients=True, head_sgd_step_changed_weight=True,
            sparse_encoder_executed=False, cuda_kernel_executed=False, real_data_accuracy_measured=False)
        model.zero_grad(set_to_none=True)
        later=model.transformer.get_cart_features(
            model.cart_voxel_emb.weight[:np.prod(cart)].T.unsqueeze(0),sphere.detach(),*cart[:2],
            ref_3d=refs.numpy(),sph_ref_3d=refs,spatial_shapes=list(spatial),
            cart_spatial_shape=list(cart),bev_pos=model.positional_encoding(torch.zeros(1,*cart)),
            spherical_pos=sphere_pos.detach())
        later[...,0].square().mean().backward()
        self.assertGreater(float(model.cart_voxel_emb.weight.grad.abs().sum()),0)
        PROBES['learned_subgraph']['cart_query_nonzero_gradient_after_first_update']=True
        model.zero_grad(set_to_none=True)

    def test_fpn_checkpoint_matches_plain_gradients_and_bn_state(self):
        for reentrant in (False,True):
            with self.subTest(reentrant=reentrant):
                self._check_fpn_checkpoint(reentrant)

    def _check_fpn_checkpoint(self,reentrant):
        torch.manual_seed(35)
        checkpointed=copy.deepcopy(self.model.occ_encoder_neck).train()
        checkpointed.checkpoint_use_reentrant=reentrant
        plain=copy.deepcopy(checkpointed)
        plain.with_cp=False
        channels=checkpointed.in_channels
        shapes=[(8,8,4),(4,4,2),(2,2,1),(1,1,1)]
        inputs=[torch.randn(2,c,*s,requires_grad=True) for c,s in zip(channels,shapes)]
        cloned=[x.detach().clone().requires_grad_() for x in inputs]
        actual=checkpointed(inputs)
        reference=plain(cloned)
        for a,b in zip(actual,reference):torch.testing.assert_close(a,b)
        sum(x.square().mean() for x in actual).backward()
        sum(x.square().mean() for x in reference).backward()
        for a,b in zip(inputs,cloned):torch.testing.assert_close(a.grad,b.grad)
        for (name,a),(other,b) in zip(checkpointed.named_parameters(),plain.named_parameters()):
            self.assertEqual(name,other)
            torch.testing.assert_close(a.grad,b.grad)
        for name,buffer in checkpointed.named_buffers():
            torch.testing.assert_close(buffer,dict(plain.named_buffers())[name],msg=name)
        PROBES.setdefault('fpn_checkpoint',{})[str(reentrant)]=dict(output_and_gradient_match_plain=True,
            batchnorm_buffers_match_plain_after_backward=True,
            use_reentrant=checkpointed.checkpoint_use_reentrant)

    def test_all_sixteen_resolved_model_constructors_and_initialization(self):
        from temporal.experiments_v2.registry import experiments
        checked=[]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'synthetic_ranker.pth'
            save_ranker(path,PairRanker(),coordinate_convention='kradar_physical_v2')
            for identifier,spec in experiments().items():
                if spec['kind']!='occupancy-training':continue
                with self.subTest(identifier=identifier):
                    cfg=config(spec['method'])
                    attention=(cfg.model.get('temporal_cfg') or {}).get('attention',{})
                    selection=attention.get('selection')
                    substituted=bool(selection and selection['method']=='learned')
                    if substituted:selection['checkpoint']=str(path)
                    model=build(cfg=cfg)
                    model.init_weights()
                    self.assertEqual(model.pts_bbox_head.__class__.__module__,
                        'projects.occ_plugin.occupancy.dense_heads.occ_head_fp16')
                    self.assertFalse(model.pts_bbox_head.fp16)
                    self.assertEqual(model.pts_bbox_head.out_channel,3)
                    for module in model.modules():
                        if isinstance(module,DeformSelfAttention3DCustom) or module.__class__.__name__=='DeformCrossAttention3DCustom':
                            self.assertEqual(int(torch.count_nonzero(module.sampling_offsets.weight)),0)
                            self.assertEqual(int(torch.count_nonzero(module.attention_weights.weight)),0)
                    checked.append(dict(id=identifier,type=model.__class__.__name__,
                        active_head=model.pts_bbox_head.__class__.__module__,fp16=False,
                        temporary_synthetic_ranker=substituted))
                    del model
                    gc.collect()
        self.assertEqual(len(checked),16)
        PROBES['model_constructors']=checked

    def test_mask37_then_actual_range_attention(self):
        cfg=config('knn_mask37')
        self.assertTrue(cfg.model.pts_middle_encoder.allow_zero_power)
        plugin=build_temporal_plugin(cfg.model.temporal_cfg)
        values=torch.rand(1,4,6,8)*100+1
        values[:,:,:,3:6]=3
        original=values.clone()
        coords=torch.tensor([[r,53+a,18] for r in (1,50) for a in (-1,0,1)])
        coords=coords[None,None].expand(1,4,-1,-1).clone()
        poses=torch.eye(4)[None,None].expand(1,4,-1,-1).clone()
        out,_=plugin(values,coords,poses=poses,valid_mask=torch.ones(1,4,dtype=torch.bool),
                     frame_offsets=torch.tensor([[-3,-2,-1,0]]))
        self.assertTrue(torch.equal(values,original))
        self.assertTrue(torch.equal(out[0,:3,[0,1,2,6,7]],torch.zeros(3,5)))
        module=MultiLayerRangeAttention(num_ranges=2,K=3,allow_zero_power=True)
        # Leaf input so the log-power gradient is observable independently.
        out=out[0].detach().requires_grad_()
        encoded=module(out,torch.zeros(6,dtype=torch.long),torch.zeros(6,dtype=torch.long))
        self.assertTrue(torch.isfinite(encoded).all())
        encoded.square().mean().backward()
        self.assertTrue(torch.isfinite(out.grad).all())
        PROBES['mask37_range_attention']=dict(resolved_allow_zero_power=True,
            near_power_zero=True, source_descriptors_unchanged=True, forward_and_backward_finite=True)

    def test_loaded_frozen_ranker_survives_full_model_initialization(self):
        torch.manual_seed(21)
        ranker=PairRanker()
        for p in ranker.parameters():nn.init.uniform_(p,-.1,.1)
        expected=copy.deepcopy(ranker.state_dict())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'synthetic_ranker.pth'
            save_ranker(path,ranker,coordinate_convention='kradar_physical_v2')
            cfg=config('learned_knn')
            cfg.model.temporal_cfg.attention.selection.checkpoint=str(path)
            model=build(cfg=cfg)
            model.init_weights()
            loaded=[m for m in model.modules() if isinstance(m,PairRanker)]
            self.assertEqual(len(loaded),1)
            model.train()
            self.assertFalse(loaded[0].training)
            self.assertTrue(all(not p.requires_grad for p in loaded[0].parameters()))
            for key,tensor in loaded[0].state_dict().items():
                self.assertTrue(torch.equal(tensor,expected[key]),key)
        PROBES['frozen_ranker']=dict(synthetic_checkpoint=True,
            exact_weights_preserved_after_parent_init=True, frozen_in_model_train_mode=True)

    def test_actual_dataset_causality_split_and_gt_samples(self):
        cfg=config('knn')
        paths={}
        record={}
        for split in ('train','val'):
            options=copy.deepcopy(cfg.data[split])
            options.pop('samples_per_gpu',None)
            options.update(pipeline=[],test_mode=True)
            dataset=custom_build_dataset(options)
            paths[split]=set()
            valid_histories=0
            for index in range(len(dataset)):
                info=dataset.get_temporal_info(index)
                order=np.asarray(info['orders_in_scene'])
                valid=np.asarray(info['valid_mask'])
                self.assertTrue(valid[-1])
                self.assertTrue((order[:-1][valid[:-1]]<order[-1]).all())
                self.assertTrue((np.asarray(info['frame_offsets'])[:-1][valid[:-1]]<0).all())
                times=np.asarray(info['timestamps'],float)
                self.assertTrue((times[:-1][valid[:-1]]<times[-1]).all())
                self.assertTrue(np.isfinite(info['ego_poses']).all())
                paths[split].update(info['radar_paths'])
                valid_histories+=int(valid[:-1].sum())
            label_counts={}
            for index in np.linspace(0,len(dataset)-1,13,dtype=int):
                raw=np.load(dataset.data_infos[index]['occ_path'],mmap_mode='r')
                labels,counts=np.unique(raw[:,-1],return_counts=True)
                for label,count in zip(labels,counts):
                    label_counts[str(int(label))]=label_counts.get(str(int(label)),0)+int(count)
            record[split]=dict(frames=len(dataset),valid_history_entries=valid_histories,
                sampled_gt_files=13,sampled_raw_label_counts=label_counts)
        self.assertFalse(paths['train'] & paths['val'])
        PROBES['dataset']=dict(**record, train_val_radar_paths_disjoint=True,
            valid_histories_strictly_past=True, sampled_gt_is_not_exhaustive=True)


if __name__=='__main__':
    unittest.main()
