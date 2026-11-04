#!/usr/bin/env python3
"""One real-frame identity/gradient preflight, without an optimizer step."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from mmdet.apis import set_random_seed
import projects.occ_plugin  # register existing model and pipeline components


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--ann-file', default='data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl')
    p.add_argument('--cache', default='data/box_z_pairs/val20')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the original RadarOcc CUDA operators')
    set_random_seed(42)
    cfg = Config.fromfile('projects/configs/baselines/RadarOcc_Small_box_z_p0_20pct.py')
    cfg.model.box_z_cfg.update(mode='Z3', diagnostic_path=None)
    cfg.data.val.update(ann_file=args.ann_file, box_cache=args.cache, test_mode=True)
    dataset = build_dataset(cfg.data.val)
    possible = [i for i, info in enumerate(dataset.data_infos)
        if dataset.box_manifest['records'][f'{dataset._scene_id(info)}_{dataset._order_in_scene(info):05d}']['pairs'] > 0
        and dataset.sequence_valid_masks[i].all()]
    if not possible:
        raise RuntimeError('No frame with matched historical boxes and three valid histories')
    batch = collate([dataset[possible[0]]], samples_per_gpu=1)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    payload = load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    supplied = {k.removeprefix('module.') for k in payload['state_dict']}
    missing = [k for k in model.state_dict() if not k.startswith('box_z_fusion.')
               and k not in supplied and not k.endswith('num_batches_tracked')]
    if missing:
        raise RuntimeError(f'Initialization does not cover frozen original model: {missing[:10]}')
    model = model.cuda()
    parallel = MMDataParallel(model, device_ids=[0])
    model.eval()
    with torch.no_grad():
        model.box_z_fusion.enabled = False
        baseline = parallel(return_loss=False, rescale=True, **batch)['pred_c'].clone()
        model.box_z_fusion.enabled = True
        initial = parallel(return_loss=False, rescale=True, **batch)['pred_c']
        identity = torch.equal(initial, baseline)
        if not identity:
            raise AssertionError(f'Initial prediction differs; max delta={(initial-baseline).abs().max()}')
    model.train()
    if any(p.requires_grad for n, p in model.named_parameters() if not n.startswith('box_z_fusion.')):
        raise AssertionError('Original model parameters are not frozen')
    # At exact identity the zero output projection blocks upstream data gradients.
    # Open it slightly solely for the gradient test; no trained weights are saved.
    with torch.no_grad():
        model.box_z_fusion.output.weight.fill_(.005)
    retained = {}
    def retain(name):
        def hook(module, inputs, output):
            if isinstance(output, dict):
                output = output.get('x', output.get('output_voxels', []))
            if isinstance(output, (tuple, list)):
                output = output[0]
            if torch.is_tensor(output) and output.requires_grad:
                output.retain_grad(); retained[name] = output
        return hook
    for name in ('box_z_fusion', 'box_z_fusion.output', 'pts_middle_encoder.range_attn',
                 'pts_middle_encoder', 'occ_encoder_neck', 'pts_bbox_head'):
        model.get_submodule(name).register_forward_hook(retain(name))
    losses = parallel(return_loss=True, **batch)
    total = sum(v.mean() if torch.is_tensor(v) else sum(t.mean() for t in v)
                for k, v in losses.items() if k.startswith('loss'))
    if not torch.isfinite(total):
        raise AssertionError('Nonfinite full-model loss')
    total.backward()
    gradient = model.box_z_fusion.z_linear.weight.grad
    print('Preflight sampling:', model.box_z_fusion.diagnostics, flush=True)
    print('Preflight Linear gradient:', gradient, flush=True)
    print('Preflight chain gradients:', {k: None if v.grad is None else float(v.grad.abs().max())
                                        for k, v in retained.items()}, flush=True)
    if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().sum() > 0:
        raise AssertionError('Z3 sampling coordinates receive no finite occupancy gradient')
    report = dict(identity_prediction_exact=identity, frozen_original=True,
        frame_index=possible[0], z3_linear_gradient_l1=float(gradient.abs().sum()),
        loss=float(total.detach()), diagnostics=model.box_z_fusion.diagnostics,
        no_optimizer_step=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
