#!/usr/bin/env python3
"""Check shared random initialization and two full-network optimizer steps.

The engineering model is discarded. Formal runs rebuild it from the same seed.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
import projects.occ_plugin


def shared_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        if key.startswith('box_z_fusion.z_linear.'):
            continue
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='projects/configs/baselines/RadarOcc_Small_box_z_p0_20pct_scratch.py')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ann-file', default='data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl')
    parser.add_argument('--cache', default='data/box_z_pairs/val20')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for original RadarOcc operators')
    cfg = Config.fromfile(args.config)
    if cfg.load_from is not None or cfg.resume_from is not None or cfg.model.box_z_cfg.freeze_original:
        raise ValueError('Scratch requires no initialization checkpoint and an unfrozen original model')
    hashes, trainable = {}, {}
    for method in ('Single-frame', 'Z0', 'Z1', 'Z2', 'Z3'):
        set_random_seed(args.seed)
        config = copy.deepcopy(cfg.model)
        config.box_z_cfg.update(mode='Z1' if method == 'Single-frame' else method,
                               enabled=method != 'Single-frame', diagnostic_path=None)
        model = build_model(config, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        model.init_weights()
        if any(not p.requires_grad for p in model.parameters()):
            raise AssertionError('Unexpected frozen parameters in scratch model')
        hashes[method] = shared_hash(model)
        trainable[method] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if method != 'Z3':
            del model
            gc.collect(); torch.cuda.empty_cache()
    if len(set(hashes.values())) != 1:
        raise AssertionError(f'Shared random initialization differs: {hashes}')
    cfg.data.val.update(ann_file=args.ann_file, box_cache=args.cache, test_mode=True)
    dataset = build_dataset(cfg.data.val)
    candidates = [i for i, info in enumerate(dataset.data_infos) if dataset.sequence_valid_masks[i].all()
        and dataset.box_manifest['records'][f'{dataset._scene_id(info)}_{dataset._order_in_scene(info):05d}']['pairs'] > 0]
    if not candidates:
        raise RuntimeError('Need a frame with matched boxes and all three historical frames')
    index = candidates[0]
    batch = collate([dataset[index]], samples_per_gpu=1)
    model.cuda().eval()
    parallel = MMDataParallel(model, device_ids=[0])
    with torch.no_grad():
        model.box_z_fusion.enabled = False
        single = parallel(return_loss=False, rescale=True, **batch)['pred_c'].clone()
        model.box_z_fusion.enabled = True
        initial = parallel(return_loss=False, rescale=True, **batch)['pred_c']
        identity = torch.equal(single, initial)
        if not identity:
            raise AssertionError('Random initialization fusion is not an exact identity')
    del single, initial
    model.train()
    if not model.pts_middle_encoder.training or not model.pts_bbox_head.training:
        raise AssertionError('Original encoder/head are not in training mode')
    name = 'pts_middle_encoder.conv_input.0.weight'
    original = model.get_parameter(name).detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    steps = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        losses = parallel(return_loss=True, **batch)
        total = sum(v.mean() if torch.is_tensor(v) else sum(t.mean() for t in v)
                    for k, v in losses.items() if k.startswith('loss'))
        if not torch.isfinite(total):
            raise AssertionError('Nonfinite scratch loss')
        total.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise AssertionError('Nonfinite scratch gradient')
        gradient = model.box_z_fusion.z_linear.weight.grad
        grad_l1 = float(gradient.abs().sum()) if gradient is not None else 0.
        core_grad = model.get_parameter(name).grad
        if core_grad is None or not core_grad.abs().sum() > 0:
            raise AssertionError('Original RadarOcc receives no data gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 35.)
        optimizer.step()
        steps.append(dict(step=step+1, loss=float(total.detach()), z3_gradient_l1=grad_l1))
    if steps[-1]['z3_gradient_l1'] <= 0:
        raise AssertionError('Z3 does not receive a data gradient after the identity projection opens')
    changed = not torch.equal(original, model.get_parameter(name).detach())
    if not changed:
        raise AssertionError('Original RadarOcc weight did not update')
    report = dict(seed=args.seed, initialization='random', freeze_original=False,
        shared_initialization_hashes=hashes, trainable_parameters=trainable,
        identity_prediction_exact=identity, original_weight_updated=changed,
        frame_index=index, optimizer_steps=steps, peak_memory_bytes=torch.cuda.max_memory_allocated(),
        discarded_engineering_model=True, formal_training_started=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
