#!/usr/bin/env python3
"""Batch-1 RadarOcc end-of-occupancy-head wall-clock latency on real validation frames.

Includes model invocation, CPU/GPU tensor scatter, P1 preparation, online KNN/Ranker,
attention, encoder and occupancy head. Excludes disk data loading and GT evaluation.
A forward hook deliberately stops the model at the end of the occupancy head.
"""
import os
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import argparse
import copy
import importlib
import json
import time
import sys
from pathlib import Path

# Expected location: <RadarOcc>/tools/temporal/benchmark_radarocc_latency.py
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model
from projects.occ_plugin.datasets.builder import custom_build_dataset, build_dataloader


class OccupancyHeadComplete(Exception):
    """Stop after occupancy prediction but before GT evaluation."""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--ann-file', required=True,
                   help='Use the SAME validation annotation PKL for all three models.')
    p.add_argument('--ranker-checkpoint', default=None)
    p.add_argument('--cache-dir', default=None)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--samples', type=int, default=40, help='Measured valid-history frames')
    p.add_argument('--include-incomplete-history', action='store_true')
    p.add_argument('--output', default=None, help='Optional JSON output')
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error('CUDA GPU is required for an RTX 5060 Ti latency benchmark.')
    if args.warmup < 0 or args.samples < 1:
        p.error('Need nonnegative warmup and positive samples')
    for path in (args.config, args.checkpoint, args.ann_file):
        if not Path(path).is_file():
            p.error(f'Not found: {path}')

    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get('plugin', False):
        mod = cfg.get('plugin_dir', '').rstrip('/').replace('/', '.')
        if not mod:
            mod = str(Path(args.config).parent).replace('/', '.')
        importlib.import_module(mod)

    cfg.model['pretrained'] = None
    cfg.model['train_cfg'] = None
    dataset_cfg = copy.deepcopy(cfg.data.val)
    dataset_cfg['ann_file'] = args.ann_file
    dataset_cfg['test_mode'] = True
    dataset_cfg.pop('samples_per_gpu', None)
    dataset = custom_build_dataset(dataset_cfg)

    # Skip the first three frames per scene in temporal experiments so every
    # comparison has a full 3-frame history. A single-frame model can use this
    # same scene/frame cohort since the annotation PKLs are identical.
    valid = getattr(dataset, 'sequence_valid_masks', None)
    if valid is not None and not args.include_incomplete_history:
        eligible = np.flatnonzero(np.asarray(valid).all(axis=1)).tolist()
    else:
        eligible = list(range(len(dataset)))
        # A non-temporal baseline has no sequence mask. Match the temporal
        # cohort by excluding each scene's first three available frames.
        if valid is None and not args.include_incomplete_history:
            infos = getattr(dataset, 'data_infos', None)
            if infos is None:
                p.error('Non-temporal dataset lacks data_infos; cannot guarantee matched cohorts.')
            scenes = {}
            for idx, info in enumerate(infos):
                scene = info.get('scene_id', info.get('scene_token'))
                if scene is None or 'order_in_scene' not in info:
                    p.error('Cannot identify temporal cohort without scene_id/order_in_scene.')
                scenes.setdefault(str(scene), []).append((int(info['order_in_scene']), idx))
            excluded = set()
            for entries in scenes.values():
                entries.sort()
                excluded.update(idx for _, idx in entries[:3])
            eligible = [idx for idx in eligible if idx not in excluded]
    wanted = args.warmup + args.samples
    if len(eligible) < wanted:
        p.error(f'Only {len(eligible)} eligible frames; need {wanted}. '
                'Decrease --samples/--warmup or add --include-incomplete-history.')

    # Do not use a distributed launcher. This is intentionally one GPU,
    # one sample per invocation, deterministic sequential order.
    from torch.utils.data import Subset
    chosen = eligible[:wanted]
    loader = build_dataloader(Subset(dataset, chosen), samples_per_gpu=1,
                              workers_per_gpu=0, dist=False, shuffle=False)

    # Select online or precomputed selector. The default learned config runs
    # online when cache_dir is not set. Never read/write cache during online run.
    temporal_cfg = cfg.model.get('temporal_cfg') or {}
    selector_cfg = (temporal_cfg.get('attention') or {}).get('selection', None)
    selector_mode = 'n/a'
    if selector_cfg is not None:
        if args.ranker_checkpoint is not None:
            selector_cfg['checkpoint'] = args.ranker_checkpoint
        if selector_cfg.get('method') == 'learned' and not Path(selector_cfg['checkpoint']).is_file():
            p.error(f'Ranker checkpoint missing: {selector_cfg.get("checkpoint")}')
        if args.cache_dir:
            if not Path(args.cache_dir).is_dir():
                p.error(f'Cache dir not found: {args.cache_dir}')
            selector_cfg['cache_dir'] = args.cache_dir
            selector_cfg['cache_mode'] = 'read'
            selector_mode = 'cache-read'
        else:
            selector_cfg.pop('cache_dir', None)
            selector_cfg.pop('cache_mode', None)
            selector_mode = 'online'
    elif args.cache_dir or args.ranker_checkpoint:
        p.error('This model config does not contain a learned selector')

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda()
    model.eval()
    parallel = MMDataParallel(model, device_ids=[0])

    # The stock RadarOcc_small.simple_test computes GT SC/SSC metrics AFTER
    # pts_bbox_head. We exclude those by stopping exactly at the head output.
    def stop_after_head(_module, _inputs, _output):
        raise OccupancyHeadComplete()

    handle = model.pts_bbox_head.register_forward_hook(stop_after_head)
    runtimes = []
    try:
        with torch.no_grad():
            for step, batch in enumerate(loader):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                try:
                    parallel(return_loss=False, rescale=True, **batch)
                except OccupancyHeadComplete:
                    pass
                else:
                    raise RuntimeError('Occupancy Head hook did not fire. Benchmark is invalid.')
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if step >= args.warmup:
                    runtimes.append(elapsed_ms)
                print(f'[{step + 1}/{wanted}] '
                      f'{"warmup" if step < args.warmup else "measure"} '
                      f'{elapsed_ms:.1f} ms', flush=True)
    finally:
        handle.remove()

    a = np.asarray(runtimes, dtype=np.float64)
    report = dict(
        config=args.config, checkpoint=args.checkpoint, ann_file=args.ann_file,
        ranker_checkpoint=args.ranker_checkpoint, selector=selector_mode,
        device=torch.cuda.get_device_name(0), batch_size=1,
        history_incomplete_skipped=not args.include_incomplete_history,
        warmup=args.warmup, samples=len(a),
        mean_ms=float(a.mean()), median_ms=float(np.median(a)),
        p95_ms=float(np.percentile(a, 95)), std_ms=float(a.std()),
        fps_from_mean=1000 / float(a.mean()),
        definition='Whole model forward to occupancy head, including host/device transfer and online P1 matching; excludes disk loading and GT metric computation.',
    )
    print('\nRESULT\n' + json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(f'Saved: {out}')


if __name__ == '__main__':
    main()
