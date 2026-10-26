#!/usr/bin/env python3
"""Warm/verify full-query selection using the actual detector's P1 preparation.

Intercept immediately after the selector: no attention, occupancy forward,
backward, optimizer, or model checkpoint initialization is required.
"""
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class SelectionReady(Exception):
    pass


def main():
    from mmcv import Config, DictAction
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--checkpoint', required=True, help='Standalone frozen ranker checkpoint')
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val'])
    parser.add_argument('--max-samples', type=int, default=0, help='Per split; 0 = all')
    parser.add_argument('--verify', action='store_true', help='Compare online indices and actual attention output')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error('max-samples must be nonnegative')

    import numpy as np
    import torch
    from mmcv.parallel import MMDataParallel
    from mmdet3d.models import build_model
    from projects.occ_plugin.datasets.builder import build_dataloader, custom_build_dataset
    from temporal.core.learned_topk import BoxTopKSelector

    if not torch.cuda.is_available():
        raise RuntimeError('Build on the same CUDA environment as full-model training')
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('plugin', False):
        importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/', '.'))
    selection = cfg.model.temporal_cfg.attention.selection
    selection.update(checkpoint=args.checkpoint, cache_dir=args.cache_dir, cache_mode='readwrite')
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')).cuda()
    # Do not call init_weights/load occupancy weights: only fixed P1 preparation
    # and the already-loaded ranker execute before the intercept.
    selectors = [module for module in model.modules() if isinstance(module, BoxTopKSelector)]
    if len(selectors) != 1:
        raise ValueError('Expected exactly one frozen P1 BoxTopKSelector')
    selector = selectors[0]
    fusion = model.temporal_plugin.fusion

    def finish(module, inputs, output):
        raise SelectionReady()

    def verify_fusion(module, inputs, kwargs):
        features, xyz, offsets, frames = inputs[:4]
        extra = dict(timestamps=kwargs.get('timestamps'), point_valid_mask=kwargs.get('point_valid_mask'))
        start = time.perf_counter()
        online = selector._select_online(features, xyz, frames, **extra)
        torch.cuda.synchronize()
        online_seconds = time.perf_counter()-start
        start = time.perf_counter()
        cached = selector(features, xyz, frames, **extra)
        torch.cuda.synchronize()
        cached_seconds = time.perf_counter()-start
        for expected, actual in zip(online, cached):
            np.testing.assert_array_equal(expected, actual)
        cached_output, _ = module.forward(*inputs, **kwargs)
        # Compare a nonzero attention residual, rather than trivial identity init.
        original_forward = selector.forward
        try:
            selector.forward = lambda *a, **k: online
            online_output, _ = module.forward(*inputs, **kwargs)
        finally:
            selector.forward = original_forward
        torch.testing.assert_close(cached_output, online_output, rtol=0, atol=0)
        print(f'verify PASS indices/masks/attention; online_selection={online_seconds:.3f}s '
              f'cached_selection={cached_seconds:.3f}s (first miss includes writing)', flush=True)
        raise SelectionReady()

    if args.verify:
        # Verification only; these temporary fusion weights are never saved.
        with torch.no_grad():
            fusion.output.weight.fill_(0.01)
        hook = fusion.register_forward_pre_hook(verify_fusion, with_kwargs=True)
    else:
        hook = selector.register_forward_hook(finish)
    parallel = MMDataParallel(model, device_ids=[0])
    reports = []
    try:
        for split in args.splits:
            dataset_cfg = copy.deepcopy(cfg.data[split])
            dataset_cfg.pop('samples_per_gpu', None)
            dataset = custom_build_dataset(dataset_cfg)
            loader = build_dataloader(dataset, 1, 0, dist=False, shuffle=False)
            parallel.train(split == 'train')
            start = time.perf_counter()
            count = 0
            for index, batch in enumerate(loader):
                selector.cache_metadata = dict(split=split, dataset_index=index,
                    annotation=str(dataset_cfg.get('ann_file', '')))
                if hasattr(dataset, 'get_temporal_info'):
                    info = dataset.get_temporal_info(index)
                    selector.cache_metadata.update(scene=info['scene_id'],
                        orders=info['orders_in_scene'], radar_paths=info['radar_paths'])
                with torch.no_grad():
                    try:
                        parallel(return_loss=(split == 'train'), **batch)
                    except SelectionReady:
                        pass
                    else:
                        raise RuntimeError('Selector was not reached; refusing incomplete cache')
                count += 1
                print(f'{split} {count}/{len(dataset)} hits={selector.cache.hits} '
                      f'misses={selector.cache.misses} elapsed={time.perf_counter()-start:.1f}s', flush=True)
                if args.max_samples and count >= args.max_samples:
                    break
            reports.append(dict(split=split, processed=count, total=len(dataset),
                complete=count == len(dataset), elapsed_seconds=time.perf_counter()-start))
    finally:
        hook.remove()
    report = dict(config=args.config, checkpoint=args.checkpoint, verify=args.verify,
        cache_dir=args.cache_dir, splits=reports, hits=selector.cache.hits, misses=selector.cache.misses)
    report_path = Path(args.cache_dir)/f'report_{time.time_ns()}.json'
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Report: {report_path}', flush=True)


if __name__ == '__main__':
    main()
