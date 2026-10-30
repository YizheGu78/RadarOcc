#!/usr/bin/env python3
"""Time real T=4 P1 inference through the occupancy head, without IoU evaluation.

Default: online box search + frozen learned selector, without selection cache.
Run from the RadarOcc project root. No weights or existing caches are changed.
"""
import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import time


class PredictionReady(Exception):
    """Stop after producing occupancy logits, before CPU metric calculation."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='projects/configs/baselines/'
                        'RadarOcc_Small_5060_true_fp32_temporal_t4_box64_learned_top8_p1.py')
    parser.add_argument('--work-dir', default='work_dirs/'
                        'radarocc_small_temporal_t4_box64_learned_top8_p1_20pct')
    parser.add_argument('--checkpoint', help='Otherwise find the unique best_SSC_mean*.pth in work-dir')
    parser.add_argument('--ann-file', default='data/annotations/'
                        'kradar_dict_val_official_temporal_doppler8_20pct.pkl')
    parser.add_argument('--ranker', help='Override standalone ranker checkpoint in config')
    parser.add_argument('--cache-dir', help='Optional strict read-only cache comparison; default is online')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--start-index', type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1 or min(args.warmup, args.start_index) < 0:
        parser.error('steps must be positive; warmup and start-index must be nonnegative')
    root = Path.cwd()
    if not (root / 'projects/occ_plugin').is_dir():
        parser.error('Run from the RadarOcc project root')
    if not args.checkpoint:
        matches = sorted(Path(args.work_dir).glob('best_SSC_mean*.pth'))
        if len(matches) != 1:
            parser.error(f'Expected one best checkpoint, found {len(matches)}; pass --checkpoint explicitly')
        args.checkpoint = str(matches[0])
    if not Path(args.checkpoint).is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')
    sys.path.insert(0, str(root))
    # Only load the user's locally trained, trusted MMCV checkpoints.
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.models import build_model
    from torch.utils.data import Subset
    from projects.occ_plugin.datasets.builder import build_dataloader, custom_build_dataset
    from temporal.core.learned_topk import BoxTopKSelector

    if not torch.cuda.is_available():
        raise RuntimeError('Run in the RadarOcc CUDA environment')
    torch.cuda.set_device(0)
    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get('plugin', False):
        importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/', '.'))
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = cfg.get('cudnn_benchmark', False)
    torch.backends.cudnn.deterministic = True
    selection = cfg.model.temporal_cfg.attention.selection
    selection.cache_dir = args.cache_dir
    selection.cache_mode = 'read'
    if args.ranker:
        selection.checkpoint = args.ranker
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg')).cuda()
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    selectors = [m for m in model.modules() if isinstance(m, BoxTopKSelector)]
    if len(selectors) != 1 or selectors[0].method != 'learned':
        raise ValueError('Expected exactly one learned P1 BoxTopKSelector')
    selector = selectors[0]
    model.record_time = False

    dataset_cfg = copy.deepcopy(cfg.data.val)
    dataset_cfg.pop('samples_per_gpu', None)
    dataset_cfg.ann_file = args.ann_file
    dataset_cfg.test_mode = True
    dataset = custom_build_dataset(dataset_cfg)
    # Avoid misleading fast samples with missing histories at scene boundaries.
    indices = [i for i in range(args.start_index, len(dataset))
               if np.asarray(dataset.get_temporal_info(i)['valid_mask']).all()]
    count = args.warmup + args.steps
    if len(indices) < count:
        raise ValueError(f'Need {count} frames with all histories, only {len(indices)} available')
    # Spread measurement frames over the remaining validation sequence.
    positions = np.linspace(0, len(indices) - 1, count, dtype=int)
    indices = [indices[i] for i in positions]
    loader = build_dataloader(Subset(dataset, indices), 1, 0, dist=False, shuffle=False)
    parallel = MMDataParallel(model, device_ids=[0])
    state = {}

    def selection_start(module, inputs):
        torch.cuda.synchronize()
        state['selector_start'] = time.perf_counter()

    def selection_end(module, inputs, output):
        torch.cuda.synchronize()
        state['selection_ms'] = 1000 * (time.perf_counter() - state['selector_start'])

    def prediction_end(module, inputs, output):
        torch.cuda.synchronize()
        raise PredictionReady()

    hooks = [selector.register_forward_pre_hook(selection_start),
             selector.register_forward_hook(selection_end),
             model.pts_bbox_head.register_forward_hook(prediction_end)]
    print(json.dumps(dict(checkpoint=args.checkpoint, annotation=args.ann_file,
        selection='cached' if args.cache_dir else 'online', cache_dir=args.cache_dir,
        gpu=torch.cuda.get_device_name(0), warmup=args.warmup, steps=args.steps,
        sampled_indices=indices), ensure_ascii=False), flush=True)
    records = []
    iterator = iter(loader)
    try:
        with torch.no_grad():
            for i in range(count):
                if i == args.warmup:
                    torch.cuda.reset_peak_memory_stats()
                load_start = time.perf_counter()
                batch = next(iterator)
                torch.cuda.synchronize()
                start = time.perf_counter()
                load_ms = 1000 * (start - load_start)
                state.clear()
                try:
                    parallel(return_loss=False, rescale=True, **batch)
                except PredictionReady:
                    pass
                else:
                    raise RuntimeError('Occupancy head was not reached')
                torch.cuda.synchronize()
                model_ms = 1000 * (time.perf_counter() - start)
                if 'selection_ms' not in state:
                    raise RuntimeError('Learned selector was not executed')
                row = dict(model_ms=model_ms, selection_ms=state['selection_ms'],
                    other_model_ms=model_ms-state['selection_ms'], load_ms=load_ms,
                    load_plus_model_ms=load_ms+model_ms)
                print(('warmup' if i < args.warmup else 'measure') + ' ' +
                      json.dumps(dict(index=indices[i], **row)), flush=True)
                if i >= args.warmup:
                    records.append(row)
    finally:
        for hook in hooks:
            hook.remove()
    means = {k: float(np.mean([r[k] for r in records])) for k in records[0]}
    report = dict(means=means,
        model_median_ms=float(np.median([r['model_ms'] for r in records])),
        model_p95_ms=float(np.percentile([r['model_ms'] for r in records], 95)),
        model_fps=1000 / means['model_ms'],
        load_plus_model_fps=1000 / means['load_plus_model_ms'],
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        notes='B=1, T=4, all histories valid, no backward/optimizer/IoU; '
              'model includes input transfer through occupancy logits; '
              'selection includes CPU search and GPU scoring; synchronization at timing boundaries')
    print('SUMMARY ' + json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
