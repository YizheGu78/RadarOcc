#!/usr/bin/env python3
"""Create deterministic 20% contiguous-per-scene K-Radar temporal PKL subsets.

Only TRAIN and VAL are subsetted. Never change the official TEST set.
The source PKL is a NuScenes-style dict containing an infos list; all
other top-level fields and per-frame records are preserved verbatim.

This script retains *every scene*, drawing a single consecutive 20% window
from each scene. TemporalKRadarDataset reconstructs [t-3,t-2,t-1,t]
using scene_id/order_in_scene: scattered frame subsampling would silently
create false cross-gap historical neighbors.
"""

import argparse
import json
import os
import pickle
import random
import tempfile
from collections import defaultdict
from pathlib import Path


DEFAULT_TRAIN = Path('data/annotations/kradar_dict_train_official_temporal_doppler8.pkl')
DEFAULT_VAL = Path('data/annotations/kradar_dict_val_official_temporal_doppler8.pkl')
FRACTION = 0.20


def get_scene_id(info):
    if 'scene_id' in info:
        return str(info['scene_id'])
    if 'scene_token' in info:
        return str(info['scene_token'])
    raise ValueError('Missing scene_id and scene_token in temporal info')


def load_and_validate(path):
    with path.open('rb') as handle:
        data = pickle.load(handle)  # Only load trusted local annotation PKLs.
    if not isinstance(data, dict) or not isinstance(data.get('infos'), list):
        raise ValueError(f'{path}: expected a NuScenes-style dict with infos list')
    if not data['infos']:
        raise ValueError(f'{path}: infos is empty')

    groups = defaultdict(list)
    seen = set()
    gaps = 0
    for index, info in enumerate(data['infos']):
        if not isinstance(info, dict):
            raise ValueError(f'{path}: infos[{index}] is not a dict')
        scene = get_scene_id(info)
        if 'order_in_scene' not in info or 'ego_pose' not in info:
            raise ValueError(f'{path}: infos[{index}] lacks order_in_scene or ego_pose')
        if not ('radar_path' in info or 'sparse_radar_path' in info):
            raise ValueError(f'{path}: infos[{index}] has no radar path')
        order = int(info['order_in_scene'])
        if order != info['order_in_scene']:
            raise ValueError(f'{path}: non-integer order_in_scene at infos[{index}]')
        key = (scene, order)
        if key in seen:
            raise ValueError(f'{path}: duplicate (scene, order)={key}')
        seen.add(key)
        groups[scene].append(order)

    for orders in groups.values():
        ordered = sorted(orders)
        gaps += sum(b != a + 1 for a, b in zip(ordered, ordered[1:]))
    return data, groups, gaps


def choose_contiguous_windows(groups, fraction=FRACTION, seed=42, frame_nums=4):
    """Select one seeded, contiguous window from *every* scene.

    Each window contains round(fraction * scene_size) samples, with at least
    frame_nums samples where available. A valid window's order_in_scene values
    must be consecutive integers, so TemporalKRadarDataset never bridges gaps.
    """
    rng = random.Random(seed)
    windows = {}
    for scene in sorted(groups):
        orders = sorted(groups[scene])
        count = min(len(orders), max(frame_nums, round(len(orders) * fraction)))
        valid_starts = [
            i for i in range(len(orders) - count + 1)
            if orders[i + count - 1] - orders[i] == count - 1
        ]
        if not valid_starts:
            raise ValueError(
                f'Scene {scene}: cannot find {count} consecutive frames out of '
                f'{len(orders)} source records. Check order_in_scene gaps; '
                'refusing to create a false temporal sequence.'
            )
        start_idx = rng.choice(valid_starts)
        windows[scene] = (orders[start_idx], orders[start_idx + count - 1])
    return windows


def write_atomic(path, contents, binary=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=str(path.parent))
    try:
        mode = 'wb' if binary else 'w'
        with os.fdopen(fd, mode, **({} if binary else {'encoding': 'utf-8'})) as handle:
            if binary:
                pickle.dump(contents, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                json.dump(contents, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write('\n')
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.remove(name)


def plan_subset(input_path, output_dir, seed):
    source, groups, source_gaps = load_and_validate(input_path)
    windows = choose_contiguous_windows(groups, seed=seed)
    retained = [
        info for info in source['infos']
        if windows[get_scene_id(info)][0] <= int(info['order_in_scene'])
        <= windows[get_scene_id(info)][1]
    ]
    expected = sum(end - start + 1 for start, end in windows.values())
    if len(retained) != expected:
        raise AssertionError('Selected window size does not match retained infos')
    output = dict(source)
    output['infos'] = retained
    output_path = output_dir / (input_path.stem + '_20pct.pkl')
    manifest_path = output_path.with_suffix('.selection.json')
    per_scene = {
        scene: {
            'source_frames': len(groups[scene]),
            'selected_frames': end - start + 1,
            'start_order_in_scene': start,
            'end_order_in_scene': end,
        }
        for scene, (start, end) in sorted(windows.items())
    }
    report = {
        'source': str(input_path),
        'subset': str(output_path),
        'fraction_requested': FRACTION,
        'fraction_actual': len(retained) / len(source['infos']),
        'seed': seed,
        'selection_unit': 'one_contiguous_window_per_scene',
        'total_frames': len(source['infos']),
        'selected_frames': len(retained),
        'total_scenes': len(groups),
        'selected_scenes': sorted(windows),
        'scene_windows': per_scene,
        'source_nonconsecutive_order_pairs': source_gaps,
        'note': ('Every source scene is retained with one contiguous window. '
                 'Original info records and metadata remain unchanged. '
                 'Window starts are reproducible for the supplied seed.'),
    }
    return output_path, manifest_path, output, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-pkl', type=Path, default=DEFAULT_TRAIN)
    parser.add_argument('--val-pkl', type=Path, default=DEFAULT_VAL)
    parser.add_argument('--output-dir', type=Path, default=Path('data/annotations'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true', help='Report selection without writing files')
    parser.add_argument('--overwrite', action='store_true', help='Allow replacing previously generated 20% outputs')
    args = parser.parse_args(argv)

    paths = [args.train_pkl, args.val_pkl]
    if paths[0].resolve() == paths[1].resolve():
        parser.error('Train and Val source PKLs must be different files')
    for path in paths:
        if not path.is_file():
            parser.error(f'Input PKL not found: {path}')

    plans = [plan_subset(path, args.output_dir, args.seed + i)
             for i, path in enumerate(paths)]
    outputs = [p for out_path, manifest, _, _ in plans for p in (out_path, manifest)]
    if len({p.resolve() for p in outputs + paths}) != len(outputs) + len(paths):
        parser.error('Output must not overlap source files or other outputs')
    if not args.dry_run and not args.overwrite:
        existing = [p for p in outputs if p.exists()]
        if existing:
            parser.error('Refusing to overwrite: ' + ', '.join(map(str, existing))
                         + ' (pass --overwrite to replace generated outputs)')

    for out_path, manifest, data, report in plans:
        print(f"{out_path}: {report['selected_frames']}/{report['total_frames']} frames "
              f"({report['fraction_actual']:.2%}), "
              f"{len(report['selected_scenes'])}/{report['total_scenes']} scenes, "
              'one consecutive window each')
        if abs(report['fraction_actual'] - FRACTION) > 0.02:
            print('  NOTE: rounding and minimum T-frame windows change the '
                  'exact frame fraction; inspect the manifest.')
        if report['source_nonconsecutive_order_pairs']:
            print(f"  WARNING: source has {report['source_nonconsecutive_order_pairs']} "
                  'nonconsecutive within-scene order pairs; selected windows remain continuous.')
        if not args.dry_run:
            write_atomic(out_path, data, binary=True)
            write_atomic(manifest, report)
            print(f'  Manifest: {manifest}')
    if args.dry_run:
        print('Dry run: no files created.')


if __name__ == '__main__':
    main()
