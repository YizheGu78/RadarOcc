#!/usr/bin/env python3
"""Create deterministic 20% scene-level K-Radar temporal PKL subsets.

Only TRAIN and VAL are subsetted. Never change the official TEST set.
The source PKL is a NuScenes-style dict containing an infos list; all
other top-level fields and per-frame records are preserved verbatim.

This script deliberately selects *whole scenes*. TemporalKRadarDataset
reconstructs [t-3,t-2,t-1,t] using scene_id/order_in_scene, so selecting
arbitrary individual frames would create false cross-gap history windows.
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


def choose_complete_scenes(groups, fraction=FRACTION, seed=42, trials=1024):
    """Return a deterministic scene set approximately matching target frame count.

    Randomized scene-prefix search avoids favoring numeric scene IDs. Each
    candidate uses *whole* scenes; exact 20% may be impossible for the source
    scene sizes, so the actual achieved fraction is always reported.
    """
    sizes = {scene: len(orders) for scene, orders in groups.items()}
    scenes = sorted(sizes)
    total = sum(sizes.values())
    target = total * fraction
    rng = random.Random(seed)
    best = None
    best_score = None
    for _ in range(max(1, trials)):
        rng.shuffle(scenes)
        chosen = []
        count = 0
        for scene in scenes:
            chosen.append(scene)
            count += sizes[scene]
            # Favor a closer frame count, then more scene diversity. The
            # seeded order breaks any remaining ties deterministically.
            score = (abs(count - target), -len(chosen))
            if best_score is None or score < best_score:
                best = frozenset(chosen)
                best_score = score
            if count >= target:
                break
    return best


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
    selected_scenes = choose_complete_scenes(groups, seed=seed)
    retained = [info for info in source['infos'] if get_scene_id(info) in selected_scenes]
    assert len(retained) == sum(len(groups[scene]) for scene in selected_scenes)
    output = dict(source)
    output['infos'] = retained
    output_path = output_dir / (input_path.stem + '_20pct.pkl')
    manifest_path = output_path.with_suffix('.selection.json')
    report = {
        'source': str(input_path),
        'subset': str(output_path),
        'fraction_requested': FRACTION,
        'fraction_actual': len(retained) / len(source['infos']),
        'seed': seed,
        'selection_unit': 'complete_scene',
        'total_frames': len(source['infos']),
        'selected_frames': len(retained),
        'total_scenes': len(groups),
        'selected_scenes': sorted(selected_scenes),
        'selected_scene_frame_counts': {
            scene: len(groups[scene]) for scene in sorted(selected_scenes)
        },
        'source_nonconsecutive_order_pairs': source_gaps,
        'note': ('PKL records and metadata are unchanged except infos filtering. '
                 'Any order gaps reported already exist in source; no new gaps are created.'),
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
              f"{len(report['selected_scenes'])}/{report['total_scenes']} complete scenes")
        if abs(report['fraction_actual'] - FRACTION) > 0.02:
            print('  NOTE: complete-scene selection cannot closely match 20%; '
                  'check the actual fraction and scene coverage.')
        if report['source_nonconsecutive_order_pairs']:
            print(f"  WARNING: source has {report['source_nonconsecutive_order_pairs']} "
                  'nonconsecutive within-scene order pairs; this subset does not introduce them.')
        if not args.dry_run:
            write_atomic(out_path, data, binary=True)
            write_atomic(manifest, report)
            print(f'  Manifest: {manifest}')
    if args.dry_run:
        print('Dry run: no files created.')


if __name__ == '__main__':
    main()
