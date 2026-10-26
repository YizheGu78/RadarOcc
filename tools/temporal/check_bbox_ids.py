#!/usr/bin/env python3
"""Read-only bbox/track-ID diagnostics; no torch or CUDA required.

Relative input/output paths are resolved from the caller's working directory.
Only an explicitly requested JSON report is written; annotations are never changed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle

import numpy as np


def inspect_frame(info, bbox_root):
    scene = str(info.get('scene_id', info.get('scene_token')))
    index = int(info['occ_frame_idx'])
    root = bbox_root / scene / 'bbox'
    paths = {name: root / f'{prefix}{index}.npy' for name, prefix in
             [('boxes', 'bbox'), ('ids', 'bbox_token'), ('categories', 'object_category')]}
    record = dict(scene=scene, order=int(info['order_in_scene']), occ_frame_idx=index,
                  radar_frame_idx=info.get('radar_frame_idx'),
                  radar_path=str(info.get('sparse_radar_path') or info.get('radar_path', '')),
                  paths={k: str(v.resolve()) for k, v in paths.items()},
                  issues=[], duplicates={}, objects=[])
    arrays = {}
    for name, path in paths.items():
        try:
            arrays[name] = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            record['issues'].append(f'{name}: {error}')
    if len(arrays) != 3:
        return record
    try:
        boxes = arrays['boxes'].reshape(-1, 7)
        ids = arrays['ids'].reshape(-1).astype(str)
        categories = arrays['categories'].reshape(-1)
    except ValueError as error:
        record['issues'].append(f'invalid shape: {error}')
        return record
    record['counts'] = dict(boxes=len(boxes), ids=len(ids), categories=len(categories))
    if not len(boxes) == len(ids) == len(categories):
        record['issues'].append('count_mismatch')
    record['duplicates'] = {token: np.flatnonzero(ids == token).tolist()
                            for token, count in Counter(ids.tolist()).items() if count > 1}
    if record['duplicates']:
        record['issues'].append('duplicate_ids')
    if not np.isfinite(boxes).all() or np.any(boxes[:, 3:6] <= 0):
        record['issues'].append('invalid_box_geometry')
    # Preserve every row, even when the three array lengths differ.
    for i in range(max(len(boxes), len(ids), len(categories))):
        record['objects'].append(dict(row=i, id=str(ids[i]) if i < len(ids) else None,
            category=str(categories[i]) if i < len(categories) else None,
            box=boxes[i].tolist() if i < len(boxes) else None))
    return record


def inspect_raw(record, label_root):
    """Mirror handle_raw_label's sorted-file indexing and b[1] parsing."""
    directory = label_root / record['scene']
    files = sorted(directory.glob('*.txt'))
    index = record['occ_frame_idx']
    result = dict(directory=str(directory.resolve()), file_count=len(files),
                  mapping='sorted *.txt indexed by occ_frame_idx; verify sensor IDs in header')
    if index < 0 or index >= len(files):
        result['error'] = 'No label at this index; check root, completeness and preprocessing version'
        return result
    path = files[index]
    result['path'] = str(path.resolve())
    try:
        lines = path.read_text().splitlines()
        result['header'] = lines[0] if lines else ''
        first = result['header'].split('*')
        rows = ([first[-1]] if len(first) == 3 else []) + lines[1:]
        if len(first) > 3:
            result['header_warning'] = 'Preprocessor would print bad data. CHECK!'
        parsed = []
        for line in rows:
            fields = line.split(',')
            if len(fields) < 10:
                raise ValueError(f'Expected at least 10 comma-separated fields: {line!r}')
            parsed.append(dict(raw_id=fields[1].strip(), id=str(int(fields[1])),
                               category=fields[2].strip(), line=line))
        counts = Counter(row['id'] for row in parsed)
        result['duplicates_after_int_conversion'] = {k: v for k, v in counts.items() if v > 1}
        result['rows'] = parsed
        result['ids_match_npy'] = ([r['id'] for r in parsed] ==
                                   [r['id'] for r in record['objects']])
    except (OSError, ValueError) as error:
        result['error'] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ann-file', type=Path,
        default=Path('data/annotations/kradar_dict_train_official_temporal_doppler8.pkl'))
    parser.add_argument('--bbox-root', type=Path, default=Path('data/K-RadarOcc/train'))
    parser.add_argument('--scene', default='11')
    parser.add_argument('--order', type=int, default=228)
    parser.add_argument('--context', type=int, default=2, help='Number of adjacent split entries on each side')
    parser.add_argument('--scan-all', action='store_true', help='Scan every frame in the annotation PKL')
    parser.add_argument('--label-root', type=Path, help='Optional raw label root containing <scene>/*.txt')
    parser.add_argument('--output', type=Path, help='Optional JSON report path (must not exist)')
    args = parser.parse_args()
    if args.context < 0:
        parser.error('--context must be nonnegative')
    if args.output and args.output.exists():
        parser.error('--output already exists; choose a new report path')
    # Use only trusted local annotation PKLs, as with the training tools.
    with args.ann_file.open('rb') as stream:
        data = pickle.load(stream)
    infos = data['infos'] if isinstance(data, dict) else data
    if args.scan_all:
        selected = infos
    else:
        scene = sorted((f for f in infos if str(f.get('scene_id', f.get('scene_token'))) == args.scene),
                       key=lambda f: int(f['order_in_scene']))
        matches = [i for i, f in enumerate(scene) if int(f['order_in_scene']) == args.order]
        if len(matches) != 1:
            parser.error(f'Expected one frame at scene={args.scene}, order={args.order}; found {len(matches)}')
        pos = matches[0]
        selected = scene[max(0, pos-args.context):pos+args.context+1]
    records = []
    bad = 0
    for info in selected:
        record = inspect_frame(info, args.bbox_root)
        bad += bool(record['issues'])
        if args.scan_all and not record['issues']:
            continue
        if args.label_root:
            record['raw_label'] = inspect_raw(record, args.label_root)
        records.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
    summary = dict(checked_frames=len(selected), abnormal_frames=bad,
                   report_scope='abnormal frames only' if args.scan_all else 'target and context frames')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(dict(annotation=str(args.ann_file.resolve()), summary=summary, frames=records),
                      stream, ensure_ascii=False, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
