#!/usr/bin/env python3
"""Scan v2.0/v2.1 raw labels directly; never modify source annotations.

No PKL, processed NPY, PyTorch or NumPy dependency. Supports the layout
<label-root>/<scene>/*.txt and records physical source line numbers (1-based).
"""
import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_ROOT = ROOT / ('data/K-Radar-RGB/K-Radar/K-Radar-official-gt/'
    'kradar_revised_label_v2_1/KRadar_revised_visibility')


def duplicate_groups(objects):
    grouped = defaultdict(list)
    for obj in objects:
        grouped[obj['id']].append(obj)
    result = {}
    for token, rows in grouped.items():
        if len(rows) < 2:
            continue
        first = rows[0]
        identical = all(o['category'] == first['category'] and
            all(abs(a-b) <= 1e-6 for a, b in zip(
                o['xyz'] + [o['yaw_degrees']] + o['raw_dimensions'],
                first['xyz'] + [first['yaw_degrees']] + first['raw_dimensions']))
            for o in rows[1:])
        flags = [o['availability'] for o in rows]
        result[token] = dict(lines=[o['line'] for o in rows],
            categories=[o['category'] for o in rows], availability=flags,
            cross_availability=len(set(flags)) > 1,
            identical_geometry_and_category=identical,
            int_conversion_collision=len({o['raw_id'] for o in rows}) > 1)
    return result


def inspect_label(path):
    result = dict(scene=path.parent.name, path=str(path.resolve()), header='',
                  radar_frame_idx=None, object_count=0, issues=[], duplicates={},
                  duplicates_only_R={}, duplicates_by_availability={}, objects=[])
    try:
        lines = path.read_text(encoding='utf-8-sig').splitlines()
    except (OSError, UnicodeError) as error:
        result['issues'].append(dict(type='read_error', message=str(error)))
        return result
    if not lines:
        result['issues'].append(dict(type='empty_file'))
        return result
    result['header'] = lines[0]
    match = re.search(r'idx\([^)]*\)\s*=\s*([^,\s]+)', lines[0])
    if match:
        result['radar_frame_idx'] = match.group(1).split('_')[0]
        if path.stem.split('_')[0] != result['radar_frame_idx']:
            result['issues'].append(dict(type='filename_header_radar_mismatch'))
    else:
        result['issues'].append(dict(type='invalid_header'))
    # Some v2.0 files append the first object to the sensor header.
    first = lines[0].split('*')
    rows = []
    if len(first) == 3:
        rows.append((1, '*' + first[-1]))
    elif len(first) > 3:
        result['issues'].append(dict(type='ambiguous_header_objects', line=1))
    rows.extend(enumerate(lines[1:], start=2))
    for line_number, line in rows:
        if not line.strip():
            continue
        try:
            fields = next(csv.reader(io.StringIO(line), skipinitialspace=True))
            if len(fields) not in (10, 11) or fields[0].strip() != '*':
                raise ValueError('Expected v2.0 (10 fields) or v2.1 (11 fields) row')
            offset = 1 if len(fields) == 11 else 0
            availability = fields[1].strip() if offset else None
            if offset and availability not in ('R', 'L', 'L1'):
                result['issues'].append(dict(type='unknown_availability', line=line_number,
                                             value=availability, text=line))
            raw_id = fields[1+offset].strip()
            token = str(int(raw_id))  # Mirror process_kradar.py's int conversion.
            geometry = [float(v) for v in fields[3+offset:]]
            if not all(math.isfinite(v) for v in geometry) or min(geometry[4:]) <= 0:
                raise ValueError('Nonfinite geometry or nonpositive dimensions')
            obj = dict(line=line_number, raw_id=raw_id, id=token,
                       label_version='v2.1' if offset else 'v2.0',
                       availability=availability, category=fields[2+offset].strip(),
                       xyz=geometry[:3], yaw_degrees=geometry[3],
                       raw_dimensions=geometry[4:],
                       full_dimensions=[2*v for v in geometry[4:]], text=line)
            result['objects'].append(obj)
        except (ValueError, csv.Error, StopIteration) as error:
            result['issues'].append(dict(type='invalid_object_row', line=line_number,
                                         message=str(error), text=line))
    result['object_count'] = len(result['objects'])
    objects = result['objects']
    result['availability_counts'] = dict(Counter(o['availability'] or 'not_provided' for o in objects))
    result['negative_id_lines'] = [o['line'] for o in objects if int(o['id']) < 0]
    result['duplicates'] = duplicate_groups(objects)
    result['duplicates_only_R'] = duplicate_groups([o for o in objects if o['availability'] == 'R'])
    result['duplicates_by_availability'] = {
        flag: duplicate_groups([o for o in objects if o['availability'] == flag])
        for flag in ('R', 'L', 'L1')}
    if result['duplicates']:
        result['issues'].append(dict(type='duplicate_ids'))
    if result['negative_id_lines']:
        result['issues'].append(dict(type='negative_id_requires_review'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label-root', type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument('--scenes', nargs='+', help='Scene directory names; default: all numeric scenes')
    parser.add_argument('--file', type=Path, help='Check one raw TXT instead of scanning directories')
    parser.add_argument('--print-objects', action='store_true', help='Print all fields for each file (verbose)')
    parser.add_argument('--output', type=Path, help='Optional JSON report; must be a new file')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('--output already exists; choose a new file')
    if args.file:
        if not args.file.is_file():
            parser.error(f'File not found: {args.file}')
        files = [args.file]
    else:
        if not args.label_root.is_dir():
            parser.error(f'Label root not found: {args.label_root}')
        scenes = args.scenes or sorted(p.name for p in args.label_root.iterdir()
                                      if p.is_dir() and p.name.isdigit())
        files = []
        for scene in scenes:
            selected = sorted((args.label_root / scene).glob('*.txt'))
            if not selected:
                parser.error(f'No TXT labels in {args.label_root / scene}')
            files.extend(selected)
        if not files:
            parser.error('No raw labels found')
    abnormal, scene_counts, issue_counts = [], Counter(), Counter()
    totals = Counter()
    availability_counts = Counter()
    for path in files:
        record = inspect_label(path)
        availability_counts.update(record.get('availability_counts', {}))
        totals['files_with_duplicate_ids'] += bool(record['duplicates'])
        totals['files_with_duplicate_ids_only_R'] += bool(record['duplicates_only_R'])
        totals['cross_availability_duplicate_groups'] += sum(
            g['cross_availability'] for g in record['duplicates'].values())
        totals['negative_id_objects'] += len(record.get('negative_id_lines', []))
        if args.print_objects:
            print(json.dumps(record, ensure_ascii=False, indent=2))
        if record['issues']:
            abnormal.append(record)
            scene_counts[record['scene']] += 1
            issue_counts.update(i['type'] for i in record['issues'])
            print(f"scene={record['scene']} file={path.name} "
                  f"issues={[i['type'] for i in record['issues']]} "
                  f"duplicate_IDs={list(record['duplicates'])} "
                  f"duplicate_IDs_only_R={list(record['duplicates_only_R'])}", flush=True)
    summary = dict(checked_files=len(files), abnormal_files=len(abnormal),
                   issues=dict(issue_counts), abnormal_files_by_scene=dict(scene_counts),
                   **dict(totals), availability_counts=dict(availability_counts),
                   report_scope='abnormal files only; all dataset splits under selected scenes')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(dict(label_root=str(args.label_root.resolve()),
                           notes=['R/L/L1 are availability attributes, not assumed track-ID namespaces.',
                                  'Negative IDs require review; not automatically treated as annotation errors.',
                                  'v2.0 availability is unknown, not inferred as R.',
                                  'Same-frame checks do not verify cross-frame identity.'], summary=summary,
                           files=abnormal), stream, ensure_ascii=False, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
