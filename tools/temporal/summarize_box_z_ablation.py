#!/usr/bin/env python3
"""Collect every validation IoU and completed validation diagnostics for Z0-Z3."""
import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np

IOU_KEYS = ['SC_non-empty', 'SSC_free', 'SSC_Background', 'SSC_Foreground', 'SSC_mean']


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def summarize(root, val_frames):
    epochs, diagnostics, summary = [], [], []
    baseline = root/'single_frame_metrics.json'
    if baseline.exists():
        values = json.loads(baseline.read_text())
        summary.append(dict(method='Single-frame', epoch=0, **{k: values.get(k) for k in IOU_KEYS}))
    for method in ('Z0', 'Z1', 'Z2', 'Z3'):
        directory = root/method
        if not directory.exists():
            continue
        by_epoch = {}
        for path in sorted(directory.glob('*.log.json')):
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if 'SSC_mean' in row and row['SSC_mean'] is not None and math.isfinite(row['SSC_mean']):
                    by_epoch[int(row['epoch'])] = dict(method=method, epoch=int(row['epoch']),
                        **{k: row.get(k) for k in IOU_KEYS})
        values = [by_epoch[k] for k in sorted(by_epoch)]
        epochs.extend(values)
        runs = defaultdict(list)
        path = directory/'eval_diagnostics.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                runs[(row['epoch'], row['eval_run'])].append(row)
        complete = {}
        for (epoch, _), records in runs.items():
            if len(records) == val_frames:
                complete[epoch] = records
        per_epoch_diag = {}
        for epoch, records in complete.items():
            n = sum(r['sample_attempted'] for r in records)
            valid = sum(r['sample_valid'] for r in records)
            queries = sum(r['fused_queries'] for r in records)
            row = dict(method=method, epoch=epoch, val_frames=len(records), sample_attempted=n,
                sample_valid=valid, valid_sampling_rate=valid/n if n else None,
                invalid_sampling_rate=1-valid/n if n else None,
                gate_mean=sum(r['gate_sum'] for r in records)/queries if queries else None,
                relative_contribution=sum(r['correction_sum'] for r in records)/queries if queries else None,
                fused_queries=queries, valid_tokens=sum(r['valid_tokens'] for r in records),
                selected_queries=sum(r['selected_queries'] for r in records))
            for key in ('alpha', 'shift', 'scale'):
                a = np.array([v for r in records for v in r['pair_values'].get(key, [])])
                for suffix, value in [('mean', a.mean() if a.size else None),
                                      ('median', np.median(a) if a.size else None),
                                      ('p95', np.quantile(a, .95) if a.size else None)]:
                    row[key+'_'+suffix] = float(value) if value is not None else None
                if key == 'shift':
                    row['abs_shift_median'] = float(np.median(abs(a))) if a.size else None
                    row['abs_shift_p95'] = float(np.quantile(abs(a), .95)) if a.size else None
            diagnostics.append(row); per_epoch_diag[epoch] = row
        if values:
            best = max(values, key=lambda r: r['SSC_mean'])
            summary.append(dict(best, selection='best_validation_SSC_mean',
                last_epoch=values[-1]['epoch'], last_SSC_mean=values[-1]['SSC_mean'],
                **{k: v for k, v in per_epoch_diag.get(best['epoch'], {}).items() if k not in ('method', 'epoch')}))
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root/'epoch_iou.csv', epochs, ['method', 'epoch']+IOU_KEYS)
    diag_fields = list(dict.fromkeys(k for row in diagnostics for k in row))
    write_csv(root/'sampling_diagnostics.csv', diagnostics, diag_fields or ['method', 'epoch'])
    fields = list(dict.fromkeys(k for row in summary for k in row))
    write_csv(root/'comparison.csv', summary, fields or ['method']+IOU_KEYS)
    (root/'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    lines = ['# P0 Z0–Z3：20% 数据对照', '',
        'IoU 保留项目原始 0–1 数值；SSC_mean 沿用项目定义（Background/Foreground 平均，不含 free）。', '',
        '| Method | Best epoch | SC_non-empty | SSC_free | SSC_Background | SSC_Foreground | SSC_mean |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append('| '+' | '.join([row['method'], str(row['epoch'])]+[
            f'{row[k]:.3f}' if row.get(k) is not None else 'N/A' for k in IOU_KEYS])+' |')
    if not summary:
        lines.append('\n尚无实际 IoU 结果；运行实验后生成。')
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('Saved IoU and diagnostics:', root)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--val-frames', type=int, default=211)
    args = p.parse_args()
    summarize(args.root, args.val_frames)
