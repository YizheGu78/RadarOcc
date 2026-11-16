#!/usr/bin/env python3
"""Evaluate saved temporal checkpoints on/off, plus an independent T00 reference."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from temporal.experiments_v2.registry import (
    CONVENTION, PROJECTION_VERSION, RUN_OUT, SERIES, VAL20, WORK, experiments)
from tools.temporal.run_geometry_v2 import file_sha256, knn_ablation, run, write_json_atomic


def choose_checkpoint(directory, explicit=None):
    """Never silently substitute latest or guess between different best runs."""
    if explicit is not None:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted({p.resolve() for p in Path(directory).glob('best_SSC_mean*.pth')
                         if p.is_file()})
    if len(candidates) > 1:
        raise ValueError('Multiple best checkpoints; specify --checkpoint ID=/path: '
                         + ', '.join(map(str, candidates)))
    return candidates[0] if candidates else None


def model_signature(model):
    temporal = model.get('temporal_cfg') or {}
    attention = temporal.get('attention') or {}
    selection = attention.get('selection') or {}
    return dict(type=model['type'], mode=temporal.get('mode'),
        selection=selection.get('method'), query_budget=temporal.get('query_budget'),
        attention={key: value for key, value in attention.items() if key != 'selection'},
        power_mask=temporal.get('power_mask'),
        box=(model.get('box_z_cfg') or {}).get('mode'),
        port=(model.get('p0_temporal_cfg') or {}).get('method'),
        unaligned=model.get('unaligned_temporal') is not None)


def validation_contract(cfg):
    val = cfg.data.val
    path = (ROOT/val.ann_file).resolve()
    if path != (ROOT/VAL20).resolve():
        raise ValueError(f'Expected the common val20 annotation, got {path}')
    occupancy = [dict(step) for step in val.pipeline if step['type'] == 'LoadOccupancy']
    if len(occupancy) != 1:
        raise ValueError('Exactly one LoadOccupancy evaluator input is required')
    return dict(annotation=str(path), occupancy=occupancy,
                load_interval=val.get('load_interval', 1),
                **{key: val.get(key) for key in ('occ_size', 'pc_range', 'classes', 'occ_root')})


def inspect_checkpoint(path, spec, expected_contract):
    import torch
    from mmcv import Config
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    saved = payload.get('meta', {}).get('config')
    if not saved or not payload.get('state_dict'):
        raise ValueError('Expected a complete occupancy checkpoint with meta.config and state_dict')
    cfg = Config.fromstring(saved, '.py')
    if cfg.get('projection_version') != PROJECTION_VERSION:
        raise ValueError('Old projection checkpoint refused')
    if cfg.get('experiment_series') != SERIES:
        raise ValueError('Checkpoint does not belong to this experiment series')
    if cfg.get('radar_coordinate_convention') != CONVENTION:
        raise ValueError('Expected kradar_physical_v2 coordinates')
    expected = Config.fromfile(str(ROOT/spec['config']))
    if model_signature(cfg.model) != model_signature(expected.model):
        raise ValueError('Saved checkpoint method disagrees with experiment ID')
    if spec['id'] != 'T00' and (cfg.data.val.get('current_only', True)
                               or cfg.data.val.get('frame_nums', 4) != 4):
        raise ValueError('Temporal comparisons require the saved real four-frame inputs')
    if validation_contract(cfg) != expected_contract:
        raise ValueError('Validation occupancy labels/grid/classes differ from the common protocol')
    # Keep saved ranker/cache policies; no new selection, GT, or model overrides.
    selection = ((cfg.model.get('temporal_cfg') or {}).get('attention') or {}).get('selection') or {}
    if selection.get('method') == 'learned' and not (ROOT/selection['checkpoint']).is_file():
        raise FileNotFoundError('Saved ranker initialization file is missing: '+selection['checkpoint'])
    if selection.get('cache_dir') and selection.get('cache_mode') == 'read':
        if not (ROOT/selection['cache_dir']).is_dir():
            raise FileNotFoundError('Saved read-only selection cache is missing: '+selection['cache_dir'])
    box_cache = cfg.data.val.get('box_cache')
    if box_cache and not (ROOT/box_cache/'manifest.json').is_file():
        raise FileNotFoundError('Validation box-pair cache is missing: '+box_cache)
    stat = path.stat()
    return dict(checkpoint=str(path), checkpoint_bytes=stat.st_size,
                checkpoint_mtime_ns=stat.st_mtime_ns,
                epoch=payload.get('meta', {}).get('epoch'), signature=model_signature(cfg.model))


def make_plan(ids, overrides, work_root, output, require_all=False):
    from mmcv import Config
    registry = experiments()
    expected_contract = validation_contract(Config.fromfile(str(ROOT/registry['T00']['config'])))
    rows = []
    for identifier in ids:
        spec = registry[identifier]
        row = dict(id=identifier, method=spec['method'], title=spec['title'],
                   output=str(output/f'{identifier}_{spec["method"]}'))
        try:
            path = choose_checkpoint(work_root/spec['method'], overrides.get(identifier))
            if path is None:
                row.update(status='missing', reason='No best_SSC_mean checkpoint; latest is never substituted')
            else:
                row.update(inspect_checkpoint(path, spec, expected_contract), status='ready')
        except Exception as error:
            row.update(status='failed', reason=f'{type(error).__name__}: {error}')
        rows.append(row)
    return dict(series=SERIES, projection_version=PROJECTION_VERSION,
                annotation_sha256=file_sha256(ROOT/VAL20), validation=expected_contract,
                checkpoint_policy='best_SSC_mean; same checkpoint for on/off; no latest fallback',
                require_all=require_all, experiments=rows)


def collect(plan, output):
    """Missing/failed runs remain explicit; never fabricate zero-valued metrics."""
    records = []
    baseline = None
    for item in plan['experiments']:
        row = {key: item.get(key, '') for key in ('id', 'method', 'status', 'checkpoint', 'epoch', 'reason')}
        if item['status'] == 'completed':
            directory = Path(item['output'])
            arms = ['reference'] if item['id'] == 'T00' else ['on', 'off']
            entries = {arm: json.loads((directory/f'{arm}_metrics.json').read_text()) for arm in arms}
            for arm, data in entries.items():
                for metric in ('SSC_mean', 'SSC_Foreground', 'SSC_Background', 'SC_non-empty',
                               'SSC1_mean', 'SSC2_mean'):
                    row[f'{arm}_{metric}'] = data['metrics'][metric]
                row[f'{arm}_history_batches'] = data['fusion_diagnostics']['batches_using_history']
                row[f'{arm}_history_queries'] = data['fusion_diagnostics']['queries_with_history']
            if item['id'] == 'T00':
                baseline = row['reference_SSC_mean']
            else:
                row['on_minus_off_pp'] = 100*(row['on_SSC_mean']-row['off_SSC_mean'])
        records.append(row)
    for row in records:
        if baseline is not None and 'on_SSC_mean' in row:
            row['T00_SSC_mean'] = baseline
            row['on_minus_T00_pp'] = 100*(row['on_SSC_mean']-baseline)
            row['off_minus_T00_pp'] = 100*(row['off_SSC_mean']-baseline)
    fields = list(dict.fromkeys(key for row in records for key in row))
    with (output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    lines = ['# 各时序方案：同一权重开启/旁路融合', '',
        f'统一验证集：`{plan["validation"]["annotation"]}`。每个时序方案使用自己的最佳SSC_mean权重，on/off不换权重。', '',
        '| ID | 方案 | 状态 | Epoch | 开启SSC_mean | 旁路SSC_mean | 开启−旁路(pp) | 开启−T00(pp) |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in records:
        def fmt(key, percent=False):
            value = row.get(key)
            return '—' if value is None else (f'{value:.2%}' if percent else f'{value:+.3f}')
        score = fmt('reference_SSC_mean', True) if row['id'] == 'T00' else fmt('on_SSC_mean', True)
        lines.append(f'| {row["id"]} | {row["method"]} | {row["status"]} | {row["epoch"]} | '
                     f'{score} | {fmt("off_SSC_mean", True)} | {fmt("on_minus_off_pp")} | {fmt("on_minus_T00_pp")} |')
    lines += ['', 'T00只验证一次独立单帧权重，表中放在开启列；它没有开/关融合之分。',
        '差值为正：开启更好；为负：旁路更好。差值很小时需结合多次训练或不确定性分析，不能直接宣称显著收益。',
        '旁路仍是联合训练后的主干，不等于独立单帧。T14两臂都保留近场power屏蔽。',
        'on_history_batches/on_history_queries在summary.csv中记录实际历史使用；零值不证明融合有效。',
        '每项protocol.json记录权重SHA256、验证标注SHA256、源码指纹和旁路位置；每项REPORT.md、comparison.csv给出各指标差值。', '',
        '## 未完成项目', '']
    lines += [f'- {r["id"]}: {r["status"]}；{r.get("reason", "")}'
              for r in records if r['status'] != 'completed'] or ['无。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json_atomic(output/'batch_plan.json', plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', nargs='+', help='Default: T00 plus T01..T15; T00 is a single-frame reference')
    parser.add_argument('--checkpoint', action='append', default=[], metavar='ID=PATH')
    parser.add_argument('--work-root', type=Path, default=ROOT/WORK)
    parser.add_argument('--output-dir', type=Path, default=ROOT/RUN_OUT/'fusion_on_off')
    parser.add_argument('--dry-run', action='store_true', help='Check checkpoints and protocol without CUDA inference')
    parser.add_argument('--require-all', action='store_true', help='Do not start inference if any selected checkpoint is missing')
    parser.add_argument('--single', help=argparse.SUPPRESS)
    parser.add_argument('--annotation-sha256', help=argparse.SUPPRESS)
    args = parser.parse_args()
    registry = experiments()
    ids = args.ids or [key for key, spec in registry.items() if spec['kind'] == 'occupancy-training']
    ids = sorted(set(ids))
    overrides = {}
    for item in args.checkpoint:
        identifier, separator, path = item.partition('=')
        if not separator or not path or identifier in overrides:
            parser.error('--checkpoint expects one unique ID=PATH per override')
        overrides[identifier] = path
    for identifier in [*ids, *overrides, *([args.single] if args.single else [])]:
        if identifier not in registry or registry[identifier]['kind'] != 'occupancy-training':
            parser.error('Unknown occupancy experiment: '+identifier)
    output = args.output_dir.resolve()
    if args.single:
        if args.single not in overrides:
            parser.error('--single requires its explicit checkpoint')
        knn_ablation(Path(overrides[args.single]), output, identifier=args.single,
                     resume=True, annotation_sha256=args.annotation_sha256)
        return
    if set(overrides)-set(ids):
        parser.error('Checkpoint overrides must belong to --ids')
    plan = make_plan(ids, overrides, args.work_root.resolve(), output, args.require_all)
    for item in plan['experiments']:
        print(item['id'], item['method'], item['status'], item.get('checkpoint', item.get('reason', '')), flush=True)
    errors = [r for r in plan['experiments'] if r['status'] == 'failed']
    missing = [r for r in plan['experiments'] if r['status'] == 'missing']
    if args.dry_run:
        print('Only preflight performed. No CUDA inference, no result files written.')
        if errors or (args.require_all and missing):
            raise SystemExit(1)
        return
    output.mkdir(parents=True, exist_ok=True)
    if args.require_all and (missing or errors):
        collect(plan, output)
        raise SystemExit('Preflight incomplete; no inference started. See REPORT.md')
    collect(plan, output)
    for item in plan['experiments']:
        if item['status'] != 'ready':
            continue
        checkpoint = Path(item['checkpoint'])
        command = [sys.executable, str(Path(__file__).resolve()), '--single', item['id'],
                   '--checkpoint', item['id']+'='+str(checkpoint), '--output-dir', item['output'],
                   '--annotation-sha256', plan['annotation_sha256']]
        log_path = output/'logs'/f'{item["id"]}_{time.time_ns()}.log'
        try:
            if (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns) != (
                    item['checkpoint_bytes'], item['checkpoint_mtime_ns']):
                raise ValueError('Checkpoint changed after preflight; stop training before paired evaluation')
            run(command, log_path=log_path)
            item['status'] = 'completed'
        except Exception as error:
            item.update(status='failed', reason=f'{type(error).__name__}: {error}; log: {log_path}')
        collect(plan, output)
    print('汇总报告：', output/'REPORT.md')
    print('汇总表格：', output/'summary.csv')
    print('完成:', sum(r['status'] == 'completed' for r in plan['experiments']),
          '缺少最佳权重:', sum(r['status'] == 'missing' for r in plan['experiments']),
          '失败:', sum(r['status'] == 'failed' for r in plan['experiments']))
    if any(r['status'] == 'failed' for r in plan['experiments']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
