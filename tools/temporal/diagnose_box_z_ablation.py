#!/usr/bin/env python3
"""Re-evaluate saved Z checkpoints, keeping exact confusion counts and IoUs.

No optimizer steps or training artifacts are changed. Selected frames also test
whether replacing historical descriptors changes the learned fusion output.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
import projects.occ_plugin


def metrics_from_counts(counts):
    metrics = {}
    for name, matrix in counts.items():
        matrix = np.asarray(matrix, dtype=np.float64)
        union = matrix.sum(0) + matrix.sum(1) - np.diag(matrix)
        ious = np.divide(np.diag(matrix), union,
                         out=np.full(len(union), np.nan), where=union != 0)
        prefix = name.replace('_metric', '').replace('_', '')
        if prefix.startswith('SSC'):
            for key, value in zip(('free', 'Background', 'Foreground'), ious):
                metrics[f'{prefix}_{key}'] = float(value)
            metrics[f'{prefix}_mean'] = float(ious[1:].mean())
        else:
            metrics[f'{prefix}_non-empty'] = float(ious[1])
    return metrics


def describe(tensor):
    values = tensor.detach().float().flatten()
    if not len(values):
        return None
    q = torch.quantile(values, values.new_tensor([0., .01, .5, .99, 1.]))
    return dict(zip(('min', 'p01', 'median', 'p99', 'max'), q.cpu().tolist()))


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def write_summary(report, output):
    lines = ['# Z0–Z3 已保存权重复查', '',
        f"验证帧数：{len(report['indices'])}；完整固定验证子集：{report['full_validation']}。",
        'IoU 使用原有评价代码产生的混淆矩阵汇总，保留未舍入值，单位0–1。', '',
        '`best` 指原实验保存的最佳文件；旧评价提前舍入，不能据此恢复全部训练轮次的精确最佳。', '',
        '| 权重 | SC_non-empty | SSC_free | SSC_Background | SSC_Foreground | SSC_mean |',
        '|---|---:|---:|---:|---:|---:|']
    keys = ('SC_non-empty', 'SSC_free', 'SSC_Background', 'SSC_Foreground', 'SSC_mean')
    for label, case in report['cases'].items():
        lines.append('| '+label+' | '+' | '.join(f'{case[k]:.8f}' for k in keys)+' |')
    lines += ['', '## 融合输出检查', '',
        '以下列出每份权重的第一个代表帧，通道倍率统计只包含实际修改的Query。',
        '历史打乱变化为相对当前原始通道值的绝对差中位数，保持采样位置和有效性掩码。', '',
        '| 权重 | 帧索引 | 五个连续通道的倍率中位数 | 历史打乱变化中位数 |',
        '|---|---:|---|---:|']
    for label, case in report['cases'].items():
        if case['probes']:
            probe = case['probes'][0]
            multipliers = ', '.join(f"{c['median']:.6f}" for c in probe['channel_multipliers'])
            delta = probe['shuffled_history_relative_output_difference']['median']
            lines.append(f"| {label} | {probe['index']} | {multipliers} | {delta:.8f} |")
    lines += ['',
        '适配器为 `F_out=F_current*(1+gate*expm1(0.5*tanh(Output(H))))`。',
        '当Gate趋近1且tanh饱和时，倍率趋近exp(±0.5)，约0.606531或1.648721。',
        '因此需要结合倍率及历史打乱检查判断是否仍对历史内容敏感。', '',
        '逐帧检查、未舍入指标和原始混淆矩阵见 `report.json`；全部指标见 `precise_iou.csv`。']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('work_dirs/box_z_p0_20pct'))
    p.add_argument('--initial', type=Path,
                   default=Path('work_dirs/radarocc_small_fp32_idfix_timealign_v2/epoch_4.pth'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, default=0, help='0: entire fixed validation subset')
    p.add_argument('--checkpoints', nargs='+', choices=['best', 'last'], default=['best'])
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    args.output.mkdir(parents=True, exist_ok=True)
    set_random_seed(42)
    cfg = Config.fromfile('projects/configs/baselines/RadarOcc_Small_box_z_p0_20pct.py')
    cfg.model.box_z_cfg.update(mode='Z3', diagnostic_path=None)
    cfg.data.val.test_mode = True
    dataset = build_dataset(cfg.data.val)
    indices = list(range(len(dataset)))
    if args.frames:
        eligible = [i for i in indices if dataset.sequence_valid_masks[i].all()]
        indices = [eligible[i] for i in np.linspace(0, len(eligible)-1, args.frames, dtype=int)]
    candidates = [i for i in indices if dataset.sequence_valid_masks[i].all() and
        dataset.box_manifest['records'][
            f'{dataset._scene_id(dataset.data_infos[i])}_{dataset._order_in_scene(dataset.data_infos[i]):05d}'
        ]['pairs'] > 0]
    probes = set(candidates[i] for i in np.linspace(0, len(candidates)-1, min(4, len(candidates)), dtype=int))
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model.cuda().eval()
    parallel = MMDataParallel(model, device_ids=[0])
    fusion = model.box_z_fusion
    capture = {}

    def before_fusion(module, inputs):
        if capture.get('enabled'):
            capture['inputs'] = inputs

    def after_output(module, inputs, output):
        if capture.get('enabled'):
            capture.setdefault('projection', []).append(output.detach())

    def after_gate(module, inputs, output):
        if capture.get('enabled'):
            capture.setdefault('gates', []).append(output.detach().sigmoid())

    def after_fusion(module, inputs, output):
        if capture.get('enabled'):
            capture['output'] = output.detach()

    fusion.register_forward_pre_hook(before_fusion)
    fusion.output.register_forward_hook(after_output)
    fusion.gate.register_forward_hook(after_gate)
    fusion.register_forward_hook(after_fusion)
    baseline_labels, baseline_logits, first_labels = {}, {}, {}
    rows, report = [], dict(indices=indices, full_validation=len(indices) == len(dataset),
        label_comparison_scope='all output voxels, including ignored GT; IoUs use original evaluation masks',
        probe_gate_saturation_scope='selected queries, including those with no valid history', cases={})
    cases = [('Single-frame', None, args.initial)]
    for kind in args.checkpoints:
        for mode in ('Z0', 'Z1', 'Z2', 'Z3'):
            if kind == 'best':
                choices = list((args.root/mode).glob('best_SSC_mean_epoch_*.pth'))
                if len(choices) != 1:
                    raise ValueError(f'Expected one saved best checkpoint: {choices}')
                path = choices[0]
            else:
                path = (args.root/mode/'latest.pth').resolve()
            cases.append((f'{mode}_{kind}', mode, path))
    reference_original = None
    for label, mode, path in cases:
        state = torch.load(path, map_location='cpu', weights_only=False)
        # Keep version metadata: sparse encoders use it to identify weight layout.
        state_dict = state['state_dict']
        if any(k.startswith('module.') for k in state_dict):
            raise ValueError('Use an unwrapped RadarOcc checkpoint')
        original = {k: v for k, v in state_dict.items() if not k.startswith('box_z_fusion.')}
        original_equal = reference_original is None or (
            original.keys() == reference_original.keys() and
            all(torch.equal(v, reference_original[k]) for k, v in original.items()))
        if reference_original is None:
            reference_original = original
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        expected_missing = (lambda k: k.startswith('box_z_fusion.') if mode is None else
                             mode != 'Z3' and k.startswith('box_z_fusion.z_linear.'))
        if unexpected or any(not expected_missing(k) and not k.endswith('num_batches_tracked') for k in missing):
            raise RuntimeError((missing, unexpected))
        if mode != 'Z3':
            with torch.no_grad():
                fusion.z_linear.weight.zero_(); fusion.z_linear.bias.zero_()
        fusion.enabled, fusion.mode = mode is not None, mode or 'Z3'
        counts, frame_probes = {}, []
        changed_from_baseline = changed_from_z0 = evaluated_voxels = 0
        for step, index in enumerate(indices):
            capture.clear(); capture['enabled'] = index in probes and mode is not None
            batch = collate([dataset[index]], samples_per_gpu=1)
            with torch.no_grad():
                result = parallel(return_loss=False, rescale=True, **batch)
            logits = result['pred_c'].detach().cpu()
            classes = logits.argmax(1).numpy().astype(np.uint8)
            for key, value in result.items():
                if '_metric' in key:
                    counts[key] = counts.get(key, 0) + np.asarray(value, dtype=np.int64)
            if mode is None:
                baseline_labels[index] = classes
                if index in probes:
                    baseline_logits[index] = logits
            else:
                changed_from_baseline += int((classes != baseline_labels[index]).sum())
                if mode == 'Z0':
                    first_labels[(label.split('_')[-1], index)] = classes
                else:
                    changed_from_z0 += int((classes != first_labels[(label.split('_')[-1], index)]).sum())
            evaluated_voxels += classes.size
            if capture['enabled']:
                features, coords, pairs = capture['inputs']
                actual = capture['output']
                valid = (actual[:, [0, 1, 2, 6, 7]] != features[-1][:, [0, 1, 2, 6, 7]]).any(-1)
                source = features[-1][:, [0, 1, 2, 6, 7]]
                ratio = actual[:, [0, 1, 2, 6, 7]] / source.clamp_min(1e-30)
                projection = torch.cat(capture['projection'])
                gates = torch.cat(capture['gates'])
                item = dict(index=index, fusion_diagnostics=dict(fusion.diagnostics),
                    output_tanh_saturated_fraction=float((projection.tanh().abs() > .99).float().mean()),
                    gate_above_095_fraction=float((gates > .95).float().mean()),
                    channel_multipliers=[describe(ratio[valid, c]) for c in range(5)],
                    max_logit_difference_from_baseline=float((logits-baseline_logits[index]).abs().max()))
                capture['enabled'] = False
                with torch.no_grad():
                    replay = parallel(return_loss=False, rescale=True, **batch)['pred_c'].detach().cpu()
                item['repeat_inference_max_logit_difference'] = float((replay-logits).abs().max())
                item['repeat_inference_changed_voxels'] = int((replay.argmax(1).numpy() != classes).sum())
                with torch.no_grad():
                    replaced = features.clone(); replaced[:3] = 0
                    altered = fusion(replaced, coords, pairs)
                delta = ((altered[:, [0, 1, 2, 6, 7]]-actual[:, [0, 1, 2, 6, 7]]) /
                         source.clamp_min(1e-30)).abs()
                item['zero_history_relative_output_difference'] = describe(delta[valid])
                with torch.no_grad():
                    shuffled = features.clone()
                    shuffled[:3] = shuffled[:3].roll(137, dims=1)
                    altered = fusion(shuffled, coords, pairs)
                delta = ((altered[:, [0, 1, 2, 6, 7]]-actual[:, [0, 1, 2, 6, 7]]) /
                         source.clamp_min(1e-30)).abs()
                item['shuffled_history_relative_output_difference'] = describe(delta[valid])
                frame_probes.append(item)
            if (step+1) % 25 == 0 or step+1 == len(indices):
                print(f'{label}: {step+1}/{len(indices)}', flush=True)
        metrics = metrics_from_counts(counts)
        rows.append(dict(method=label, epoch=state.get('meta', {}).get('epoch'), frames=len(indices),
            changed_voxels_vs_baseline=changed_from_baseline,
            changed_voxels_vs_Z0=changed_from_z0 if mode is not None else None,
            evaluated_voxels=evaluated_voxels, **metrics))
        report['cases'][label] = dict(checkpoint=str(path), original_weights_equal_to_initial=original_equal,
            **rows[-1], confusion_counts={k: v.tolist() for k, v in counts.items()}, probes=frame_probes)
        (args.output/'report.json').write_text(json.dumps(finite_json(report), indent=2, allow_nan=False)+'\n')
        with (args.output/'precise_iou.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(finite_json(rows))
        write_summary(report, args.output)
        print(label, json.dumps(metrics), flush=True)
        del state, state_dict, original


if __name__ == '__main__':
    main()
