#!/usr/bin/env python3
"""Export complete per-epoch and best validation IoUs for sparse-query runs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.temporal.summarize_box_z_ablation import (ALL_IOU_KEYS, read_epoch_metrics,
                                                   select_best, write_csv)


def summarize(root):
    epochs, best = [], []
    for method in ('Q64', 'Q32'):
        directory = root/method
        values = read_epoch_metrics(directory, method)
        epochs.extend(values)
        if values:
            chosen = select_best(values, directory)
            chosen['per_range'] = int(method[1:])
            chosen['reduction_rate'] = 1-int(method[1:])/250
            best.append(chosen)
            write_csv(directory/'best_val.csv', [chosen], list(chosen))
            (directory/'best_val.json').write_text(json.dumps(chosen, indent=2)+'\n')
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root/'epoch_iou.csv', epochs, ['method', 'epoch']+ALL_IOU_KEYS+['metric_source'])
    write_csv(root/'comparison.csv', best, ['method', 'epoch', 'per_range', 'reduction_rate']+ALL_IOU_KEYS+['checkpoint', 'metric_source'])
    (root/'comparison.json').write_text(json.dumps(best, indent=2)+'\n')
    lines = ['# 20% 数据：稀疏时序 Query + Learned Top-8', '',
             '最佳轮次按验证集 SSC_mean 选择。IoU 为 0–1 原始值；缩减率不是运行加速比。', '',
             '| 方法 | 最佳 epoch | SC | SSC Background | SSC Foreground | SSC mean |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in best:
        lines.append('| '+' | '.join([row['method'], str(row['epoch'])]+[
            f'{row[k]:.6%}' for k in ('SC_non-empty', 'SSC_Background', 'SSC_Foreground', 'SSC_mean')])+' |')
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    summarize(p.parse_args().root)
