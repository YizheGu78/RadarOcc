#!/usr/bin/env python3
"""Reproducible learned-module CPU audit, explicitly excluding CUDA/spconv."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def source_fingerprints():
    paths=set()
    for directory in ('temporal','projects/unaligned_temporal'):
        paths.update((ROOT/directory).rglob('*.py'))
    for relative in (
        'projects/occ_plugin/occupancy/detectors/radarocc_self_small.py',
        'projects/occ_plugin/occupancy/voxel_encoder/sparse_lidar_enc.py',
        'projects/occ_plugin/occupancy/backbones/resnet3d.py',
        'projects/occ_plugin/occupancy/necks/fpn3d.py',
        'projects/occ_plugin/occupancy/dense_heads/occ_head_fp16.py',
        'projects/occ_plugin/occupancy/dense_heads/lovasz_softmax.py',
        'projects/occ_plugin/utils/position_embedding.py',
        'projects/occ_plugin/utils/semkitti.py',
        'projects/occ_plugin/datasets/pipelines/loading.py',
        'projects/occ_plugin/datasets/temporal_kradar_dataset.py',
        'tests/test_model_integrity.py',
        'tests/test_deform_cuda_gradients.py',
        'third_party/VoxFormer/deform_attn_3d/csrc/ms_deform_attn_cuda_kernel.cuh',
        'third_party/VoxFormer/deform_attn_3d/csrc/ms_deform_attn_cuda.cu',
        'third_party/VoxFormer/deform_attn_3d/setup.py',
        'tools/temporal/audit_model_integrity.py'):
        paths.add(ROOT/relative)
    paths.update((ROOT/'projects/occ_plugin/occupancy/deformable_transformer').glob('*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/model_audit/20261112')
    parser.add_argument('--require-pass',action='store_true')
    args=parser.parse_args()
    import torch
    from temporal.experiments_v2.registry import PROJECTION_VERSION, SERIES
    torch.set_num_threads(1)
    torch.manual_seed(42)
    spec=importlib.util.spec_from_file_location('radarocc_model_integrity_tests',
                                               ROOT/'tests/test_model_integrity.py')
    module=importlib.util.module_from_spec(spec)
    started=time.monotonic()
    spec.loader.exec_module(module)
    suite=unittest.defaultTestLoader.loadTestsFromModule(module)
    scalar_spec=importlib.util.spec_from_file_location('radarocc_cuda_scalar_tests',
        ROOT/'tests/test_deform_cuda_gradients.py')
    scalar_module=importlib.util.module_from_spec(scalar_spec)
    scalar_spec.loader.exec_module(scalar_module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(scalar_module))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    output=args.output_dir
    supplemental={}
    for name in ('regression_suite','centerformer_real_frames','rodnet_tdc_real_frames'):
        path=output/(name+'.json')
        if path.exists():
            payload=json.loads(path.read_text())
            if payload.get('projection_version')==PROJECTION_VERSION:
                supplemental[name]=payload
    report=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,
        failures=[dict(test=str(t),traceback=message) for t,message in result.failures],
        errors=[dict(test=str(t),traceback=message) for t,message in result.errors],
        skipped=[dict(test=str(t),reason=reason) for t,reason in result.skipped],
        projection_version=PROJECTION_VERSION,series=SERIES,
        generated_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        cuda_available=torch.cuda.is_available(),whole_model_cuda_executed=False,
        whole_model_verified=False, training_checkpoint_saved=False,
        probes=module.PROBES,supplemental=supplemental,source_sha256=source_fingerprints())
    output.mkdir(parents=True,exist_ok=True)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    findings=[
        ('所有Small训练','position_embedding.py','位置编码输出原为B,C,Z,H,W，与特征B,C,H,W,Z展平错配；已修正并逐格验证。'),
        ('所有Small从头初始化','transformer_3D.py','父初始化覆盖3D注意力的零偏移/零logit矩阵；已让实际注册的3D类恢复专用初始化。'),
        ('CUDA采样偏移训练梯度','ms_deform_attn_cuda_kernel.cuh','三线性反向的坐标梯度漏乘另外两轴的插值权重；两种helper均已修复，旧公式保留注释。实际C++ helper在CPU上与autograd/有限差分比较，GPU并行执行仍待V13。'),
        ('所有启用checkpoint的FPN','fpn3d.py','反向重算重复更新BatchNorm统计；已用临时运行缓冲区隔离重算，保留batch统计与梯度。'),
        ('T14近场屏蔽','configs/knn_mask37.py','实际配置未启用零power编码，屏蔽后log10(0)产生非有限值；已启用已有安全编码。'),
        ('GT存在忽略标签时','loading.py / radarocc_self_small.py','加载器把255压成前景，近距离SC又把255当占用；已分别修正，原始GT文件不改。'),
        ('全忽略/单有效体素等边界','semkitti.py / lovasz_softmax.py','CE可NaN、semantic可除零、Lovasz单点维度丢失；已修复并检查四项损失的标量与梯度。'),
        ('非方形网格边界','occ_head_fp16.py / radarocc_self_small.py','GT缩放率原以Y除预测X；已统一用X。当前128×128会掩盖此问题。'),
        ('CPU路径','3D attention / Small构造函数','原CPU fallback是2D；已加入独立oracle核验的3D实现，未使用的构造常量改为可迁移buffer。CUDA核实现未改。'),
        ('B>1的注意力接口','deformable_self_attention_3D_custom.py','首队列索引混入其他batch，且残差开关不可达；已修正。当前整网仍要求B=1，不能用此项解释此前B=1掉点。')]
    lines=['# 当前Small与时序模型：模型层复查（2026-11-12）','',
        f'本次CPU审计：**{"通过" if report["passed"] else "失败"}**，{result.testsRun}项；源码版本`{PROJECTION_VERSION}`。',
        '证据为[report.json](report.json)中的逐项结果、真实类名、梯度和源码SHA256；修正前部分复现记录见[before.json]('+os.path.relpath(ROOT/'outputs/model_audit/20261112/before.json',output)+')。','',
        '审计覆盖当前Small和16种T配置的实际构造/初始化、位置编码、3D注意力、ResNet3D/FPN、实际注册的OccHead、四项损失、归一化、评价、T14串接、冻结ranker及数据时序约束。**这不等于真实数据的CUDA整网已经验证通过。**','',
        '| 影响范围 | 位置 | 已确认问题与处理 |','| --- | --- | --- |']
    lines.extend('| '+' | '.join(row)+' |' for row in findings)
    lines.extend(['',
        '真实解码子图没有把学习层替换成恒等层：由合成`[1,144,3,5,2]`稀疏编码器输出，经过真实球坐标注意力、Cartesian注意力与残差、ResNet3D、FPN、OccHead和四种损失；Cartesian网格缩小为16×12×8，引用点人为放在有效插值区域，检查有限非零梯度及一次临时SGD更新。`extract_feat`边界由该子图提供，未执行spconv。此测试不测真实投影覆盖率、真实数据精度或显存需求。',
        '在零初始化偏移/权重矩阵时，Cartesian query embedding首步梯度为零是预期的：其输入只影响偏移/logit。测试确认这些矩阵首步能更新，随后query embedding得到非零梯度。无需为了首步非零而改成随机偏移。',
        '四项损失使用`L / abs(L.detach()).clamp_min(1e-3)`，日志可显示每项1、总和4；实际梯度不为零。保留此训练协议，比较收敛需看raw_*和验证指标。',
        'T00～T15都从实际配置构造并执行父初始化，未绕开`.cuda()`调用。LearnedKNN的构造检查使用临时合成ranker，专门验证冻结和权重保留；不表示正式P03已训练完成。','',
        '数据检查遍历train/val的全部序列元信息，检查有效历史的序号、时间戳均早于当前帧，train/val的雷达输入路径不重叠。另各抽13个GT文件统计原始标签；这是抽样，不是全量GT内容扫描。未从抽样中观察到0/255，不能因此断定损失和忽略标签边界导致了已有实验掉点。','',
        '所有替换的原有模型语句留作注释；未重新生成或反号LiDAR GT，也未改原始雷达文件。Full/Strict、fine/cascade、多卡SyncBN通信、混合精度等未启用路径不在本次已通过范围内。','',
        '仍需在GPU执行V13及各方法`run Txx --smoke`：实际CUDA采样核的前向/反向、spconv、真实数据完整尺寸和显存。V13采用单卡MMDataParallel；多卡DDP/SyncBN通信还需单独验证，不能把V13视为多卡检查。没有完成GPU检查及正式训练，就不能保证所有模型一定正确，不能据CPU测试推断IoU会提高多少。','',
        '```bash','source use_radarocc5060.sh',
        'python tools/temporal/run_geometry_v2.py run V14  # 本次CPU模型审计',
        'python tools/temporal/run_geometry_v2.py run V13  # GPU整网检查',
        'python tools/temporal/run_geometry_v2.py run T01 --smoke  # KNN自身的整网接线',
        '```',''])
    if supplemental:
        lines.extend(['补充运行记录（与上述CPU模型审计分别执行）：',''])
        regression=supplemental.get('regression_suite')
        if regression:
            lines.append(f'- 全套回归：{regression["passed"]}项通过、{regression["skipped"]}项CUDA测试跳过；[日志](pytest.log)、[统计](regression_suite.json)。')
        for name in ('centerformer_real_frames','rodnet_tdc_real_frames'):
            if name in supplemental:
                p=supplemental[name]
                lines.append(f'- {p["experiment_id"]} {p["method"]}：真实四帧、每帧{p["points"]:,}个P1 token；初始恒等、描述符未修改、梯度有限；offset梯度和={p["offset_gradient"]:.8g}。[记录]({name}.json)。仅模块检查。')
        lines.append('')
    (output/'REPORT.md').write_text('\n'.join(lines))
    print('Model integrity report:',output/'REPORT.md',flush=True)
    if args.require_pass and not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
