#!/usr/bin/env python3
"""List, run and document corrected geometry experiments without overwriting v1."""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from temporal.experiments_v2.registry import (experiments, COORD, OUT, CONVENTION,
    SERIES, PROJECTION_VERSION, WORK, RUN_OUT)
from tools.temporal.audit_model_integrity import source_fingerprints


def run(command, log_path=None):
    command = [sys.executable if s == 'python' else str(s) for s in command]
    print('+ '+shlex.join(command), flush=True)
    # Keep the original child traceback; nested CalledProcessError messages
    # alone do not tell the user which CUDA check actually failed.
    log_path = Path(log_path) if log_path is not None else (
        ROOT/RUN_OUT/'command_logs'/f'{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns()}.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print('完整输出日志：'+str(log_path), flush=True)
    with log_path.open('w') as stream:
        stream.write('+ '+shlex.join(command)+'\n')
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    stream.write(line)
                    stream.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
    if code:
        print(f'子命令失败（exit={code}），首条报错和完整Traceback保存在：{log_path}',
              file=sys.stderr, flush=True)
        raise subprocess.CalledProcessError(code, command)
    return log_path


def check_checkpoint(path, current_only=False):
    import torch
    from mmcv import Config
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    config = payload.get('meta', {}).get('config', '')
    if not config:
        raise ValueError('Checkpoint has no saved training configuration')
    saved = Config.fromstring(config, '.py')
    if saved.get('projection_version') != PROJECTION_VERSION:
        raise ValueError('Old projection checkpoint refused; start this series from scratch, not resume/load old weights')
    if not current_only and CONVENTION not in config:
        raise ValueError('Expected a full-model checkpoint trained with kradar_physical_v2; legacy resume/eval refused')


def model_smoke(identifier, index=0):
    """Real one-batch CUDA forward/backward + val inference, with no weight save."""
    import copy
    import importlib
    import numpy as np
    import torch
    from mmcv import Config
    spec = experiments()[identifier]
    if spec['kind'] != 'occupancy-training':
        raise ValueError('model-smoke requires a T training ID')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; whole-model smoke check was NOT run. No training started.')
    missing = [p for p in spec['required'] if not (ROOT/p).exists()]
    if missing:
        raise FileNotFoundError('Prepare this method first: '+', '.join(missing))
    cfg = Config.fromfile(str(ROOT/spec['config']))
    if cfg.load_from is not None or cfg.resume_from is not None:
        raise ValueError('Smoke checks must use scratch configs')
    if identifier == 'T00':
        run(['python', 'tools/temporal/audit_projection_chain.py', '--require-pass',
             '--check-cuda', '--output-dir', RUN_OUT+'/projection_audit_cuda'])
    from mmcv.parallel import MMDataParallel, collate
    from mmdet.apis import set_random_seed
    from mmdet3d.models import build_model
    from projects.occ_plugin.datasets.builder import custom_build_dataset
    set_random_seed(int(cfg.seed), deterministic=True)
    torch.cuda.set_device(0)
    importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/', '.'))
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    wrapped = MMDataParallel(model.cuda(), device_ids=[0])
    def sample(split):
        options = copy.deepcopy(cfg.data[split])
        options.pop('samples_per_gpu', None)
        options['test_mode'] = split != 'train'
        dataset = custom_build_dataset(options)
        if index < 0 or index >= len(dataset):
            raise IndexError(f'{split} index {index} exceeds dataset length {len(dataset)}')
        return collate([dataset[index]], samples_per_gpu=1)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    wrapped.train()
    losses = wrapped(return_loss=True, **sample('train'))
    loss, logs = model._parse_losses(losses)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite whole-model loss')
    loss.backward()
    gradients = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for _, g in gradients):
        raise RuntimeError('Missing or nonfinite whole-model gradients')
    if not any(bool(torch.count_nonzero(g)) for _, g in gradients):
        raise RuntimeError('All whole-model gradients are zero')
    gradient_count = len(gradients)
    loss_value = float(loss.detach())
    model.zero_grad(set_to_none=True)
    del gradients, losses, loss
    wrapped.eval()
    head_shapes = []
    def inspect_head(module, inputs, output):
        logits = output.get('output_voxels', [])
        tensors = [v for v in logits if torch.is_tensor(v)]
        if not tensors or not all(bool(torch.isfinite(v).all()) for v in tensors):
            raise RuntimeError('Missing or nonfinite raw occupancy logits before argmax')
        head_shapes.extend([list(v.shape) for v in tensors])
    handle = model.pts_bbox_head.register_forward_hook(inspect_head)
    try:
        with torch.inference_mode():
            prediction = wrapped(return_loss=False, rescale=True, **sample('val'))
    finally:
        handle.remove()
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, (np.ndarray, float, np.floating)):
            return bool(np.isfinite(value).all())
        return True
    if prediction is None or not head_shapes or not finite(prediction):
        raise RuntimeError('Nonfinite validation output')
    torch.cuda.synchronize()
    result = dict(id=identifier, series=SERIES, projection_version=PROJECTION_VERSION,
        completed=True, device=torch.cuda.get_device_name(0), config=spec['config'], index=index,
        scratch=True, optimizer_steps=0, checkpoint_saved=False,
        loss=loss_value, log_vars=logs, parameters_with_gradients=gradient_count,
        finite_gradients=True, nonzero_gradients=True, finite_validation_output=True,
        raw_occupancy_logit_shapes=head_shapes,
        train_annotation=cfg.data.train.ann_file, val_annotation=cfg.data.val.ann_file,
        peak_memory_MiB=torch.cuda.max_memory_allocated()/1024**2,
        elapsed_seconds=time.time()-started,
        source_sha256=source_fingerprints())
    path = ROOT/RUN_OUT/'model_smoke'/f'{identifier}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('Whole-model one-batch check:', path)


@contextmanager
def fusion_bypass(plugin, enabled):
    """Keep every parameter loaded; replace only the temporal forward output."""
    if not enabled:
        yield
        return
    had_instance_forward = 'forward' in plugin.__dict__
    original = plugin.__dict__.get('forward')

    def current_frame_only(features, coords, *args, **kwargs):
        return features[:, -1], dict(used_history=False, fusion_bypassed=True)

    plugin.forward = current_frame_only
    try:
        yield
    finally:
        if had_instance_forward:
            plugin.forward = original
        else:
            del plugin.forward


def knn_ablation(checkpoint_path, output, dry_run=False):
    """Paired full-val inference of one checkpoint, with only fusion bypassed."""
    import copy
    import importlib
    import numpy as np
    import torch
    from mmcv import Config

    checkpoint_path = checkpoint_path.resolve()
    check_checkpoint(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
    saved_config = payload.get('meta', {}).get('config', '')
    if CONVENTION not in saved_config:
        raise ValueError('Expected a v2 KNN full-model checkpoint')
    cfg = Config.fromstring(saved_config, '.py')
    temporal_cfg = cfg.model.get('temporal_cfg') or {}
    attention = temporal_cfg.get('attention') or {}
    if (temporal_cfg.get('mode') != 'ego_attention' or attention.get('selection') is not None
            or temporal_cfg.get('query_budget') or temporal_cfg.get('power_mask')):
        raise ValueError('ablate-knn currently supports the plain KNN T01 configuration only')
    if cfg.data.val.get('current_only', True):
        raise ValueError('Expected the same real four-frame validation inputs in both arms')
    if output.exists():
        raise FileExistsError(f'Use a new output directory; existing paired results are retained: {output}')
    seed = int(payload.get('meta', {}).get('seed', 42))
    protocol = dict(checkpoint=str(checkpoint_path), checkpoint_epoch=payload.get('meta', {}).get('epoch'),
        checkpoint_bytes=checkpoint_path.stat().st_size,
        annotation=str(cfg.data.val.ann_file), seed=seed, coordinate_convention=CONVENTION,
        config_source='checkpoint.meta.config', arms=['on', 'off'],
        projection_version=PROJECTION_VERSION, series=SERIES,
        off_behavior='Same loaded model and four-frame inputs; temporal output replaced by current descriptors',
        training=False, output=str(output))
    if dry_run:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --dry-run; no full inference was started.')

    from mmcv.parallel import MMDataParallel
    from mmdet.apis import set_random_seed
    from mmdet3d.models import build_model
    from projects.occ_plugin.datasets.builder import build_dataloader, custom_build_dataset
    from projects.occ_plugin.occupancy.apis.test import custom_single_gpu_test

    set_random_seed(seed, deterministic=True)
    torch.cuda.set_device(0)
    importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/', '.'))
    dataset_cfg = copy.deepcopy(cfg.data.val)
    dataset_cfg.update(test_mode=True)
    dataset_cfg.pop('samples_per_gpu', None)
    dataset = custom_build_dataset(dataset_cfg)
    loader = build_dataloader(dataset, 1, 0, dist=False, shuffle=False, seed=seed)
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.train_cfg = None
    model = build_model(model_cfg, test_cfg=cfg.get('test_cfg'))
    # Preserve OrderedDict._metadata: spconv uses version=2 to identify its
    # kernel layout. Converting to a plain dict would trigger a wrong transpose.
    state = payload['state_dict']
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state, 'module.')
    # No discarded/missing temporal weights: both arms use this exact same model.
    model.load_state_dict(state, strict=True)
    model.CLASSES = payload.get('meta', {}).get('CLASSES', dataset.CLASSES)
    parallel = MMDataParallel(model.cuda(), device_ids=[0])
    parallel.eval()
    protocol['validation_samples'] = len(dataset)
    output.mkdir(parents=True)
    (output/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')

    def serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        return value

    results = {}
    for arm in ('on', 'off'):
        set_random_seed(seed, deterministic=True)
        print(f'\nFusion {arm}: evaluating {len(dataset)} identical validation samples', flush=True)
        started = time.time()
        with fusion_bypass(model.temporal_plugin, enabled=arm == 'off'):
            aggregate = custom_single_gpu_test(parallel, loader)
            metrics = serializable(dataset.evaluate(aggregate))
        results[arm] = metrics
        (output/f'{arm}_metrics.json').write_text(json.dumps(dict(arm=arm, completed=True,
            samples=len(dataset), elapsed_seconds=time.time()-started, metrics=metrics,
            confusion=serializable(aggregate)), ensure_ascii=False, indent=2)+'\n')
    fields = ('SSC_mean', 'SSC_Foreground', 'SSC_Background', 'SC_non-empty', 'SSC1_mean', 'SSC2_mean')
    rows = [dict(metric=key, on=results['on'][key], off=results['off'][key],
                 on_minus_off_pp=100*(results['on'][key]-results['off'][key])) for key in fields]
    with (output/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    text = ['# 同一KNN checkpoint：融合开/关验证', '',
        f'Checkpoint: `{checkpoint_path}`；验证样本：{len(dataset)}。', '',
        '| 指标 | 开启融合 | 关闭融合 | 开启−关闭（百分点） |', '| --- | --- | --- | --- |']
    for row in rows:
        text.append(f'| {row["metric"]} | {row["on"]:.2%} | {row["off"]:.2%} | {row["on_minus_off_pp"]:+.3f} |')
    text += ['', '开启高于关闭：该checkpoint的融合分支在这次验证中有正收益；低于关闭：推理时绕过融合反而更好。',
        '关闭融合仍使用经过KNN联合训练的主干，不等于独立训练的单帧模型，也不能单靠这个消融判断几何方向对错。',
        '本轮仅接受修正后重新训练的checkpoint；不得把旧主干权重的迁移结果混入此对比。']
    (output/'REPORT.md').write_text('\n'.join(text)+'\n')
    print('Paired evaluation complete:', output/'REPORT.md')


def document(output):
    """Update the existing table, keeping earlier records separate from this run."""
    output.mkdir(parents=True, exist_ok=True)
    registry = dict(sorted(experiments().items(),
        key=lambda item: (('V', 'P', 'T').index(item[0][0]), int(item[0][1:]))))
    previous = {}
    if (output/'experiments.csv').exists():
        with (output/'experiments.csv').open() as stream:
            previous = {r['id']: r for r in csv.DictReader(stream)}
    metrics = ('SSC_mean', 'SC_non_empty', 'SSC_Foreground', 'SSC1_mean', 'SSC2_mean',
               'runtime', 'checkpoint')
    rows = []
    for spec in registry.values():
        identifier = spec['id']
        old = previous.get(identifier, {})
        same_series = (old.get('series') == SERIES and
                       old.get('projection_version') == PROJECTION_VERSION)
        historical = old.get('historical_record', '') if same_series else (
            json.dumps(old, ensure_ascii=False) if old else '')
        row = dict(id=identifier, experiment=spec['title'], kind=spec['kind'], series=SERIES,
            projection_version=PROJECTION_VERSION, rerun=spec['rerun'], comparison=spec['comparison'],
            purpose=spec['purpose'], procedure=spec['procedure'], readout=spec['readout'],
            next_step=spec['next_step'], priority=spec['priority'], gt_usage=spec['gt_usage'],
            prerequisites=', '.join(spec['depends']),
            command=f'python tools/temporal/run_geometry_v2.py run {identifier}',
            expanded_command=shlex.join(spec['command']), smoke_command='',
            validation_command='', resume_command='', notes=spec['note'],
            status='本轮待运行', **{key:old.get(key, '') if same_series else '' for key in metrics},
            historical_record=historical)
        if spec['kind'] == 'occupancy-training':
            row['smoke_command'] = f'python tools/temporal/run_geometry_v2.py run {identifier} --smoke'
            row['validation_command'] = f'python tools/temporal/run_geometry_v2.py val {identifier} --checkpoint {spec["work_dir"]}/best_SSC_mean_epoch_N.pth'
            row['resume_command'] = f'python tools/temporal/run_geometry_v2.py resume {identifier} --checkpoint {spec["work_dir"]}/latest.pth'
        records = ROOT/RUN_OUT/'execution_records'
        completed = records/f'{identifier}_run.json'
        if completed.exists() and json.loads(completed.read_text()).get('projection_version') == PROJECTION_VERSION:
            row['status'] = '本轮命令已完成；训练结果查看日志'
        elif (ROOT/RUN_OUT/'model_smoke'/f'{identifier}.json').exists():
            smoke = json.loads((ROOT/RUN_OUT/'model_smoke'/f'{identifier}.json').read_text())
            if smoke.get('completed') and smoke.get('projection_version') == PROJECTION_VERSION:
                row['status'] = '本轮整网单批次通过；正式训练待运行'
        elif spec['rerun'].startswith('同协议') and (ROOT/OUT/'execution_records'/f'{identifier}_run.json').exists():
            row['status'] = '历史有产物；按复用条件核对'
        if identifier == 'V12':
            path = ROOT/RUN_OUT/'projection_audit/report.json'
            if path.exists():
                audit = json.loads(path.read_text())
                counts = audit['counts']
                if (audit.get('projection_version') == PROJECTION_VERSION and
                        not counts['fail'] and not counts['review']):
                    row['status'] = '本轮CPU投影审计通过；GPU另看V13'
        if identifier == 'V14':
            path = ROOT/RUN_OUT/'model_integrity/report.json'
            if path.exists():
                audit = json.loads(path.read_text())
                if audit.get('passed') and audit.get('projection_version') == PROJECTION_VERSION:
                    row['status'] = '本轮CPU实际学习层/梯度审计通过；GPU另看V13'
        if identifier == 'V00':
            path = ROOT/RUN_OUT/'checks.json'
            if path.exists() and json.loads(path.read_text()).get('passed'):
                row['status'] = '本轮时序几何回归通过'
        rows.append(row)
    with (output/'experiments.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/'experiments.json').write_text(json.dumps(list(registry.values()), ensure_ascii=False, indent=2)+'\n')
    text = [f'# RadarOcc主干投影修正后：从头训练安排表（{SERIES}）', '',
        '本表在原V/P/T安排表上更新，保留V00～V11、P01～P07、T00～T14编号；新增V12/V13/V14检查和T15无对齐memory对照。', '',
        '**本轮所有occupancy模型从头训练，T00单帧也必须重跑。** 沿用之前20%连续窗口、seed42、15 epochs协议；“从头”指不加载旧模型，不代表把数据改成全量训练集。', '',
        f'共同主干版本：`{PROJECTION_VERSION}`；训练目录：`{WORK}/<method>/`。',
        f'新检查/单批次记录/融合消融：`{RUN_OUT}/`；旧日志、checkpoint和离线报告保留。', '',
        '## 新RadarOcc代码和各时序方案在哪里', '',
        '- [Small主干源码](../../projects/occ_plugin/occupancy/detectors/radarocc_self_small.py)：角度反号、采样[R,A,E]、BNC重排及残差投影；替换的旧代码留作注释。',
        '- [稀疏编码器](../../projects/occ_plugin/occupancy/voxel_encoder/sparse_lidar_enc.py)：Range Attention后过滤越界坐标及对应特征。保持56层稀疏高度和14层输出，E原始bin仅5～32进入spconv，其余仍参与前面的Range Attention。',
        '- [Cross-Attention批次](../../projects/occ_plugin/occupancy/deformable_transformer/encoder_3D.py)：reference和value均为B，注意力入口检查形状。',
        '- 本轮模型复查还修正位置编码轴序、3D注意力初始化、FPN检查点重复更新BN、T14零power编码及损失/GT忽略边界；详见[模型审计报告](../../'+RUN_OUT+'/model_integrity/REPORT.md)。实际注册的occupancy head为occ_head_fp16.py，当前仍按FP32运行。',
        '- [共享训练配置](base.py)和[全部模型配置](configs/)：T00是修正后的RadarOcc Small；其他T项在同一主干上接入时序。',
        '- 当前表针对Small及继承它的时序模型；Full/Strict不在本轮安排中。CenterFormer/RODNet是移植的时序模块，最终任务仍为occupancy。', '',
        '共同执行顺序：原始NPZ → 当前/历史P1选点 → 时序融合入口（此前讨论的P0，代码标作P1）→ Range Attention → Sparse Encoder → 球坐标Self-Attention → 修正后的Cartesian投影 → occupancy。', '',
        '| 方案 | 配置 | 时序实现 |', '| --- | --- | --- |',
        '| T00单帧 | [current_only.py](configs/current_only.py) | 不融合历史帧 |',
        '| T01普通KNN、T02距离Top8 | [knn.py](configs/knn.py)、[distance_top8.py](configs/distance_top8.py) | [local_attention.py](../fusion/local_attention.py) |',
        '| T03/T04/T05 LearnedKNN及预算 | [learned_knn.py](configs/learned_knn.py)、[learned_q32.py](configs/learned_q32.py)、[learned_q64.py](configs/learned_q64.py) | [learned_topk.py](../core/learned_topk.py)与local_attention |',
        '| T06～T09 power区域/Z映射 | [box_z1.py](configs/box_z1.py)及box_z0/2/3.py | [box_z_p0.py](../fusion/box_z_p0.py) |',
        '| T10 CenterFormer | [centerformer.py](configs/centerformer.py) | [centerformer/](../centerformer/) |',
        '| T11 RODNet TDC | [rodnet_tdc.py](configs/rodnet_tdc.py) | [rodnet/](../rodnet/) |',
        '| T12/T13 RA-ADTA | [ra_adta.py](configs/ra_adta.py)、[ra_adta_radius.py](configs/ra_adta_radius.py) | [ra_adta.py](../fusion/ra_adta.py) |',
        '| T14近场屏蔽、T15无对齐memory | [knn_mask37.py](configs/knn_mask37.py)、[unaligned_memory.py](configs/unaligned_memory.py) | power_mask与[unaligned_memory.py](../fusion/unaligned_memory.py) |', '',
        '## 先做哪些：建议按阶段执行', '',
        '| 阶段 | ID | 做什么、目的 |', '| --- | --- | --- |',
        '| 0：公共检查 | V00 → V12/V14 → V13 | 时序几何、CPU投影、真实学习层与梯度、CUDA整网单批次；CPU不能代替GPU验证 |',
        '| 1：基本对照 | T00 → T01 | 先得到新单帧对照，再检验普通KNN是否带来收益；T01前核对V11 |',
        '| 2：LearnedKNN | P01/P02 → P03 → P04；T02/T03/T04 | 重建或核对监督、训练新ranker；比较同候选池距离排序/学习排序以及Q32预算 |',
        '| 3：其他主方法 | T07、T10、T11、T12 | 分别比较power区域、CenterFormer、RODNet TDC和RA-ADTA；各自准备步骤见下表 |',
        '| 4：补充消融 | T05/T06/T08/T09/T13/T14/T15 | 查询预算、高度映射、搜索方式、近场屏蔽、无对齐memory；按研究问题选做 |', '',
        'CUDA反向梯度也已修正：跨机器同步源码后，先重编译三维采样扩展。V13会核对.so的源码指纹，拒绝旧二进制；梯度失败仍会中止，未放宽检查。',
        '```bash', 'source use_radarocc5060.sh',
        '(cd third_party/VoxFormer/deform_attn_3d && FORCE_CUDA=1 MAX_JOBS=2 python setup.py build_ext --inplace)',
        '```', '',
        '第一轮主训练共9项：T00/T01/T02/T03/T04/T07/T10/T11/T12；其余7项是按需要补的对照。单卡逐项执行，不要求一次把全部模型跑完。',
        'V/P用于检查和准备，不产生occupancy训练结果；只有P03训练独立ranker，只有T项训练完整模型。命令只执行指定ID，不会自动运行前置项，也不会自动开始下一模型。', '',
        '```bash', 'source use_radarocc5060.sh',
        'python tools/temporal/run_geometry_v2.py run V00',
        'python tools/temporal/run_geometry_v2.py run V12',
        'python tools/temporal/run_geometry_v2.py run V14  # CPU：实际学习层/损失/梯度与16配置初始化',
        'python tools/temporal/run_geometry_v2.py run V13  # 需要GPU；失败时先解决该问题',
        'python tools/temporal/run_geometry_v2.py run T00 --dry-run',
        'python tools/temporal/run_geometry_v2.py run T00',
        '```', '',
        '## 哪些必须重做、哪些可以复用', '',
        '| 产物 | 本轮处理 | 判断条件 |', '| --- | --- | --- |',
        '| 所有旧occupancy checkpoint/验证成绩 | 所有T模型重新训练 | 单帧投影也已改变；旧权重不能用于resume或作为新主干的公平对照 |',
        '| V00、V12、V13、V14 | 必须重做 | 直接检查当前源码；无GPU时V13尚未完成 |',
        '| P01/P02 ranker监督、P05/P06区域配对 | 有合格v2产物可复用，否则重建 | 坐标为kradar_physical_v2，manifest完整，标注/角度表/策略/查询预算匹配，非smoke产物 |',
        '| P03/P04 ranker训练和评分 | 本轮默认在新目录重新训练/验证 | P01/P02不依赖主干投影；旧正号ranker不能复用；本轮配置统一指向新ranker |',
        '| V01～V11已完成的离线几何报告 | 相同v2协议可复用 | 主干修正未改变原始P1输入；坐标、帧范围、K、半径、提框/匹配参数改变时需重算 |',
        '| P07选点缓存 | 可跳过；若使用需随新ranker重建 | 默认在线选点；本轮新缓存路径与旧缓存隔离 |',
        '| 原始雷达NPZ、LiDAR occupancy GT、位姿、数据划分 | 保留 | 本轮不重新生成GT，不反号GT；仅新模型投影与物理坐标统一 |', '',
        'V01/V10及此前scene3方向框/候选64报告属于test场景历史诊断，不是必须重复的训练步骤，也不能再用该场景选择新参数。',
        'P01/P02沿用完整train/val的旧ranker协议，额外使用完整训练集bbox监督；occupancy训练本身为20%。这不是整个方法总共只用20%标签的协议。',
        'power配对仍使用预设B门限（Y=4/5/3m，中心距离=4/6/8m）；V04只在train内选参。若改门限，须同步缓存和配置，不自动采用扫描最优值。', '',
        '## 模型之间具体比什么', '',
        '| 对照 | 回答的问题 |', '| --- | --- |',
        '| T00 ↔ T01 | 最基础的历史KNN融合能否超过修正后的单帧？ |',
        '| T02 ↔ T03 | 固定大对称框、64候选和Top8，仅学习排序有没有用？ |',
        '| T03 ↔ T04 ↔ T05 | 每range时序查询250/32/64的精度、时间和显存如何权衡？ |',
        '| T06 ↔ T07 ↔ T08 ↔ T09 | 同一power区域和历史配对下，高度映射方式有何影响？ |',
        '| T07/T10/T11/T12 ↔ T00/T01/T03 | 各时序方案在相同主干与训练协议下的总体效果 |',
        '| T12 ↔ T13；T01 ↔ T14 | RA-ADTA搜索方式、近场power屏蔽各自有什么作用？ |',
        '| T15 ↔ T00，再与T01对照 | 不显式自车对齐的历史memory能否提供收益？结构不同，不能只归因于对齐 |', '',
        '主要记录同一验证集的SSC_mean/SSC_Foreground/SC与SSC1/SSC2；同时记录耗时、显存和ranker额外预训练成本。单批次smoke的时间/显存不是完整训练基准。',
        '每项均按验证SSC_mean选best，验证命令仍评估同一val20。最终测试或全量数据实验另行固定协议；不要把当前验证成绩当作独立最终测试成绩。', '',
        '## 实验总表', '',
        '| ID | 实验 | 目的 | 本轮处理 | 直接前置 | 命令 |', '| --- | --- | --- | --- | --- | --- |']
    for row in rows:
        text.append(f'| {row["id"]} | {row["experiment"]} | {row["purpose"]} | {row["rerun"]} | {row["prerequisites"] or "—"} | `{row["command"]}` |')
    text += ['', '## 训练、检查和看结果的通用用法', '',
        f'新单帧日志在`{WORK}/current_only/*.log.json`，普通KNN在`{WORK}/knn/*.log.json`，其他方法目录见逐项说明。',
        f'新ranker权重为`{WORK}/ranker/best.pth`；所有新T模型的load_from/resume_from均为None。',
        'Q32/Q64只限制参与时序融合的当前查询，每range250个P1输入仍保留；不是只向主干输入32/64个点。', '',
        '```bash',
        'python tools/temporal/run_geometry_v2.py run T01 --dry-run  # 看完整训练命令',
        'python tools/temporal/run_geometry_v2.py run T01 --smoke    # 整网单批次，不训练权重',
        'python tools/temporal/run_geometry_v2.py run T01',
        f'python tools/temporal/run_geometry_v2.py val T01 --checkpoint {WORK}/knn/best_SSC_mean_epoch_N.pth',
        f'python tools/temporal/run_geometry_v2.py resume T01 --checkpoint {WORK}/knn/latest.pth',
        'python tools/temporal/run_geometry_v2.py collect', '```', '',
        '将N替换为实际best文件名。resume只用于本轮被中断的训练，不用于旧正号/旧主干模型。',
        '配置和checkpoint保存projection_version；新入口会拒绝旧投影checkpoint。表格本轮指标与历史记录分开：experiments.csv的historical_record保留旧行，collect只扫描新训练目录。',
        '已有v2缓存按上表核对后可跳过生成步骤。V/P的--smoke仅用于小样本调试，不能替代正式缓存。', '',
        '## 只检查K=8及复核之前的候选64结论', '',
        'V01/V10默认跑K=8/32/64三组。若已有相同v2结果无需重复；只想重做K=8时可运行：', '',
        '```bash', 'python tools/temporal_vis/audit_dynamic_objects.py box --scene 3 \\',
        '  --start-order 80 --end-order 200 --stride 5 --max-frames 25 --k 8 \\',
        f'  --coordinate-convention {CONVENTION} --output-dir {RUN_OUT}/direction_boxes_k8_only', '```', '',
        '若复核此前correct_direction的候选8/32/64实验，保持相同帧、框长度4/6/8、反向0.5、Y±1.5、Z±1，显式加入--coordinate-convention kradar_physical_v2并使用新输出目录；不要混同T03实际使用的大对称框。', '',
        '## 同一新KNN checkpoint开/关融合', '',
        '```bash', 'python tools/temporal/run_geometry_v2.py ablate-knn \\',
        f'  --checkpoint {WORK}/knn/best_SSC_mean_epoch_N.pth', '```', '',
        f'结果写入`{RUN_OUT}/knn_fusion_ablation_<checkpoint名>/REPORT.md`。用同一新权重和val20先on再off；off仍是联合训练过的主干，不能替代独立T00。',
        '该命令不训练参数；旧投影checkpoint不允许混入本轮正式对比。', '',
        '## 每项做什么、为什么做、看什么及完整命令', '']
    for row in rows:
        text += [f'### {row["id"]} {row["experiment"]}', '',
            f'**运行定位：** {row["priority"]}。直接前置：{row["prerequisites"] or "无"}。', '',
            f'**本轮处理：** {row["rerun"]}。当前记录：{row["status"]}。', '',
            f'**目的：** {row["purpose"]}', '', f'**具体做什么：** {row["procedure"]}', '',
            f'**产物与怎么看：** {row["readout"]}', '', f'**对比谁：** {row["comparison"]}', '',
            f'**和后续实验的关系：** {row["next_step"]}', '', f'**GT怎么用：** {row["gt_usage"]}', '']
        if row['notes']:
            text += [f'**配置备注：** {row["notes"]}。', '']
        text += ['**运行命令：**', '', '```bash', row['command'], '```', '',
                 '**展开后的实际命令：**', '', '```bash', row['expanded_command']]
        for field in ('smoke_command', 'validation_command', 'resume_command'):
            if row[field]:
                text.append(row[field])
        text += ['```', '']
    (output/'EXPERIMENTS.md').write_text('\n'.join(text)+'\n')
    print('Experiment table:', output/'EXPERIMENTS.md')


def direction_sweep(smoke=False, large_box=False):
    for k in [8, 32, 64]:
        name = 'large_symmetric' if large_box else 'direction_boxes'
        directory = OUT+('/smoke' if smoke else '')+f'/{name}_k{k}'
        cmd = ['python', 'tools/temporal_vis/audit_dynamic_objects.py', 'box',
            '--scene', '3', '--start-order', '80', '--end-order', '200', '--stride', '5',
            '--max-frames', '1' if smoke else '25', '--k', str(k), '--output-dir', directory, *COORD]
        if large_box:
            cmd += ['--box-methods', 'symmetric', '--box-lengths', '8', '12', '16', '--box-reverse', '0']
        if (ROOT/directory).exists():
            raise FileExistsError('Use another output directory; v2 results are not overwritten: '+directory)
        run(cmd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['list', 'table', 'run', 'val', 'resume', 'direction-sweep', 'collect', 'ablate-knn', 'model-smoke'])
    p.add_argument('experiment', nargs='?')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--large-box', action='store_true', help='direction-sweep: original large symmetric box')
    p.add_argument('--index', type=int, default=0, help='model-smoke: real train/val sample index')
    p.add_argument('--output-dir', type=Path, default=ROOT/'temporal/experiments_v2')
    args = p.parse_args()
    if args.action == 'ablate-knn':
        if args.checkpoint is None:
            p.error('ablate-knn requires --checkpoint pointing to a v2 T01 checkpoint')
        if args.experiment is not None or args.smoke or args.large_box:
            p.error('ablate-knn takes --checkpoint, optional --output-dir and --dry-run only')
        output = args.output_dir
        if output == ROOT/'temporal/experiments_v2':
            output = ROOT/RUN_OUT/f'knn_fusion_ablation_{args.checkpoint.stem}'
        knn_ablation(args.checkpoint, output, args.dry_run)
        return
    registry = experiments()
    if args.action == 'list':
        for row in registry.values():
            print(row['id'], row['title'], '| 前置:', ','.join(row['depends']) or '无')
        return
    if args.action == 'table':
        document(args.output_dir); return
    if args.action == 'collect':
        collect(args.output_dir); return
    if args.action == 'direction-sweep':
        direction_sweep(args.smoke, args.large_box); return
    if args.experiment not in registry:
        p.error('Specify an experiment ID; use list')
    row = registry[args.experiment]
    if args.action == 'model-smoke':
        if row['kind'] != 'occupancy-training':
            p.error('model-smoke requires a T experiment ID')
        if args.dry_run:
            print(json.dumps(dict(id=row['id'], config=row['config'], series=SERIES,
                output=RUN_OUT+'/model_smoke/'+row['id']+'.json', optimizer_steps=0,
                required=row['required']), ensure_ascii=False, indent=2))
        else:
            model_smoke(row['id'], args.index)
        return
    command = row['command']
    if args.smoke:
        if row['id'] in ('V01', 'V10'):
            command = command+['--smoke']
        elif row['smoke_command']:
            command = row['smoke_command']
        else:
            p.error('This experiment has no smoke shortcut; use V00 or --dry-run')
    if args.action in ('val', 'resume'):
        if row['kind'] != 'occupancy-training' or args.checkpoint is None:
            p.error('val/resume require an occupancy-training ID and --checkpoint')
        if not args.dry_run:
            check_checkpoint(args.checkpoint, row['method'] == 'current_only')
        if args.action == 'val':
            command = ['bash', 'tools/dist_test.sh', row['config'], str(args.checkpoint), '1', '--eval', 'ssc']
        else:
            command = command+['--resume-from', str(args.checkpoint)]
    missing = [path for path in row['required'] if not (ROOT/path).exists()]
    if args.dry_run:
        print(shlex.join(command))
        if missing:
            print('Required artifacts not yet available:', ', '.join(missing))
        return
    if missing:
        raise FileNotFoundError('Run prerequisites '+', '.join(row['depends'])+' first; missing: '+', '.join(missing))
    if row['id'] in ('V04', 'V09'):
        report = json.loads((ROOT/row['required'][0]).read_text())
        if report['config'].get('coordinate_convention') != CONVENTION:
            raise ValueError('A v2 experiment must not inherit an old-coordinate reference report')
    if row['id'] in ('P03', 'P04'):
        for path in row['required']:
            if not path.endswith('manifest.json'):
                continue
            if json.loads((ROOT/path).read_text()).get('coordinate_convention') != CONVENTION:
                raise ValueError('Rebuild ranker cache with kradar_physical_v2 before training/evaluation')
    if args.action == 'run' and row['kind'] == 'occupancy-training':
        if (ROOT/row['work_dir']/'latest.pth').exists():
            raise FileExistsError('A run already exists; use resume with its v2 latest.pth')
    if args.action == 'run' and row['id'] == 'P03' and (ROOT/WORK/'ranker/best.pth').exists():
        raise FileExistsError('This series already has a ranker; evaluate it with P04 rather than overwriting it')
    if row['kind'] in ('occupancy-training', 'ranker-training') or row['id'] == 'P07':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; use CPU validation or --dry-run. No long training was started.')
    start = time.time()
    run(command)
    status_dir = ROOT/RUN_OUT/'execution_records'
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir/f'{row["id"]}_{args.action}{"_smoke" if args.smoke else ""}.json').write_text(
        json.dumps(dict(id=row['id'], action=args.action, smoke=args.smoke,
            elapsed_seconds=time.time()-start, command=command, completed=True,
            coordinate_convention=CONVENTION, series=SERIES,
            projection_version=PROJECTION_VERSION), ensure_ascii=False, indent=2)+'\n')


def collect(output):
    """Populate a new results file from real val logs; never invent unrun scores."""
    registry = experiments(); rows = []
    for spec in registry.values():
        if spec['kind'] != 'occupancy-training':
            continue
        work = ROOT/spec['work_dir']; values = []
        for path in work.glob('*.log.json'):
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('mode') == 'val' and 'SSC_mean' in row:
                    values.append(dict(row, log=str(path)))
        best = max(values, key=lambda r: r['SSC_mean']) if values else {}
        rows.append(dict(id=spec['id'], experiment=spec['title'], series=SERIES,
            status='本轮已有验证' if best else '本轮待运行',
            best_epoch=best.get('epoch', ''), SSC_mean=best.get('SSC_mean', ''),
            SC_non_empty=best.get('SC_non-empty', ''), SSC_Foreground=best.get('SSC_Foreground', ''),
            SSC1_mean=best.get('SSC1_mean', ''), SSC2_mean=best.get('SSC2_mean', ''),
            source_log=best.get('log', '')))
    output.mkdir(parents=True, exist_ok=True)
    with (output/'results_from_logs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(output/'results_from_logs.csv')


if __name__ == '__main__':
    main()
