"""One source of truth for the experiment table and executable commands."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONVENTION = 'kradar_physical_v2'
SERIES = 'projection_fixed_20261112'
PROJECTION_VERSION = 'small_physical_model_audit_20261112'
WORK = 'work_dirs/geometry_v2/' + SERIES
RUN_OUT = 'outputs/geometry_v2/' + SERIES
TRAIN20 = 'data/annotations/kradar_dict_train_official_temporal_doppler8_20pct.pkl'
VAL20 = 'data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl'
TRAIN = 'data/annotations/kradar_dict_train_official_temporal_doppler8.pkl'
VAL = 'data/annotations/kradar_dict_val_official_temporal_doppler8.pkl'
TEST = 'data/annotations/kradar_dict_test_official_temporal_doppler8.pkl'
RANKER = WORK + '/ranker/best.pth'
OUT = 'outputs/geometry_v2'
COORD = ['--coordinate-convention', CONVENTION]


def python(script, *args):
    return ['python', script, *map(str, args)]


def entry(identifier, title, command, depends=(), required=(), note='', smoke=None):
    return dict(id=identifier, title=title, kind='validation', command=command,
                depends=list(depends), required=list(required), note=note, smoke_command=smoke)


def experiments():
    rows = [entry('V00', '坐标/位姿/GT/反向采样与旧模式回归',
        python('tools/temporal/check_geometry_v2.py', '--output', RUN_OUT+'/checks.json'),
        note='CPU，读取真实角度表/帧；不启动训练')]
    rows.append(entry('V12', '修正后Small投影链CPU审计',
        python('tools/temporal/audit_projection_chain.py', '--require-pass',
               '--output-dir', RUN_OUT+'/projection_audit'), ['V00'],
        note='执行当前Small源码；不读取旧checkpoint，不训练模型'))
    rows.append(entry('V13', '修正后单帧整网CUDA单批次检查',
        python('tools/temporal/run_geometry_v2.py', 'model-smoke', 'T00'), ['V12', 'V14'],
        note='真实训练/验证数据；整网前向、反向和推理；不保存权重，不执行长期训练'))
    rows.append(entry('V14', '实际学习层、初始化、损失和梯度CPU审计',
        python('tools/temporal/audit_model_integrity.py', '--require-pass',
               '--output-dir', RUN_OUT+'/model_integrity'), ['V00'],
        note='16配置实际构造/初始化；真实学习层缩小网格反向；不代替CUDA整网验证'))
    rows.append(entry('V01', '场景3：三种方向搜索框 × K=8/32/64',
        python('tools/temporal/run_geometry_v2.py', 'direction-sweep'), ['V00'],
        note='旧协议复现诊断，scene3属于test；禁止用它调最终参数'))
    rows.append(entry('V10', '场景3：大对称搜索框 × K=8/32/64',
        python('tools/temporal/run_geometry_v2.py', 'direction-sweep', '--large-box'), ['V00'],
        note='X半宽4/6/8m；旧scene3诊断，不用于新参数选择'))
    knn = python('tools/temporal_vis/audit_dynamic_objects.py', 'match', '--ann-file', TRAIN,
        '--scene', '11', '--start-order', '80', '--end-order', '200', '--stride', '5',
        '--radius', '2', '--k', '8', '--output-dir', OUT+'/knn_matching', *COORD)
    smoke = knn.copy(); smoke[smoke.index(OUT+'/knn_matching')] = OUT+'/smoke/knn_matching'; smoke += ['--max-frames', '1']
    rows.append(entry('V11', '普通KNN：GT同目标召回/精度与有效邻居', knn, ['V00'], smoke=smoke))
    motion = python('tools/temporal_vis/audit_neighborhood_motion.py', '--ann-file', TRAIN,
        '--scene', '11', '--start-order', '80', '--end-order', '200', '--stride', '5',
        '--output-dir', OUT+'/motion', *COORD)
    smoke = motion.copy(); smoke[smoke.index(OUT+'/motion')] = OUT+'/smoke/motion'; smoke += ['--max-frames', '1']
    rows.append(entry('V02', 'Doppler方向/静态与运动查询前置验证', motion, ['V00'], smoke=smoke))
    proposal = python('tools/temporal_vis/benchmark_region_matching.py', '--ann-file', TRAIN,
        '--scenes', '11', '--start-order', '80', '--end-order', '200', '--stride', '5',
        '--seed-percentile', '97', '--merge-space', 'xy', '--max-box-size-m', '20', '6', '6',
        '--gt-space', 'xy', '--match-space', 'xy', '--gt-purity', '0', '--min-gt-points', '1',
        '--output-dir', OUT+'/regions_train', *COORD)
    smoke = proposal.copy(); smoke += ['--max-frames-per-scene', '1']; smoke[smoke.index(OUT+'/regions_train')] = OUT+'/smoke/regions_train'
    rows.append(entry('V03', '训练场景power框、权重/匹配参考报告', proposal, ['V00'], smoke=smoke,
        note='仅train场景；GT用于评分，框与匹配不读GT'))
    rows.append(entry('V04', '区域门限扫描（训练场景内部时间留出）',
        python('tools/temporal_vis/sweep_region_matching_gates.py', '--reference-report', OUT+'/regions_train/report.json',
               '--output-dir', OUT+'/gate_sweep'), ['V03'], [OUT+'/regions_train/report.json'],
        note='继承v2参考报告；不读取旧scene3报告'))
    quality = python('tools/temporal_vis/audit_region_quality.py', '--ann-files', VAL,
        '--start-order', '80', '--end-order', '200', '--stride', '5',
        '--output-dir', OUT+'/region_abc_val', *COORD)
    smoke = quality.copy(); smoke += ['--max-frames-per-scene', '1']; smoke[smoke.index(OUT+'/region_abc_val')] = OUT+'/smoke/region_abc_val'
    rows.append(entry('V05', '区域门限A/B/C与严格GT纯度验证', quality, ['V00'], smoke=smoke,
        note='A=原门限，B=只放宽Y，C=放宽Y和中心；与V01方向框不同'))
    budget = python('tools/temporal_vis/visualize_query_budget.py', '--ann-files', VAL,
        '--scenes', '1', '2', '--output-dir', OUT+'/query_budget', *COORD)
    smoke = budget.copy(); smoke[smoke.index(OUT+'/query_budget')] = OUT+'/smoke/query_budget'; smoke += ['--frames-per-scene', '1', '--no-plots']
    rows.append(entry('V06', 'Q32/Q64预算、Doppler投影与GT覆盖', budget, ['V00'], smoke=smoke,
        note='Q32指每range最多32个时序查询，原单帧250个token仍保留'))
    rows.append(entry('V09', '冻结区域门限的跨验证场景检查',
        python('tools/temporal_vis/validate_region_matching_gates.py', '--reference-report', OUT+'/regions_train/report.json',
            '--ann-files', VAL, '--exclude-scenes', '11', '--y-gates', '4', '5', '3',
            '--center-gates', '4', '6', '8', '--output-dir', OUT+'/gates_cross_val'),
        ['V03'], [OUT+'/regions_train/report.json'], note='继承v2训练场景报告；固定B门限，禁止用旧报告替代'))
    for split, ann in [('train', TRAIN), ('val', VAL)]:
        path = 'data/geometry_v2/ranker/'+split
        command = python('tools/temporal/train_top8_ranker.py', 'prepare', '--ann-file', ann,
            '--split', split, '--output-dir', path, *COORD)
        if split == 'val':
            command += ['--max-queries-per-frame', '0']
        smoke = command.copy(); smoke[smoke.index(path)] = 'data/geometry_v2/smoke/ranker/'+split
        smoke += ['--scenes', '11' if split == 'train' else '1', '--max-frames-per-scene', '4', '--max-queries-per-frame', '32']
        rows.append(entry('P01' if split == 'train' else 'P02', '重建选点器'+split+'监督', command,
            ['V00'], note='保持旧ranker完整train/val预训练协议；occupancy仍20%', smoke=smoke))
    rows.append(dict(entry('P03', 'Learned Top8选点器独立预训练',
        python('tools/temporal/train_top8_ranker.py', 'train', '--train-cache', 'data/geometry_v2/ranker/train',
            '--val-cache', 'data/geometry_v2/ranker/val', '--work-dir', WORK+'/ranker',
            '--epochs', '20', '--seed', '42'), ['P01', 'P02'],
        ['data/geometry_v2/ranker/train/manifest.json', 'data/geometry_v2/ranker/val/manifest.json']), kind='ranker-training'))
    rows.append(entry('P04', '新ranker验证：距离Top8/学习Top8/候选64',
        python('tools/temporal/train_top8_ranker.py', 'evaluate', '--cache', 'data/geometry_v2/ranker/val',
            '--checkpoint', RANKER, '--output', RUN_OUT+'/ranker_val.csv'), ['P03'],
        [RANKER, 'data/geometry_v2/ranker/val/manifest.json']))
    for identifier, split, ann in [('P05', 'train20', TRAIN20), ('P06', 'val20', VAL20)]:
        target = 'data/geometry_v2/box_pairs/'+split
        command = python('tools/temporal/cache_box_z_pairs.py', '--ann-file', ann, '--output-dir', target, *COORD)
        smoke = command.copy(); smoke[smoke.index(target)] = 'data/geometry_v2/smoke/box_pairs/'+split; smoke += ['--max-frames', '2']
        rows.append(entry(identifier, '重建power框历史配对 '+split, command, ['V05'],
            note='无GT输入；默认固定B门限，尚不宣称新最优', smoke=smoke))
    rows.append(entry('P07', '选点缓存预热（可选）',
        python('tools/temporal/cache_top8_selection.py', 'temporal/experiments_v2/configs/learned_knn.py',
            '--checkpoint', RANKER, '--cache-dir', 'data/geometry_v2/'+SERIES+'/top8_selection'), ['P03'], [RANKER],
        note='需CUDA；训练默认在线选点，不依赖缓存预热'))
    rows.append(entry('V07', 'CenterFormer真实四帧前向/梯度',
        ['python', '-m', 'temporal.check_p0', '--method', 'centerformer', '--device', 'cpu', '--config',
         'temporal/experiments_v2/configs/centerformer.py'], ['V00']))
    rows.append(entry('V08', 'RODNet TDC真实四帧前向/梯度',
        ['python', '-m', 'temporal.check_p0', '--method', 'rodnet', '--device', 'cpu', '--config',
         'temporal/experiments_v2/configs/rodnet_tdc.py'], ['V00']))
    train = [('T00', 'current_only', '修正后当前帧对照（主干投影也已修改，需重训）', [], []),
        ('T01', 'knn', '普通球形KNN，2m，8点/历史帧', ['V11'], []),
        ('T02', 'distance_top8', '大对称框64→距离Top8公平对照', ['V00'], []),
        ('T03', 'learned_knn', 'LearnedKNN，全部时序查询', ['P04'], [RANKER]),
        ('T04', 'learned_q32', 'LearnedKNN，32查询/range', ['P04', 'V06'], [RANKER]),
        ('T05', 'learned_q64', 'LearnedKNN，64查询/range预算对照', ['P04', 'V06'], [RANKER])]
    for i in range(4):
        train.append((f'T{6+i:02d}', f'box_z{i}', f'power框选区域，Z{i}'+('主方案' if i==1 else '映射消融'), ['P05', 'P06'],
            ['data/geometry_v2/box_pairs/train20/manifest.json', 'data/geometry_v2/box_pairs/val20/manifest.json']))
    train += [('T10', 'centerformer', 'CenterFormer移植', ['V07'], []),
              ('T11', 'rodnet_tdc', 'RODNet TDC移植', ['V08'], []),
              ('T12', 'ra_adta', 'RA-ADTA各向异性KNN（原配置）', ['V00'], []),
              ('T13', 'ra_adta_radius', 'RA-ADTA radius-anchor消融（可选）', ['V00'], []),
              ('T14', 'knn_mask37', 'KNN近场power屏蔽消融（可选）', ['V00'], []),
              ('T15', 'unaligned_memory', '无显式自车对齐的历史memory对照（可选）', ['V00'], [])]
    for identifier, name, title, dependencies, required in train:
        config = f'temporal/experiments_v2/configs/{name}.py'
        rows.append(dict(id=identifier, title=title, kind='occupancy-training', method=name,
            config=config, command=['bash', 'tools/dist_train.sh', config, '1', '--seed', '42'],
            depends=list(dict.fromkeys(['V13', *dependencies])), required=required,
            note='修正主干；20%窗口；15 epochs；seed42；scratch；新目录',
            work_dir=WORK+'/'+name,
            smoke_command=python('tools/temporal/run_geometry_v2.py', 'model-smoke', identifier)))
    for row in rows:
        row.update(EXPERIMENT_DETAILS[row['id']])
        row['series'] = SERIES
        row['projection_version'] = PROJECTION_VERSION
        row['comparison'] = COMPARISONS.get(row['id'], '前置检查或数据准备，不直接比较occupancy IoU')
        if row['id'].startswith('T'):
            row['rerun'] = '本轮从头重训；不得resume旧投影checkpoint'
        elif row['id'] == 'P03':
            row['rerun'] = '本轮默认新目录重训ranker；合格v2权重不因主干投影自动失效'
        elif row['id'] in ('V00', 'V12', 'V13', 'V14', 'P04'):
            row['rerun'] = '本轮必须重做'
        elif row['id'] == 'P07':
            row['rerun'] = '可选；新ranker权重需要新的选点缓存'
        else:
            row['rerun'] = '同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做'
        # Keep detailed descriptions in the same registry, with paths matching
        # the executable config rather than the preceding training series.
        if row['id'].startswith('T'):
            row['readout'] = row['readout'].replace('work_dirs/geometry_v2/', WORK+'/')
        if row['id'] == 'P03':
            row['readout'] = row['readout'].replace('work_dirs/geometry_v2/ranker', WORK+'/ranker')
        if row['id'] == 'P04':
            row['readout'] = row['readout'].replace('outputs/geometry_v2/ranker_val.csv', RUN_OUT+'/ranker_val.csv')
    return {r['id']: r for r in rows}


def details(purpose, procedure, readout, next_step, priority, gt_usage):
    return dict(purpose=purpose, procedure=procedure, readout=readout,
                next_step=next_step, priority=priority, gt_usage=gt_usage)


# These explanations are part of the registry so regenerating the table keeps them.
EXPERIMENT_DETAILS = {
    'V14': details(
        '检查投影之外的初始化、位置编码、注意力、解码器、损失和实际梯度，避免只证明形状能接上。',
        '实际构造并初始化16个T配置；用独立八邻域oracle检查CPU三维采样；真实Self/Cross Attention、ResNet3D、FPN、OccHead在合成缩小网格前向/反向及临时SGD更新；检查checkpoint的BN统计、GT忽略边界、T14屏蔽、冻结ranker和全部序列因果关系。未执行spconv/CUDA，临时ranker不代表正式预训练完成。',
        RUN_OUT+'/model_integrity/REPORT.md和report.json；passed=true、failures/errors为空。whole_model_verified保持false，整网另看V13。',
        '通过后执行V13；所有T实验使用相同已修正学习层；不能用CPU梯度测试预测真实IoU。',
        '公共必做：模型数值与梯度', '合成GT检查损失/评价边界；真实GT只抽样统计，未训练实际模型。'),
    'V12': details(
        '确认新单帧主干的物理坐标、采样轴序、特征重排、残差投影及有效稀疏索引一致。',
        '执行当前Small源码的CPU方法，用独立物理公式、非对称位置编码网格、真实Scene3第80帧检查18项契约。学习层与稀疏卷积使用明确的CPU夹具，不把它称为整网验证。',
        RUN_OUT+'/projection_audit/REPORT.md和report.json；必须fail=0、review=0，CUDA是否运行另列。',
        '通过后运行V13；该审计只证明几何/张量对应，不预测修正后的IoU。',
        '公共必做：主干投影', 'GT用于加载轴序和评价器检查，不训练参数。'),
    'V13': details(
        '在长期训练前确认真实CUDA采样、spconv、新残差投影和occupancy损失能接通并得到有限梯度。',
        '先核对.so中的梯度版本/源码指纹，比较实际CUDA采样核与独立插值的前向/反向，再构造修正后的T00，从头初始化，取一个真实train样本做loss.backward，另取一个val样本推理。不执行optimizer.step，不加载或保存模型权重。首次同步CUDA梯度修正后须重编译扩展，不能只更新Python。',
        RUN_OUT+'/model_smoke/T00.json；失败先看projection_audit_cuda/cuda_failure.json中的stage/error，以及command_logs/内完整子进程日志。无CUDA时明确未完成；完整检查通过才有completed=true。',
        '所有T实验共用此主干检查；每种时序模型准备好产物后，还可运行run Txx --smoke检查自己的整网接线。',
        '公共必做：GPU整网接线', '训练样本GT用于一次occupancy损失反向；验证标签只用于评价输出。'),
    'V00': details(
        '确认这次修正确实统一了雷达物理坐标、LiDAR GT 和历史位姿，避免后面的实验继续用错坐标。',
        '运行坐标专项回归：真实角度表的 R/A/E→XYZ、非对称 GT 框归属、含转弯/横移的历史对齐、XYZ→原始网格反向采样、旧产物拒绝检查及配置解析；另取训练场景中的真实帧统计 GT 框内点数。CPU，不训练模型。',
        RUN_OUT+'/checks.json 的 passed、失败项和 actual_GT_frames。此项检查时序几何；单帧主干另看V12/V13。',
        '所有修正版实验的公共起点；若失败，先查失败原因，再做选点器和整网训练。',
        '公共必做', '读取 GT 做几何检查，不用 GT 训练参数。'),
    'V01': details(
        '重新回答：搜索历史点时，把框沿物体运动方向伸长，是否比对称框更容易找到同一物体的点？',
        '使用 test 中的场景3，当前帧序号80～200、间隔5，共25帧；比较等体积的对称框、正确方向框、反向框。方向来自 GT 位移，属于理想方向诊断。默认依次运行 K=8、32、64。',
        'outputs/geometry_v2/direction_boxes_k*/ 下的报告；看同目标命中、框外漏检、框内有正确点但 Top-K 没选中的比例及所选点纯度。结合样本数看结果。',
        '只用于重算之前的方向框结论；不是 T01/T03 必需输入，也不能把 GT 方向当成推理时已知条件。',
        '可选：复核历史结论', 'GT 同时提供理想运动方向和评分身份；不是可直接部署的选点方案。'),
    'V10': details(
        '重新检查 LearnedKNN 使用的大对称候选框，是否能覆盖足够多的同目标历史点，以及 K 大小带来多少差异。',
        '同样使用场景3的25帧，只比较大对称框；t-1/t-2/t-3 的 X 半宽分别4/6/8m，Y半宽1.5m、Z半宽1m。默认跑 K=8、32、64。',
        'outputs/geometry_v2/large_symmetric_k*/ 的覆盖、Top-K 命中和纯度；区分框本身漏掉目标与距离排序把正确点挤掉。',
        '辅助解释 T02/T03；不生成选点器权重或训练缓存。场景3来自 test，仅作历史诊断，不据此调新参数。',
        '可选：复核候选框', 'GT 用于判断同目标与评分；对称框构造和距离排序不读取 GT。'),
    'V11': details(
        '在花时间重训普通 KNN 前，确认修正后的2m近邻搜索实际能否选到同一目标的历史回波。',
        '在训练场景11、序号80～200、间隔5上做离线匹配；历史帧先做自车对齐，每个查询在2m内最多选8个近邻。不训练 attention 或 occupancy。',
        'outputs/geometry_v2/knn_matching/；看同目标召回、所选点精度、有效近邻数，以及 t-1/t-2/t-3 的退化。',
        '为 T01 提供选点质量解释；几何召回高不等于最终 occupancy IoU 一定高，后者由 T01 检验。',
        '普通KNN路线前置', 'GT 只用于评分，不参与 KNN 搜索。'),
    'V02': details(
        '检查 Doppler 加自车补偿能否可靠判断运动方向，解释方向框和 Q32 查询筛选是否有物理依据。',
        '使用训练场景11，比较对称框、GT理想方向、单个token的Doppler方向、邻域投票方向；同时统计 Doppler 符号假设及方向不确定的样本。',
        'outputs/geometry_v2/motion/；看有判定样本的方向准确率、反向率、不确定比例及相应历史点命中。方向准确率必须和判定覆盖率一起看。',
        '辅助解释 V06/T04/T05；不训练新方向网络，也不会自动修改 Q32 的筛选参数。',
        '建议：使用Doppler查询预算时复核', '实际 Doppler 预测不读 GT；GT 用于理想方向对照和事后评分。'),
    'V03': details(
        '检查从高power回波提取出的区域是否有目标意义，以及这些区域在历史帧之间能否配上。',
        '在训练场景11提取高power种子区域（97百分位），按XY组织区域并比较匹配代价权重，生成可复用参考报告。这里的框是回波区域框，不能直接视作车辆真实外形。',
        'outputs/geometry_v2/regions_train/report.json 及同目录报告；看区域数量、覆盖、匹配正确/错误/未知数、Recall和Precision。当前GT归属是宽松XY投影，至少1个点即可，需结合V05严格审计。',
        '给 V04 和 V09 提供相同区域提取配置及参考数据；不生成整网训练用的 P05/P06 配对缓存。',
        'power框路线前置', '提框和匹配不读 GT；GT 用于离线打分和比较权重。'),
    'V04': details(
        '检查历史区域匹配门限是否过紧或过宽，找到召回与错配之间的合理取舍。',
        '继承 V03 的训练场景和区域配置，扫描Y差、XY中心距离及代价权重。当前帧序号≤140用于选参，后续当前帧用于时间留出检查；相邻历史帧可能重叠，不是独立场景验证。',
        'outputs/geometry_v2/gate_sweep/ 的 gate_sweep.csv、summary.md 和推荐组合；同时看选参段与留出段的Recall、保守Precision/F1、正确目标被门限排除的数量。',
        '只输出参数研究结论，不会自动写回 P05/P06 或训练配置；采用新门限时，需要同步另建缓存和对应配置。',
        '可选：需要重新选择门限时做', 'GT 用于训练场景内的参数评分；匹配器输入不含 GT。'),
    'V05': details(
        '验证宽松XY投影下的高Recall是否掩盖了混入其他目标或背景的问题，并比较区域门限A/B/C。',
        '在验证集对比A原门限、B只放宽Y、C放宽Y和中心距离；同时用更严格XYZ归属检查区域支持点、区域框及配对纯度。这里的A/B/C是区域匹配门限，不是V01的三种方向框。',
        'outputs/geometry_v2/region_abc_val/summary.md、report.json及低纯度样例；看宽松/严格Recall、目标覆盖、support/box purity、保守Precision。宽松与严格Recall分母不同，不能直接作增减比较。',
        '解释power框匹配是否可信，是 P05/P06 的前置审计。默认仍固定B门限，不会根据本次验证集结果自动挑最优参数。',
        'power框路线前置', 'GT 只负责归属、纯度与评分，不用于提框、配对或训练缓存生成。'),
    'V06': details(
        '确认每range只选32/64个时序查询时，预算确实生效，以及省下计算时是否丢掉太多目标回波。',
        '在验证场景1、2上运行当前的Doppler/自车补偿/空间分布查询筛选，展示Q32和Q64的查询位置与GT覆盖。原单帧每range250个token和历史候选输入仍保留。',
        'outputs/geometry_v2/query_budget/；看各range查询数、筛选类别、空间分布、GT目标/点覆盖。这里检查选哪些当前查询，不是在评估历史邻居选得对不对。',
        '为 T04/T05 提供筛选正确性和覆盖解释；是否保持最终精度还要比较 T03/T04/T05。',
        'Q32/Q64路线前置', '查询筛选不读 GT；GT 仅用于可视化和覆盖评分。'),
    'V07': details(
        '在完整训练前，排除 CenterFormer 移植模块接线、数值和反向传播的问题。',
        '读取一组真实四帧雷达输入，在CPU上执行模块前向和两次小优化更新；检查初始恒等输出、有限数值、可学习偏移的非零梯度及输入未被原地修改。',
        '控制台检查结果：identity、finite_gradients、offset_gradient 和有效历史支持。两次更新使用描述符数值目标，不是occupancy训练。',
        '通过后再跑 T10；通过只说明该模块在这个样本上可计算、可求梯度，不说明整网GPU训练或最终IoU已验证。',
        'CenterFormer路线前置', '不加载occupancy GT；不使用bbox监督。'),
    'V08': details(
        '在完整训练前，排除 RODNet 时序可变形卷积TDC移植模块的接线和梯度问题。',
        '与V07使用同类真实四帧检查，在CPU上前向并做两次小优化更新，检查TDC偏移分支有梯度、输出有限和原始描述符未被修改。',
        '控制台的identity、finite_gradients、offset_gradient和历史支持统计；检查目标不是occupancy损失。',
        '通过后再跑 T11；这是模块可运行检查，不是检测效果、完整GPU训练或IoU证明。',
        'RODNet路线前置', '不加载occupancy GT；不使用bbox监督。'),
    'V09': details(
        '检查固定的区域门限在其他验证场景是否仍然有效，避免只在训练场景11上看起来合理。',
        '继承V03的区域配置，在验证场景比较原门限、只放宽Y、只改中心门限及固定B组合。B的t-1/t-2/t-3门限为Y=[4,5,3]m、XY中心距离=[4,6,8]m。',
        'outputs/geometry_v2/gates_cross_val/summary.md；看分场景和汇总Recall、保守Precision/F1、明确错配和未知身份数，留意是否只有个别场景受益。',
        '给P05/P06默认固定B门限提供跨场景检查；不会读取V04推荐值，也不会自动把验证集最优组合写回训练。',
        '建议：power框跨场景复核', 'GT 只用于评分；区域提取和匹配不读 GT。'),
    'P01': details(
        '用正确的雷达XYZ和GT归属，重新生成LearnedKNN选点器的训练样本，替换受旧坐标影响的监督。',
        '遍历完整训练集四帧窗口，生成历史对齐后的候选框、最多64候选及同一GT实例的正负标签，保存相对几何/描述符等特征；每帧每历史对默认最多512个合格查询。不训练参数。',
        'data/geometry_v2/ranker/train/manifest.json及分片；检查坐标版本、帧数、查询/候选数量、正负样本与排除统计。',
        '与P02一起供P03训练；必须重建，不能把旧ranker缓存复制到新目录当作修正后的数据。',
        'LearnedKNN路线必做', '使用完整训练集bbox产生监督标签；这是额外标注预算，不能称为整个方法只用20%标签。'),
    'P02': details(
        '重新生成正确坐标下的选点器验证样本，使新选点器能够在一致的验证集合上选best和比较。',
        '在完整验证集构造与P01同策略的历史候选和GT标签；默认不限制每帧合格查询数。不执行选点器参数更新。',
        'data/geometry_v2/ranker/val/manifest.json及分片；核对坐标、策略、样本数量，并确保train/val连同历史帧没有重叠。',
        '供P03每轮验证和P04最终选点比较；验证标签用于评分与选best，不用于梯度训练。',
        'LearnedKNN路线必做', '使用验证bbox提供评分标签，不是独立test结果。'),
    'P03': details(
        '重新训练一个能从64个候选中挑选8个历史点的打分器，修复旧坐标导致的错误监督。',
        '只训练独立ranker，用P01的GT正负标签监督候选打分；20轮、seed42，以P02验证Recall@8保存best。此步骤不训练RadarOcc occupancy主模型。',
        'work_dirs/geometry_v2/ranker/best.pth、每轮val CSV与best_val.csv；看训练损失及验证Recall@8，检查checkpoint记录kradar_physical_v2。',
        'P04检查通过后，将这个权重冻结供T03/T04/T05使用；三项occupancy训练共享这一次ranker预训练。',
        'LearnedKNN路线必做；仅训练选点器', '训练集bbox监督ranker损失；验证bbox负责评分和选best。'),
    'P04': details(
        '确认新ranker相对于纯距离排序到底改善了多少选点质量，区分候选池不足和排序失败。',
        '在同一P02验证缓存上比较距离Top8、学习Top8和整个候选Top64；不更新权重。候选64表示搜索候选池的覆盖参考，不是一个同成本的8点算法。',
        'outputs/geometry_v2/ranker_val.csv；比较同目标Recall、所选点Precision、不同历史间隔的结果与候选池可达覆盖。',
        '若64候选本来就找不到同目标，先查框/对齐；若候选有正确点但学习排序仍很差，先查ranker，再跑T03/T04/T05。无预设必须超过多少的阈值。',
        'LearnedKNN路线必做', 'GT 只用于评价已冻结ranker；推理输入没有GT标签。'),
    'P05': details(
        '为power框融合模型准备修正坐标后的训练帧区域与历史配对，避免训练时重复做区域提取。',
        '对20%训练连续窗口提取power区域、对齐历史并用固定B门限配对，保存current/history框、配对标记和逆变换；不训练参数，不读取GT。',
        'data/geometry_v2/box_pairs/train20/manifest.json及分片；检查complete、坐标版本、门限、帧数和GT未使用标记。',
        '供T06～T09共同读取；四个高度方案共享这份缓存。若改变匹配门限，需生成另一路径的缓存并同步配置。',
        'power框路线必做', '不读取bbox或occupancy GT，不在这里产生GT监督标签。'),
    'P06': details(
        '为power框融合模型准备与训练相同规则的验证配对，确保验证时没有换策略或使用GT框。',
        '对20%验证连续窗口执行与P05一致的无GT区域提取、历史对齐和固定B配对，不训练参数。',
        'data/geometry_v2/box_pairs/val20/manifest.json及分片；核对策略与train20一致、数据划分正确且缓存完整。',
        '与P05一起供T06～T09训练过程中的验证使用；不是使用验证GT修正搜索区域。',
        'power框路线必做', '配对生成不读GT；后续occupancy验证另用该划分的occupancy标签评分。'),
    'P07': details(
        '提前缓存冻结ranker的选点结果，减少后续反复在线打分的开销；不研究新算法。',
        '使用T03配置和P03权重，在CUDA上预计算选点并写入独立缓存。不更新ranker或occupancy模型。',
        'data/geometry_v2/top8_selection/中的缓存产物与完整性记录；关注是否命中、是否对应同一配置/坐标/权重。',
        '可以完全跳过：当前T03/T04/T05默认在线选点，执行P07不会自动把这些训练配置切换为读缓存。',
        '可选：工程加速', '冻结ranker推理不读GT。'),
    'T00': details(
        '提供相同20%数据、15轮训练协议下的当前帧对照，用于衡量时序模块的净收益。',
        '从头训练current-only RadarOcc，不融合历史帧；数据和优化设置与其他T实验保持同一套协议。',
        'work_dirs/geometry_v2/current_only/；以验证SSC_mean为主，辅看Foreground、SC、SSC1/SSC2，作为其他T实验对照。',
        '旧单帧权重不能替代新主干的独立训练对照。本轮配置已经自动使用独立work_dir，保留旧日志。',
        '主对照：修正主干后重训', '使用occupancy训练标签；无额外ranker bbox预训练。'),
    'T01': details(
        '重算最基础几何时序融合在正确坐标下能带来多少occupancy收益。',
        '历史帧自车对齐后，对每个当前查询在2m球内每历史帧最多选8个近邻，再用attention融合；从头联合训练时序模块和occupancy网络。无独立选点器。',
        'work_dirs/geometry_v2/knn/的验证日志与best；和同协议T00比较最终SSC_mean/Foreground，结合V11解释邻居质量。',
        '这是你列出的普通KNN重跑主项；也作为T14屏蔽实验和其他时序模型的基础参照。',
        '主训练：普通KNN', 'occupancy损失使用训练标签；近邻搜索本身不读GT。'),
    'T02': details(
        '给LearnedKNN建立同候选框、同64候选、同8点数量的纯距离对照，隔离学习排序的作用。',
        '使用与T03相同的大对称历史框和最多64候选，按距离选8点；其余融合与训练设置按对应配置运行，不加载ranker。',
        'work_dirs/geometry_v2/distance_top8/；重点与T03比较，才能回答学习排序是否改善最终occupancy。T01的球形候选域不同，T01→T03差异不能只归因于ranker。',
        '研究LearnedKNN改进来源时建议跑；不必先训练P03才能做这项。',
        '建议：LearnedKNN公平对照', 'occupancy损失使用标签；框内搜索和距离排序不读GT。'),
    'T03': details(
        '重算完整LearnedKNN方案在正确GT监督和几何坐标下的最终效果。',
        '冻结P03新ranker，从每个历史帧最多64候选中选8点；不启用Q32/Q64当前查询预算，随后训练时序融合与occupancy主模型。',
        'work_dirs/geometry_v2/learned_knn/；与T02比较学习选点收益，与T04/T05比较查询预算影响；独立记录显存和迭代时间才能评估计算代价。',
        '这是你列出的LearnedKNN主训练，也作为Q32/Q64预算方案的全查询参照；不能使用旧坐标ranker权重。',
        '主训练：LearnedKNN', '训练使用occupancy标签；额外bbox监督已在P03使用，冻结选点时不读GT。'),
    'T04': details(
        '验证每range只让32个当前查询参与LearnedKNN融合，是否能在减少时序计算时保留最终精度。',
        '在T03基础上启用Q32查询筛选；只对被选查询搜索历史8点并融合，其余当前token仍保留给单帧主网络。使用相同新ranker，从头训练occupancy模型。',
        'work_dirs/geometry_v2/learned_q32/；与T03/T05比较SSC_mean、Foreground和实际时间/显存，结合V06目标覆盖解释精度变化。',
        '对应你说的“每个range保留32个点”时序预算实验；实际不是把整个原始输入压成32点，也不是每查询只留32个历史邻居。',
        '主训练：Q32', '查询筛选与冻结ranker不读GT；使用occupancy训练标签和P03已有bbox预训练。'),
    'T05': details(
        '提供Q32与全查询之间的中间预算，判断32是否过少以及增至64能否恢复精度。',
        '与T04相同，只把每range时序查询上限从32改为64；同一ranker、相同每查询历史Top8和occupancy训练协议。',
        'work_dirs/geometry_v2/learned_q64/；与T03/T04共同观察预算—精度—计算量取舍。不要仅因查询数减半就宣称总训练耗时减半。',
        '预算消融项；如果当前只想重跑你关注的Q32，可以先跳过，之后再补。',
        '可选：查询预算对照', '与T04一致；筛选过程不读GT。'),
    'T06': details(
        '检查在高度不确定时，沿历史区域的整个高度多取样，是否比指定单一对应高度更有效。',
        'power框配对后，XY按当前/历史框的中心与尺寸映射；Z0在历史框内均匀取8个高度，有效样本聚合成该历史帧特征，再训练融合与occupancy。',
        'work_dirs/geometry_v2/box_z0/；与T07～T09比较SSC和Foreground，并看z_valid_sampling_rate、有效融合查询数，区分采不到回波与融合无收益。',
        '高度映射消融；与其他Z方案共享P05/P06，框和配对不变，只改变高度取样方式。',
        '可选：power框Z0消融', '提框和配对无GT；网络使用occupancy标签训练。'),
    'T07': details(
        '检验power框先定位并配对区域，再保持物理高度采历史回波的主要方案。',
        '使用P05/P06；XY按两框映射，Z1直接保留当前查询的物理Z，不按框的高度比例缩放。通过逆自车变换回到历史雷达网格采样，再训练融合与occupancy。',
        'work_dirs/geometry_v2/box_z1/；与T00/T01/T03比较总体收益，与T06/T08/T09比较高度处理；关注采样有效率和GT-free区域匹配审计。',
        '这是你列出的“按回波power框选目标”的主要重跑项。power区域是雷达回波区域，不能直接当成GT车辆长方体。',
        '主训练：power框Z1', '提框和配对不使用GT；occupancy损失使用训练标签。'),
    'T08': details(
        '验证按当前框与历史框的相对高度比例建立对应，是否优于Z1保持绝对高度。',
        '与T07共享框、配对和XY映射；Z2把查询在当前框内的相对高度，按历史框中心和高度尺寸映射过去，不额外学习高度变换参数。',
        'work_dirs/geometry_v2/box_z2/；重点对比T07的SSC、Foreground和有效采样率，判断回波区域高度变化是否值得做比例映射。',
        '用于区分固定物理Z与框归一化Z两种假设；如果只跑主power方案，可后补。',
        '可选：power框Z2消融', '框和对应不读GT；使用occupancy标签训练。'),
    'T09': details(
        '检查固定框高度映射不够准确时，学习额外高度缩放与平移能否带来收益。',
        '在Z2框比例映射上增加可学习高度缩放exp(alpha)和偏移beta，并对二者加正则；共享P05/P06，从头训练融合与occupancy。',
        'work_dirs/geometry_v2/box_z3/；看SSC/Foreground、有效采样率、z_alpha_mean、z_shift_mean、z_scale_mean及相应正则损失。参数有变化本身不等于效果变好。',
        '和T08对比学习高度修正的增益；和T07比较最终是否值得增加这份复杂度。',
        '可选：power框Z3消融', 'occupancy标签训练映射参数；无GT框指导采样位置，另有参数正则。'),
    'T10': details(
        '评估移植的CenterFormer时序注意力，相比KNN或power框融合是否更有效。',
        '四帧先按物理坐标对齐、组织特征网格，以当前token为query做空间感知融合和多尺度可变形注意力，再接回现有前端位置；从头训练整个occupancy模型。',
        'work_dirs/geometry_v2/centerformer/；比较SSC/Foreground及实际训练成本，结合V07检查与有效历史支持判断模块是否工作。',
        '属于之前移植方案的修正版重跑；移植的是时序组件并做了适配，不是完整CenterFormer检测器，没有新增检测热图选点头。',
        '补充主训练：移植CenterFormer', '使用occupancy标签；不额外训练CenterFormer检测头或使用GT热图选query。'),
    'T11': details(
        '评估RODNet的TDC时序可变形卷积，在当前雷达输入上是否比逐点KNN attention更合适。',
        '四帧物理对齐后构成特征网格，按物理高度切片执行共享权重的因果TDC，学习空间采样偏移，再采回当前查询并训练occupancy。只用当前和历史帧。',
        'work_dirs/geometry_v2/rodnet_tdc/；和T10/T01等比较SSC/Foreground及训练代价，检查TDC偏移和有效历史支持。',
        '属于之前RODNet移植方案重跑；当前移植的是TDC，不是完整RODNet，也不是额外的Doppler速度预测网络。',
        '补充主训练：移植RODNet TDC', '使用occupancy训练标签；不使用未来帧或检测GT作为融合输入。'),
    'T12': details(
        '重新评估之前RA-ADTA方案：按雷达方向分辨率组织各向异性邻域、学习采样位置是否改善occupancy。',
        '使用修正物理坐标及历史自车对齐，保持此前anisotropic_knn搜索、8候选、每头2个采样点等参数，从头训练RA-ADTA与occupancy模型。',
        'work_dirs/geometry_v2/ra_adta/；与T01/T03等比较最终SSC/Foreground，辅看有效采样及训练开销。此项不是在P03 ranker上继续训练。',
        '补齐先前依赖雷达XYZ的另一条方案；如果此前做过该方案，应按修正版本重跑才可比较。',
        '补充主训练：RA-ADTA', '使用occupancy标签；无需bbox ranker预训练。'),
    'T13': details(
        '单独检验RA-ADTA的候选搜索方式，判断原各向异性KNN设置是否限制了采样效果。',
        '与T12保持同样主要网络和采样设置，只把search_mode换为radius_anchor，独立从头训练。',
        'work_dirs/geometry_v2/ra_adta_radius/；主要与T12比较最终SSC/Foreground、有效历史支持和计算代价。',
        '搜索方式消融，不是修正坐标必需的新模型；先完成T12再按需要补。',
        '可选：RA-ADTA消融', '使用occupancy训练标签；不依赖P03选点器。'),
    'T14': details(
        '检查近车区域强回波是否干扰普通KNN融合，屏蔽它们能否改善occupancy。',
        '在T01基础上，对当前与历史各帧水平距离≤3.7m的token屏蔽power通道，并将这些点标为无效候选；其余2m/8近邻和训练协议保持对应设置，从头训练。',
        'work_dirs/geometry_v2/knn_mask37/；主要与T01比较SSC/Foreground及近场效果，同时留意屏蔽是否损失近处真实目标信息。',
        '可选的近场处理消融；3.7m是预设对照半径，不是本次修正后已证明的最优值。',
        '可选：近场屏蔽消融', '屏蔽只由几何距离决定，不读GT；训练使用occupancy标签。'),
    'T15': details(
        '补回之前无显式自车对齐的历史memory方案，检验历史描述符本身是否能提供收益。',
        '同样使用四帧和修正后的Small主干；将历史token压为256个memory token，再更新当前描述符，不按物理XYZ执行自车对齐。独立从头训练。',
        'work_dirs/geometry_v2/unaligned_memory/；和T00比较历史特征的净收益，和T01/T03比较精度及计算开销。',
        '这是不同结构的时序对照；与KNN的差异不能全部归因于是否做自车对齐。',
        '可选：补回已有无对齐方案', '使用occupancy标签；memory选择不使用GT。'),
}

COMPARISONS = {
    'T00': '所有T模型的共同单帧对照；首次必须重新训练',
    'T01': 'T01 vs T00：基础KNN时序收益',
    'T02': 'T03 vs T02：同候选64/保留8点，隔离学习排序的作用；T02 vs T01改变了候选搜索域',
    'T03': 'T03 vs T02检验学习排序；T03 vs T00检验完整方案',
    'T04': 'T04 vs T03/T05：Q32/Q250/Q64的精度、时间和显存',
    'T05': 'T05 vs T04/T03：同ranker、同候选，仅改变当前查询预算',
    'T06': 'T06/T07/T08/T09：同区域/配对缓存，仅比较Z映射方案',
    'T07': 'T07 vs T00/T01/T03：power区域融合；另和Z0/Z2/Z3对照',
    'T08': 'T08 vs T07：相对高度映射 vs 保持物理高度',
    'T09': 'T09 vs T08：学习高度缩放/平移的增益',
    'T10': 'T10 vs T00/T01/T03：当前主干上的CenterFormer时序移植',
    'T11': 'T11 vs T10/T01/T00：当前主干上的RODNet TDC移植',
    'T12': 'T12 vs T01/T03：RA-ADTA采样方案',
    'T13': 'T13 vs T12：仅替换RA-ADTA搜索方式',
    'T14': 'T14 vs T01：近场power屏蔽；不是新增独立主方法',
    'T15': 'T15 vs T00：无显式对齐的历史memory；与KNN比较时同时存在结构差异',
}
