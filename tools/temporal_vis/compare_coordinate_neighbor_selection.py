#!/usr/bin/env python3
"""Paired old/corrected temporal-neighbour inspection, without editing models.

Keep the native positive-angle basis for BOTH selector inputs. Only replace the
inconsistent history transform M by S M S. Render selected token identities at
their consistently transformed physical coordinates M S p, on both sides.
The frozen legacy ranker is reused; this is not a corrected/retrained model.
"""
import argparse
import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from temporal.core.ego_alignment import RadarBinGeometry, radar_to_lidar_default, align_to_current
from temporal.core.learned_topk import BoxTopKSelector
from temporal.core.sparse_topk import select_query_subset
from temporal.fusion.local_attention import local_neighbors
from tools.temporal_vis.vis_alignment import read_infos, load_p1
from tools.temporal_vis.p1_reflection_regions import RegionConfig, extract_regions

METHODS = [('knn', 'KNN：2m 内最近8点'), ('distance', 'Box64 → 距离Top8'),
           ('learned', 'Box64 → Learned Top8（原权重）')]
SIGN = np.array([1., -1., -1.])
COLORS = dict(common='#536273', old='#d94646', corrected='#176fd1', power='#e5a11a')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               default=serial, allow_nan=False)+'\n')


def choose_frames(annotation, seed):
    """Scene-uniform, then frame-uniform sampling, before inspecting any power."""
    scenes = defaultdict(list)
    for info in read_infos(annotation):
        scenes[str(info.get('scene_id', info['scene_token']))].append(info)
    eligible = {}
    for scene, rows in scenes.items():
        rows.sort(key=lambda x: int(x['order_in_scene']))
        sequences = []
        for pos in range(3, len(rows)):
            seq = rows[pos-3:pos+1]
            if not all(f.get('ego_pose_valid') and f.get('timestamp_valid')
                       and np.isfinite(f['timestamp_sec'])
                       and Path(f.get('sparse_radar_path') or f['radar_path']).is_file()
                       for f in seq):
                continue
            if not np.all(np.diff([f['timestamp_sec'] for f in seq]) > 0):
                continue
            sequences.append(seq)
        if sequences:
            eligible[scene] = sequences
    rng = np.random.default_rng(seed)
    choices = rng.choice(sorted(eligible, key=int), 5, replace=False).tolist()
    sequences = [eligible[s][int(rng.integers(len(eligible[s])))] for s in choices]
    sampling = dict(seed=seed, generator='numpy.default_rng / PCG64',
        rule='uniform 5 distinct eligible scenes; uniform current frame within each scene',
        annotation=str(annotation), annotation_sha256=digest(annotation),
        eligible_frames_by_scene={s: len(v) for s, v in eligible.items()},
        sequences=[dict(scene=s, orders=[int(f['order_in_scene']) for f in seq],
                        radar_frames=[int(f['radar_frame_idx']) for f in seq],
                        timestamps=[float(f['timestamp_sec']) for f in seq],
                        paths=[f.get('sparse_radar_path') or f['radar_path'] for f in seq])
                   for s, seq in zip(choices, sequences)])
    return sequences, sampling


def frame_data(info, geometry):
    path = Path(info.get('sparse_radar_path') or info['radar_path'])
    coords, _ = load_p1(path)
    with np.load(path, allow_pickle=False) as raw:
        features = raw['power_val'][:8, :175*250].T.copy().astype(np.float32)
    # Exactly the existing P1 per-range torch sort, including ties.
    order = torch.sort(torch.from_numpy(features[:, 2].copy()).reshape(175, 250),
                       descending=True, dim=1).indices.numpy()
    order = (order + np.arange(175)[:, None]*250).reshape(-1)
    return geometry.to_xyz(coords[order]), features[order], coords[order], order


def compare_ids(old, corrected):
    """Identity includes history slot, so points from different frames differ."""
    common = np.intersect1d(old, corrected)
    denom = max(len(old), len(corrected))
    union = len(np.union1d(old, corrected))
    return dict(old_count=len(old), corrected_count=len(corrected), shared_count=len(common),
                retention=len(common)/denom if denom else 1.,
                jaccard=len(common)/union if union else 1.,
                changed=not np.array_equal(np.sort(old), np.sort(corrected)))


def select(features, xyz, times, queries, selectors):
    frames = np.ones((1, 4), bool)
    point_mask = np.ones(features.shape[:3], bool)
    point_mask[0, -1] = False
    point_mask[0, -1, queries] = True
    ix, valid = local_neighbors(xyz, frames, k=8, radius_m=2., point_valid_mask=point_mask)
    result = {'knn': (ix[0, queries], valid[0, queries])}
    for name, selector in selectors.items():
        ix, valid = select_query_subset(selector, features, xyz, frames, times,
                                       None, [queries])
        result[name] = (ix[0], valid[0])
    return result


def setup_plotting():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    matplotlib.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
        'font.size': 10, 'axes.unicode_minus': False, 'figure.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False})


def rectangle(ax, low, high, axes=(0, 1), color=COLORS['power'], label=None, width=1.8):
    from matplotlib.patches import Rectangle
    a, b = axes
    ax.add_patch(Rectangle((low[a], low[b]), high[a]-low[a], high[b]-low[b],
                           fill=False, edgecolor=color, lw=width))
    if label:
        ax.text(low[a], high[b], label, color='#8b5700', fontsize=9,
                bbox=dict(facecolor='white', alpha=.85, edgecolor='none'), va='bottom')


def power_background(ax, xyz, power, ids, axes, size=3):
    from matplotlib.colors import Normalize
    a, b = axes
    values = np.log10(np.maximum(power[ids], 1e-20))
    low, high = np.percentile(values, [5, 99.5]) if len(values) else (0, 1)
    return ax.scatter(xyz[ids, a], xyz[ids, b], c=values, s=size, cmap='YlOrBr',
                      norm=Normalize(low, max(high, low+1e-6)), alpha=.65, linewidths=0)


def frame_overview(path, xyz, power, regions, result, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), layout='constrained')
    visible = np.flatnonzero(result['seed_point_mask'])
    for ax, dims in zip(axes, [(0, 1), (0, 2)]):
        artist = power_background(ax, xyz, power, visible, dims, 5)
        for j, region in enumerate(regions):
            rectangle(ax, region['box_min'], region['box_max'], dims,
                      label=f"R{j+1}  Q={region['query_id']}")
            q = xyz[region['query_id']]
            ax.scatter(q[dims[0]], q[dims[1]], marker='*', s=120, c='black', zorder=5)
        ax.set(xlabel='X 前方 (m)', ylabel='Y 左方 (m)' if dims[1] == 1 else 'Z 上方 (m)',
               xlim=(8, 70), ylim=(-35, 35) if dims[1] == 1 else (-4, 6))
        ax.set_aspect('equal' if dims[1] == 1 else 'auto')
        ax.grid(alpha=.15)
    fig.colorbar(artist, ax=axes, label='log10(max of stored Top3 powers)', shrink=.65)
    fig.suptitle(title+'\n当前帧高 power 点与回波区域框；星号为固定查询点。框不是 GT。', fontsize=15)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def detail_plot(path, xyz, features, physical_history, old_assumed, region,
                query_row, selections, title, history_slot=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(3, 4, figsize=(21, 13), layout='constrained')
    qid = region['query_id']
    query = xyz[qid]
    low, high = np.asarray(region['box_min']), np.asarray(region['box_max'])
    # Shared limits across old/corrected, expanded to include ALL selected points.
    gathered = [low[None], high[None], query[None]]
    chosen = {}
    slot_range = slice(None) if history_slot is None else slice(history_slot*8, (history_slot+1)*8)
    for method, _ in METHODS:
        for version in ('old', 'corrected'):
            ids, good = selections[version][method]
            ix = ids[query_row, slot_range][good[query_row, slot_range]]
            chosen[method, version] = ix
            gathered.append(physical_history[ix])
            if version == 'old':
                gathered.append(old_assumed[ix])
    all_points = np.concatenate(gathered)
    view_low, view_high = all_points.min(0)-.7, all_points.max(0)+.7
    view_low = np.minimum(view_low, query-[2, 2, 1.5])
    view_high = np.maximum(view_high, query+[2, 2, 1.5])
    context = np.flatnonzero(((xyz >= view_low) & (xyz <= view_high)).all(1))
    for row, (method, method_title) in enumerate(METHODS):
        old = chosen[method, 'old']; fixed = chosen[method, 'corrected']
        common = np.intersect1d(old, fixed)
        metric = compare_ids(old, fixed)
        for col, (version, dims) in enumerate([('old', (0, 1)), ('corrected', (0, 1)),
                                              ('old', (0, 2)), ('corrected', (0, 2))]):
            ax = axes[row, col]
            a, b = dims
            ax.scatter(xyz[context, a], xyz[context, b], c='#f0dfb5', s=3, alpha=.5, linewidths=0)
            rectangle(ax, low, high, dims)
            for frame, marker in enumerate(['s', '^', 'o']):
                ix = chosen[method, version]
                ix = ix[ix//len(xyz) == frame]
                for shared in (True, False):
                    subset = ix[np.isin(ix, common) == shared]
                    if not len(subset):
                        continue
                    color = COLORS['common'] if shared else COLORS[version]
                    p = physical_history[subset]
                    ax.scatter(p[:, a], p[:, b], s=45, marker=marker, c=color,
                               edgecolors='white', linewidths=.4, zorder=5)
                if version == 'old' and len(ix):
                    assumed = old_assumed[ix]; physical = physical_history[ix]
                    ax.scatter(assumed[:, a], assumed[:, b], s=44, marker=marker,
                               facecolors='none', edgecolors='#d78484', lw=.8, zorder=3)
                    for pp, aa in zip(physical, assumed):
                        ax.plot([aa[a], pp[a]], [aa[b], pp[b]], c='#d78484', alpha=.45, lw=.65)
            ax.scatter(query[a], query[b], marker='*', s=160, c='black', zorder=7)
            ax.set(xlim=(view_low[a], view_high[a]), ylim=(view_low[b], view_high[b]),
                   xlabel='X (m)', ylabel=('Y (m)' if b == 1 else 'Z (m)'))
            ax.set_aspect('equal')
            ax.grid(alpha=.17)
            ax.set_title(f"{'不修正' if version == 'old' else '修正对齐'} · {'XY' if b == 1 else 'XZ'}"
                         f" · n={len(chosen[method, version])}", fontsize=11)
        axes[row, 0].text(-.23, .5, method_title+f"\n邻居保留率 {metric['retention']:.1%}",
            transform=axes[row, 0].transAxes, rotation=90, ha='center', va='center', fontsize=11)
    legend = [Line2D([], [], marker='*', color='none', markerfacecolor='black', markersize=11, label='同一当前查询点'),
              Line2D([], [], color=COLORS['power'], label='当前帧高 power 区域框'),
              Line2D([], [], marker='o', color='none', markerfacecolor=COLORS['common'], label='两边都选中'),
              Line2D([], [], marker='o', color='none', markerfacecolor=COLORS['old'], label='仅旧算法选中'),
              Line2D([], [], marker='o', color='none', markerfacecolor=COLORS['corrected'], label='仅修正后选中'),
              Line2D([], [], marker='o', color='#d78484', markerfacecolor='none', label='旧算法误算的位置 → 实心为物理位置')]
    fig.legend(handles=legend, loc='outside lower center', ncol=3, fontsize=10)
    scope = '全部3帧历史（每帧最多8点）' if history_slot is None else f'历史槽 {history_slot-3}（最多8点）'
    fig.suptitle(title+f" · Q={qid} · {scope}\n"
        '两侧实心点均按同一物理坐标显示；□ 最旧帧 / △ 中间帧 / ○ 最近帧。', fontsize=15)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_plot(path, scene_results):
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(5, 3, figsize=(18, 22), layout='constrained')
    for row, result in enumerate(scene_results):
        with Image.open(path.parent/result['overview']) as im:
            axes[row, 0].imshow(im)
        axes[row, 0].axis('off')
        axes[row, 0].set_title(result['title'], fontsize=12)
        for col, field in enumerate(['retention', 'changed_fraction'], 1):
            values = [result['metrics'][m][field] for m, _ in METHODS]
            axes[row, col].barh(['KNN', 'Box距离Top8', 'Learned Top8'], values,
                               color=['#718b45', '#8e6fbe', '#327fbe'])
            axes[row, col].set_xlim(0, 1.15)
            for j, value in enumerate(values):
                axes[row, col].text(value+.02, j, f'{value:.1%}', va='center')
            axes[row, col].set_title('平均邻居保留率（越高越一致）' if col == 1 else '查询–历史帧对：邻居集合变化比例')
            axes[row, col].grid(axis='x', alpha=.15)
    fig.suptitle('随机5场景：坐标修正前后实际取点对比\n统计当前帧所有高 power 查询；每个查询分别比较3帧历史，等权平均。', fontsize=17)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def five_scene_selection_plot(output, scene_results):
    """One direct visual comparison for all five preselected R1 anchors."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(5, 4, figsize=(21, 23), layout='constrained')
    for row, meta in enumerate(scene_results):
        stem = Path(meta['overview']).stem.removesuffix('_power')
        with np.load(output/(stem+'_selection.npz')) as data:
            xyz = data['current_physical_xyz']; hist = data['history_physical_xyz']
            r = meta['regions'][0]; qid = r['query_id']; q = xyz[qid]
            qi = int(np.flatnonzero(data['query_ids'] == qid)[0])
            selected = {}
            for method in ['knn', 'learned']:
                for version in ['old', 'corrected']:
                    selected[method, version] = data[version+'_'+method+'_indices'][qi][
                        data[version+'_'+method+'_valid'][qi]]
            limits = np.concatenate([np.asarray([r['box_min'], r['box_max']]),
                                      *[hist[v] for v in selected.values()]])
            low, high = limits.min(0)-.6, limits.max(0)+.6
            context = np.flatnonzero(((xyz >= low) & (xyz <= high)).all(1))
            for col, (method, version) in enumerate([('knn', 'old'), ('knn', 'corrected'),
                                                    ('learned', 'old'), ('learned', 'corrected')]):
                ax = axes[row, col]
                common = np.intersect1d(selected[method, 'old'], selected[method, 'corrected'])
                ix = selected[method, version]
                ax.scatter(xyz[context, 0], xyz[context, 1], s=3, color='#efdfb9', alpha=.6, linewidths=0)
                rectangle(ax, np.asarray(r['box_min']), np.asarray(r['box_max']))
                for frame, marker in enumerate(['s', '^', 'o']):
                    for shared in [True, False]:
                        ids = ix[(ix//len(xyz) == frame) & (np.isin(ix, common) == shared)]
                        color = COLORS['common'] if shared else COLORS[version]
                        ax.scatter(hist[ids, 0], hist[ids, 1], marker=marker, s=55,
                                   c=color, edgecolors='white', linewidths=.35, zorder=4)
                ax.scatter(q[0], q[1], marker='*', s=170, c='black', zorder=6)
                stat = compare_ids(selected[method, 'old'], selected[method, 'corrected'])
                title = ('KNN' if method == 'knn' else 'Learned Top8')+(' · 不修正' if version == 'old' else ' · 修正对齐')
                ax.set_title(title+f"\n同一查询点的邻居保留率：{stat['retention']:.1%}", fontsize=12)
                ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel='X (m)', ylabel='Y (m)')
                ax.set_aspect('equal'); ax.grid(alpha=.15)
            axes[row, 0].text(-.23, .5, f"场景 {meta['scene']}\nradar {meta['radar_frames'][-1]:05d}\nR1 · Q={qid}",
                transform=axes[row, 0].transAxes, ha='center', va='center', rotation=90, fontsize=12)
    fig.legend(handles=[
        Line2D([], [], color=COLORS['power'], label='高 power 回波区域框'),
        Line2D([], [], marker='*', color='none', markerfacecolor='black', markersize=12, label='固定当前查询点'),
        Line2D([], [], marker='o', color='none', markerfacecolor=COLORS['common'], label='共同选中'),
        Line2D([], [], marker='o', color='none', markerfacecolor=COLORS['old'], label='仅不修正时选中'),
        Line2D([], [], marker='o', color='none', markerfacecolor=COLORS['corrected'], label='仅修正后选中')],
        loc='outside lower center', ncol=5, fontsize=11)
    fig.suptitle('随机五场景：旧 KNN / LearnedKNN 实际拿到哪些历史 token？\n'
        '每帧固定最高峰值功率区域 R1 中最强查询点；全部三帧历史。两侧点均画在统一物理坐标。\n'
        '这里的保留率仅针对图中单个查询；全帧高 power 查询统计见网页表格。', fontsize=17)
    fig.savefig(output/'five_scene_selection.png', dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ann-file', type=Path, default=ROOT/'data/annotations/kradar_dict_train_official_temporal_doppler8_20pct.pkl')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'work_dirs/p1_top8_ranker/best.pth')
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    setup_plotting()
    sequences, sampling = choose_frames(args.ann_file, args.seed)
    # Save random draw BEFORE loading any radar points or ranking differences.
    write_json(args.output/'sampling.json', sampling)
    print(json.dumps(sampling['sequences'], ensure_ascii=False), flush=True)
    geometry = RadarBinGeometry.from_file(ROOT/'data/K-Radar-official-meta/resources/info_arr.mat')
    ext = radar_to_lidar_default()
    selectors = dict(distance=BoxTopKSelector(method='distance'),
                     learned=BoxTopKSelector(method='learned', checkpoint=args.checkpoint))
    config = RegionConfig(x_limits=(8., 70.), y_limits=(-35., 35.), z_limits=(-4., 6.))
    results, csv_rows = [], []
    for seq in sequences:
        current = seq[-1]
        scene = str(current.get('scene_id', current['scene_token']))
        stem = f"scene{scene}_order{int(current['order_in_scene']):05d}"
        title = f"场景 {scene} · order {current['order_in_scene']} · radar {current['radar_frame_idx']:05d}"
        loaded = [frame_data(f, geometry) for f in seq]
        native = np.stack([f[0] for f in loaded])[None]
        features = torch.from_numpy(np.stack([f[1] for f in loaded])[None])
        poses = np.stack([np.asarray(f['ego_pose']) for f in seq])[None]
        times = np.array([[f['timestamp_sec'] for f in seq]])
        old, transforms = align_to_current(native, poses, ext)
        physical, _ = align_to_current(native*SIGN, poses, ext)
        fixed = physical*SIGN
        # Independent homogeneous composition for the corrected native transform.
        s = np.diag([1., -1., -1., 1.])
        direct = s @ transforms @ s
        check = np.einsum('btij,btnj->btni', direct[..., :3, :3], native)+direct[..., None, :3, 3]
        np.testing.assert_allclose(fixed, check, atol=1e-10, rtol=0)
        xyz = physical[0, -1]
        power = loaded[-1][1][:, :3].max(1).astype(float)
        proposals = extract_regions(xyz, power, config)
        # Region choice is independent of history, GT and before/after changes.
        regions = sorted([r for r in proposals['regions'] if r['accepted']],
                         key=lambda r: (-r['peak_power'], -r['support_count'], r['id']))[:3]
        if not regions:
            raise ValueError(f'{stem}: no accepted power regions; keep original random draw, do not resample')
        for r in regions:
            ids = r['support_point_ids']
            r['query_id'] = int(ids[np.argmax(power[ids])])
        high_queries = np.flatnonzero(proposals['seed_point_mask'])
        queries = np.unique(np.r_[high_queries, [r['query_id'] for r in regions]])
        selections = dict(old=select(features, old, times, queries, selectors),
                          corrected=select(features, fixed, times, queries, selectors))
        metrics, per_query = {}, []
        high_set = set(high_queries.tolist())
        for method, _ in METHODS:
            rows = []
            for qi, qid in enumerate(queries):
                for frame in range(3):
                    sl = slice(frame*8, (frame+1)*8)
                    oi, ov = selections['old'][method]
                    fi, fv = selections['corrected'][method]
                    stat = compare_ids(oi[qi, sl][ov[qi, sl]], fi[qi, sl][fv[qi, sl]])
                    entry = dict(scene=scene, order=int(current['order_in_scene']), method=method,
                        query_id=int(qid), history_slot=frame-3,
                        history_order=int(seq[frame]['order_in_scene']),
                        dt_sec=float(times[0, frame]-times[0, -1]),
                        high_power_query=int(qid) in high_set, **stat)
                    per_query.append(entry)
                    if int(qid) in high_set:
                        rows.append(entry)
            metrics[method] = dict(n_queries=len(high_queries), n_query_history_pairs=len(rows),
                retention=float(np.mean([r['retention'] for r in rows])),
                jaccard=float(np.mean([r['jaccard'] for r in rows])),
                changed_fraction=float(np.mean([r['changed'] for r in rows])),
                old_mean_count=float(np.mean([r['old_count'] for r in rows])),
                corrected_mean_count=float(np.mean([r['corrected_count'] for r in rows])))
            csv_rows.append(dict(scene=scene, order=int(current['order_in_scene']),
                                 radar_frame=int(current['radar_frame_idx']), method=method, **metrics[method]))
        overview = stem+'_power.png'
        frame_overview(args.output/overview, xyz, power, regions, proposals, title)
        physical_hist = physical[0, :3].reshape(-1, 3)
        assumed_hist = (old[0, :3]*SIGN).reshape(-1, 3)
        details = []
        for ri, region in enumerate(regions, 1):
            qi = int(np.flatnonzero(queries == region['query_id'])[0])
            images = {}
            for slot in [None, 0, 1, 2]:
                suffix = 'all' if slot is None else f'history{slot-3}'
                name = f'{stem}_R{ri}_{suffix}.png'
                detail_plot(args.output/name, xyz, features, physical_hist, assumed_hist,
                            region, qi, selections, title+f' · R{ri}', slot)
                images[suffix] = name
            details.append(dict(region_id=ri, source_proposal_id=region['id'],
                query_id=region['query_id'], query_original_npz_id=int(loaded[-1][3][region['query_id']]),
                box_min=region['box_min'], box_max=region['box_max'], support_count=region['support_count'],
                peak_power=region['peak_power'], images=images))
        displacement = np.linalg.norm(old[0, :3]-fixed[0, :3], axis=-1)
        result = dict(scene=scene, title=title, overview=overview, regions=details, metrics=metrics,
            orders=[int(f['order_in_scene']) for f in seq],
            radar_frames=[int(f['radar_frame_idx']) for f in seq],
            history_dt_sec=(times[0, :3]-times[0, -1]).tolist(),
            transform_difference_m=[dict(median=float(np.median(d)), p95=float(np.percentile(d, 95)))
                                    for d in displacement],
            proposal_config=proposals['config'], proposal_thresholds=proposals['thresholds'],
            input_sources=[dict(path=str(f.get('sparse_radar_path') or f['radar_path']),
                                sha256=digest(f.get('sparse_radar_path') or f['radar_path'])) for f in seq])
        write_json(args.output/(stem+'.json'), result)
        with (args.output/(stem+'_queries.csv')).open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=list(per_query[0])); writer.writeheader(); writer.writerows(per_query)
        payload = dict(query_ids=queries, high_power_query_ids=high_queries,
            current_physical_xyz=xyz, history_physical_xyz=physical_hist,
            old_assumed_physical_xyz=assumed_hist, native_coords=np.stack([f[2] for f in loaded]),
            sorted_to_original_npz=np.stack([f[3] for f in loaded]),
            relative_transforms=transforms[0], timestamps=times[0], current_power=power)
        for version in selections:
            for method in selections[version]:
                payload[version+'_'+method+'_indices'], payload[version+'_'+method+'_valid'] = selections[version][method]
        np.savez_compressed(args.output/(stem+'_selection.npz'), **payload)
        results.append(result)
        print(stem, json.dumps(metrics), flush=True)
    summary_plot(args.output/'summary.png', results)
    five_scene_selection_plot(args.output, results)
    with (args.output/'summary.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    manifest = dict(sampling=sampling, scenes=results, checkpoint=str(args.checkpoint),
        checkpoint_sha256=digest(args.checkpoint), script_sha256=digest(__file__),
        correction='native-basis selectors: old M p; corrected S M S p; same frozen ranker',
        display='all selected identities drawn at M S p; legacy assumed location shown hollow',
        power='max of three stored peak powers; proposals: 8m bands P95 seeds / P80 support plus MAD gates',
        region_selection='top 3 accepted regions by peak power, then support count; strongest member is anchor',
        metric='mean over all P95/MAD high-power queries and 3 histories; |intersection|/max(count_old,count_corrected); both empty=1',
        gt_used=False, model_or_cache_edited=False,
        limitations=['Same legacy ranker, no retraining; not corrected-method performance.',
                     'Neighbour changes and physical alignment conventions do not measure true object correspondence or IoU.',
                     'CPU FP32 ranker selection; near ties may differ on a GPU.',
                     'Power boxes are reflection proposals, not vehicle or ground-truth boxes.'])
    write_json(args.output/'manifest.json', manifest)
    render_html(args.output, results, sampling)
    print('OUTPUT', args.output, flush=True)


def render_html(output, results, sampling):
    sections = []
    for s in results:
        rows = ''.join('<tr><td>'+label+'</td><td>'+f"{s['metrics'][m]['retention']:.1%}"+
            '</td><td>'+f"{s['metrics'][m]['changed_fraction']:.1%}"+'</td></tr>' for m, label in METHODS)
        regions = []
        for region in s['regions']:
            main = region['images']['all']
            links = ' · '.join(f'<a href="{name}" target="_blank">{key}</a>' for key, name in region['images'].items())
            opened = ' open' if region['region_id'] == 1 else ''
            regions.append(f'<details{opened}><summary>R{region["region_id"]}：Q={region["query_id"]}；'
                f'{region["support_count"]} 个高 power 支撑点</summary><p>{links}</p><a href="{main}" target="_blank">'
                f'<img loading="lazy" src="{main}"></a></details>')
        sections.append(f'<section id="scene{s["scene"]}"><h2>{s["title"]}</h2>'
            f'<p>历史 order：{s["orders"][:3]}；时间差：{[round(v,3) for v in s["history_dt_sec"]]} 秒。'
            f'统计 {s["metrics"]["knn"]["n_queries"]} 个当前高 power 查询。</p>'
            f'<a href="{s["overview"]}" target="_blank"><img loading="lazy" src="{s["overview"]}"></a>'
            '<table><tr><th>方法</th><th>平均邻居保留率</th><th>查询–历史帧对的集合变化比例</th></tr>'+rows+'</table>'
            +''.join(regions)+'</section>')
    nav = ' · '.join(f'<a href="#scene{s["scene"]}">场景{s["scene"]}</a>' for s in results)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>坐标修正前后 KNN / LearnedKNN 取点对比</title><style>
body{font:16px/1.7 system-ui,sans-serif;color:#243247;background:#f5f7fb;margin:0 auto;max-width:1500px;padding:24px}
section,.card{background:white;border:1px solid #dce3ed;border-radius:12px;padding:22px;margin:24px 0}img{width:100%;height:auto}
a{color:#176fd1}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{padding:10px;border-bottom:1px solid #dce3ed;text-align:left}
summary{cursor:pointer;font-weight:650;padding:16px;background:#f0f4fa;margin-top:12px}h1{font-size:29px}nav{position:sticky;top:0;background:#fff;padding:12px;z-index:2}
</style><h1>随机5场景：坐标修正前后实际取点</h1>'''+f'<nav>{nav}</nav>'
    page += f'''<div class="card"><p>随机种子 <b>{sampling['seed']}</b>；从当前20%训练标注中均匀抽5个不同场景，再各抽1帧，使用该标注中的前三个历史条目。先保存抽样，再读取回波，不按效果重抽。</p>
<p><b>比较内容：</b>同一输入、同一高 power 框、同一查询、同一旧 ranker 权重；旧对齐 M 与坐标一致的 S M S。保留原正角度基底，避免同时改变 LearnedKNN 的位置输入基底。</p>
<p><b>读图：</b>金框＝当前帧高 power 回波区域；黑星＝查询；灰点＝两边共同选择；红点＝仅旧算法选中；蓝点＝仅修正后选中。实心历史点两边都画在正确物理位置。旧侧空心点和连线显示算法误算的位置。</p>
<p>高 power：每8m距离带 P95 和 MAD 门限种子、P80 支撑；选峰值功率最高的3个合格回波区域，每框最强点作为详细示例。统计使用全部高 power 查询，非只用示例点。完整候选检索仍使用全部43,750个历史token，不先滤高power。</p>
<p>每帧图中展开 R1–R3 查看 XY/XZ 对照；链接 all 为三帧合并，history-3/-2/-1 可单独查看。邻居保留率＝交集数量 / 两侧较大的有效数量，双空集合计1。框不是 GT 或车辆框。</p>
<p><b>结论边界：</b>这些图验证取点身份与坐标差异，不能单凭取点变化证明物体匹配或IoU提高。LearnedKNN尚未重训；这里只隔离几何对齐修正。</p>
<p><a href="five_scene_selection.png">五场景实际取点对照图</a> · <a href="sampling.json">抽样清单</a> · <a href="manifest.json">方法和全部来源</a> · <a href="summary.csv">统计CSV</a> · <a href="summary.png">五场景统计图</a></p></div>
<a href="five_scene_selection.png" target="_blank"><img src="five_scene_selection.png" alt="五场景实际取点对照图"></a>'''
    page += ''.join(sections)+'</html>'
    (output/'index.html').write_text(page)


if __name__ == '__main__':
    main()
