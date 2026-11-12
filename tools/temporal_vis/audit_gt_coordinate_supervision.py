#!/usr/bin/env python3
"""Audit actual persisted ranker targets and GT membership, preserving old data."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from temporal.core.ego_alignment import RadarBinGeometry, radar_to_lidar_default
from tools.temporal_vis.audit_dynamic_objects import transform
from tools.temporal.train_top8_ranker import ranker_memberships
from tools.temporal_vis.vis_p1_reflection_regions import load_gt
from tools.temporal_vis.compare_coordinate_neighbor_selection import (
    frame_data, setup_plotting, write_json, digest)


def label_tokens(labels, ids):
    tokens = np.full(len(labels), '__background__', dtype='<U64')
    tokens[labels == -2] = '__ambiguous__'
    good = labels >= 0
    tokens[good] = ids[labels[good]]
    return tokens


def feature_lookup(features):
    result = defaultdict(list)
    for i, row in enumerate(features):
        result[row.tobytes()].append(i)
    return result


def cache_audit(cache_root, shard, frames, old_tokens, fixed_tokens, old_xyz, poses, ext):
    """Match saved float32 descriptors to source points, not guessed indices."""
    with np.load(cache_root/shard['file']) as data:
        cache = {k: data[k] for k in data.files}
    hist_slot = 3-int(shard['slot'])
    qlookup = feature_lookup(frames[-1][1])
    hlookup = feature_lookup(frames[hist_slot][1])
    M = np.linalg.inv(poses[-1] @ ext) @ (poses[hist_slot] @ ext)
    aligned = transform(old_xyz[hist_slot], M)
    query_ids = np.full(len(cache['query']), -1, int)
    history_ids = np.full(cache['valid'].shape, -1, int)
    positive_old = np.zeros_like(cache['positive'])
    positive_fixed = np.zeros_like(cache['positive'])
    loss_valid_old = np.zeros_like(cache['loss_valid'])
    loss_valid_fixed = np.zeros_like(cache['loss_valid'])
    evaluated = np.zeros_like(cache['positive'])
    valid_fixed_query = np.zeros(len(query_ids), bool)
    old_correct_query_same_object = np.zeros(len(query_ids), bool)
    for row, features in enumerate(cache['query']):
        candidates = qlookup.get(features.tobytes(), [])
        if len(candidates) != 1:
            continue
        q = candidates[0]
        query_ids[row] = q
        old_target = old_tokens[-1][q]
        target = fixed_tokens[-1][q]
        valid_fixed_query[row] = not target.startswith('__')
        old_correct_query_same_object[row] = old_target == target
        for col in np.flatnonzero(cache['valid'][row]):
            candidates = hlookup.get(cache['history'][row, col].tobytes(), [])
            # Resolve repeated identical descriptors using stored relative XYZ.
            candidates = [h for h in candidates if np.allclose(
                aligned[h]-old_xyz[-1][q], cache['relative_xyz'][row, col], atol=1e-8, rtol=0)]
            if len(candidates) != 1:
                continue
            h = candidates[0]
            history_ids[row, col] = h
            evaluated[row, col] = True
            positive_old[row, col] = old_tokens[hist_slot][h] == old_target
            loss_valid_old[row, col] = old_tokens[hist_slot][h] != '__ambiguous__'
            loss_valid_fixed[row, col] = (valid_fixed_query[row]
                and fixed_tokens[hist_slot][h] != '__ambiguous__')
            positive_fixed[row, col] = (valid_fixed_query[row]
                and fixed_tokens[hist_slot][h] == target)
    # Persisted labels must reproduce under the actual legacy coordinate path.
    np.testing.assert_array_equal(positive_old[evaluated], cache['positive'][evaluated])
    np.testing.assert_array_equal(loss_valid_old[evaluated], cache['loss_valid'][evaluated])
    retained = evaluated & valid_fixed_query[:, None]
    changes = retained & (cache['positive'] != positive_fixed)
    result = dict(scene=str(shard['scene']), order=int(shard['order']),
        history_order=int(shard['history_order']), slot=int(shard['slot']), cache_file=shard['file'],
        cached_queries=len(query_ids), uniquely_reconstructed_queries=int((query_ids >= 0).sum()),
        valid_cached_pairs=int(cache['valid'].sum()), reconstructed_pairs=int(evaluated.sum()),
        persisted_positive_matches_legacy=True,
        persisted_loss_valid_matches_legacy=True,
        changed_loss_valid_flags=int((evaluated & (cache['loss_valid'] != loss_valid_fixed)).sum()),
        corrected_query_has_unique_GT=int(valid_fixed_query.sum()),
        corrected_query_same_GT_as_legacy=int(old_correct_query_same_object.sum()),
        corrected_query_background_or_ambiguous=int(((query_ids >= 0) & ~valid_fixed_query).sum()),
        pairs_with_corrected_eligible_query=int(retained.sum()), changed_labels_in_those_pairs=int(changes.sum()),
        old_positive_becomes_negative=int((retained & cache['positive'] & ~positive_fixed).sum()),
        old_negative_becomes_positive=int((retained & ~cache['positive'] & positive_fixed).sum()),
        old_positive_pairs_from_now_background_queries=int((evaluated & cache['positive'] & ~valid_fixed_query[:, None]).sum()))
    arrays = dict(query_sorted_ids=query_ids, history_sorted_ids=history_ids,
        reconstructed_mask=evaluated, cached_positive=cache['positive'], corrected_positive=positive_fixed,
        corrected_valid_query=valid_fixed_query, cached_loss_valid=cache['loss_valid'],
        corrected_loss_valid=loss_valid_fixed)
    return result, arrays


def draw_gt(ax, gt, dims, selected=None):
    from scipy.spatial import ConvexHull
    for i, (corners, token) in enumerate(zip(gt['corners'], gt['ids'])):
        points = corners[:, dims]
        hull = ConvexHull(points)
        polygon = points[np.r_[hull.vertices, hull.vertices[0]]]
        ax.plot(*polygon.T, c='#158148', lw=2.1 if i == selected else 1.1)
        ax.text(*points.mean(0), f'GT {token}', c='#116c3c', fontsize=8, clip_on=True,
                bbox=dict(facecolor='white', alpha=.7, edgecolor='none'))


def render_gt(path, old, fixed, old_labels, fixed_labels, power, gt, title):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    # Example object chosen by corrected contained-point count, not label-change rate.
    counts = np.bincount(fixed_labels[fixed_labels >= 0], minlength=len(gt['ids']))
    chosen = int(counts.argmax()) if len(counts) else None
    fig, axes = plt.subplots(3, 2, figsize=(16, 15), layout='constrained')
    strongest = np.argsort(-power, kind='stable')[:4000]
    for col, (xyz, labels, name) in enumerate([(old, old_labels, '旧代码：E · p_native'),
                                               (fixed, fixed_labels, '坐标一致：E · S · p_native')]):
        for row, dims in enumerate([(0, 1), (0, 1), (0, 2)]):
            ax = axes[row, col]
            a, b = dims
            if row == 0 or chosen is None:
                low = np.array([0., -35., -5.]); high = np.array([75., 35., 8.])
                ids = strongest
            else:
                corners = gt['corners'][chosen]
                low, high = corners.min(0)-[3, 3, 2], corners.max(0)+[3, 3, 2]
                mask = ((xyz >= low) & (xyz <= high)).all(1)
                ids = np.flatnonzero(mask)
            inside = labels[ids] >= 0
            ax.scatter(xyz[ids[~inside], a], xyz[ids[~inside], b], s=3 if row == 0 else 7,
                       color='#c77a27', alpha=.45, linewidths=0)
            ax.scatter(xyz[ids[inside], a], xyz[ids[inside], b], s=5 if row == 0 else 10,
                       color='#2879c2', alpha=.65, linewidths=0)
            draw_gt(ax, gt, dims, chosen)
            ax.set(xlim=(low[a], high[a]), ylim=(low[b], high[b]),
                   xlabel='LiDAR X (m)', ylabel='LiDAR Y (m)' if b == 1 else 'LiDAR Z (m)')
            ax.set_aspect('equal'); ax.grid(alpha=.15)
            scope = '全景（显示最强4000点）' if row == 0 else ('同一 GT 的 XY 局部' if row == 1 else '同一 GT 的 XZ 局部')
            ax.set_title(name+'\n'+scope, fontsize=12)
    fig.suptitle(title+'\n两侧绿色 GT 框完全相同；只改变雷达点用于 GT 判定的坐标。\n'
        f"全部43,750点：旧框内 {(old_labels >= 0).sum():,}；修正框内 {(fixed_labels >= 0).sum():,}；"
        f"归属标签变化 {(old_labels != fixed_labels).sum():,}。", fontsize=16)
    fig.legend(handles=[Line2D([], [], c='#158148', label='原 LiDAR GT（不反号、不移动）'),
        Line2D([], [], marker='o', c='none', markerfacecolor='#2879c2', label='按该侧坐标落入唯一 GT'),
        Line2D([], [], marker='o', c='none', markerfacecolor='#c77a27', label='背景或重叠框归属不明确')],
        loc='outside lower center', ncol=3)
    fig.savefig(path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sampling', type=Path, default=ROOT/'outputs/coordinate_neighbor_comparison/random5_seed20261111/sampling.json')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/gt_coordinate_audit/20261111/gt_and_loss')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    setup_plotting()
    geometry = RadarBinGeometry.from_file(ROOT/'data/K-Radar-official-meta/resources/info_arr.mat')
    ext = radar_to_lidar_default(); sign = np.array([1., -1., -1.])
    run = json.loads((ROOT/'work_dirs/p1_top8_ranker/run.json').read_text())
    cache = ROOT/run['arguments']['train_cache']
    manifest = json.loads((cache/'manifest.json').read_text())
    assert digest(cache/'manifest.json') == run['train_manifest_sha256']
    sampling = json.loads(args.sampling.read_text())
    import pickle
    with (ROOT/'data/annotations/kradar_dict_train_official_temporal_doppler8.pkl').open('rb') as stream:
        infos = pickle.load(stream)
    if isinstance(infos, dict):
        infos = infos['infos']
    indexed = {(str(f.get('scene_id', f['scene_token'])), int(f['order_in_scene'])): f for f in infos}
    reports, cache_reports = [], []
    for choice in sampling['sequences']:
        scene = choice['scene']; seq = [indexed[scene, order] for order in choice['orders']]
        frames = [frame_data(f, geometry) for f in seq]
        native = [f[0] for f in frames]; poses = [np.asarray(f['ego_pose']) for f in seq]
        tokens_old, tokens_fixed, labels_old, labels_fixed = [], [], [], []
        for info, xyz in zip(seq, native):
            gt = load_gt(info, scene, ROOT/'data/K-RadarOcc/train', np.eye(4))
            old_labels = ranker_memberships(transform(xyz, ext), gt['boxes'], gt['ids'])
            new_labels = ranker_memberships(transform(xyz*sign, ext), gt['boxes'], gt['ids'])
            labels_old.append(old_labels); labels_fixed.append(new_labels)
            tokens_old.append(label_tokens(old_labels, gt['ids']))
            tokens_fixed.append(label_tokens(new_labels, gt['ids']))
        current = seq[-1]; stem = f"scene{scene}_order{current['order_in_scene']:05d}"
        image = stem+'_gt.png'
        render_gt(args.output/image, transform(native[-1], ext), transform(native[-1]*sign, ext),
            labels_old[-1], labels_fixed[-1], frames[-1][1][:, :3].max(1), gt,
            f"场景 {scene} · order {current['order_in_scene']} · radar {current['radar_frame_idx']:05d}")
        stats = dict(scene=scene, order=int(current['order_in_scene']), radar_frame=int(current['radar_frame_idx']),
            image=image, n_points=len(native[-1]), unique_GT_count=len(gt['ids']),
            old_inside_unique_GT=int((labels_old[-1] >= 0).sum()),
            corrected_inside_unique_GT=int((labels_fixed[-1] >= 0).sum()),
            membership_changed=int((tokens_old[-1] != tokens_fixed[-1]).sum()),
            old_GT_to_background=int(((labels_old[-1] >= 0) & (labels_fixed[-1] == -1)).sum()),
            old_background_to_GT=int(((labels_old[-1] == -1) & (labels_fixed[-1] >= 0)).sum()),
            old_GT_to_different_GT=int(((labels_old[-1] >= 0) & (labels_fixed[-1] >= 0)
                                      & (tokens_old[-1] != tokens_fixed[-1])).sum()),
            gt_source=gt['source'])
        reports.append(stats)
        shards = [s for s in manifest['shards'] if str(s['scene']) == scene and s['order'] == current['order_in_scene']]
        for shard in shards:
            assert int(shard['history_order']) == choice['orders'][3-int(shard['slot'])]
            row, arrays = cache_audit(cache, shard, frames, tokens_old, tokens_fixed, native, poses, ext)
            cache_reports.append(row)
            np.savez_compressed(args.output/f"{stem}_slot{shard['slot']}_cache_audit.npz", **arrays)
            print(json.dumps(row), flush=True)
        print(json.dumps(stats), flush=True)
    write_json(args.output/'gt_membership.json', reports)
    write_json(args.output/'actual_training_cache_audit.json', cache_reports)
    write_json(args.output/'provenance.json', dict(sampling=str(args.sampling),
        sampling_sha256=digest(args.sampling), train_cache=str(cache),
        train_cache_manifest_sha256=digest(cache/'manifest.json'), train_run_matches=True,
        gt_modified=False, correction='E S p_native vs legacy E p_native', script_sha256=digest(__file__)))
    rows = ''.join(f'<h2>场景 {r["scene"]} / radar {r["radar_frame"]:05d}</h2>'
        f'<p>全部 P1：旧框内 {r["old_inside_unique_GT"]}；修正框内 {r["corrected_inside_unique_GT"]}；'
        f'归属变化 {r["membership_changed"]}。</p><a href="{r["image"]}"><img src="{r["image"]}"></a>' for r in reports)
    (args.output/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>GT 坐标与训练监督核查</title>
<style>body{font:17px/1.7 system-ui;max-width:1450px;margin:auto;padding:25px;color:#243247}img{width:100%}a{color:#176fd1}</style>
<h1>GT 保持原样：旧雷达点坐标与修正坐标的实际归属</h1>
<p>沿用上一次随机5帧，未重抽。绿色是完全相同的预处理 LiDAR GT 框；左侧显示旧代码实际用于 GT 归属的雷达坐标，右侧修正雷达角度反号。X/Y/Z 都采用 LiDAR 坐标轴。</p>
<p>框沿用选点器训练与 ABC 评估使用的扩张 bbox 近似归属，不是 occupancy 体素 GT。图中局部对象按修正后框内点数最多选择，不按差异大小选择。</p>
<p>统计包括全部43,750点；全景仅显示最强4000点。重复 ID 和重叠框按原训练规则排除。没有修改模型、缓存、权重或 GT 文件。</p>
<p><a href="gt_membership.json">归属统计</a> · <a href="actual_training_cache_audit.json">实际训练缓存核查</a> · <a href="provenance.json">来源及哈希</a></p>'''+rows+'</html>')


if __name__ == '__main__':
    main()
