#!/usr/bin/env python3
"""Controlled P1 seed sweep; reuse the existing extractor without changing it."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from temporal.core.ego_alignment import RadarBinGeometry, radar_to_lidar_default
from tools.temporal_vis.p1_reflection_regions import RegionConfig, extract_regions, npz_payload
from tools.temporal_vis import vis_p1_reflection_regions as vis


def write_json(path, data):
    path.write_text(json.dumps(data, default=vis.json_default, ensure_ascii=False,
                               allow_nan=False, indent=2) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def covered_points(regions, count):
    covered = np.zeros(count, bool)
    for region in regions:
        if region['accepted']:
            covered[region['box_point_ids']] = True
    return covered


def final_height_filter(result, xyz, limit):
    """Diagnostic-only final rejection: do not alter seeds, growth, or merging."""
    filtered = dict(result)
    filtered['regions'] = [dict(region) for region in result['regions']]
    rejected = []
    for region in filtered['regions']:
        if region['accepted'] and limit is not None and region['box_size'][2] > limit + 1e-8:
            region.update(accepted=False, status='final_z_size_limit')
            rejected.append(region['id'])
    owner = np.full(len(xyz), -1, np.int32)
    distance = np.full(len(xyz), np.inf)
    for region in filtered['regions']:
        if not region['accepted']:
            continue
        indices = region['box_point_ids']
        candidate = np.sum(((xyz[indices] - region['center']) / region['box_size']) ** 2, axis=1)
        better = candidate < distance[indices]
        owner[indices[better]] = region['id']
        distance[indices[better]] = candidate[better]
    filtered['box_region_ids'] = owner
    filtered['stats'] = dict(result['stats'],
        n_accepted=sum(r['accepted'] for r in filtered['regions']),
        n_box_points=int((owner >= 0).sum()),
        box_fraction_all_p1=float(np.mean(owner >= 0)), n_final_z_rejected=len(rejected))
    return filtered, rejected


def height_comparison_plot(cases, xyz, gt, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon, Rectangle
    from scipy.spatial import ConvexHull
    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4.5),
                             sharex=True, sharey=True, layout='constrained', squeeze=False)
    eligible = cases[0]['result']['analysis_mask']
    target = eligible & (gt['point_labels'] == 0)
    background = np.flatnonzero(eligible)[::3]
    for ax, case in zip(axes[0], cases):
        result = case['result']
        ax.scatter(xyz[background, 0], xyz[background, 1], color='.75', s=1, alpha=.4)
        ax.scatter(xyz[target, 0], xyz[target, 1], s=9, facecolors='none',
                   edgecolors='#00aabb', lw=.5)
        for region in result['regions']:
            if region['accepted']:
                low, high = region['box_min'], region['box_max']
                ax.add_patch(Rectangle(low[:2], *(high-low)[:2], fill=False,
                                       edgecolor='#2563eb', lw=1.3))
        for i, corners in enumerate(gt['corners']):
            hull = ConvexHull(corners[:, :2])
            ax.add_patch(Polygon(corners[hull.vertices, :2], closed=True, fill=False,
                                 edgecolor=vis.GT_COLOR, lw=1.7))
        covered = (result['box_region_ids'] >= 0) & target
        label = 'No final Z cap' if case['cap'] is None else f'Final Z cap = {case["cap"]:g} m'
        ax.set_title(f'{label}\naccepted={result["stats"]["n_accepted"]}; '
                     f'GT:0 tokens covered={covered.sum()}/{target.sum()}')
        ax.set_xlim(0, 60); ax.set_ylim(-10, 10); ax.set_aspect('equal'); ax.grid(alpha=.15)
        ax.set_xlabel('X forward (m)'); ax.set_ylabel('Y left (m)')
    fig.legend(handles=[Line2D([0],[0],color='#2563eb',label='Accepted proposal boxes'),
        Line2D([0],[0],color=vis.GT_COLOR,label='GT footprint (overlay only)'),
        Line2D([0],[0],marker='o',ls='',markerfacecolor='none',markeredgecolor='#00aabb',
                   label='Radar tokens inside GT:0')], loc='outside lower center', ncols=3)
    fig.suptitle('P97 / Seed MAD 3 | Identical XY grouping and X/Y caps (20 m / 6 m)\n'
                 'Only the final box-height rejection is switched; no regrouping or clipping')
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def run_height_ablation(source, output, limits):
    source = source.resolve()
    manifest = json.loads((source/'manifest.json').read_text())
    if (not manifest['completed'] or manifest['coordinate_convention'] != 'kradar_physical_v2'
            or manifest['extractor_sha256'] != digest(ROOT/'tools/temporal_vis/p1_reflection_regions.py')):
        raise ValueError('Expected a complete, current physical-coordinate seed sweep')
    if output.exists():
        raise FileExistsError('Use a new ablation output directory: '+str(output))
    if not limits or len(set(limits)) != len(limits) or any(not np.isfinite(v) or v <= 0 for v in limits):
        raise ValueError('Final Z caps must be distinct finite positive heights')
    output.mkdir(parents=True)
    write_json(output/'manifest.json', dict(completed=False, source=str(source)))
    scene_id, order = manifest['scene'], manifest['order']
    _, scene = vis.load_scene(ROOT, Path(manifest['annotation']), scene_id)
    info = next(row for row in scene if int(row['order_in_scene']) == order)
    stem = f'scene{scene_id}_order{order:05d}'
    gt_source = None
    rows, target_rows, invariant_checks = [], [], []
    by_cap = {'none': []} | {f'{cap:g}': [] for cap in limits}
    stored_for_view = []
    reference_support = None
    for mad in (3., 4.):
        for percentile in (95., 97., 98.):
            name = f'p{int(percentile)}_mad{mad:.1f}'.replace('.', '_')
            directory = source/'xy_20x6'/name
            record = json.loads((directory/'result.json').read_text())
            with np.load(directory/(stem+'.npz'), allow_pickle=False) as payload:
                xyz, power, coords = payload['xyz'], payload['power_channel2'], payload['coords_rae']
                expected_owner = payload['box_region_ids']
                expected_labels = payload['point_gt_box_rows']
            config = RegionConfig(**record['config'])
            base = extract_regions(xyz, power, config)
            assert base['stats'] == record['stats'], 'Baseline extractor changed'
            np.testing.assert_array_equal(base['box_region_ids'], expected_owner)
            if reference_support is None:
                reference_support = base['support_mask'].copy()
            np.testing.assert_array_equal(reference_support, base['support_mask'])
            if gt_source is None:
                gt_source = vis.load_gt(info, scene_id, ROOT/'data/K-RadarOcc/train',
                                        radar_to_lidar_default())
            for cap in (None, *limits):
                filtered, rejected = final_height_filter(base, xyz, cap)
                np.testing.assert_array_equal(filtered['support_region_ids'], base['support_region_ids'])
                assert filtered['fine_regions'] is base['fine_regions']
                assert all(a['source_seed_ids'] == b['source_seed_ids']
                           and np.array_equal(a['box_min'], b['box_min'])
                           and np.array_equal(a['box_max'], b['box_max'])
                           for a,b in zip(filtered['regions'], base['regions']))
                if cap is None:
                    np.testing.assert_array_equal(filtered['box_region_ids'], expected_owner)
                gt = vis.gt_diagnostics(filtered, xyz, dict(gt_source), gt_space='xyz')
                np.testing.assert_array_equal(gt['point_labels'], expected_labels)
                cap_name = 'none' if cap is None else f'{cap:g}'
                case_dir = output/('zcap_'+cap_name)/name
                case_dir.mkdir(parents=True)
                title = f'Scene {scene_id} | order {order} | XY 20x6 | P{percentile:g} MAD{mad:.1f} | final Z cap={cap_name}'
                vis.render(filtered, xyz, power, case_dir/(stem+'.png'), title,
                    rgb_path=record['rgb']['path'], max_boxes=128, gt=gt)
                arrays = npz_payload(filtered, xyz, power, coords)
                arrays.update(point_gt_box_rows=gt['point_labels'], gt_ids=gt['ids'],
                              gt_corners_radar=gt['corners'])
                np.savez_compressed(case_dir/(stem+'.npz'), **arrays)
                case_record = dict(config=record['config'], final_z_cap_m=cap,
                    rule='Reject an already-merged accepted region iff its full box Z size exceeds cap; do not change merging, box geometry or original points.',
                    stats=filtered['stats'], rejected_region_ids=rejected,
                    regions=[{k:v for k,v in r.items() if not k.endswith('_point_ids')}
                             for r in filtered['regions']], gt_used_for_extraction=False)
                write_json(case_dir/'result.json', case_record)
                row = dict(seed_percentile=percentile, seed_mad=mad, final_z_cap=cap_name,
                    high_threshold_tokens=base['stats']['n_seed_points'], local_seeds=base['stats']['seeds_used'],
                    fine_accepted=base['stats']['n_fine_accepted'], merged_groups=base['stats']['n_regions'],
                    rejected_by_final_z=len(rejected), final_accepted=filtered['stats']['n_accepted'],
                    boxed_tokens=filtered['stats']['n_box_points'], image=str((case_dir/(stem+'.png')).relative_to(output)))
                rows.append(row)
                for i, token in enumerate(gt['ids']):
                    target = filtered['analysis_mask'] & (gt['point_labels'] == i)
                    covered = target & (filtered['box_region_ids'] >= 0)
                    target_rows.append(dict(seed_percentile=percentile, seed_mad=mad,
                        final_z_cap=cap_name, object_id=str(token), gt_tokens=int(target.sum()),
                        covered_tokens=int(covered.sum())))
                case = dict(percentile=percentile, mad=mad, config=config, result=filtered,
                            title=f'Scene {scene_id} | order {order} | XY 20x6 | final Z cap={cap_name}', cap=cap)
                by_cap[cap_name].append(case)
                if percentile == 97 and mad == 3:
                    stored_for_view.append(case)
                invariant_checks.append(dict(seed_percentile=percentile, seed_mad=mad, final_z_cap=cap_name,
                    identical_initial_regions=True, identical_support_assignment=True,
                    identical_pre_filter_box_geometry=True, gt_used_to_filter=False))
                print(f'Z cap={cap_name} P{percentile:g} MAD{mad:.1f}: '
                      f'accepted={row["final_accepted"]}, rejected={len(rejected)}', flush=True)
    for cap_name, cases in by_cap.items():
        comparison_plot(cases, xyz, gt, output/f'zcap_{cap_name}_comparison.png', 'after')
        comparison_plot(cases, xyz, gt, output/f'zcap_{cap_name}_detail_comparison.png', 'after', detail=True)
    height_comparison_plot(stored_for_view, xyz, gt, output/'z_gate_comparison.png')
    write_csv(output/'summary.csv', rows)
    write_csv(output/'gt_token_coverage.csv', target_rows)
    write_json(output/'controlled_variable_checks.json', invariant_checks)
    write_json(output/'manifest.json', dict(completed=True, source=str(source), scene=scene_id, order=order,
        coordinate_convention='kradar_physical_v2', only_changed_parameter='final box-height acceptance gate',
        caps_m=[None,*limits], cases=len(rows), all_baselines_reproduced=True,
        seeds_growth_merging_and_geometry_identical_within_each_cap_triplet=True,
        original_points_and_heights_retained=True, gt_used_for_extraction=False,
        extractor_sha256=digest(ROOT/'tools/temporal_vis/p1_reflection_regions.py'),
        runner_sha256=digest(__file__), production_training_config_modified=False))
    lines=['# 最终Z尺寸筛除：单变量对照','',
        '固定XY合并、X≤20m、Y≤6m及每一组Seed参数；只在合并完成后按框高度决定是否接受。高度含原有padding。',
        '不裁点、不重新生长/合并、不移动框、不反号GT。无Z上限、4m上限、6m上限分别评价。',
        '', '| Seed分位 | MAD | 最终Z上限 | 合并区域 | Z删除区域 | 最终有效框 | 框内点 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+' | '.join(str(row[k]) for k in ('seed_percentile','seed_mad','final_z_cap',
            'merged_groups','rejected_by_final_z','final_accepted','boxed_tokens'))+' |')
    lines += ['', 'GT覆盖详见gt_token_coverage.csv；分母为分析ROI内真实落在GT三维框内的雷达点。不是历史匹配Recall或occupancy IoU。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    links=''.join(f'<p>Z上限 {html.escape(cap)}：<a href="zcap_{cap}_comparison.png">六组合全景</a> · '
                  f'<a href="zcap_{cap}_detail_comparison.png">六组合局部</a></p>' for cap in by_cap)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(row[k]))+'</td>' for k in
        ('seed_percentile','seed_mad','final_z_cap','merged_groups','rejected_by_final_z','final_accepted','boxed_tokens'))+'</tr>' for row in rows)
    record_link = html.escape(os.path.relpath(ROOT/'temporal/experiments_v2/POWER_REGION_EXPERIMENT_RECORD.md', output))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Final Z gate ablation</title>'
        '<style>body{font:16px sans-serif;margin:24px;background:#f5f5f5}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}</style>'
        '<h1>最终Z尺寸限制：严格单变量对照</h1><p>固定当前XY合并规则及XY尺寸，单独对比无Z上限、4m、6m；下图固定P97/MAD3。</p>'
        '<p><a href="'+record_link+'">完整实验记录：目的、方法、控制变量、结果与结论</a></p>'
        '<a href="z_gate_comparison.png"><img src="z_gate_comparison.png"></a>'+links+
        '<p><a href="REPORT.md">报告</a> · <a href="summary.csv">全部统计</a> · <a href="gt_token_coverage.csv">GT点覆盖</a></p>'
        '<table><tr><th>Seed分位</th><th>MAD</th><th>Z上限</th><th>合并区域</th><th>Z删除区域</th><th>最终有效框</th><th>框内点</th></tr>'+table+'</table>')
    print('Completed:',output/'index.html',flush=True)


def comparison_plot(cases, xyz, gt, destination, stage, detail=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon, Rectangle
    from scipy.spatial import ConvexHull

    fig, axes = plt.subplots(2, 3, figsize=(18, 7 if detail else 14),
                             layout='constrained', sharex=True, sharey=True)
    bounds = ((0, 60), (-10, 10)) if detail else ((0, 85), (-60, 60))
    first = cases[0]['result']
    indices = np.flatnonzero(first['analysis_mask'])
    if len(indices) > 10000:
        indices = indices[np.linspace(0, len(indices) - 1, 10000, dtype=int)]
    target_ids = np.flatnonzero(first['analysis_mask'] & (gt['point_labels'] >= 0))
    for case in cases:
        row = (3., 4.).index(case['mad'])
        col = (95., 97., 98.).index(case['percentile'])
        ax = axes[row, col]
        result = case['result']
        ax.scatter(xyz[indices, 0], xyz[indices, 1], c='.75', s=.5, alpha=.35,
                   rasterized=True)
        if stage == 'seeds':
            high = np.flatnonzero(result['seed_point_mask'])
            ax.scatter(xyz[high, 0], xyz[high, 1], c='#ff8800', s=3, alpha=.6)
            peaks = np.asarray(result['peak_cells'])
            cfg = case['config']
            peak_xyz = cfg.lower + (peaks + .5) * cfg.spacing
            ax.scatter(peak_xyz[:, 0], peak_xyz[:, 1], c='red', marker='+', s=27)
            label = (f'high-power tokens={result["stats"]["n_seed_points"]}; '
                     f'local seeds={result["stats"]["seeds_used"]}')
        else:
            regions = result['fine_regions'] if stage == 'before' else result['regions']
            accepted = [r for r in regions if r['accepted']]
            for region in accepted:
                low, high = region['box_min'], region['box_max']
                ax.add_patch(Rectangle(low[:2], *(high - low)[:2], fill=False,
                                       edgecolor='#2563eb', lw=1.1))
                if (bounds[0][0] <= region['center'][0] <= bounds[0][1]
                        and bounds[1][0] <= region['center'][1] <= bounds[1][1]):
                    ax.text(*region['center'][:2], str(region['id']), fontsize=6,
                            color='#1d4ed8', clip_on=True)
            label = (f'accepted={len(accepted)}; '
                     f'boxed tokens={int(covered_points(regions, len(xyz)).sum())}')
        ax.scatter(xyz[target_ids, 0], xyz[target_ids, 1], s=8, facecolors='none',
                   edgecolors='#00aabb', lw=.5, alpha=.7)
        for i, corners in enumerate(gt['corners']):
            hull = ConvexHull(corners[:, :2])
            ax.add_patch(Polygon(corners[hull.vertices, :2], closed=True, fill=False,
                                 edgecolor=vis.GT_COLOR, lw=1.5))
            position = corners[:, :2].min(axis=0)
            if (bounds[0][0] <= position[0] <= bounds[0][1]
                    and bounds[1][0] <= position[1] <= bounds[1][1]):
                ax.text(*position, 'GT:' + str(gt['ids'][i]), fontsize=7,
                        color=vis.GT_COLOR, va='top')
        ax.set_title(f'Seed percentile={case["percentile"]:g}, MAD={case["mad"]:.1f}\n{label}',
                     fontsize=11)
        ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1])
        ax.set_aspect('equal'); ax.grid(alpha=.15)
        ax.set_xlabel('X forward (m)'); ax.set_ylabel('Y left (m)')
    handles = [Line2D([0], [0], color=vis.GT_COLOR, label='GT footprint (overlay only)'),
               Line2D([0], [0], marker='o', ls='', markerfacecolor='none',
                      markeredgecolor='#00aabb', label='Radar tokens inside 3D GT')]
    if stage == 'seeds':
        handles += [Line2D([0], [0], marker='.', ls='', color='#ff8800',
                          label='High-threshold tokens'),
                    Line2D([0], [0], marker='+', ls='', color='red',
                           label='Selected local seed cells')]
    else:
        handles += [Line2D([0], [0], color='#2563eb', label='Accepted proposal boxes')]
    fig.legend(handles=handles, loc='outside lower center', ncols=2, fontsize=9)
    titles = dict(seeds='Seed distribution', before='Before: fixed-size fine boxes',
                  after='After: accepted support-enclosing boxes')
    fig.suptitle(cases[0]['title'] + ' | ' + titles[stage]
                 + '\nSame physical XYZ and support mask; only seed percentile / MAD vary',
                 fontsize=13)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='3')
    parser.add_argument('--order', type=int, default=85)
    parser.add_argument('--ann-file', type=Path)
    parser.add_argument('--from-sweep', type=Path, help='Add final-Z-gate controls to an existing completed seed sweep')
    parser.add_argument('--z-caps', type=float, nargs='+', default=[4., 6.])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; existing sweeps are retained: ' + str(output))
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/radarocc_seed_sweep_mpl')
    os.environ.setdefault('XDG_CACHE_HOME', '/tmp/radarocc_seed_sweep_xdg')
    if args.from_sweep is not None:
        run_height_ablation(args.from_sweep, output, args.z_caps)
        return
    annotation, scene = vis.load_scene(ROOT, args.ann_file, args.scene)
    matches = [row for row in scene if int(row['order_in_scene']) == args.order]
    if len(matches) != 1:
        raise ValueError('Expected exactly one matching scene/order')
    info = matches[0]
    radar_path = Path(info.get('sparse_radar_path') or info['radar_path'])
    if not radar_path.is_absolute():
        radar_path = ROOT / radar_path
    coords, power = vis.read_p1(radar_path)
    bins = ROOT / 'data/K-Radar-official-meta/resources/info_arr.mat'
    geometry = RadarBinGeometry.from_file(bins, coordinate_convention='kradar_physical_v2')
    xyz = geometry.to_xyz(coords)
    camera = vis.camera_directory(ROOT, args.scene, None)
    rgb_path, rgb_mapping = vis.alignment.resolve_rgb(info, scene, camera_dir=camera)
    protocols = {
        'xyz_12x4x4': RegionConfig(merge_space='xyz', max_box_size=(12., 4., 4.)),
        'xy_20x6': RegionConfig(merge_space='xy', max_box_size=(20., 6., 6.))}
    output.mkdir(parents=True)
    write_json(output/'manifest.json', dict(completed=False,
        arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}))
    cases_by_protocol, rows, threshold_rows, target_rows, checks = {}, [], [], [], []
    reference_support = reference_analysis = None
    gt_source = None
    stem = f'scene{args.scene}_order{args.order:05d}'
    for protocol, base in protocols.items():
        cases = []
        for mad in (3., 4.):
            for percentile in (95., 97., 98.):
                config = replace(base, seed_percentile=percentile, seed_mad=mad).validate()
                result = extract_regions(xyz, power, config)
                changed = {k for k in asdict(base) if asdict(base)[k] != asdict(config)[k]}
                assert changed <= {'seed_percentile', 'seed_mad'}
                if reference_support is None:
                    reference_support = result['support_mask'].copy()
                    reference_analysis = result['analysis_mask'].copy()
                np.testing.assert_array_equal(reference_support, result['support_mask'])
                np.testing.assert_array_equal(reference_analysis, result['analysis_mask'])
                if gt_source is None:
                    gt_source = vis.load_gt(info, args.scene, ROOT/'data/K-RadarOcc/train',
                                            radar_to_lidar_default())
                gt = vis.gt_diagnostics(result, xyz, dict(gt_source), gt_space='xyz')
                name = f'p{int(percentile)}_mad{mad:.1f}'.replace('.', '_')
                directory = output/protocol/name
                directory.mkdir(parents=True)
                title = f'Scene {args.scene} | order {args.order} | {protocol} | P{percentile:g} MAD{mad:.1f}'
                vis.render(result, xyz, power, directory/(stem+'.png'), title,
                           rgb_path=rgb_path, max_boxes=128, gt=gt)
                arrays = npz_payload(result, xyz, power, coords)
                arrays.update(point_gt_box_rows=gt['point_labels'], gt_ids=gt['ids'],
                              gt_corners_radar=gt['corners'])
                np.savez_compressed(directory/(stem+'.npz'), **arrays)
                record = dict(config=asdict(config), stats=result['stats'], thresholds=result['thresholds'],
                    regions=[{k:v for k,v in r.items() if not k.endswith('_point_ids')}
                             for r in result['regions']],
                    gt_used_for_extraction=False, gt_membership_space='xyz',
                    image=stem+'.png', arrays=stem+'.npz', rgb=rgb_mapping)
                write_json(directory/'result.json', record)
                covered = covered_points(result['regions'], len(xyz))
                targets = []
                for i, token in enumerate(gt['ids']):
                    target = result['analysis_mask'] & (gt['point_labels'] == i)
                    targets.append(dict(protocol=protocol, seed_percentile=percentile, seed_mad=mad,
                        object_id=str(token), gt_tokens=int(target.sum()), covered_tokens=int((target&covered).sum())))
                target_rows.extend(targets)
                row = dict(protocol=protocol, seed_percentile=percentile, seed_mad=mad,
                    n_support=result['stats']['n_support'], high_threshold_tokens=result['stats']['n_seed_points'],
                    local_seeds=result['stats']['seeds_used'], fine_accepted=result['stats']['n_fine_accepted'],
                    merge_inputs=result['stats']['n_merge_input_regions'], groups=result['stats']['n_regions'],
                    final_accepted=result['stats']['n_accepted'], boxed_tokens=result['stats']['n_box_points'],
                    rejected=json.dumps(dict(Counter(r['status'] for r in result['regions'] if not r['accepted'])),sort_keys=True),
                    image=str((directory/(stem+'.png')).relative_to(output)))
                rows.append(row)
                radius = np.linalg.norm(xyz, axis=1)
                for band in result['thresholds']:
                    selected = result['analysis_mask'] & (radius>=band['range_min_m']) & (radius<band['range_max_m'])
                    quantile = float(np.percentile(power[selected], percentile))
                    mad_threshold = band['median'] + mad * band['robust_sigma']
                    threshold_rows.append(dict(protocol=protocol, seed_percentile=percentile, seed_mad=mad,
                        **band, percentile_threshold=quantile, mad_threshold=mad_threshold,
                        controlling_gate='percentile' if quantile>=mad_threshold else 'MAD'))
                cases.append(dict(percentile=percentile, mad=mad, config=config, result=result,
                                  title=f'Scene {args.scene} | order {args.order} | {protocol}'))
                print(f'{protocol} P{percentile:g} MAD{mad:.1f}: high={row["high_threshold_tokens"]} '
                      f'seeds={row["local_seeds"]} before={row["fine_accepted"]} after={row["final_accepted"]}', flush=True)
        for percentile in (95., 97., 98.):
            a,b = [c for c in cases if c['percentile']==percentile]
            def box_geometry(case):
                return np.asarray(sorted(tuple(np.r_[r['box_min'], r['box_max']])
                                         for r in case['result']['regions'] if r['accepted'])).reshape(-1, 6)
            geometry_a, geometry_b = box_geometry(a), box_geometry(b)
            checks.append(dict(protocol=protocol, percentile=percentile,
                changed_high_threshold_tokens=int(np.count_nonzero(a['result']['seed_point_mask'] != b['result']['seed_point_mask'])),
                identical_selected_seed_cells=np.array_equal(a['result']['peak_cells'],b['result']['peak_cells']),
                identical_final_box_assignments=np.array_equal(a['result']['box_region_ids'],b['result']['box_region_ids']),
                identical_accepted_box_geometry=(geometry_a.shape == geometry_b.shape
                    and bool(np.allclose(geometry_a, geometry_b, rtol=0, atol=1e-9))),
                identical_boxed_point_union=np.array_equal(a['result']['box_region_ids'] >= 0,
                                                         b['result']['box_region_ids'] >= 0)))
        cases_by_protocol[protocol] = cases
        for stage in ('seeds','before','after'):
            comparison_plot(cases, xyz, gt, output/protocol/(stage+'_comparison.png'), stage)
        comparison_plot(cases, xyz, gt, output/protocol/'after_detail_comparison.png', 'after', detail=True)
    write_csv(output/'summary.csv', rows)
    write_csv(output/'thresholds_by_range.csv', threshold_rows)
    write_csv(output/'gt_token_coverage.csv', target_rows)
    write_json(output/'parameter_effect_checks.json', checks)
    write_json(output/'manifest.json', dict(completed=True, scene=args.scene, order=args.order,
        coordinate_convention='kradar_physical_v2', azimuth_sign=-1, elevation_sign=-1,
        annotation=str(annotation), annotation_sha256=digest(annotation), radar_path=str(radar_path),
        bins_sha256=digest(bins), extractor_sha256=digest(ROOT/'tools/temporal_vis/p1_reflection_regions.py'),
        runner_sha256=digest(__file__), configs={k:asdict(v) for k,v in protocols.items()},
        varied_parameters=dict(seed_percentile=[95,97,98],seed_mad=[3.,4.]),
        support_masks_identical=True, analysis_masks_identical=True, gt_used_for_extraction=False,
        model_run=False, configurations_per_protocol=6, protocols_compared_independently=True,
        note='Single-frame historical diagnostic; scene 3 is test. No training configuration is changed or selected by this sweep.'))
    lines = ['# 修正坐标后的 Seed 参数对比实验','',
        f'场景 {args.scene}，order_in_scene={args.order}；原始雷达 {radar_path.name}。每个协议六组组合。',
        '两组协议分开比较：旧 XYZ 最大框12×4×4m；当前 XY 合并限制X≤20m、Y≤6m，不单独以Z超限拒绝。',
        '统一物理坐标(-1,-1)。组内仅改变Seed Percentile和Seed MAD；Support=80分位/0.5MAD，距离分段8m，其他区域生长、合并和框约束全部固定。已逐点检查所有组合的Support和分析ROI掩码完全一致。',
        '', '阈值：sigma=1.4826×median(|power−median(power)|)；Seed=max(分位阈值, median+Seed MAD×sigma)。MAD未必实际控制阈值。',
        '', '| 协议 | Seed分位 | MAD | 高门限点 | 局部种子 | 合并前有效框 | 合并后区域 | 最终有效框 | 框内全部点 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+' | '.join(str(r[k]) for k in ('protocol','seed_percentile','seed_mad','high_threshold_tokens','local_seeds','fine_accepted','groups','final_accepted','boxed_tokens'))+' |')
    lines += ['', '## MAD 3→4的实际影响','',
              '| 协议 | Seed分位 | 门限点变化数 | 种子位置完全一致 | 最终框几何一致 | 框覆盖点集一致 |',
              '|---|---:|---:|---|---|---|']
    for r in checks:
        lines.append('| '+' | '.join(str(r[k]) for k in ('protocol','percentile','changed_high_threshold_tokens','identical_selected_seed_cells','identical_accepted_box_geometry','identical_boxed_point_union'))+' |')
    lines += ['', '区域编号是局部编号；编号变化不代表框几何变化。框几何比较忽略编号、使用1e-9m容差，覆盖点集逐点比较。',
              '', 'GT只用于3D标注叠加与点覆盖诊断，不参与提框。GT点覆盖不是跨帧匹配Recall或occupancy IoU；本实验不据此选择最佳训练参数。',
              '', '运行方式：', '', '```bash', 'source use_radarocc5060.sh',
              f'python tools/temporal_vis/sweep_reflection_seeds.py --scene {args.scene} --order {args.order} --output-dir <新的输出目录>', '```']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    table_rows=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in
        ('protocol','seed_percentile','seed_mad','high_threshold_tokens','local_seeds','fine_accepted','final_accepted','boxed_tokens'))
        +f'<td><a href="{html.escape(r["image"])}">六面板原图</a></td></tr>' for r in rows)
    page='''<!doctype html><meta charset="utf-8"><title>P1 seed parameter sweep</title>
<style>body{font:16px sans-serif;margin:24px;background:#f5f5f5;color:#222}button,select{font:inherit;padding:8px;margin:4px}img{max-width:100%;background:white}table{border-collapse:collapse;background:white}td,th{padding:8px;border:1px solid #bbb}p{max-width:1100px}a{color:#164fb5}</style>
<h1>场景 SCENE，第 ORDER 帧：Seed Percentile × Seed MAD</h1>
<p><a href="EXPERIMENT_RECORD">完整实验记录：Seed敏感性与最终Z限制对照</a></p>
<p>每张对比图：列=95/97/98分位，行=MAD 3/4。组内Support、区域生长、合并、尺寸约束固定；所有输入使用修正物理坐标。绿色GT只作对照，蓝框为雷达提框。</p>
<label>固定协议 <select id="protocol"><option value="xyz_12x4x4">原XYZ 12×4×4m</option><option value="xy_20x6">当前XY：X≤20m、Y≤6m</option></select></label>
<label>查看阶段 <select id="stage"><option value="seeds">种子分布</option><option value="before">合并前初始框</option><option value="after">合并后最终框</option><option value="after_detail">最终框前方局部</option></select></label>
<p><a id="open" target="_blank">打开完整分辨率图片</a> · <a href="REPORT.md">实验报告</a> · <a href="summary.csv">统计CSV</a> · <a href="thresholds_by_range.csv">逐距离段阈值</a> · <a href="gt_token_coverage.csv">GT内点覆盖诊断</a></p>
<img id="comparison" alt="六组参数对比图">
<h2>全部组合</h2><table><thead><tr><th>协议</th><th>Seed分位</th><th>MAD</th><th>高门限点</th><th>局部种子</th><th>合并前有效框</th><th>最终有效框</th><th>框内点</th><th>原图</th></tr></thead><tbody>TABLE</tbody></table>
<p>本次仅重做同帧可视化诊断，不修改训练参数。场景3属于test，不能用这张图宣称参数泛化最优或时序匹配有效。</p>
<script>function update(){const path=document.querySelector('#protocol').value+'/'+document.querySelector('#stage').value+'_comparison.png';document.querySelector('#comparison').src=path;document.querySelector('#open').href=path}document.querySelectorAll('select').forEach(x=>x.addEventListener('change',update));update();</script>'''
    record_link = html.escape(os.path.relpath(ROOT/'temporal/experiments_v2/POWER_REGION_EXPERIMENT_RECORD.md', output))
    page=page.replace('SCENE',html.escape(str(args.scene))).replace('ORDER',str(args.order)).replace('TABLE',table_rows).replace('EXPERIMENT_RECORD',record_link)
    (output/'index.html').write_text(page)
    print('Completed:',output/'index.html',flush=True)


if __name__ == '__main__':
    main()
