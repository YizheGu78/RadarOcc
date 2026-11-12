#!/usr/bin/env python3
"""Single-frame spatial density versus Doppler similarity; no model or GT.

Uses the original NPZ membership of RadarOcc's 175 x 250 P0 tokens.
Neighboring Doppler bins are circular. Raw-spectrum checks are optional.
"""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.temporal_vis.audit_doppler_queries import read_frame
from tools.temporal_vis.neighborhood_motion import load_doppler_table
from tools.temporal_vis.vis_alignment import RadarBinGeometry
from temporal.core.ego_alignment import add_coordinate_argument
from tools.temporal_vis.vis_p1_reflection_regions import annotation_scenes


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def peak_bins(features, n_bins):
    powers, bins = features[:, :3], features[:, 3:6]
    valid = (np.isfinite(features).all(1) & (powers >= 0).all(1)
             & (powers.max(1) > 0) & (bins == np.floor(bins)).all(1)
             & (bins >= 0).all(1) & (bins < n_bins).all(1))
    safe = np.clip(np.nan_to_num(bins, nan=0, posinf=0, neginf=0), 0, n_bins-1).astype(int)
    strongest = safe[np.arange(len(features)), np.argmax(powers, axis=1)]
    return strongest, valid


def cell_histograms(inverse, bins, n_cells, n_bins):
    return np.bincount(inverse*n_bins+bins, minlength=n_cells*n_bins).reshape(n_cells, n_bins)


def band_counts(hist):
    return np.roll(hist, 1, axis=-1) + hist + np.roll(hist, -1, axis=-1)


def cosine_pair_mean(vectors):
    """Mean over distinct unordered pairs, without an NxN allocation."""
    vectors = np.asarray(vectors, float)
    norm = np.linalg.norm(vectors, axis=1)
    unit = vectors[norm > 0]/norm[norm > 0, None]
    if len(unit) < 2:
        return None
    return float((np.square(unit.sum(0)).sum()-len(unit))/(len(unit)*(len(unit)-1)))


def analyze(xyz, coords, features, table, side, minimum, coherence, scope, rng, shuffles):
    strongest, valid = peak_bins(features, len(table))
    radius = np.linalg.norm(xyz, axis=1)
    mask = valid & ((xyz >= [0, -60, -5]) & (xyz < [85, 60, 8])).all(1)
    if scope == 'r_ge_8m':
        mask &= radius >= 8
    ids = np.flatnonzero(mask)
    keys, inverse, counts = np.unique(np.floor(xyz[ids]/side).astype(int), axis=0,
                                      return_inverse=True, return_counts=True)
    hist = cell_histograms(inverse, strongest[ids], len(keys), len(table))
    near = band_counts(hist)
    dense = counts >= minimum
    coherent = dense & (near.max(1) >= coherence*counts)
    exact = dense & (hist.max(1) >= coherence*counts)
    total = hist.sum(0)
    # Preserve per-frame Doppler frequency AND its coarse range dependence.
    # This control tests spatial association, not whether a group is one object.
    strata = np.floor(radius[ids]/8).astype(int)
    groups = [np.flatnonzero(strata == k) for k in np.unique(strata)]
    null_cells, null_tokens = [], []
    for _ in range(shuffles):
        shuffled = strongest[ids].copy()
        for group in groups:
            shuffled[group] = rng.permutation(shuffled[group])
        nh = cell_histograms(inverse, shuffled, len(keys), len(table))
        qualified = dense & (band_counts(nh).max(1) >= coherence*counts)
        null_cells.append(int(qualified.sum()))
        null_tokens.append(int(counts[qualified].sum()))
    stat = dict(scope=scope, cube_m=side, n_input=len(xyz), n_invalid=int((~valid).sum()),
                n_roi=len(ids), n_dense_cells=int(dense.sum()),
                n_exact_coherent_cells=int(exact.sum()), n_near_coherent_cells=int(coherent.sum()),
                n_tokens_in_dense=int(counts[dense].sum()),
                n_tokens_in_near_coherent_cells=int(counts[coherent].sum()),
                coherent_token_fraction=float(counts[coherent].sum()/max(len(ids), 1)),
                range_shuffle_cells_mean=float(np.mean(null_cells)) if shuffles else None,
                range_shuffle_tokens_mean=float(np.mean(null_tokens)) if shuffles else None,
                global_exact_fraction=float(total.max()/max(total.sum(), 1)),
                global_near_fraction=float(band_counts(total).max()/max(total.sum(), 1)))
    order = np.argsort(inverse, kind='stable')
    starts = np.r_[0, np.cumsum(counts)]
    rows = []
    for j in np.flatnonzero(dense):
        member = ids[order[starts[j]:starts[j+1]]]
        main, center = int(hist[j].argmax()), int(near[j].argmax())
        sets, frequencies = np.unique(np.sort(features[member, 3:6].astype(int), axis=1),
                                      axis=0, return_counts=True)
        q = int(frequencies.argmax())
        peak_power = features[member, :3].max(1)
        p10, p90 = np.percentile(peak_power, [10, 90])
        low = keys[j]*side
        rows.append(dict(scope=scope, cube_m=side, cell_id=int(j),
                         x_min=float(low[0]), y_min=float(low[1]), z_min=float(low[2]),
                         n_tokens=len(member), top1_mode_bin=main, top1_mode_mps=float(table[main]),
                         top1_exact_count=int(hist[j, main]), top1_exact_fraction=float(hist[j, main]/counts[j]),
                         near_center_bin=center, near_center_mps=float(table[center]),
                         top1_near_count=int(near[j, center]), top1_near_fraction=float(near[j, center]/counts[j]),
                         top3_set_mode=','.join(map(str, sets[q])),
                         top3_set_fraction=float(frequencies[q]/counts[j]),
                         peak_power_p90_p10_db=float(10*np.log10(p90/p10)),
                         coherent=bool(coherent[j])))
    return stat, rows, dict(ids=ids, inverse=inverse, keys=keys, counts=counts,
                            near_fraction=near.max(1)/counts, main=strongest[ids],
                            main_fraction=hist.max(1)/counts)


def render(path, xyz, table, cube, rows, frame, minimum, coherence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout='constrained')
    points = xyz[cube['ids']]
    ax = axes[0, 0]
    artist = ax.scatter(points[:, 0], points[:, 1], c=table[cube['main']], s=1.5,
                        cmap='twilight', vmin=table.min(), vmax=table.max(), rasterized=True)
    fig.colorbar(artist, ax=ax, label='Strongest Doppler peak (m/s, folded)')
    ax.set_title('All valid P0 returns; XY projection (heights overlap)')
    ax = axes[0, 1]
    artist = ax.scatter(points[:, 0], points[:, 1], c=cube['near_fraction'][cube['inverse']],
                        s=1.5, vmin=0, vmax=1, cmap='viridis', rasterized=True)
    fig.colorbar(artist, ax=ax, label='Fraction in best circular bin +/- 1')
    ax.set_title('Doppler agreement within each 4 x 4 x 4 m XYZ cell')
    examples = sorted((r for r in rows if r['coherent']), key=lambda r: -r['n_tokens'])
    ex = examples[0] if examples else max(rows, key=lambda r: r['n_tokens'])
    low = np.array([ex['x_min'], ex['y_min'], ex['z_min']])
    member = np.all((xyz >= low) & (xyz < low+ex['cube_m']), axis=1)
    member_ids = np.flatnonzero(member)
    # Lookup only the tokens in this scope, not other projected heights.
    belongs = np.isin(cube['ids'], member_ids)
    for ax in axes[0]:
        ax.set(xlim=(0, 85), ylim=(-60, 60), xlabel='X forward (m)', ylabel='Y left (m)')
        ax.set_aspect('equal')
        ax.add_patch(Rectangle(low[:2], ex['cube_m'], ex['cube_m'], fill=False, ec='red', lw=1.7))
    ax = axes[1, 0]
    artist = ax.scatter(points[belongs, 0], points[belongs, 1], c=table[cube['main'][belongs]],
                        cmap='twilight', vmin=table.min(), vmax=table.max(), s=13)
    ax.set(xlabel='X (m)', ylabel='Y (m)', title=f'Example: n={ex["n_tokens"]}, Z=[{low[2]:g},{low[2]+4:g}) m')
    ax.set_aspect('equal')
    ax = axes[1, 1]
    hist = np.bincount(cube['main'][belongs], minlength=len(table))
    ax.bar(table, hist, width=abs(table[1]-table[0])*.9)
    ax.set(xlabel='Folded Doppler (m/s)', ylabel='Token count',
           title=f'Exact mode {ex["top1_exact_fraction"]:.1%}; +/-1 bin {ex["top1_near_fraction"]:.1%}')
    fig.suptitle(f'{frame} | no GT/model; range >= 8 m\n'
                 f'Dense = >= {minimum} tokens/cube; coherent = >= {coherence:.0%} in +/-1 bin')
    fig.savefig(path, dpi=135)
    plt.close(fig)


def raw_check(raw_path, geometry, table, output):
    from tools.visualize_radarocc_sparse_rd_ra import load_raw
    scene = raw_path.parent.parent.name
    frame = int(raw_path.stem.split('_')[-1])
    sparse = ROOT/f'data/RadarOcc_8doppler/{scene}/radar_tensor_8doppler/EAsparse_{frame:05d}.npz'
    coords, xyz, features = read_frame(sparse, geometry)
    raw = load_raw(raw_path)
    spectra = np.asarray(raw[:, coords[:, 0], coords[:, 2], coords[:, 1]].T, float)
    # Check association and axis order against saved peak powers, not filename alone.
    bins = features[:, 3:6].astype(int)
    paired = np.take_along_axis(spectra, bins, axis=1)
    if not np.allclose(paired, features[:, :3], rtol=1e-5, atol=1e-6):
        raise ValueError(f'Raw/NPZ peak powers disagree: {raw_path}')
    _, rows, cube = analyze(xyz, coords, features, table, 4., 30, .8, 'r_ge_8m',
                            np.random.default_rng(42), 0)
    selected = sorted((r for r in rows if r['coherent']), key=lambda r: -r['n_tokens'])[:5]
    result = []
    for ex in selected:
        member = cube['ids'][cube['inverse'] == ex['cell_id']]
        result.append(dict(**ex, full_64bin_pair_cosine_mean=cosine_pair_mean(spectra[member])))
    global_ids = cube['ids']
    out = dict(scene=scene, radar_frame_id=frame, raw_path=str(raw_path), sparse_path=str(sparse),
               raw_npz_peak_powers_match=True, n_spectral_bins=spectra.shape[1],
               roi_pair_cosine_mean=cosine_pair_mean(spectra[global_ids]), examples=result)
    if selected:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Use the second-largest dense coherent region when available, chosen
        # by token count, independently of its full-spectrum cosine score.
        ex = selected[min(1, len(selected)-1)]
        member = cube['ids'][cube['inverse'] == ex['cell_id']]
        example_spectra = spectra[member]
        normalized = example_spectra/example_spectra.max(1, keepdims=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
        for row in normalized[np.linspace(0, len(member)-1, min(24, len(member)), dtype=int)]:
            axes[0].plot(table, row, alpha=.3, lw=.8)
        axes[0].plot(table, normalized.mean(0), color='black', lw=2, label='Mean over all region tokens')
        axes[0].legend(fontsize=8)
        axes[0].set(xlabel='Folded Doppler (m/s)', ylabel='Power / per-token maximum',
                    title='24 evenly spaced examples; complete 64-bin spectra')
        peak_db = 10*np.log10(example_spectra.max(1))
        axes[1].hist(peak_db, bins=30)
        axes[1].set(xlabel='10 log10(stored peak power)', ylabel='Token count',
                    title=f'Peak-power p90/p10 = {ex["peak_power_p90_p10_db"]:.1f} dB')
        fig.suptitle(f'Scene {scene}, radar {frame:05d}: XYZ cell '
                     f'[{ex["x_min"]:g},{ex["y_min"]:g},{ex["z_min"]:g}] + [0,4) m; n={len(member)}\n'
                     f'Best +/-1 bin agreement {ex["top1_near_fraction"]:.1%}; '
                     f'mean distinct-pair cosine {result[min(1, len(result)-1)]["full_64bin_pair_cosine_mean"]:.3f}')
        out['image'] = f'raw_scene{scene}_radar{frame:05d}.png'
        fig.savefig(output/out['image'], dpi=140)
        plt.close(fig)
    (output/f'raw_scene{scene}_radar{frame:05d}.json').write_text(json.dumps(out, indent=2)+'\n')
    del raw
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_coordinate_argument(p)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scenes', nargs='+', default=['1', '2', '3', '15', '22'])
    p.add_argument('--start', type=int, default=80)
    p.add_argument('--stop', type=int, default=200)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--min-tokens', type=int, default=30)
    p.add_argument('--coherence', type=float, default=.8)
    p.add_argument('--shuffles', type=int, default=10)
    p.add_argument('--raw-check', type=Path, nargs='*', default=[])
    args = p.parse_args()
    if (args.stride < 1 or args.stop < args.start or args.min_tokens < 2
            or not 0 < args.coherence <= 1 or args.shuffles < 0):
        p.error('Invalid selection, density, coherence, or shuffle parameters')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        p.error('Choose a new or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/radarocc_mpl')
    geometry = RadarBinGeometry.from_file(ROOT/'data/K-Radar-official-meta/resources/info_arr.mat', coordinate_convention=args.coordinate_convention)
    table = load_doppler_table(ROOT/'data/K-Radar-official-meta/resources/arr_doppler.mat')
    inventory = [annotation_scenes(ROOT/f'data/annotations/kradar_dict_{s}_official_temporal_doppler8.pkl')
                 for s in ['train', 'val', 'test']]
    rng = np.random.default_rng(42)
    stats, regions, sources, images = [], [], [], []
    for scene in args.scenes:
        matches = [g[scene] for g in inventory if scene in g]
        if len(matches) != 1:
            raise ValueError(f'Scene {scene}: expected one annotation source')
        selected = [r for r in matches[0] if args.start <= int(r['order_in_scene']) <= args.stop
                    and (int(r['order_in_scene'])-args.start) % args.stride == 0]
        expected = len(range(args.start, args.stop+1, args.stride))
        if len(selected) != expected:
            raise ValueError(f'Scene {scene}: selected {len(selected)}, expected {expected}')
        for info in selected:
            order = int(info['order_in_scene'])
            path = Path(info.get('sparse_radar_path') or info['radar_path'])
            if not path.is_absolute():
                path = ROOT/path
            coords, xyz, features = read_frame(path, geometry)
            base = dict(scene=scene, order=order, radar_frame_id=info.get('radar_frame_id'))
            sources.append(dict(**base, path=str(path)))
            for scope in ['all_roi', 'r_ge_8m']:
                for side in [2., 4.]:
                    stat, rows, cube = analyze(xyz, coords, features, table, side,
                                              args.min_tokens, args.coherence, scope, rng, args.shuffles)
                    stats.append(dict(**base, **stat))
                    regions.extend(dict(**base, **r) for r in rows)
                    if scope == 'r_ge_8m' and side == 4 and (order == args.start or (scene == '3' and order == 85)):
                        image = f'scene{scene}_order{order:05d}.png'
                        render(output/image, xyz, table, cube, rows, f'Scene {scene}, order {order}',
                               args.min_tokens, args.coherence)
                        images.append(image)
            print(f'scene {scene} order {order}: analyzed {len(xyz)} tokens', flush=True)
    write_csv(output/'frame_metrics.csv', stats)
    write_csv(output/'regions.csv', regions)
    summaries = []
    for scene in args.scenes + ['ALL']:
        for scope in ['all_roi', 'r_ge_8m']:
            for side in [2., 4.]:
                group = [s for s in stats if (scene == 'ALL' or s['scene'] == scene)
                         and s['scope'] == scope and s['cube_m'] == side]
                n_roi = sum(s['n_roi'] for s in group)
                summaries.append(dict(scene=scene, scope=scope, cube_m=side, frames=len(group),
                    frames_with_near_coherent_cells=sum(s['n_near_coherent_cells'] > 0 for s in group),
                    n_roi=n_roi, n_dense_cells=sum(s['n_dense_cells'] for s in group),
                    n_near_coherent_cells=sum(s['n_near_coherent_cells'] for s in group),
                    coherent_token_fraction=sum(s['n_tokens_in_near_coherent_cells'] for s in group)/max(n_roi, 1),
                    shuffled_coherent_token_fraction=(sum(s['range_shuffle_tokens_mean'] for s in group)/max(n_roi, 1)
                                                       if args.shuffles else None)))
    write_csv(output/'summary.csv', summaries)
    raws = [raw_check(path.resolve(), geometry, table, output) for path in args.raw_check]
    images.extend(raw['image'] for raw in raws if 'image' in raw)
    manifest = dict(parameters=vars(args), n_frames=len(sources), sources=sources, raw_checks=raws,
                    n_bins=len(table), bin_step_mps=float(np.median(np.diff(table))),
                    stored_npz_n_tokens=64000, analyzed_p0_n_tokens=43750,
                    roi_xyz=[[0,85],[-60,60],[-5,8]], gt_used=False, model_run=False,
                    cube_origin=[0,0,0], adjacent_bins_circular=True,
                    scope_note='r_ge_8m is a reported sensitivity slice, not a sensor-validity claim')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str)+'\n')
    text = ['# Single-frame spatial Doppler similarity', '',
            'No GT, model, temporal alignment, Doppler sign selection, or token deletion.',
            'Each source NPZ stores 64,000 selected spatial returns; this audit uses the same 43,750 as P0.',
            'Doppler8 stores three unsorted peak powers, their paired bin IDs, mean, and variance. It is not a full spectrum.',
            f'Dense XYZ cells: >= {args.min_tokens} tokens. Agreement: >= {args.coherence:.0%} strongest peaks in best circular bin +/-1.',
            f'64 Doppler bins; one-bin step {abs(table[1]-table[0]):.9f} m/s; +/-1 denotes three bins, not identical spectra.',
            'Both 2m and 4m cubes reported. Grid origin [0,0,0]; fixed cells can split a physical cluster.',
            'Counts are representation density, not independent measurements, objects, or true motion.',
            'all_roi includes near-range bin-0 concentrations; r_ge_8m reports results after excluding radius <8m without claiming its cause.',
            'The null shuffles strongest bins within 8m radial bands of each scope/frame, retaining density and coarse range dependence.',
            'Power variation is p90/p10 in dB; top3_set_fraction compares unordered bin sets, not amplitudes.',
            'Optional raw checks verify raw/NPZ peak powers and measure L2-normalized full-64-bin mean pairwise cosine.', '',
            'See summary.csv, frame_metrics.csv, regions.csv, manifest.json and raw_scene*.json.', '']
    (output/'README.md').write_text('\n'.join(text))
    page = ['<!doctype html><meta charset="utf-8"><title>Local Doppler similarity</title>',
            '<style>body{font:16px sans-serif;max-width:1400px;margin:24px auto}img{width:100%}details{margin:20px 0}</style>',
            '<h1>单帧局部空间 Doppler 相似性</h1>',
            '<p>不使用 GT 或模型。统计当前 P0 的 43,750 个 token；颜色为最强峰 Doppler，非真实物体速度。</p>',
            '<p>下图展示距雷达至少 8 米的回波，统计表同时保留全 ROI 结果。每个立方体独立保留 Z。</p>',
            '<p><a href="README.md">统计定义</a> · <a href="summary.csv">汇总</a> · '
            '<a href="frame_metrics.csv">每帧</a> · <a href="regions.csv">局部区域</a></p>']
    for image in images:
        page.append(f'<details open><summary>{html.escape(image)}</summary><img src="{image}"></details>')
    (output/'index.html').write_text('\n'.join(page))
    print('OUTPUT', output, flush=True)


if __name__ == '__main__':
    main()
