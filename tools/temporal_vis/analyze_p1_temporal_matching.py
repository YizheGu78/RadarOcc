#!/usr/bin/env python3
"""Audit real RadarOcc P1 temporal matching: 2 m radius, up to K per history frame.

No training, checkpoint, PyTorch or CUDA is required. Designed for
YizheGu78/RadarOcc @ radarocc-5060-reproduction (2026-10-14).

Reuses the existing tools/temporal_vis/vis_alignment.py for:
* official radar-bin spherical -> Cartesian conversion;
* per-frame P1 sparse token membership (175 ranges x 250 = 43750 tokens);
* ego-pose alignment in the current radar coordinate frame.

Outputs summary_by_range.csv, per_frame_by_range.csv, inspect_candidates.csv,
matching_summary.png, and a few manual-inspection BEV examples.

IMPORTANT: Geometry alone cannot establish whether two tokens belong to the
same physical structure. BEV panels are samples requiring RGB/GT/instance
annotation or manual review. Candidate count is computed *before* the K cap.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2] if Path(__file__).parent.name == 'temporal_vis' else Path.cwd()
sys.path.insert(0, str(ROOT))
from tools.temporal_vis import vis_alignment as va  # reuse repo's validated geometry/loader

BANDS = [(0., 12.8, '00_12.8m'), (12.8, 25.6, '12.8_25.6m'),
         (25.6, 51.2, '25.6_51.2m'), (51.2, float('inf'), 'over_51.2m')]


def args_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--ann-file', type=Path, help='Temporal frame-level annotation pkl')
    g.add_argument('--debug-file', type=Path, help='Existing alignment NPZ from vis_alignment.py or TemporalPlugin')
    p.add_argument('--bins', type=Path, default=Path('data/K-Radar-official-meta/resources/info_arr.mat'))
    p.add_argument('--scene', default='3', help='For --ann-file, scene ID')
    p.add_argument('--start-order', type=int, default=80, help='Inclusive order_in_scene (NOT radar frame ID)')
    p.add_argument('--end-order', type=int, default=200, help='Inclusive order_in_scene')
    p.add_argument('--stride', type=int, default=10, help='Use every n-th matched order')
    p.add_argument('--max-frames', type=int, default=20)
    p.add_argument('--frame-nums', type=int, default=4)
    p.add_argument('--radius', type=float, default=2.0)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--ego-mask-radius', type=float, default=0.0,
                   help='0: current P1 baseline keeps ego returns; 3.7: mask37-like XY self-return mask')
    p.add_argument('--azimuth-sign', type=int, choices=(-1, 1), default=1)
    p.add_argument('--elevation-sign', type=int, choices=(-1, 1), default=1)
    p.add_argument('--extrinsic', type=Path, help='Optional radar_to_lidar 4x4 NPY; default repo convention')
    p.add_argument('--output-dir', type=Path, default=Path('outputs/temporal_matching_audit'))
    p.add_argument('--max-plots', type=int, default=6, help='Number of BEV manual inspection examples')
    p.add_argument('--rgb-image', type=Path, help='Optional RGB reference for one chosen frame (only when testing one frame)')
    args = p.parse_args()
    if args.radius <= 0 or args.k < 1 or args.frame_nums < 2 or args.stride < 1 or args.max_frames < 1 or args.max_plots < 0:
        p.error('radius>0, k>=1, frame-nums>=2, stride>=1, max-frames>=1, max-plots>=0 required')
    if args.end_order < args.start_order:
        p.error('end-order must be >= start-order')
    if args.ego_mask_radius < 0:
        p.error('ego-mask-radius must be >=0')
    return args


def build_from_annotation(args):
    infos = va.read_infos(args.ann_file)
    scene = va.scene_infos(infos, args.scene)
    if not scene:
        raise ValueError(f'Scene {args.scene} not found in {args.ann_file}')
    geometry = va.RadarBinGeometry.from_file(args.bins, azimuth_sign=args.azimuth_sign,
                                             elevation_sign=args.elevation_sign)
    extrinsic = np.load(args.extrinsic, allow_pickle=False) if args.extrinsic else None
    selected = [(i, info) for i, info in enumerate(scene)
                if args.start_order <= int(info['order_in_scene']) <= args.end_order]
    selected = selected[::args.stride][:args.max_frames]
    if not selected:
        raise ValueError('No selected current orders in this scene/split. Try --start-order and --end-order.')
    for pos, info in selected:
        slots = range(pos - args.frame_nums + 1, pos + 1)
        valid = np.array([j >= 0 for j in slots], dtype=bool)
        sequence = [scene[max(j, 0)] for j in slots]
        if any(not frame.get('ego_pose_valid', False) for frame, active in zip(sequence, valid) if active):
            print(f'[skip] scene {args.scene} order {info["order_in_scene"]}: invalid ego pose')
            continue
        loaded = [va.load_p1(f.get('sparse_radar_path') or f['radar_path']) for f in sequence]
        raw = geometry.to_xyz(np.stack([item[0] for item in loaded]))
        poses = np.stack([f['ego_pose'] for f in sequence])
        aligned, _ = va.align_to_current(raw[None], poses[None], extrinsic, valid[None])
        power = np.stack([item[1] for item in loaded])
        yield dict(raw=raw, xyz=aligned[0], valid=valid, offsets=np.arange(-args.frame_nums+1, 1),
                   power=power, label=f'scene{args.scene}_order{int(info["order_in_scene"]):05d}',
                   orders=[int(f['order_in_scene']) for f in sequence])


def build_from_debug(args):
    with np.load(args.debug_file, allow_pickle=False) as d:
        xyz = d['coords_aligned_xyz']
        raw = d['coords_raw_xyz'] if 'coords_raw_xyz' in d else None
        valid = d['valid_mask'] if 'valid_mask' in d else None
        offsets = d['frame_offsets'] if 'frame_offsets' in d else None
        power = d['power_scores'] if 'power_scores' in d else None
        if xyz.ndim != 4 or xyz.shape[-1] != 3:
            raise ValueError('coords_aligned_xyz must be [B,T,N,3]')
        for b in range(min(len(xyz), args.max_frames)):
            t, n, _ = xyz[b].shape
            yield dict(xyz=xyz[b].copy(), raw=raw[b].copy() if raw is not None else xyz[b].copy(),
                       valid=valid[b].astype(bool) if valid is not None else np.ones(t, dtype=bool),
                       offsets=offsets[b] if offsets is not None else np.arange(1-t,1),
                       power=power[b] if power is not None else np.zeros((t,n)),
                       label=f'{args.debug_file.stem}_batch{b}', orders=[])


def save_csv(path, rows, columns):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def pick_example(counts, selected, xyz_cur, hist_xyz, q_ids, h_ids, good, band, label, history_slot, k, radius):
    """Geometry-based inspection sample; NEVER a same-object ground truth."""
    if len(q_ids) == 0:
        return None
    # 'crowded': many candidates, chosen K spread; 'sparse': under K, including zero.
    which = np.flatnonzero((counts > k) & (selected == k)) if band == '00_12.8m' else np.flatnonzero(counts < k)
    if not which.size:
        which = np.arange(len(q_ids))
    # deterministic sampling with representative, not ground-truth based rank
    if band == '00_12.8m':
        p = int(which[np.argmax(counts[which])])
    else:
        p = int(which[np.argmin(counts[which])])
    qid = int(q_ids[p]); anchor = xyz_cur[qid].copy()
    return dict(label=label, band=band, history_slot=int(history_slot), query_id=qid,
                query_xyz=anchor, candidate_count=int(counts[p]), selected_count=int(selected[p]),
                score=float(counts[p]) if band == '00_12.8m' else float(k-counts[p]),
                radius=radius)


def render_example(sample, xyz, eligible, power, k, output, rgb_image=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    current = xyz[-1]; history = xyz[sample['history_slot']]
    anchor = sample['query_xyz']; radius = sample['radius']
    tree = cKDTree(history[eligible[sample['history_slot']]])
    hist_ids = np.flatnonzero(eligible[sample['history_slot']])
    nn_dist, nn_idx = tree.query(anchor, k=np.arange(1,k+1), distance_upper_bound=radius, workers=1)
    valid = np.isfinite(nn_dist) & (nn_idx < len(hist_ids))
    selected_ids = hist_ids[nn_idx[valid]]
    all_idx = hist_ids[tree.query_ball_point(anchor, radius)]
    near_hist = hist_ids[tree.query_ball_point(anchor, max(8.,3*radius))]
    cur_ids = np.flatnonzero(eligible[-1]); cur_tree = cKDTree(current[cur_ids]); near_cur = cur_ids[cur_tree.query_ball_point(anchor, max(8.,3*radius))]
    show_rgb = rgb_image is not None
    fig, axs = plt.subplots(1, 3 if show_rgb else 2, figsize=(17 if show_rgb else 12, 5.6))
    axs = np.atleast_1d(axs)
    if show_rgb:
        from PIL import Image
        with Image.open(rgb_image) as image:
            axs[0].imshow(np.asarray(image.convert('RGB')))
        axs[0].axis('off'); axs[0].set_title('RGB reference (not projected)')
    for ax, halfspan, heading in zip(axs[-2:], [max(8.,3*radius), radius+1.], ['Local context', f'{radius:g} m (3D) search + K nearest']):
        ax.scatter(current[near_cur,0],current[near_cur,1],s=3,c='#c8c8c8',alpha=.45,label='Current-frame P1')
        ax.scatter(history[near_hist,0],history[near_hist,1],s=4,c='#4287b9',alpha=.22,label='Aligned history P1')
        if all_idx.size:
            ax.scatter(history[all_idx,0],history[all_idx,1],s=17,c='#f2aa4c',alpha=.65,label='All radius candidates')
        if selected_ids.size:
            ax.scatter(history[selected_ids,0],history[selected_ids,1],s=37,c='#b01e2e',marker='x',linewidths=1.5,
                       label=f'Selected up to K={k}')
        ax.scatter([anchor[0]],[anchor[1]],s=130,c='#167a39',marker='*',edgecolors='white',linewidths=.5,
                   zorder=8,label='Current query')
        ax.add_patch(Circle(anchor[:2],radius,fill=False,lw=1.4,linestyle='--',color='black'))
        ax.set_xlim(anchor[0]-halfspan, anchor[0]+halfspan)
        ax.set_ylim(anchor[1]-halfspan, anchor[1]+halfspan)
        ax.set_aspect('equal');ax.set_xlabel('X forward (m)');ax.set_ylabel('Y left (m)')
        ax.set_title(heading);ax.grid(alpha=.2)
    axs[-1].legend(loc='upper left', fontsize=8)
    fig.suptitle(f"{sample['label']} | {sample['band']} | history slot {sample['history_slot']} | "
                 f"candidates={len(all_idx)}, selected={len(selected_ids)}\n"
                 '3D radius projected to BEV; physical/semantic identity NOT verified', fontsize=11)
    fig.tight_layout(rect=(0,0,1,.90))
    fig.savefig(output,dpi=145)
    plt.close(fig)
    rows=[]
    for rank, (idx, distance) in enumerate(zip(selected_ids,nn_dist[valid]),1):
        h=history[idx]
        rows.append(dict(frame=sample['label'], range_bin=sample['band'], history_slot=sample['history_slot'],
                         query_id=sample['query_id'], query_x=anchor[0],query_y=anchor[1],query_z=anchor[2],
                         query_range=float(np.linalg.norm(anchor)), candidate_count=len(all_idx),rank=rank,
                         neighbor_id=int(idx),distance_m=float(distance),hist_x=h[0],hist_y=h[1],hist_z=h[2],
                         power=float(power[sample['history_slot'],idx]),
                         note='Geometric neighbor only, NOT same-object ground truth'))
    if not rows:
        rows.append(dict(frame=sample['label'], range_bin=sample['band'], history_slot=sample['history_slot'],
                         query_id=sample['query_id'],query_x=anchor[0],query_y=anchor[1],query_z=anchor[2],
                         query_range=float(np.linalg.norm(anchor)),candidate_count=0,rank=0,neighbor_id=-1,
                         distance_m='',hist_x='',hist_y='',hist_z='',power='',note='NO HISTORY MATCH'))
    return rows


def summarize_arrays(block):
    counts = np.asarray(block['candidate_count'],dtype=np.int64)
    selected = np.asarray(block['matched_count'],dtype=np.int64)
    nearest = np.asarray(block['nearest_m'],dtype=float)
    kth = np.asarray(block['kth_m'],dtype=float)
    v1=nearest[np.isfinite(nearest)];vk=kth[np.isfinite(kth)]
    n=len(counts)
    return dict(n_query_history_pairs=n,mean_radius_candidates=float(counts.mean()) if n else None,
                median_radius_candidates=float(np.median(counts)) if n else None,
                p90_radius_candidates=float(np.percentile(counts,90)) if n else None,
                max_radius_candidates=int(counts.max()) if n else None,
                mean_matched_after_k=float(selected.mean()) if n else None,
                frac_less_than_k=float((selected<block['k']).mean()) if n else None,
                frac_zero_neighbor=float((selected==0).mean()) if n else None,
                mean_nearest_distance_m=float(v1.mean()) if len(v1) else None,
                mean_kth_distance_when_full_m=float(vk.mean()) if len(vk) else None)


def metric_plot(path, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=[b[2] for b in BANDS]
    ycount=[next((r.get('mean_radius_candidates') for r in rows if r['range_bin']==name),None) for name in labels]
    ymiss=[next((r.get('frac_less_than_k') for r in rows if r['range_bin']==name),None) for name in labels]
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(len(labels))
    for a,y,title in zip(ax,[ycount,ymiss],['Mean raw candidates inside radius','Fraction with fewer than K']):
        a.bar(x,[float(z) if z is not None else 0 for z in y]);a.set_xticks(x,labels,rotation=18)
        a.set_title(title);a.grid(axis='y',alpha=.2)
    ax[0].set_ylabel('Candidate tokens / query / history frame')
    ax[1].set_ylim(0,1);ax[1].set_ylabel('Fraction')
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def main():
    args=args_parser();args.output_dir.mkdir(parents=True,exist_ok=True)
    accum={band[2]:dict(candidate_count=[],matched_count=[],nearest_m=[],kth_m=[],k=args.k) for band in BANDS}
    per_frame=[];inspect=[];nframes=0;plot_count=0
    iterator=build_from_debug(args) if args.debug_file else build_from_annotation(args)
    for frame in iterator:
        xyz=np.asarray(frame['xyz'],dtype=np.float64)
        raw=np.asarray(frame['raw'],dtype=np.float64)
        valid=np.asarray(frame['valid'],dtype=bool)
        if xyz.shape!=raw.shape or xyz.ndim!=3 or xyz.shape[-1]!=3 or len(valid)!=len(xyz):
            raise ValueError('Expected raw and ego-aligned data as [T,N,3] with valid[T]')
        if not np.isfinite(xyz).all() or not valid[-1]:
            raise ValueError('Nonfinite xyz or invalid current frame')
        nframes+=1;t,n,_=xyz.shape
        eligible=np.ones((t,n),dtype=bool)
        if args.ego_mask_radius:
            eligible=np.hypot(raw[...,0],raw[...,1])>args.ego_mask_radius
        cur_id=np.flatnonzero(eligible[-1])
        if len(cur_id)==0:
            raise ValueError('No eligible current queries')
        qxyz=xyz[-1,cur_id]
        # Physical radar range from current radar origin, not x-forward coordinate.
        qrange=np.linalg.norm(raw[-1,cur_id],axis=1)
        print(f"[frame {nframes}] {frame['label']}: queries={len(cur_id)}, valid_history={int(valid[:-1].sum())}",flush=True)
        for j in range(t-1):
            if not valid[j]:
                continue
            hist_id=np.flatnonzero(eligible[j])
            if len(hist_id)==0:
                continue
            tree=cKDTree(xyz[j,hist_id])
            counts=np.asarray(tree.query_ball_point(qxyz,args.radius,return_length=True,workers=1),dtype=np.int64)
            d, idx=tree.query(qxyz,k=np.arange(1,args.k+1),distance_upper_bound=args.radius,workers=1)
            if d.ndim==1: d=d[:,None]
            good=np.isfinite(d) & (idx<len(hist_id))
            matched=good.sum(axis=1)
            nearest=np.where(good[:,0],d[:,0],np.nan)
            kth=np.where(good[:,-1],d[:,-1],np.nan)
            # Sanity: mirrors the actual local_neighbors() geometric cardinality.
            if not np.array_equal(matched,np.minimum(counts,args.k)):
                raise AssertionError('KDTree query and query_ball_point disagree; check distance boundary precision')
            for lo,hi,name in BANDS:
                mask=(qrange>=lo)&(qrange<hi)
                if not mask.any():
                    continue
                block=dict(candidate_count=counts[mask],matched_count=matched[mask],
                           nearest_m=nearest[mask],kth_m=kth[mask],k=args.k)
                for key in ('candidate_count','matched_count','nearest_m','kth_m'):
                    accum[name][key].append(block[key])
                result=summarize_arrays(block)
                per_frame.append(dict(frame=frame['label'],history_slot=j,frame_gap=int(frame['offsets'][-1]-frame['offsets'][j]),
                                      range_bin=name,**result))
                # One example per range/per history/frame until plot limit (as read).
                if plot_count<args.max_plots:
                    ids=np.flatnonzero(mask)
                    example=pick_example(counts[mask],matched[mask],xyz[-1],xyz[j],cur_id[ids],hist_id,good[mask],
                                         name,frame['label'],j,args.k,args.radius)
                    if example is not None:
                        path=args.output_dir/f"example_{plot_count+1:02d}_{frame['label']}_{name}_history{j}.png"
                        image=args.rgb_image if nframes==1 and args.max_frames==1 else None
                        inspect.extend(render_example(example,xyz,eligible,frame['power'],args.k,path,image))
                        plot_count+=1
    if nframes==0:
        raise ValueError('No valid frames processed')
    summary=[]
    for band in BANDS:
        name=band[2];block=accum[name]
        if not block['candidate_count']:
            continue
        combo={key:(np.concatenate(value) if isinstance(value,list) else value) for key,value in block.items()}
        summary.append(dict(range_bin=name,**summarize_arrays(combo)))
    stats_columns=['range_bin','n_query_history_pairs','mean_radius_candidates','median_radius_candidates',
                   'p90_radius_candidates','max_radius_candidates','mean_matched_after_k','frac_less_than_k',
                   'frac_zero_neighbor','mean_nearest_distance_m','mean_kth_distance_when_full_m']
    save_csv(args.output_dir/'summary_by_range.csv',summary,stats_columns)
    save_csv(args.output_dir/'per_frame_by_range.csv',per_frame,
             ['frame','history_slot','frame_gap']+stats_columns)
    if inspect:
        save_csv(args.output_dir/'inspect_candidates.csv',inspect,list(inspect[0]))
    metric_plot(args.output_dir/'matching_summary.png',summary)
    meta=dict(n_frames=nframes,radius_m=args.radius,k=args.k,frame_nums=args.frame_nums,
              masked_radius_m=args.ego_mask_radius,analysis='P1 center-based geometric search',
              unit='one query x one valid history frame',
              range_type='Euclidean distance from current radar origin',
              ground_truth_of_same_object='NOT available in this diagnostic',
              note='High raw candidate count, low count or BEV spread do not establish a different physical structure.')
    (args.output_dir/'README_results.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'\nProcessed {nframes} frames. Outputs: {args.output_dir.resolve()}')
    for row in summary:
        print(f"  {row['range_bin']:>12s}: raw_mean={row['mean_radius_candidates']:.2f}, "
              f"under_K={row['frac_less_than_k']:.1%}, zero={row['frac_zero_neighbor']:.1%}")
    print('BEV examples are for manual inspection; structural/instance membership is NOT automatically established.')


if __name__=='__main__':
    main()