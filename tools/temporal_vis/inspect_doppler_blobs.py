#!/usr/bin/env python3
"""Inspect spatial shapes of single-frame similar-Doppler connected components.

Components use measured spatial adjacency and a fixed three-bin Doppler band.
No cuboid size prior, GT, camera, or tracking enters extraction. GT is optional
post-extraction evidence; drawing an enclosing cuboid does not prove a vehicle.
"""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.temporal_vis.audit_doppler_queries import read_frame
from tools.temporal_vis.audit_local_doppler_similarity import peak_bins
from tools.temporal_vis.benchmark_region_matching import gt_memberships
from tools.temporal_vis.neighborhood_motion import load_doppler_table
from tools.temporal_vis import vis_alignment as va
from tools.temporal_vis.vis_p1_reflection_regions import annotation_scenes, load_gt, camera_directory


def extract(xyz, features, percentile=0, min_tokens=30, min_cells=6):
    """Bound Doppler span globally, preventing velocity-chain percolation."""
    bins, valid = peak_bins(features, 64)
    radius = np.linalg.norm(xyz, axis=1)
    eligible = valid & ((xyz >= [0,-60,-5]) & (xyz < [85,60,8])).all(1) & (radius >= 8)
    if percentile:
        power = features[:, :3].max(1)
        keep = np.zeros(len(xyz), bool)
        for band in np.unique(np.floor(radius[eligible]/8).astype(int)):
            ids = np.flatnonzero(eligible & (radius//8 == band))
            keep[ids] = power[ids] >= np.percentile(power[ids], percentile)
        eligible &= keep
    ids = np.flatnonzero(eligible)
    cells = np.floor((xyz[ids]-[0,-60,-5])/[.5,.5,1.]).astype(int)
    candidates = []
    shape = (170,240,13)
    for center in range(64):
        near = np.abs((bins[ids]-center+32)%64-32) <= 1
        selected, spatial = ids[near], cells[near]
        grid = np.zeros(shape, bool)
        grid[tuple(spatial.T)] = True
        labels, _ = ndimage.label(grid, structure=np.ones((3,3,3), bool))
        point_labels = labels[tuple(spatial.T)]
        counts = np.bincount(point_labels)
        cell_counts = np.bincount(labels.reshape(-1))
        for label in np.flatnonzero(counts >= min_tokens):
            if label == 0 or cell_counts[label] < min_cells:
                continue
            member = selected[point_labels == label]
            candidates.append((member, center, int(cell_counts[label])))
    candidates.sort(key=lambda r: (-len(r[0]), r[1], int(r[0].min())))
    used = np.zeros(len(xyz), bool)
    groups = []
    for member, center, n_cells in candidates:
        # Suppress substantially duplicated components from overlapping bands.
        # Saved groups need not form a disjoint segmentation.
        if used[member].mean() > .5:
            continue
        used[member] = True
        hist = np.bincount(bins[member], minlength=64)
        low, high = xyz[member].min(0), xyz[member].max(0)
        groups.append(dict(id=len(groups), point_ids=member.tolist(), n_tokens=len(member), n_cells=n_cells,
                           band_center_bin=center, main_bin=int(hist.argmax()),
                           mode_fraction=float(hist.max()/len(member)), observed_min=low.tolist(),
                           observed_max=high.tolist(), observed_span_xyz=(high-low).tolist()))
    return groups, eligible, bins


def evaluate(groups, xyz, gt):
    if gt is None:
        return
    memberships = {space: gt_memberships(xyz, gt['boxes'], gt['radar_to_lidar'], space)
                   for space in ['xy','xyz']}
    for group in groups:
        member = np.asarray(group['point_ids'])
        for space, labels in memberships.items():
            observed = labels[member]
            counts = np.bincount(observed[observed >= 0], minlength=len(gt['ids']))
            winner = int(counts.argmax()) if len(counts) else None
            group[f'gt_{space}_id'] = str(gt['ids'][winner]) if winner is not None and counts[winner] else None
            group[f'gt_{space}_fraction'] = float(counts.max()/len(member)) if len(counts) else 0.


def gt_projection(ax, gt, axes=(0,1)):
    if gt is None:
        return
    for corners, identifier in zip(gt['corners'], gt['ids']):
        points = corners[:, axes]
        hull = ConvexHull(points)
        polygon = points[np.r_[hull.vertices, hull.vertices[0]]]
        ax.plot(*polygon.T, color='#168038', lw=1.2, ls='--')
        ax.text(*points.mean(0), f'GT {identifier}', color='#168038', fontsize=7, clip_on=True)


def render(path, xyz, bins, eligible, groups, table, gt, rgb, title, percentile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig = plt.figure(figsize=(18,10), layout='constrained')
    axes = [fig.add_subplot(2,3,i+1) for i in range(5)]
    three = fig.add_subplot(2,3,6, projection='3d')
    if rgb:
        with Image.open(rgb) as image:
            axes[0].imshow(image)
        axes[0].set_title('RGB reference only: no calibrated projection overlay')
    axes[0].axis('off')
    ids = np.flatnonzero(eligible)
    artist = axes[1].scatter(xyz[ids,0],xyz[ids,1],c=table[bins[ids]],s=2,cmap='turbo',
                            vmin=table.min(),vmax=table.max())
    fig.colorbar(artist,ax=axes[1],label='Folded strongest-peak Doppler (m/s)',fraction=.03)
    axes[1].set_title(f'Single-frame measured returns, n={len(ids):,}\n'
                      + ('All powers' if not percentile else f'Power >= range-band P{percentile:g}'))
    gt_projection(axes[1],gt)
    axes[2].scatter(xyz[ids,0],xyz[ids,1],color='.85',s=1)
    for group in groups[:12]:
        member = np.asarray(group['point_ids'])
        color = plt.get_cmap('tab20')(group['id']%20)
        axes[2].scatter(xyz[member,0],xyz[member,1],s=4,color=color)
        position = xyz[member,:2].mean(0)
        axes[2].text(*position,f'C{group["id"]}',fontsize=7,color=color)
    gt_projection(axes[2],gt)
    axes[2].set_title(f'{len(groups)} Doppler-connected groups; showing largest 12\n'
                      'No fitted vehicle boxes or size constraints')
    for ax in axes[1:3]:
        ax.set(xlim=(0,85),ylim=(-40,40),xlabel='X forward (m)',ylabel='Y left (m)')
        ax.set_aspect('equal')
    if groups:
        # GT affects only which extracted example is illustrated. Save this
        # selection rule explicitly, and report all groups including failures.
        example = max(groups,key=lambda r:(r.get('gt_xy_fraction',0),r['n_tokens'])) if gt else groups[0]
        member = np.asarray(example['point_ids'])
        points = xyz[member]
        low, high = points.min(0), points.max(0)
        context = ((xyz[:,:2] >= low[:2]-2)&(xyz[:,:2] <= high[:2]+2)).all(1)
        axes[3].scatter(xyz[context,0],xyz[context,1],s=2,color='.85')
        axes[3].scatter(points[:,0],points[:,1],c=table[bins[member]],s=9,cmap='turbo',
                        vmin=table.min(),vmax=table.max())
        gt_projection(axes[3],gt)
        axes[3].set(xlim=(low[0]-2,high[0]+2),ylim=(low[1]-2,high[1]+2),xlabel='X (m)',ylabel='Y (m)',
                    title=f'C{example["id"]}: actual XY footprint, n={len(member)}\n'
                          f'GT XY overlap={example.get("gt_xy_fraction",0):.1%} (illustration selected after extraction)')
        axes[3].set_aspect('equal')
        axes[4].scatter(xyz[context,0],xyz[context,2],s=2,color='.85')
        axes[4].scatter(points[:,0],points[:,2],c=table[bins[member]],s=9,cmap='turbo',
                        vmin=table.min(),vmax=table.max())
        gt_projection(axes[4],gt,(0,2))
        axes[4].set(xlim=(low[0]-2,high[0]+2),ylim=(-5,8),xlabel='X (m)',ylabel='Z up (m)',
                    title='Actual XZ distribution; green = vehicle GT projection\n'
                          f'GT cuboid overlap={example.get("gt_xyz_fraction",0):.1%}')
        axes[4].set_aspect('equal')
        three.scatter(*points.T,c=table[bins[member]],s=7,cmap='turbo',vmin=table.min(),vmax=table.max())
        corners=np.array(list(itertools.product(*zip(low,high))))
        bits=np.array(list(itertools.product((0,1),repeat=3)))
        for i in range(8):
            for j in range(i+1,8):
                if np.count_nonzero(bits[i]!=bits[j])==1:
                    three.plot(*corners[[i,j]].T,color='.4',lw=.7,ls=':')
        span=high-low
        three.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',
                  title=f'Observed span: {span[0]:.1f} x {span[1]:.1f} x {span[2]:.1f} m\n'
                        'Dotted lines = measured enclosing bounds only')
        half = max(float(span.max())+1, 4)/2
        center = (low+high)/2
        three.set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half),
                  zlim=(center[2]-half,center[2]+half))
        three.set_box_aspect((1,1,1))
        from matplotlib.ticker import MaxNLocator
        for axis in [three.xaxis,three.yaxis,three.zaxis]:
            axis.set_major_locator(MaxNLocator(4))
        three.view_init(elev=22,azim=-55)
    fig.suptitle(title+'\nSpatial adjacency 0.5 x 0.5 x 1 m voxels, 26-connected, fixed three-bin band; '
                 'GT/camera used only after extraction',fontsize=12)
    fig.savefig(path,dpi=125)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    va.geometry_module.add_coordinate_argument(parser)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scenes',nargs='+',default=['3','1','2'])
    parser.add_argument('--orders',type=int,nargs='+',default=[80,85])
    parser.add_argument('--no-gt',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Use a new or empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/radarocc_mpl')
    geometry=va.RadarBinGeometry.from_file(ROOT/'data/K-Radar-official-meta/resources/info_arr.mat', coordinate_convention=args.coordinate_convention)
    table=load_doppler_table(ROOT/'data/K-Radar-official-meta/resources/arr_doppler.mat')
    inventory=[annotation_scenes(ROOT/f'data/annotations/kradar_dict_{split}_official_temporal_doppler8.pkl')
               for split in ['train','val','test']]
    frame_rows=[]
    for scene in args.scenes:
        matches=[g[scene] for g in inventory if scene in g]
        if len(matches)!=1:raise ValueError(f'Scene {scene} must have one annotation source')
        rows=matches[0]
        for order in args.orders:
            info=next(r for r in rows if int(r['order_in_scene'])==order)
            coords,xyz,features=read_frame(Path(info.get('sparse_radar_path') or info['radar_path']),geometry)
            # Both extraction variants complete before GT/RGB are loaded.
            variants=[(p,*extract(xyz,features,p)) for p in [0,80]]
            gt=None if args.no_gt else load_gt(info,scene,ROOT/'data/K-RadarOcc/train',
                                               va.geometry_module.radar_to_lidar_default())
            directory=camera_directory(ROOT,scene,None)
            rgb=va.resolve_rgb(info,rows,camera_dir=directory)[0] if directory else None
            for percentile,groups,eligible,bins in variants:
                evaluate(groups,xyz,gt)
                stem=f'scene{scene}_order{order:05d}_power{percentile}'
                render(out/f'{stem}.png',xyz,bins,eligible,groups,table,gt,rgb,
                       f'Scene {scene}, order {order}, radar {info.get("radar_frame_id")}',percentile)
                data=dict(scene=scene,order=order,radar_frame_id=info.get('radar_frame_id'),
                          power_percentile=percentile,n_eligible=int(eligible.sum()),groups=groups,
                          rgb=str(rgb),gt_used_for_extraction=False,
                          illustration_selection='highest GT XY purity, then count; diagnostic only' if gt else 'largest count',
                          sparse_path=info.get('sparse_radar_path') or info['radar_path'])
                (out/f'{stem}.json').write_text(json.dumps(data,indent=2)+'\n')
                best=max(groups,key=lambda r:(r.get('gt_xy_fraction',0),r['n_tokens'])) if groups else None
                summary=dict(scene=scene,order=order,power_percentile=percentile,n_eligible=int(eligible.sum()),
                             n_groups=len(groups),largest_n=groups[0]['n_tokens'] if groups else 0,
                             largest_xyz_span=str(groups[0]['observed_span_xyz']) if groups else '',
                             best_gt_xy_fraction=best.get('gt_xy_fraction') if best else None,
                             best_gt_xyz_fraction=best.get('gt_xyz_fraction') if best else None,
                             best_group_n=best['n_tokens'] if best else 0,image=f'{stem}.png')
                frame_rows.append(summary)
                print(summary,flush=True)
    with (out/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(frame_rows[0]));writer.writeheader();writer.writerows(frame_rows)
    page=['<!doctype html><meta charset="utf-8"><title>Doppler spatial shapes</title>',
          '<style>body{font:16px sans-serif;max-width:1650px;margin:24px auto}img{width:100%}details{margin:24px 0}</style>',
          '<h1>单帧：相近 Doppler 的回波是否自然呈车辆形状？</h1>',
          '<p>先按相近Doppler及空间连通提取区域，不预设车辆尺寸、不用GT。对照全功率与距离带P80强回波。</p>',
          '<p>绿色虚线为现有预处理车辆GT。示例按提取后的GT俯视重叠比例选取，不能当作无偏检测结果；俯视重叠不证明覆盖了整辆车。</p>',
          '<p>灰色点线长方体仅为实际点分布的包围范围，任何点集都可被包围，不证明是车辆。</p>',
          '<p>上排依次为RGB参考、Doppler俯视图、最大连通区域；下排为示例XY、XZ和3D点分布。</p>',
          '<p><a href="summary.csv">统计表</a></p>']
    for row in frame_rows:
        page.append(f'<details open><summary>Scene {row["scene"]}, order {row["order"]}, '
                    f'power percentile {row["power_percentile"]}</summary><img loading="lazy" src="{row["image"]}"></details>')
    (out/'index.html').write_text('\n'.join(page))
    (out/'manifest.json').write_text(json.dumps(dict(args=vars(args),voxel_m=[.5,.5,1],
        spatial_connectivity=26,band_half_width_bins=1,min_tokens=30,min_cells=6,
        roi_xyz=[[0,85],[-60,60],[-5,8]],min_radius_m=8,vehicle_size_prior=False,
        azimuth_sign=geometry.azimuth_sign,elevation_sign=geometry.elevation_sign,gt_used_for_extraction=False),indent=2,default=str)+'\n')


if __name__=='__main__':main()
