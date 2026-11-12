#!/usr/bin/env python3
"""Single raw 4D tensor: spatial clusters with identical/nearby Top-3 Doppler.

No sparse radar NPZ, eight-channel descriptor, fixed per-range top-k, learned
model or GT is used for extraction. GT/RGB are optional post-hoc references.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/radarocc_mpl')
from tools.visualize_radarocc_sparse_rd_ra import load_raw
from tools.temporal_vis.audit_local_doppler_similarity import cosine_pair_mean
from tools.temporal_vis.neighborhood_motion import load_doppler_table
from tools.temporal_vis import vis_alignment as va
from tools.temporal_vis.vis_p1_reflection_regions import annotation_scenes, load_gt, camera_directory
from tools.temporal_vis.inspect_doppler_blobs import evaluate, gt_projection

OFFICIAL_COORDINATES = 'https://github.com/kaist-avelab/K-Radar/blob/main/tools/radar_film/get_pw_dist.py'


def top3_from_raw(raw, chunk=65536):
    if raw.ndim != 4 or raw.shape[0] != 64 or np.iscomplexobj(raw):
        raise ValueError('Expected nonnegative real DREA power, D=64')
    flat = raw.reshape(64, -1)
    n = flat.shape[1]
    bins = np.empty((n,3), np.uint8)
    powers = np.empty((n,3), np.float64)
    total = np.empty(n, np.float64)
    for start in range(0,n,chunk):
        part = flat[:,start:start+chunk]
        if not np.isfinite(part).all() or (part < 0).any():
            raise ValueError('Raw power contains nonfinite/negative values')
        indices = np.argpartition(part,-3,axis=0)[-3:]
        values = np.take_along_axis(part,indices,axis=0)
        order = np.argsort(values,axis=0)[::-1]
        stop = start + part.shape[1]
        bins[start:stop] = np.take_along_axis(indices,order,axis=0).T
        powers[start:stop] = np.take_along_axis(values,order,axis=0).T
        total[start:stop] = part.sum(0)
    return bins,powers,total


def top3_band_mask(bins,powers,center,mode):
    inside = np.abs((bins.astype(np.int16)-center+32)%64-32) <= 1
    if mode == 'exact':
        # Three distinct largest bins in a three-bin band = the same unordered
        # Top-3 set for every member, including wrap-around [63,0,1].
        return inside.all(1)
    if mode != 'approx':
        raise ValueError('Mode must be exact or approx')
    return (inside.sum(1) >= 2) & ((powers*inside).sum(1) >= .8*powers.sum(1))


def extract(raw,bins,powers,total,xyz,ranges,percentile,mode,min_range=8,min_points=30):
    shape = raw.shape[1:]
    cutoff = np.percentile(powers[:,0].reshape(shape),percentile,axis=(1,2))
    cutoff_per_point = np.repeat(cutoff,shape[1]*shape[2])
    strong = (ranges >= min_range) & (powers[:,0] > 0) & (powers[:,0] >= cutoff_per_point)
    candidates=[]
    for center in range(64):
        mask = strong & top3_band_mask(bins,powers,center,mode)
        labels,_ = ndimage.label(mask.reshape(shape),structure=np.ones((3,3,3),bool))
        counts = np.bincount(labels.ravel())
        for label in np.flatnonzero(counts >= min_points):
            if label:
                candidates.append((np.flatnonzero(labels.ravel()==label),center))
    candidates.sort(key=lambda item:(-len(item[0]),item[1],int(item[0].min())))
    used=np.zeros(len(bins),bool)
    assignment=np.full(len(bins),-1,np.int32)
    groups=[]
    flat=raw.reshape(64,-1)
    for member,center in candidates:
        if used[member].mean() > .5:
            continue
        used[member]=True
        gid=len(groups)
        assignment[member]=gid
        sets,counts=np.unique(np.sort(bins[member],axis=1),axis=0,return_counts=True)
        k=int(counts.argmax())
        low,high=xyz[member].min(0),xyz[member].max(0)
        spectra=flat[:,member].T
        power=powers[member,0]
        q10,q90=np.percentile(power,[10,90])
        groups.append(dict(id=gid,point_ids=member.tolist(),n_tokens=len(member),
            band_center_bin=center,top3_set_mode=sets[k].tolist(),top3_set_mode_fraction=float(counts[k]/len(member)),
            observed_min=low.tolist(),observed_max=high.tolist(),observed_span_xyz=(high-low).tolist(),
            center_xyz=xyz[member].mean(0).tolist(),
            median_top3_spectrum_power_fraction=float(np.median(powers[member].sum(1)/total[member])),
            mean_full_spectrum_pair_cosine=cosine_pair_mean(spectra),
            peak_power_p90_p10_db=float(10*np.log10(q90/q10))))
    return groups,strong,assignment,cutoff


def references(raw_path,geometry):
    scene=raw_path.parent.parent.name
    frame=int(raw_path.stem.split('_')[-1])
    matches=[]
    for split in ['train','val','test']:
        path=ROOT/f'data/annotations/kradar_dict_{split}_official_temporal_doppler8.pkl'
        rows=annotation_scenes(path).get(scene,[])
        for info in rows:
            if int(info.get('radar_frame_id',-1))==frame:
                matches.append((info,rows,path))
    if len(matches)!=1:
        return None,None,dict(reason=f'Expected one associated annotation entry, found {len(matches)}')
    info,rows,path=matches[0]
    gt=load_gt(info,scene,ROOT/'data/K-RadarOcc/train',va.geometry_module.radar_to_lidar_default())
    directory=camera_directory(ROOT,scene,None)
    rgb,mapping=va.resolve_rgb(info,rows,camera_dir=directory) if directory else (None,None)
    return gt,rgb,dict(annotation=str(path),scene=scene,radar_frame_id=frame,
                       order_in_scene=int(info['order_in_scene']),rgb_mapping=mapping,
                       gt_note='Existing processed GT, reference only; not the extraction boundary')


def scatter(ax,xyz,ids,color,table,size=2):
    return ax.scatter(xyz[ids,0],xyz[ids,1],c=color,s=size,cmap='turbo',
                      vmin=table.min(),vmax=table.max(),rasterized=True)


def overview(out,xyz,bins,powers,variants,table,gt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(18,11),layout='constrained')
    ids=np.flatnonzero(variants['p95_exact']['strong'])
    for rank,ax in enumerate(axes[0]):
        artist=scatter(ax,xyz,ids,table[bins[ids,rank]],table)
        gt_projection(ax,gt)
        ax.set_title(f'Raw Doppler Top-{rank+1}: same {len(ids):,} P95 spatial cells')
        fig.colorbar(artist,ax=ax,label='Folded Doppler (m/s)',fraction=.035)
    for ax,key in zip(axes[1],['p95_exact','p99_exact','p99_approx']):
        variant=variants[key]
        strong_ids=np.flatnonzero(variant['strong'])
        ax.scatter(xyz[strong_ids,0],xyz[strong_ids,1],s=1,color='.8',alpha=.5)
        for group in variant['groups'][:20]:
            member=np.asarray(group['point_ids'])
            color=plt.get_cmap('tab20')(group['id']%20)
            ax.scatter(xyz[member,0],xyz[member,1],s=4,color=color)
            ax.text(*np.asarray(group['center_xyz'])[:2],f'C{group["id"]}',fontsize=7,color=color)
        gt_projection(ax,gt)
        ax.set_title(f'{key}: {len(variant["groups"])} groups; largest 20 shown\n'
                     'No vehicle-size prior, no GT used for clustering')
    for ax in axes.ravel():
        ax.set(xlim=(0,118),ylim=(-45,45),xlabel='X forward (m)',ylabel='Y left (m)')
        ax.set_aspect('equal')
        ax.grid(alpha=.12)
    fig.suptitle('Raw tesseract_00034.mat: Top-3 Doppler and spatial components\n'
                 'Official azimuth/elevation sign conversion; green = post-hoc processed GT; '
                 'statistics use full height/FOV, displayed XY is a crop',fontsize=12)
    fig.savefig(out/'overview.png',dpi=140)
    plt.close(fig)


def detail(out,key,group,raw,xyz,bins,powers,table,gt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    member=np.asarray(group['point_ids'])
    pts=xyz[member]
    low,high=pts.min(0),pts.max(0)
    fig=plt.figure(figsize=(17,10),layout='constrained')
    axes=[fig.add_subplot(2,3,i+1) for i in range(3)]
    for ax,(a,b) in zip(axes,[(0,1),(0,2),(1,2)]):
        ax.scatter(pts[:,a],pts[:,b],c=table[bins[member,0]],s=13,cmap='turbo',vmin=table.min(),vmax=table.max())
        gt_projection(ax,gt,(a,b))
        ax.set(xlim=(low[a]-1.5,high[a]+1.5),ylim=(low[b]-1.5,high[b]+1.5),
               xlabel=f'{"XYZ"[a]} (m)',ylabel=f'{"XYZ"[b]} (m)',title=f'Actual {"XYZ"[a]}{"XYZ"[b]} support; no cuboid fitting')
        ax.set_aspect('equal')
    ax=fig.add_subplot(2,3,4,projection='3d')
    ax.scatter(*pts.T,c=table[bins[member,0]],s=9,cmap='turbo',vmin=table.min(),vmax=table.max())
    span=high-low;center=(high+low)/2;half=max(span.max()+1,4)/2
    ax.set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half),
           zlim=(center[2]-half,center[2]+half),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',
           title=f'Actual 3D span {span[0]:.2f} x {span[1]:.2f} x {span[2]:.2f} m')
    ax.set_box_aspect((1,1,1));ax.view_init(22,-55)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.set_major_locator(MaxNLocator(4))
    spectra=raw.reshape(64,-1)[:,member].T
    normalized=spectra/spectra.max(1,keepdims=True)
    ax=fig.add_subplot(2,3,5)
    for row in normalized[np.linspace(0,len(member)-1,min(24,len(member)),dtype=int)]:
        ax.plot(table,row,lw=.7,alpha=.25)
    ax.plot(table,normalized.mean(0),color='black',lw=2,label='All-member mean')
    for b in group['top3_set_mode']:ax.axvline(table[b],ls=':',lw=1,color='red')
    ax.set(xlabel='Folded Doppler (m/s)',ylabel='Power / per-cell peak',title='Complete raw 64-bin spectra; red = common Top-3 set')
    ax.legend(fontsize=8)
    ax=fig.add_subplot(2,3,6)
    ordered_ids=member[np.argsort(xyz[member,0])]
    histogram=np.zeros((len(ordered_ids),64))
    histogram[np.arange(len(ordered_ids))[:,None],bins[ordered_ids]]=powers[ordered_ids]/powers[ordered_ids,0,None]
    artist=ax.imshow(histogram,origin='lower',aspect='auto',cmap='magma',vmin=0,vmax=1,
                     extent=(-.5,63.5,-.5,len(ordered_ids)-.5),interpolation='nearest')
    fig.colorbar(artist,ax=ax,label='Relative Top-3 power',fraction=.035)
    ax.set(xlabel='Doppler bin index',ylabel='Spatial cell, sorted by X',title='Per-cell Top-3 bins and their relative power')
    fig.suptitle(f'{key} C{group["id"]}: n={len(member)}, common Top-3 set={group["top3_set_mode"]} '
                 f'({group["top3_set_mode_fraction"]:.1%})\n'
                 f'Full-spectrum pair cosine={group["mean_full_spectrum_pair_cosine"]:.3f}; '
                 f'post-hoc GT XY={group.get("gt_xy_fraction",0):.1%}, XYZ={group.get("gt_xyz_fraction",0):.1%}',fontsize=12)
    name=f'{key}_C{group["id"]}.png'
    fig.savefig(out/name,dpi=135);plt.close(fig)
    return name


def interactive(out,xyz,bins,variant,table,gt):
    import plotly.graph_objects as go
    ids=np.flatnonzero(variant['strong'])
    assignment=variant['assignment'][ids]
    custom=np.column_stack([bins[ids],assignment])
    traces=[]
    for rank in range(3):
        traces.append(go.Scatter3d(x=xyz[ids,0],y=xyz[ids,1],z=xyz[ids,2],mode='markers',
            name=f'Top-{rank+1}',visible=rank==0,customdata=custom,
            marker=dict(size=2,color=table[bins[ids,rank]],colorscale='Turbo',cmin=table.min(),cmax=table.max(),
                        colorbar=dict(title='Doppler m/s'),opacity=.8),
            hovertemplate='XYZ=(%{x:.2f},%{y:.2f},%{z:.2f}) m<br>Top3 bins=%{customdata[0]},%{customdata[1]},%{customdata[2]}<br>Group=%{customdata[3]}<extra></extra>'))
    traces.append(go.Scatter3d(x=xyz[ids,0],y=xyz[ids,1],z=xyz[ids,2],mode='markers',name='Top3 clusters',visible=False,
        customdata=custom,marker=dict(size=2,color=assignment,colorscale='Turbo',opacity=.8,
                                      colorbar=dict(title='Group ID; -1 unassigned')),
        hovertemplate='XYZ=(%{x:.2f},%{y:.2f},%{z:.2f}) m<br>Top3 bins=%{customdata[0]},%{customdata[1]},%{customdata[2]}<br>Group=%{customdata[3]}<extra></extra>'))
    if gt:
        for corners,identifier in zip(gt['corners'],gt['ids']):
            if corners[:,0].min()>118:continue
            # load_gt corner order is itertools.product of the three bits.
            bits=np.array(list(__import__('itertools').product((0,1),repeat=3)))
            lines=[]
            for i in range(8):
                for j in range(i+1,8):
                    if np.count_nonzero(bits[i]!=bits[j])==1:lines.extend([corners[i],corners[j],[None]*3])
            xx,yy,zz=zip(*lines)
            traces.append(go.Scatter3d(x=xx,y=yy,z=zz,mode='lines',name=f'Processed GT {identifier}',
                                       line=dict(color='green',width=3),hoverinfo='name'))
    buttons=[]
    for i,label in enumerate(['主峰 Top1','次峰 Top2','第三峰 Top3','Top3 连通簇']):
        buttons.append(dict(label=label,method='update',args=[dict(visible=[j==i for j in range(4)]+[True]*(len(traces)-4))]))
    fig=go.Figure(traces)
    fig.update_layout(title='原始帧00034：P99强回波的Top3（拖动旋转，滚轮缩放；绿色GT仅事后参考）',
        scene=dict(xaxis_title='X forward (m)',yaxis_title='Y left (m)',zaxis_title='Z up (m)',aspectmode='data'),
        updatemenus=[dict(buttons=buttons,direction='right',x=0,y=1.1)],height=850)
    fig.write_html(out/'interactive_3d.html',include_plotlyjs=True,full_html=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=Path('data/K-Radar/11/radar_tensor_8doppler/tesseract_00034.mat'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--min-range',type=float,default=8)
    parser.add_argument('--min-points',type=int,default=30)
    parser.add_argument('--no-references',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):parser.error('Use a new or empty output directory')
    if args.min_range<0 or args.min_points<2:parser.error('Invalid range/count')
    out.mkdir(parents=True,exist_ok=True)
    raw_path=args.raw.resolve();raw=load_raw(raw_path)
    bins,powers,total=top3_from_raw(raw)
    geometry=va.RadarBinGeometry.from_file(ROOT/'data/K-Radar-official-meta/resources/info_arr.mat',
                                         azimuth_sign=-1,elevation_sign=-1)
    table=load_doppler_table(ROOT/'data/K-Radar-official-meta/resources/arr_doppler.mat')
    r,e,a=np.indices(raw.shape[1:])
    coords=np.column_stack([r.ravel(),a.ravel(),e.ravel()])
    xyz=geometry.to_xyz(coords);ranges=geometry.bins[0][r.ravel()]
    variants={}
    for percentile in [95,99]:
        for mode in ['exact','approx']:
            key=f'p{percentile}_{mode}'
            groups,strong,assignment,cutoff=extract(raw,bins,powers,total,xyz,ranges,percentile,mode,args.min_range,args.min_points)
            variants[key]=dict(groups=groups,strong=strong,assignment=assignment,cutoff=cutoff)
            print(key,'strong',int(strong.sum()),'groups',len(groups),flush=True)
    # Read metadata/GT/RGB only after all raw-only extraction variants finish.
    gt,rgb,ref=(None,None,{}) if args.no_references else references(raw_path,geometry)
    for v in variants.values():evaluate(v['groups'],xyz,gt)
    from PIL import Image
    if rgb:
        with Image.open(rgb) as image:image.convert('RGB').save(out/'rgb_reference.jpg',quality=90)
    overview(out,xyz,bins,powers,variants,table,gt)
    illustrations=[]
    for key in ['p95_exact','p99_exact','p99_approx']:
        groups=variants[key]['groups']
        # Include largest clusters regardless of GT; also illustrate best GT
        # overlap, explicitly marked as a post-hoc choice.
        ids=[0,1]
        if gt and groups:ids.append(max(groups,key=lambda g:(g.get('gt_xyz_fraction',0),g['n_tokens']))['id'])
        for gid in dict.fromkeys(ids):
            if gid>=len(groups):continue
            group=groups[gid]
            name=detail(out,key,group,raw,xyz,bins,powers,table,gt)
            illustrations.append(dict(image=name,variant=key,group=gid,
                                      selection='largest' if gid in [0,1] else 'highest post-hoc GT XYZ overlap'))
    interactive(out,xyz,bins,variants['p99_approx'],table,gt)
    rows=[]
    for key,v in variants.items():
        (out/f'{key}.json').write_text(json.dumps(dict(variant=key,groups=v['groups'],cutoff_by_range=v['cutoff'].tolist(),
                                                      gt_used_for_extraction=False),indent=2)+'\n')
        for group in v['groups']:
            rows.append(dict(variant=key,**{k:(json.dumps(val) if isinstance(val,list) else val)
                                           for k,val in group.items() if k!='point_ids'}))
    if rows:
        with (out/'clusters.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(out/'raw_top3.npz',spatial_shape=raw.shape[1:],top3_bins=bins,top3_powers=powers,
                        total_spectrum_power=total,**{f'{k}_assignment':v['assignment'] for k,v in variants.items()})
    sha=hashlib.sha256()
    with raw_path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):sha.update(chunk)
    report=dict(source=str(raw_path),sha256=sha.hexdigest(),key='arrDREA',shape=raw.shape,
        n_spatial_cells=len(bins),n_doppler_bins=64,bin_step_mps=float(table[1]-table[0]),
        top3_definition='Three largest Doppler-bin powers, sorted descending, not three separate local maxima',
        sparse_npz_read=False,eight_dim_descriptor_used=False,fixed_spatial_topk=False,gt_used_for_extraction=False,
        min_range_m=args.min_range,min_points=args.min_points,connectivity='26-connected on original R,E,A sampling grid',
        power_selection='P95/P99 of strongest-bin power within each raw range over all E,A cells',
        exact='All three distinct Top3 bins in the same circular three-bin band: identical unordered set',
        approx='At least two Top3 bins in that band and >=80% of Top3 power inside it',
        coordinate_signs=dict(azimuth=-1,elevation=-1,source=OFFICIAL_COORDINATES),
        references=ref,illustrations=illustrations,
        variants={key:dict(n_strong_cells=int(v['strong'].sum()),n_clusters=len(v['groups']),
                           n_unique_cluster_cells=int((v['assignment']>=0).sum())) for key,v in variants.items()})
    (out/'manifest.json').write_text(json.dumps(report,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')
    page=['<!doctype html><meta charset="utf-8"><title>原始雷达帧 Top3 Doppler 簇</title>',
          '<style>body{font:16px sans-serif;max-width:1600px;margin:24px auto}img{width:100%}details{margin:20px 0}</style>',
          '<h1>tesseract_00034.mat：从原始64-bin频谱检查Top3空间簇</h1>',
          '<p>完整64×256×37×107张量，未读取稀疏NPZ或8维描述符。Top3为三个最大bin，不要求三个独立局部峰。</p>',
          '<p>exact：簇内前三峰的bin集合完全一致；approx：同一三bin带内至少含两个Top3峰，且占Top3功率至少80%。</p>',
          '<p>P95/P99是每个距离单元内的功率分位阈值，不固定点数；统计保留完整高度/FOV，簇分析排除距离8米以内。</p>',
          f'<p>XYZ按<a href="{OFFICIAL_COORDINATES}">官方代码</a>将方位角、俯仰角取负。GT仅事后对照，不参与提取或形状限制。</p>',
          '<p><a href="interactive_3d.html">可旋转3D：切换Top1/Top2/Top3/簇</a> · <a href="clusters.csv">所有簇统计</a> · '
          '<a href="manifest.json">参数与来源</a> · <a href="raw_top3.npz">原始Top3数组</a></p>',
          '<p>总览：上排相同强回波位置的Top1、Top2、Top3；下排为不同规则的空间连通簇。绿色为已有预处理GT。</p>',
          '<img src="overview.png">']
    if rgb:page+=['<details><summary>同帧关联RGB参考（无投影框）</summary><img src="rgb_reference.jpg"></details>']
    for item in illustrations:
        page.append(f'<details open><summary>{item["variant"]} C{item["group"]}：{item["selection"]}</summary>'
                    f'<img loading="lazy" src="{item["image"]}"></details>')
    (out/'index.html').write_text('\n'.join(page))
    print('OUTPUT',out,flush=True)


if __name__=='__main__':main()
