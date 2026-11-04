#!/usr/bin/env python3
"""Cache GT-free XY box pairs once for all Z0-Z3 runs; no GT files are read."""
import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.temporal_vis import vis_alignment as va
from tools.temporal_vis.p1_reflection_regions import RegionConfig, extract_regions
from tools.temporal_vis.benchmark_region_matching import aligned_box, select_match
from temporal.core.ego_alignment import radar_to_lidar_default, validate_transform


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ann-file', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--bins', type=Path, default=Path('data/K-Radar-official-meta/resources/info_arr.mat'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-frames', type=int, default=0, help='Smoke cache only; incomplete caches cannot train')
    args = p.parse_args()
    cfg = RegionConfig(seed_percentile=97, seed_mad=3, merge_space='xy', xy_box_gap=(.5, .5),
                       box_size=(6, 3, 3), max_box_size=(20, 6, 6))
    ext = radar_to_lidar_default()
    policy = dict(format='box_z_pairs_v1', ann_sha256=sha(args.ann_file), bins_sha256=sha(args.bins),
        region_config=asdict(cfg), y_gates=[4, 5, 3], center_gates=[4, 6, 8],
        weights=[1, 0], radar_to_lidar=ext.tolist(), history_order='t-1,t-2,t-3',
        extractor_sha256=sha(ROOT/'tools/temporal_vis/p1_reflection_regions.py'),
        matcher_sha256=sha(ROOT/'tools/temporal_vis/benchmark_region_matching.py'), gt_used=False)
    policy = json.loads(json.dumps(policy))
    infos = va.read_infos(args.ann_file)
    groups, records, sources = {}, {}, {}
    for info in infos:
        scene = str(info.get('scene_id', info.get('scene_token')))
        groups.setdefault(scene, []).append(info)
    for scene in groups:
        groups[scene].sort(key=lambda i: int(i['order_in_scene']))
        orders = [int(i['order_in_scene']) for i in groups[scene]]
        if any(b-a != 1 for a, b in zip(orders, orders[1:])):
            raise ValueError('Use consecutive temporal subset windows, not isolated sampled frames')
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError('Use a new cache directory or --resume')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = args.output_dir/'protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_text()) != policy:
            raise ValueError('Cache protocol mismatch; choose a new directory')
    else:
        protocol.write_text(json.dumps(policy, indent=2)+'\n')
    geometry = va.RadarBinGeometry.from_file(args.bins)

    @lru_cache(maxsize=16)
    def frame(path):
        with np.load(path, allow_pickle=False) as z:
            n = 175*250
            coords = np.stack((np.repeat(np.arange(175), 250), z['azimuth_ind'][:n], z['elevation_ind'][:n]), -1)
            result = extract_regions(geometry.to_xyz(coords), z['power_val'][2, :n], cfg)
        return [r for r in result['regions'] if r['accepted']]

    def radar_path(info):
        path = Path(info.get('sparse_radar_path') or info['radar_path']).resolve()
        stat = path.stat()
        sources[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        return str(path)

    processed = 0
    for scene, scene_infos in groups.items():
        for pos, info in enumerate(scene_infos):
            if args.max_frames and processed >= args.max_frames:
                break
            order = int(info['order_in_scene'])
            key = f'{scene}_{order:05d}'
            path = args.output_dir/f'{key}.npz'
            seq = [scene_infos[max(0, pos-gap)] for gap in (3, 2, 1, 0)]
            paths = [radar_path(i) for i in seq]
            for i in seq:
                if not i.get('ego_pose_valid', False):
                    raise ValueError(f'Invalid ego pose: {scene}/{order}')
                validate_transform(np.asarray(i['ego_pose']))
            if not path.exists():
                regions = frame(paths[-1])
                current = np.array([np.r_[r['center'], r['box_size']] for r in regions]).reshape(-1, 6)
                history = np.broadcast_to(current[:, None], (len(current), 3, 6)).copy()
                matched = np.zeros((len(current), 3), bool)
                inverse = np.repeat(np.eye(4)[None], 3, axis=0)
                world_current = np.asarray(info['ego_pose']) @ ext
                for j in range(3):
                    if pos < j+1:
                        continue
                    old = scene_infos[pos-j-1]
                    relative = np.linalg.inv(world_current) @ (np.asarray(old['ego_pose']) @ ext)
                    inverse[j] = np.linalg.inv(relative)
                    candidates = []
                    for r in frame(paths[2-j]):
                        center, size = aligned_box(r['center'], r['box_size'], relative)
                        candidates.append(dict(id=r['id'], aligned_center=center, aligned_size=size))
                    for k, r in enumerate(regions):
                        h = select_match(r, candidates, j+1, 1., 0., [4, 5, 3], [4, 6, 8], match_space='xy')
                        if h is not None:
                            history[k, j] = np.r_[h['aligned_center'], h['aligned_size']]
                            matched[k, j] = True
                with path.with_suffix('.tmp').open('wb') as f:
                    np.savez_compressed(f, current=current.astype(np.float32), history=history.astype(np.float32),
                        matched=matched, inverse=inverse, radar_paths=np.array(paths),
                        source_fingerprints=np.array(json.dumps([sources[p] for p in paths])))
                path.with_suffix('.tmp').replace(path)
            with np.load(path, allow_pickle=False) as z:
                if z['radar_paths'].tolist() != paths:
                    raise ValueError(f'Sequence mismatch: {key}')
                if json.loads(str(z['source_fingerprints'])) != [sources[p] for p in paths]:
                    raise ValueError(f'Inputs changed after caching: {key}; use a new directory')
                count = len(z['current']); pair_count = int(z['matched'].sum())
            records[key] = dict(file=path.name, radar_paths=paths, boxes=count, pairs=pair_count)
            processed += 1
            if processed % 20 == 0:
                print(f'Cached {processed}/{len(infos)} scene={scene} order={order}', flush=True)
        frame.cache_clear()
    manifest = dict(policy=policy, records=records, sources=sources, complete=processed == len(infos),
                    frames=processed, ann_file=str(args.ann_file.resolve()))
    (args.output_dir/'manifest.tmp').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.output_dir/'manifest.tmp').replace(args.output_dir/'manifest.json')
    print(f'Cache frames={processed}, complete={manifest["complete"]}: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
