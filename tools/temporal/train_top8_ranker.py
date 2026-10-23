#!/usr/bin/env python3
"""Prepare bbox-proxy pairs, pretrain/evaluate a standalone 64 -> 8 scorer.

No MMCV/spconv dependency. GT exists only in cache targets and evaluation;
the shared inference ranker accepts descriptors, relative XYZ and seconds only.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from temporal.core.box_candidates import box_candidates
from temporal.core.learned_topk import (DEFAULT_POLICY, FORMAT, PairRanker,
    load_ranker, masked_topk, ranking_loss, save_ranker)
from tools.temporal_vis.audit_dynamic_objects import Dataset, memberships, transform
from tools.temporal_vis import directional_box_matching as boxes
from tools.temporal_vis import vis_alignment as va


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def prepare(args):
    if args.split == 'train' and 'test' in args.ann_file.name.lower():
        raise ValueError('A test annotation file cannot be declared a training cache')
    if args.output_dir.exists():
        raise FileExistsError('Use a new cache output directory; existing caches are never overwritten')
    args.output_dir.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    scenes = args.scenes or sorted({str(f.get('scene_id', f.get('scene_token')))
                                   for f in va.read_infos(args.ann_file)})
    shards, used_frames, exclusions = [], set(), defaultdict(int)
    raw_hash = digest_file(args.ann_file)
    for scene in scenes:
        cfg = SimpleNamespace(ann_file=args.ann_file, scene=scene, bbox_root=args.bbox_root,
            bins=args.bins, extrinsic=args.extrinsic, ego_mask_radius=0.,
            start_order=args.start_order, end_order=args.end_order, stride=args.stride,
            max_frames=args.max_frames_per_scene or sys.maxsize, history_mode='dataset',
            gaps=[1, 2, 3])
        data = Dataset(cfg)

        @lru_cache(maxsize=4)
        def p1_frame(order):
            info = data.by_order[order]
            path = info.get('sparse_radar_path') or info['radar_path']
            with np.load(path, allow_pickle=False) as raw:
                values = raw['power_val'][:8, :175*250].T.copy().astype(np.float32)
            coords, _ = va.load_p1(path)
            xyz = data.geometry.to_xyz(coords)
            # Mirror the model's per-range descriptor sort. All 250/250
            # candidates remain, including range-zero echoes (no ego mask).
            order_ids = torch.sort(torch.from_numpy(values[:, 2].copy()).reshape(175, 250),
                                   dim=1, descending=True).indices.numpy()
            order_ids = (order_ids + np.arange(175)[:, None]*250).reshape(-1)
            values, xyz = values[order_ids], xyz[order_ids]
            if values.shape[1] != 8 or not np.isfinite(values).all():
                raise ValueError(f'Invalid raw Doppler8 descriptors at {scene}/{order}')
            frame_boxes, _, _ = data.annotation(order)
            return xyz, values, memberships(transform(xyz, data.ext), frame_boxes)

        for order in data.selected():
            current, current_features, labels = p1_frame(order)
            current_boxes, object_ids, _ = data.annotation(order)
            current_pose = data.pose(order)
            for _, historical_order in data.histories(order):
                slot = data.positions[order]-data.positions[historical_order]
                history, history_features, history_labels = p1_frame(historical_order)
                history_boxes, history_ids, _ = data.annotation(historical_order)
                history_pose = data.pose(historical_order)
                dt = data.time(historical_order)-data.time(order)
                if not np.isfinite(dt) or dt >= 0:
                    raise ValueError('Real historical timestamps must precede the current frame')
                history = transform(history, np.linalg.inv(current_pose @ data.ext)
                                    @ (history_pose @ data.ext))
                queries, history_object = [], []
                # Same Stage A paired cohort. GT defines supervision/evaluation
                # membership only; neither direction nor displacement is a scorer input.
                for object_index, token in enumerate(object_ids):
                    matching = np.flatnonzero(history_ids == token)
                    if not len(matching):
                        exclusions['no_same_id_historical_box'] += 1
                        continue
                    delta, center = boxes.oracle_delta(current_boxes[object_index],
                        history_boxes[matching[0]], current_pose, history_pose, data.ext)
                    if center[0] <= 0 or abs(delta[0]) <= 1e-6:
                        exclusions['stage_a_front_or_near_zero_dx'] += 1
                        continue
                    ids = np.flatnonzero(labels == object_index)
                    queries.extend(ids.tolist())
                    history_object.extend([int(matching[0])]*len(ids))
                if not queries or not len(history):
                    continue
                query_ids = np.asarray(queries, dtype=np.int64)
                target_objects = np.asarray(history_object, dtype=np.int64)
                if args.max_queries_per_frame and len(query_ids) > args.max_queries_per_frame:
                    choice = np.sort(rng.choice(len(query_ids), args.max_queries_per_frame, replace=False))
                    query_ids, target_objects = query_ids[choice], target_objects[choice]
                half = (DEFAULT_POLICY['x_half_by_history'][slot-1], 1.5, 1.)
                candidates, valid = box_candidates(current[query_ids], history, half, k=64)
                positive = (history_labels[candidates] == target_objects[:, None]) & valid
                eligible = np.zeros(len(query_ids), dtype=bool)
                covered = np.zeros(len(query_ids), dtype=bool)
                for target in np.unique(target_objects):
                    rows = target_objects == target
                    same = history_labels == target
                    eligible[rows] = same.any()
                    if same.any():
                        _, present = box_candidates(current[query_ids[rows]], history[same], half, k=1)
                        covered[rows] = present[:, 0]
                filename = f'pairs_{len(shards):07d}.npz'
                np.savez_compressed(args.output_dir / filename,
                    query=current_features[query_ids], history=history_features[candidates],
                    relative_xyz=history[candidates]-current[query_ids, None],
                    delta_t=np.full(len(query_ids), dt, dtype=np.float32), valid=valid,
                    positive=positive, loss_valid=valid & (history_labels[candidates] != -2),
                    eligible=eligible, covered=covered,
                    slot=np.full(len(query_ids), slot, dtype=np.int64))
                shards.append(dict(file=filename, count=len(query_ids), scene=str(scene),
                                   order=order, history_order=historical_order, slot=slot))
                used_frames.update((f'{scene}/{order}', f'{scene}/{historical_order}'))
            print(f'prepare scene={scene} order={order} shards={len(shards)}', flush=True)
        p1_frame.cache_clear()
        data.p1.cache_clear()
        data.annotation.cache_clear()
    if not shards:
        raise ValueError('No paired queries; check annotations, scenes, poses and bbox paths')
    write_json(args.output_dir/'manifest.json', dict(format=FORMAT, split=args.split,
        policy=DEFAULT_POLICY, annotation_sha256=raw_hash, bins_sha256=digest_file(args.bins),
        extrinsic_sha256=digest_file(args.extrinsic) if args.extrinsic else 'repository_default',
        frames=sorted(used_frames), shards=shards, exclusions=dict(exclusions),
        sampling=dict(seed=args.seed, stride=args.stride, max_queries_per_frame=args.max_queries_per_frame,
                      max_frames_per_scene=args.max_frames_per_scene, start_order=args.start_order,
                      end_order=args.end_order),
        source_sha256=digest_file(__file__)))


class PairCache(TorchDataset):
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if self.manifest['format'] != FORMAT or self.manifest['policy'] != DEFAULT_POLICY:
            raise ValueError('Incompatible cache format/search/descriptor/time policy')
        self.shards = self.manifest['shards']
        self.ends = np.cumsum([s['count'] for s in self.shards])
        if not len(self.ends) or self.ends[-1] <= 0:
            raise ValueError('Empty pair cache')

    def __len__(self):
        return int(self.ends[-1])

    @lru_cache(maxsize=2)
    def shard(self, index):
        with np.load(self.root/self.shards[index]['file'], allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}

    def __getitem__(self, index):
        shard = int(np.searchsorted(self.ends, index, side='right'))
        local = index-(int(self.ends[shard-1]) if shard else 0)
        return {key: np.array(value[local], copy=True) for key, value in self.shard(shard).items()}


class ShardShuffleSampler(torch.utils.data.Sampler):
    """Shuffle within shards and shuffle shards, avoiding random NPZ decompression per row."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        self.epoch += 1
        for shard in rng.permutation(len(self.dataset.shards)):
            start = int(self.dataset.ends[shard-1]) if shard else 0
            yield from (start+rng.permutation(self.dataset.shards[shard]['count'])).tolist()

    def __len__(self):
        return len(self.dataset)


def ensure_disjoint(train, val):
    if train.manifest['split'] != 'train' or val.manifest['split'] != 'val':
        raise ValueError('Training requires explicit train and val caches; test is evaluation only')
    overlap = set(train.manifest['frames']) & set(val.manifest['frames'])
    if overlap:
        raise ValueError(f'Train/val leakage including history frames: {sorted(overlap)[:5]}')
    for key in ('bins_sha256', 'extrinsic_sha256'):
        if train.manifest[key] != val.manifest[key]:
            raise ValueError(f'Train/val geometry mismatch: {key}')


def on_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def score_batch(model, batch):
    return model(batch['query'], batch['history'], batch['relative_xyz'], batch['delta_t'])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = defaultdict(lambda: defaultdict(int))
    for batch in loader:
        batch = on_device(batch, device)
        order, good = masked_topk(score_batch(model, batch), batch['valid'], 8)
        selections = [('distance_top8', batch['positive'][:, :8], batch['valid'][:, :8]),
                      ('learned_top8', batch['positive'].gather(1, order), good),
                      ('pool_top64', batch['positive'], batch['valid'])]
        for slot in batch['slot'].unique().tolist():
            rows = (batch['slot'] == slot) & batch['eligible']
            for method, pos, valid in selections:
                result = totals[(int(slot), method)]
                result['eligible'] += int(rows.sum())
                result['covered'] += int((batch['covered'] & rows).sum())
                success = (pos & valid).any(1)
                result['success'] += int((success & rows).sum())
                result['selected'] += int(valid[rows].sum())
                result['same'] += int((pos & valid)[rows].sum())
    rows = []
    for (slot, method), counts in sorted(totals.items()):
        n = counts['eligible']
        rows.append(dict(history_slot=slot, method=method, **dict(counts),
            coverage=counts['covered']/n if n else None,
            recall=counts['success']/n if n else None,
            precision=counts['same']/counts['selected'] if counts['selected'] else None,
            outside_box=n-counts['covered'], topk_miss=counts['covered']-counts['success'],
            mean_selected=counts['selected']/n if n else None))
    return rows


def write_metrics(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    training, validation = PairCache(args.train_cache), PairCache(args.val_cache)
    ensure_disjoint(training, validation)
    args.work_dir.mkdir(parents=True, exist_ok=False)
    model = PairRanker(args.hidden_dims).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(training, batch_size=args.batch_size,
                        sampler=ShardShuffleSampler(training, args.seed), num_workers=0)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=0)
    write_json(args.work_dir/'run.json', dict(arguments={k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()}, policy=DEFAULT_POLICY,
        train_manifest_sha256=digest_file(args.train_cache/'manifest.json'),
        val_manifest_sha256=digest_file(args.val_cache/'manifest.json')))
    best = -1.
    for epoch in range(1, args.epochs+1):
        model.train()
        loss_sum, updates = 0., 0
        for batch in loader:
            batch = on_device(batch, args.device)
            if not (batch['positive'] & batch['loss_valid']).any():
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = ranking_loss(score_batch(model, batch), batch['positive'], batch['loss_valid'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite ranking loss; inspect raw descriptors/cache')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += float(loss.detach())
            updates += 1
        if not updates:
            raise ValueError('No queries with a valid positive; no ranking supervision')
        rows = evaluate(model, val_loader, args.device)
        scored = [row for row in rows if row['method'] == 'learned_top8']
        denominator = sum(row['eligible'] for row in scored)
        if not denominator:
            raise ValueError('No eligible validation queries')
        metric = sum(row['success'] for row in scored)/denominator
        write_metrics(args.work_dir/f'val_epoch_{epoch:03d}.csv', rows)
        metadata = dict(epoch=epoch, val_recall_at8=metric,
            train_manifest_sha256=digest_file(args.train_cache/'manifest.json'),
            val_manifest_sha256=digest_file(args.val_cache/'manifest.json'))
        save_ranker(args.work_dir/'last.pth', model, **metadata)
        if metric > best:
            best = metric
            save_ranker(args.work_dir/'best.pth', model, **metadata)
            write_metrics(args.work_dir/'best_val.csv', rows)
        print(f'epoch={epoch} loss={loss_sum/updates:.6f} val_recall@8={metric:.4%} best={best:.4%}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    cache = commands.add_parser('prepare')
    cache.add_argument('--ann-file', type=Path, required=True)
    cache.add_argument('--bbox-root', type=Path, default=Path('data/K-RadarOcc/train'))
    cache.add_argument('--bins', type=Path, default=Path('data/K-Radar-official-meta/resources/info_arr.mat'))
    cache.add_argument('--extrinsic', type=Path)
    cache.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    cache.add_argument('--scenes', nargs='+')
    cache.add_argument('--start-order', type=int, default=0)
    cache.add_argument('--end-order', type=int, default=sys.maxsize)
    cache.add_argument('--stride', type=int, default=1)
    cache.add_argument('--max-frames-per-scene', type=int, default=0, help='0: all selected frames')
    cache.add_argument('--max-queries-per-frame', type=int, default=512, help='Per history pair; 0: all eligible cohort queries')
    cache.add_argument('--seed', type=int, default=42)
    cache.add_argument('--output-dir', type=Path, required=True)
    fit = commands.add_parser('train')
    fit.add_argument('--train-cache', type=Path, required=True)
    fit.add_argument('--val-cache', type=Path, required=True)
    fit.add_argument('--work-dir', type=Path, default=Path('work_dirs/p1_top8_ranker'))
    fit.add_argument('--epochs', type=int, default=20)
    fit.add_argument('--batch-size', type=int, default=256)
    fit.add_argument('--hidden-dims', type=int, default=64)
    fit.add_argument('--lr', type=float, default=1e-3)
    fit.add_argument('--seed', type=int, default=42)
    fit.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    test = commands.add_parser('evaluate')
    test.add_argument('--cache', type=Path, required=True)
    test.add_argument('--checkpoint', type=Path, required=True)
    test.add_argument('--output', type=Path, required=True)
    test.add_argument('--batch-size', type=int, default=256)
    test.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.stride <= 0 or min(args.max_frames_per_scene, args.max_queries_per_frame) < 0:
            parser.error('Positive stride and nonnegative sampling caps required')
    elif args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    elif args.command == 'train' and (args.epochs <= 0 or args.lr <= 0 or args.hidden_dims <= 0):
        parser.error('Positive epochs, learning rate and hidden dims required')
    return args


def main():
    args = parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'train':
        train(args)
    else:
        data = PairCache(args.cache)
        model, _ = load_ranker(args.checkpoint, DEFAULT_POLICY)
        model.to(args.device)
        rows = evaluate(model, DataLoader(data, batch_size=args.batch_size, num_workers=0), args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_metrics(args.output, rows)
        print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
