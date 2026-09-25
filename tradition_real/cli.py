"""Run from repository root: python -m tradition_real {preprocess,train,evaluate}."""
import argparse
from collections import Counter
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import re
import joblib
import numpy as np
from .config import Config
from .frame_fusion import CacheReader, CacheWriter, signature
from .pipeline import Pipeline
from .adapters.dataset import Dataset
from .adapters.paths import _camera_map
from .semantics.random_forest import RandomForest, train, training_target, annotation_sha256, FORMAT
from .semantics.features import FEATURE_NAMES
from .evaluation.radarocc_metrics import RadarOccMetricAccumulator, radarocc_metric_dict


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['preprocess', 'train', 'evaluate'])
    p.add_argument('--annotation', required=True, help='Train split for train; held-out test/validation split for evaluate')
    p.add_argument('--radar-root', default='data/K-Radar_rpc')
    p.add_argument('--pose-root', default='data/K-RadarOcc')
    p.add_argument('--calib-root', help='Raw mode default: data/K-Radar_calib; cache mode: optional validation')
    p.add_argument('--radar-z', type=float, help='Radar origin Z; defaults to cache/model value or -0.7')
    p.add_argument('--gt-root')
    p.add_argument('--gt-order', choices=['xyz', 'zyx'], default='xyz')
    p.add_argument('--repo-root', default='.')
    p.add_argument('--output', required=True, help='preprocess: data/frame_fusion_octomap/SPLIT; train/evaluate: work_dirs/tradition_real/RUN')
    p.add_argument('--model', help='Required for evaluate; train defaults to OUTPUT/random_forest.joblib')
    p.add_argument('--config', help='JSON Config overrides; defaults to model/cache configuration')
    p.add_argument('--frame-fusion-root', help='Read cached mapping + 42D; no RPC/pose/DBSCAN needed')
    p.add_argument('--temporal-window', type=int, help='Optional config override; must match RF when evaluating')
    p.add_argument('--scenes', nargs='+', help='Limit data scenes; omit for complete split evaluation')
    p.add_argument('--max-frames', type=int, help='Smoke test only; omit for complete split')
    p.add_argument('--n-estimators', type=int, default=200)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--max-depth', type=int, default=18, help='0 means unlimited')
    p.add_argument('--min-samples-leaf', type=int, default=2)
    p.add_argument('--class-weight', choices=['balanced_subsample', 'balanced', 'none'], default='balanced_subsample')
    p.add_argument('--n-jobs', type=int, default=-1)
    p.add_argument('--positive-fraction', type=float, default=.2)
    p.add_argument('--negative-fraction', type=float, default=.05)
    p.add_argument('--save-predictions', action='store_true')
    p.add_argument('--video-scene', default='3', help='Rendering only, does NOT filter evaluation')
    p.add_argument('--camera-dir', help='Enables video for video-scene, all its frames by default')
    p.add_argument('--video-start', type=int, default=0, help='Inclusive zero-based scene ordinal')
    p.add_argument('--video-end', type=int, help='Inclusive scene ordinal; omitted = last frame')
    p.add_argument('--video-fps', type=float, default=10.)
    p.add_argument('--video-background-prediction-root', help='RadarOcc pred_c.npy root for the original blue visualization base')
    p.add_argument('--keep-frames', action='store_true', help='Keep the original renderer intermediate PNG frames')
    return p


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 <= args.negative_fraction < args.positive_fraction <= 1:
        raise ValueError('Require 0 <= negative-fraction < positive-fraction <= 1')
    if args.video_start < 0 or (args.video_end is not None and args.video_end < args.video_start) or args.video_fps <= 0:
        raise ValueError('Invalid video range or frame rate')
    if args.command == 'evaluate' and not args.model:
        raise ValueError('Evaluation requires --model trained with tradition_real')
    if args.command != 'evaluate' and args.camera_dir:
        raise ValueError('Video is available for evaluate')
    if args.command == 'preprocess' and args.frame_fusion_root:
        raise ValueError('preprocess reads RPC inputs; --frame-fusion-root is for train/evaluate')
    if args.n_estimators < 1 or args.max_depth < 0 or args.min_samples_leaf < 1 or args.n_jobs == 0:
        raise ValueError('Invalid RF hyperparameters')
    cache = CacheReader(args.frame_fusion_root) if args.frame_fusion_root else None
    bundle = joblib.load(args.model) if args.command == 'evaluate' else None
    if bundle is not None and (not isinstance(bundle, dict) or bundle.get('format') != FORMAT):
        raise ValueError('Old tradition / Autoware-GM2019 RF is incompatible; train OctoMap tradition_real first')
    raw_config = dict(bundle['config'] if bundle else (cache.manifest['config'] if cache else {}))
    if args.config:
        raw_config.update(json.loads(Path(args.config).read_text()))
    if args.radar_z is None:
        args.radar_z = cache.manifest['signature']['radar_z'] if cache else (bundle['metadata']['radar_z'] if bundle else -.7)
    for key in ('shape_xyz', 'min_xyz'):
        if key in raw_config:
            raw_config[key] = tuple(raw_config[key])
    legacy_keys = {'p_free', 'prior', 'occupied_threshold', 'free_threshold'} & raw_config.keys()
    if legacy_keys:
        raise ValueError(f'Old Autoware/GM2019 config fields {sorted(legacy_keys)}; use OctoMap config and rebuild cache')
    cfg = Config(**raw_config)
    if args.temporal_window is not None:
        cfg = replace(cfg, temporal_window=args.temporal_window)
    if cfg.shape_xyz != (128, 128, 14) or cfg.min_xyz != (0., -25.6, -2.6) or cfg.resolution != .4:
        raise ValueError('Dataset CLI requires the standard RadarOcc Small evaluation grid')
    output = Path(args.output).resolve()
    # Prevent accidental overwriting of source trees with results.
    source_dirs = [Path(__file__).resolve().parent, Path(__file__).resolve().parents[1] / 'tradition']
    if any(output == s or s in output.parents for s in source_dirs):
        raise ValueError('Choose an output directory under work_dirs, outside source code')
    if cache and (output == cache.root or cache.root in output.parents):
        raise ValueError('Training/evaluation output must be outside the input cache')
    output.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(args.annotation, args.radar_root, args.pose_root, args.calib_root or 'data/K-Radar_calib',
                      args.repo_root, args.gt_root, args.gt_order, args.radar_z,
                      args.scenes, args.max_frames, load_mapping=cache is None,
                      load_gt=args.command != 'preprocess')
    if cache:
        cache.validate(cfg, args.radar_z, args.annotation, dataset.infos, args.calib_root)
    model = RandomForest.load(args.model, cfg) if bundle else None
    keys = [(str(i['scene_token']), str(i['lidar_token'])) for i in dataset.infos]
    preprocessing_signature = signature(cfg, args.radar_z)
    if model:
        if model.metadata.get('preprocessing_signature', preprocessing_signature) != preprocessing_signature:
            raise ValueError('RF preprocessing source/signature differs from this cache/run; retrain RF')
        # Same-scene calibration must agree whenever train/test subsets share a scene.
        if cache:
            for scene, calibration in cache.manifest['calibrations'].items():
                saved = model.metadata.get('calibrations', {}).get(scene)
                if saved and (saved.get('translation') != calibration['translation'] or
                              ('sha256' in saved and saved['sha256'] != calibration['sha256'])):
                    raise ValueError(f'RF/cache calibration mismatch: {scene}')
        overlap = set(keys) & {tuple(k) for k in model.metadata.get('training_frames', [])}
        if overlap or model.metadata.get('annotation_sha256') == annotation_sha256(args.annotation):
            raise ValueError('Evaluation overlaps RF training data; use the held-out test split')
        if model.metadata.get('radar_z') != args.radar_z or model.metadata.get('gt_order') != args.gt_order:
            raise ValueError('Calibration Z or GT coordinate convention differs from training')
    manifest_path = Path(__file__).with_name('OCTOMAP_SOURCE_MANIFEST.json')
    run = {'status': 'running', 'command': args.command, 'config': cfg.signature(),
           'annotation': str(Path(args.annotation).resolve()),
           'annotation_sha256': annotation_sha256(args.annotation),
           'source_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
           'selected_frames': len(keys), 'processed_frames': 0,
           'unknown_export': cfg.unknown_export, 'radar_z': args.radar_z,
           'gt_order': args.gt_order, 'video_frames': 0,
           'input_mode': 'frame_fusion' if cache else 'rpc',
           'frame_fusion_root': str(cache.root) if cache else None,
           'preprocessing_signature': preprocessing_signature}
    # Execution provenance is separate from the algorithm/cache signature:
    # C++ and Python run the same pinned OcTree rules and share existing caches.
    if cache:
        run['octomap_execution'] = {'backend': 'cache', 'octomap_executed': False}
    else:
        from .octomap.backend import execution_info
        run['octomap_execution'] = execution_info()
        print(f"OctoMap backend: {run['octomap_execution']['backend']} (CPU)", flush=True)
    cache_writer = CacheWriter(output, cfg, args.radar_z, args.annotation, dataset.infos) if args.command == 'preprocess' else None
    if cache_writer:
        cache_writer.manifest['execution'] = run['octomap_execution']
    _json(output / 'pipeline_config.json', cfg.signature())
    _json(output / 'run.json', run)
    pipeline, accumulator = Pipeline(cfg), RadarOccMetricAccumulator()
    features, targets, counts, frame_records = [], [], Counter(), []
    video = None
    cameras = {}
    if args.camera_dir:
        from tradition.visualization.radarocc_video import RadarOccStyleVideoRenderer
        cameras = _camera_map(dataset.infos_all, args.video_scene, Path(args.camera_dir))
        if not any(scene == args.video_scene for scene, _ in keys):
            raise ValueError('Video scene is not present in the selected evaluation frames')
        video = RadarOccStyleVideoRenderer(
            output, args.video_scene, fps=args.video_fps,
            no_rotate=True, keep_frames=args.keep_frames,
            background_prediction_root=args.video_background_prediction_root,
        )
    try:
        for number, frame in enumerate(dataset, 1):
            if cache:
                mapped, cached_record = cache.load(frame)
                record = dict(cached_record)
                record['gt_path'] = frame['gt_path']
            else:
                mapped = pipeline.map_frame(frame['rpc'], frame['pose'], frame['translation'], frame['scene'], frame['ordinal'])
                record = {k: frame[k] for k in ['scene', 'token', 'ordinal', 'rpc_path', 'pose_path', 'calibration']}
                record['gt_path'] = frame.get('gt_path')
                record['rpc_points'] = len(frame['rpc'])
            record.update({'occupied_voxels': int(mapped.occupied.sum()),
                           'unknown_voxels': int((~(mapped.occupied | mapped.free)).sum()),
                           'proposals': len(mapped.candidates)})
            frame_records.append(record)
            if args.command == 'preprocess':
                cache_writer.add(mapped, frame)
            elif args.command == 'train':
                for candidate in mapped.candidates:
                    target, outcome = training_target(candidate, frame['gt'], args.positive_fraction, args.negative_fraction)
                    counts[outcome] += 1
                    if target is not None:
                        features.append(candidate['features'])
                        targets.append(target)
            else:
                dense, native = pipeline.predict(mapped, model)
                accumulator.update(dense, frame['gt'])
                if args.save_predictions:
                    if not all(re.fullmatch(r'[A-Za-z0-9_.-]+', frame[k]) for k in ['scene', 'token']):
                        raise ValueError('Unsafe scene or token filename')
                    target_dir = output / 'predictions' / frame['scene']
                    target_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(target_dir / (frame['token'] + '.npz'),
                        labels_xyz=dense, native_labels_xyz=native,
                        occupancy_probability=mapped.probability.astype(np.float32),
                        observed_mask=mapped.observed, unknown_mask=native == 255,
                        bev_probability=mapped.bev_probability.astype(np.float32))
                if video and frame['scene'] == args.video_scene and frame['ordinal'] >= args.video_start and (args.video_end is None or frame['ordinal'] <= args.video_end):
                    octomap_native = np.full(cfg.shape_xyz, 255, np.uint8)
                    octomap_native[mapped.free] = 0
                    octomap_native[mapped.occupied] = 1
                    video.add_frame(dense, octomap_native, frame['gt'], cameras[frame['token']], frame['token'])
            run['processed_frames'] = number
            print(f"[{number}/{len(keys)}] scene={frame['scene']} token={frame['token']} "
                  f"occupied={record['occupied_voxels']} proposals={record['proposals']}", flush=True)
        if video and video.frame_count == 0:
            raise ValueError('Requested video range has no frames')
        if video:
            video.finish()
            run['video_frames'] = video.frame_count
        if args.command == 'train':
            metadata = {**run, 'training_frames': keys, 'target_counts': dict(counts),
                        'positive_fraction': args.positive_fraction, 'negative_fraction': args.negative_fraction,
                        'calibrations': cache.manifest['calibrations'] if cache else {s: {'translation': t.tolist(), **c} for s, (t, c) in dataset.calibrations.items()}}
            metadata['status'] = 'completed'
            model_path = Path(args.model) if args.model else output / 'random_forest.joblib'
            if any(model_path.resolve() == s or s in model_path.resolve().parents for s in source_dirs):
                raise ValueError('Model output must be outside source code')
            run['training'] = train(features, targets, model_path, cfg, metadata, args.n_estimators, args.seed,
                                    args.max_depth or None, args.min_samples_leaf,
                                    None if args.class_weight == 'none' else args.class_weight, args.n_jobs)
            run['model'] = str(model_path.resolve())
            np.savez_compressed(output / 'training_features.npz', features=np.asarray(features),
                                targets=np.asarray(targets), feature_names=FEATURE_NAMES)
            print(f"Saved model: {model_path}", flush=True)
        elif args.command == 'evaluate':
            metrics = radarocc_metric_dict(accumulator.results())
            metrics = {k: float(v) if np.isfinite(v) else None for k, v in metrics.items()}
            _json(output / 'metrics.json', metrics)
            with (output / 'metrics.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['Metric', 'TraditionalReal'])
                writer.writerows(metrics.items())
            np.savez_compressed(output / 'confusions.npz', **{f'range_{r:g}': cm for r, cm in accumulator._confusions.items()})
            print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)
        if cache_writer:
            cache_writer.finish()
            print(f'Saved frame fusion cache: {output}', flush=True)
        run['status'] = 'completed'
    except Exception as error:
        if cache_writer:
            cache_writer.finish(error)
        run['status'], run['error'] = 'failed', str(error)
        raise
    finally:
        if video:
            run['video_frames'] = video.frame_count
        _json(output / 'frames.json', frame_records)
        _json(output / 'run.json', run)


if __name__ == '__main__':
    main()
