"""Run from repository root: python -m trodition_real {train,evaluate}."""
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
from .pipeline import Pipeline
from .adapters.dataset import Dataset
from .adapters.paths import _camera_map
from .semantics.random_forest import RandomForest, train, training_target, annotation_sha256, FORMAT
from .semantics.features import FEATURE_NAMES
from .evaluation.radarocc_metrics import RadarOccMetricAccumulator, radarocc_metric_dict


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['train', 'evaluate'])
    p.add_argument('--annotation', required=True, help='Train split for train; test split for evaluate')
    p.add_argument('--radar-root', default='data/K-Radar_rpc')
    p.add_argument('--pose-root', default='data/K-RadarOcc')
    p.add_argument('--calib-root', default='data/K-Radar_calib')
    p.add_argument('--radar-z', type=float, default=-.7, help='Radar origin Z in LiDAR coordinates')
    p.add_argument('--gt-root')
    p.add_argument('--gt-order', choices=['xyz', 'zyx'], default='xyz')
    p.add_argument('--repo-root', default='.')
    p.add_argument('--output', required=True, help='Separate work_dirs/trodition_real output directory')
    p.add_argument('--model', help='Required for evaluate; train defaults to OUTPUT/random_forest.joblib')
    p.add_argument('--config', help='JSON Config; evaluation defaults to the model configuration')
    p.add_argument('--temporal-window', type=int, help='Optional config override; must match RF when evaluating')
    p.add_argument('--scenes', nargs='+', help='Limit data scenes; omit for complete split evaluation')
    p.add_argument('--max-frames', type=int, help='Smoke test only; omit for complete split')
    p.add_argument('--n-estimators', type=int, default=200)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--positive-fraction', type=float, default=.2)
    p.add_argument('--negative-fraction', type=float, default=.05)
    p.add_argument('--save-predictions', action='store_true')
    p.add_argument('--video-scene', default='3', help='Rendering only, does NOT filter evaluation')
    p.add_argument('--camera-dir', help='Enables video for video-scene, all its frames by default')
    p.add_argument('--video-start', type=int, default=0, help='Inclusive zero-based scene ordinal')
    p.add_argument('--video-end', type=int, help='Inclusive scene ordinal; omitted = last frame')
    p.add_argument('--video-fps', type=float, default=10.)
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
        raise ValueError('Evaluation requires --model trained with trodition_real')
    if args.command == 'train' and args.camera_dir:
        raise ValueError('Video is available for evaluate')
    bundle = joblib.load(args.model) if args.command == 'evaluate' else None
    if bundle is not None and (not isinstance(bundle, dict) or bundle.get('format') != FORMAT):
        raise ValueError('Old tradition RF is incompatible; train trodition_real first')
    raw_config = json.loads(Path(args.config).read_text()) if args.config else (bundle['config'] if bundle else {})
    for key in ('shape_xyz', 'min_xyz'):
        if key in raw_config:
            raw_config[key] = tuple(raw_config[key])
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
    output.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(args.annotation, args.radar_root, args.pose_root, args.calib_root,
                      args.repo_root, args.gt_root, args.gt_order, args.radar_z,
                      args.scenes, args.max_frames)
    model = RandomForest.load(args.model, cfg) if bundle else None
    keys = [(str(i['scene_token']), str(i['lidar_token'])) for i in dataset.infos]
    if model:
        overlap = set(keys) & {tuple(k) for k in model.metadata.get('training_frames', [])}
        if overlap or model.metadata.get('annotation_sha256') == annotation_sha256(args.annotation):
            raise ValueError('Evaluation overlaps RF training data; use the held-out test split')
        if model.metadata.get('radar_z') != args.radar_z or model.metadata.get('gt_order') != args.gt_order:
            raise ValueError('Calibration Z or GT coordinate convention differs from training')
    manifest_path = Path(__file__).with_name('SOURCE_MANIFEST.json')
    run = {'status': 'running', 'command': args.command, 'config': cfg.signature(),
           'annotation': str(Path(args.annotation).resolve()),
           'annotation_sha256': annotation_sha256(args.annotation),
           'source_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
           'selected_frames': len(keys), 'processed_frames': 0,
           'unknown_export': cfg.unknown_export, 'radar_z': args.radar_z,
           'gt_order': args.gt_order, 'video_frames': 0}
    _json(output / 'pipeline_config.json', cfg.signature())
    _json(output / 'run.json', run)
    pipeline, accumulator = Pipeline(cfg), RadarOccMetricAccumulator()
    features, targets, counts, frame_records = [], [], Counter(), []
    video = None
    cameras = {}
    if args.camera_dir:
        from .visualization import Video
        cameras = _camera_map(dataset.infos_all, args.video_scene, Path(args.camera_dir))
        if not any(scene == args.video_scene for scene, _ in keys):
            raise ValueError('Video scene is not present in the selected evaluation frames')
        video = Video(output / f'scene_{args.video_scene}.mp4', args.video_fps)
    try:
        for number, frame in enumerate(dataset, 1):
            mapped = pipeline.map_frame(frame['rpc'], frame['pose'], frame['translation'], frame['scene'], frame['ordinal'])
            record = {k: frame[k] for k in ['scene', 'token', 'ordinal', 'rpc_path', 'gt_path', 'pose_path', 'calibration']}
            record.update({'rpc_points': len(frame['rpc']), 'occupied_voxels': int(mapped.occupied.sum()),
                           'unknown_voxels': int((~(mapped.occupied | mapped.free)).sum()),
                           'proposals': len(mapped.candidates)})
            frame_records.append(record)
            if args.command == 'train':
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
                    video.add(native, frame['gt'], cameras[frame['token']], frame['token'])
            run['processed_frames'] = number
            print(f"[{number}/{len(keys)}] scene={frame['scene']} token={frame['token']} "
                  f"occupied={record['occupied_voxels']} proposals={record['proposals']}", flush=True)
        if video and video.frames == 0:
            raise ValueError('Requested video range has no frames')
        if args.command == 'train':
            metadata = {**run, 'training_frames': keys, 'target_counts': dict(counts),
                        'positive_fraction': args.positive_fraction, 'negative_fraction': args.negative_fraction,
                        'calibrations': {s: {'translation': t.tolist(), **c} for s, (t, c) in dataset.calibrations.items()}}
            metadata['status'] = 'completed'
            model_path = Path(args.model) if args.model else output / 'random_forest.joblib'
            if any(model_path.resolve() == s or s in model_path.resolve().parents for s in source_dirs):
                raise ValueError('Model output must be outside source code')
            run['training'] = train(features, targets, model_path, cfg, metadata, args.n_estimators, args.seed)
            run['model'] = str(model_path.resolve())
            np.savez_compressed(output / 'training_features.npz', features=np.asarray(features),
                                targets=np.asarray(targets), feature_names=FEATURE_NAMES)
            print(f"Saved model: {model_path}", flush=True)
        else:
            metrics = radarocc_metric_dict(accumulator.results())
            metrics = {k: float(v) if np.isfinite(v) else None for k, v in metrics.items()}
            _json(output / 'metrics.json', metrics)
            with (output / 'metrics.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['Metric', 'TraditionalReal'])
                writer.writerows(metrics.items())
            np.savez_compressed(output / 'confusions.npz', **{f'range_{r:g}': cm for r, cm in accumulator._confusions.items()})
            print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)
        run['status'] = 'completed'
    except Exception as error:
        run['status'], run['error'] = 'failed', str(error)
        raise
    finally:
        if video:
            video.close()
            run['video_frames'] = video.frames
        _json(output / 'frames.json', frame_records)
        _json(output / 'run.json', run)


if __name__ == '__main__':
    main()
