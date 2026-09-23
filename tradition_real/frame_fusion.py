"""Versioned, GT-free mapping/42D cache. Never use pickle inside NPZ files."""
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from .pipeline import MappingResult
from .semantics.features import FEATURE_NAMES

FORMAT = 'tradition-real-frame-fusion-v1'
# Changes to deterministic preprocessing invalidate old cache/model pairings.
SOURCE_FILES = ('config.py', 'pipeline.py', 'adapters/dataset.py',
                'adapters/paths.py', 'adapters/pose_reader.py', 'adapters/layers.py',
                'autoware/costmap.py', 'gm2019/fusion.py', 'semantics/features.py')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def signature(config, radar_z):
    preprocessing = config.signature()
    for key in ('foreground_threshold', 'unknown_export'):
        preprocessing.pop(key)  # RF/export settings do not change cached mapping.
    base = Path(__file__).parent
    return {'config': preprocessing, 'radar_z': radar_z,
            'feature_names': list(FEATURE_NAMES),
            'source_sha256': {name: sha256(base / name) for name in SOURCE_FILES}}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def frame_path(scene, token):
    if not all(re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*', str(x)) for x in (scene, token)):
        raise ValueError('Unsafe scene/token in cache')
    return f'{scene}/frame_{token}.npz'


def save_frame(path, mapped, frame, cache_signature):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    native = np.full(mapped.probability.shape, 255, np.uint8)
    native[mapped.free] = 0
    native[mapped.occupied] = 1
    candidates = mapped.candidates
    features = np.stack([c['features'] for c in candidates]) if candidates else np.empty((0, 42), np.float64)
    voxels = np.concatenate([c['voxels'] for c in candidates]) if candidates else np.empty((0, 3), np.int32)
    offsets = np.r_[0, np.cumsum([len(c['voxels']) for c in candidates])].astype(np.int64)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, native_labels_xyz=native,
            occupancy_probability=mapped.probability, observed_mask=mapped.observed,
            bev_probability=mapped.bev_probability, proposal_features=features,
            proposal_voxels=voxels.astype(np.int32), proposal_offsets=offsets,
            scene=str(frame['scene']), token=str(frame['token']), ordinal=frame['ordinal'],
            format=FORMAT, signature=json.dumps(cache_signature, sort_keys=True))
    os.replace(temporary, path)


def load_frame(path, frame, cache_signature):
    with np.load(path, allow_pickle=False) as data:
        if str(data['format']) != FORMAT or json.loads(str(data['signature'])) != cache_signature:
            raise ValueError(f'Cache frame format/config mismatch: {path}')
        if any(str(data[key].item()) != str(frame[key]) for key in ('scene', 'token', 'ordinal')):
            raise ValueError(f'Cache frame identity mismatch: {path}')
        native, probability = data['native_labels_xyz'], data['occupancy_probability']
        observed, bev = data['observed_mask'], data['bev_probability']
        features, voxels, offsets = data['proposal_features'], data['proposal_voxels'], data['proposal_offsets']
    shape = tuple(cache_signature['config']['shape_xyz'])
    if (native.shape != shape or probability.shape != shape or observed.shape != shape
            or observed.dtype != np.bool_ or bev.shape != shape[:2]
            or not np.isin(native, [0, 1, 255]).all()):
        raise ValueError(f'Invalid cached occupancy grid: {path}')
    if any(not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) for p in (probability, bev)):
        raise ValueError(f'Invalid cached probabilities: {path}')
    if features.ndim != 2 or features.shape[1] != 42 or not np.isfinite(features).all():
        raise ValueError(f'Invalid cached 42D features: {path}')
    if (voxels.ndim != 2 or voxels.shape[1] != 3 or not np.issubdtype(voxels.dtype, np.integer)
            or offsets.shape != (len(features) + 1,) or not np.issubdtype(offsets.dtype, np.integer)
            or offsets[0] != 0 or offsets[-1] != len(voxels) or np.any(np.diff(offsets) <= 0)
            or np.any(voxels < 0) or np.any(voxels >= np.asarray(shape))):
        raise ValueError(f'Invalid cached proposal offsets/voxels: {path}')
    if (np.any(native[tuple(voxels.T)] != 1) or len(voxels) != int((native == 1).sum())
            or len(np.unique(voxels, axis=0)) != len(voxels)
            or np.any((native != 255) & ~observed)):
        raise ValueError(f'Cached proposals must partition occupied voxels: {path}')
    candidates = [{'features': features[i], 'voxels': voxels[offsets[i]:offsets[i+1]]}
                  for i in range(len(features))]
    return MappingResult(probability, observed, native == 1, native == 0, bev, candidates)


class CacheWriter:
    def __init__(self, root, config, radar_z, annotation, infos):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / 'manifest.json').exists() or any(self.root.rglob('*.npz')):
            raise ValueError('Cache output already contains a cache; use a new output directory')
        self.manifest = {'format': FORMAT, 'status': 'running', 'config': config.signature(),
                         'signature': signature(config, radar_z),
                         'annotation_sha256': sha256(annotation), 'expected_frames': len(infos),
                         'frames': [], 'calibrations': {}}
        atomic_json(self.root / 'manifest.json', self.manifest)

    def add(self, mapped, frame):
        relative = frame_path(frame['scene'], frame['token'])
        path = self.root / relative
        save_frame(path, mapped, frame, self.manifest['signature'])
        record = {key: frame[key] for key in ('scene', 'token', 'ordinal', 'rpc_path', 'pose_path', 'calibration')}
        record.update(path=relative, sha256=sha256(path), rpc_points=len(frame['rpc']),
                      rpc_sha256=sha256(frame['rpc_path']), pose_sha256=sha256(frame['pose_path']))
        self.manifest['frames'].append(record)
        scene = frame['scene']
        calibration = {**frame['calibration'], 'translation': frame['translation'].tolist(),
                       'sha256': sha256(frame['calibration']['path'])}
        if scene in self.manifest['calibrations'] and self.manifest['calibrations'][scene] != calibration:
            raise ValueError(f'Calibration changed during preprocessing: {scene}')
        self.manifest['calibrations'][scene] = calibration

    def finish(self, error=None):
        complete = len(self.manifest['frames']) == self.manifest['expected_frames']
        self.manifest['status'] = 'completed' if error is None and complete else 'failed'
        if error is not None:
            self.manifest['error'] = str(error)
        atomic_json(self.root / 'manifest.json', self.manifest)


class CacheReader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        m = self.manifest
        if m.get('format') != FORMAT or m.get('status') != 'completed':
            raise ValueError('Cache format unsupported or preprocessing incomplete/failed')
        self.records = {(r['scene'], r['token']): r for r in m['frames']}
        if len(self.records) != len(m['frames']) or len(self.records) != m['expected_frames']:
            raise ValueError('Cache manifest has duplicate or missing frames')

    def validate(self, config, radar_z, annotation, infos, calib_root=None):
        m = self.manifest
        if m['signature'] != signature(config, radar_z):
            raise ValueError('Cache preprocessing config, radar_z, feature schema or source mismatch; rebuild cache')
        if m['annotation_sha256'] != sha256(annotation):
            raise ValueError('Cache annotation mismatch; use the exact preprocessing split')
        scenes = set()
        for info in infos:
            key = (str(info['scene_token']), str(info['lidar_token']))
            record = self.records.get(key)
            if record is None or record['ordinal'] != info['_rpc_sequence_ordinal']:
                raise ValueError(f'Cache is missing frame or has wrong ordinal: {key}')
            if record['path'] != frame_path(*key):
                raise ValueError(f'Invalid cache frame path: {key}')
            if not (self.root / record['path']).is_file():
                raise FileNotFoundError(f'Missing cached frame: {record["path"]}')
            scenes.add(key[0])
        # Raw inputs need not exist for cache consumption. When explicitly
        # supplied, calibration is an assertion against the saved snapshot.
        if calib_root is not None:
            from .adapters.dataset import scene_translation
            for scene in scenes:
                translation, calibration = scene_translation(calib_root, scene, radar_z)
                saved = m['calibrations'][scene]
                if translation.tolist() != saved['translation'] or sha256(calibration['path']) != saved['sha256']:
                    raise ValueError(f'Cache calibration mismatch: scene {scene}')

    def load(self, frame):
        record = self.records[(frame['scene'], frame['token'])]
        path = self.root / record['path']
        if sha256(path) != record['sha256']:
            raise ValueError(f'Cache frame checksum mismatch: {path}')
        return load_frame(path, frame, self.manifest['signature']), record
