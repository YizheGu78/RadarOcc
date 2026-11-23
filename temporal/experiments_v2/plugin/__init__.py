"""Versioned experiments using the corrected RadarOcc Small projection."""
import hashlib
import fcntl
import os
import tempfile
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from mmdet.datasets import DATASETS
from projects.occ_plugin.datasets.box_z_dataset import BoxZTemporalDataset
from projects.occ_plugin.datasets.temporal_kradar_dataset import TemporalKRadarDataset
from temporal.p0_plugin import RadarOccP0Temporal
from projects.unaligned_temporal import RadarOccSmallUnalignedTemporal, UnalignedTemporalKRadarDataset
from temporal.core.ego_alignment import KRADAR_COORDINATES


@DATASETS.register_module()
class BoxZTemporalDatasetV2(BoxZTemporalDataset):
    """Load frozen pairs, or generate the same GT-free pairs per sample."""
    def __init__(self, box_cache=None, box_bins_path='data/K-Radar-official-meta/resources/info_arr.mat',
                 expected_y_gates=(4, 5, 3), expected_center_gates=(4, 6, 8),
                 box_pair_mode='cache', **kwargs):
        if box_pair_mode not in ('cache', 'online', 'readwrite'):
            raise ValueError('box_pair_mode must be cache, online or readwrite')
        TemporalKRadarDataset.__init__(self, **kwargs)
        self.box_pair_mode = box_pair_mode
        if box_pair_mode in ('online', 'readwrite'):
            from temporal.core.ego_alignment import RadarBinGeometry, radar_to_lidar_default
            from tools.temporal_vis.p1_reflection_regions import RegionConfig
            gates = np.asarray([expected_y_gates, expected_center_gates], dtype=float)
            if gates.shape != (2, 3) or not np.isfinite(gates).all() or (gates <= 0).any():
                raise ValueError('Expected three finite positive gates per history axis')
            if self.frame_nums != 4 or self.current_only:
                raise ValueError('Online box pairs require four real temporal inputs')
            self.box_geometry = RadarBinGeometry.from_file(
                box_bins_path, coordinate_convention=KRADAR_COORDINATES)
            self.box_extrinsic = radar_to_lidar_default()
            self.box_y_gates, self.box_center_gates = gates.tolist()
            self.box_region_config = RegionConfig(seed_percentile=97, seed_mad=3,
                merge_space='xy', xy_box_gap=(.5, .5), box_size=(6, 3, 3),
                max_box_size=(20, 6, 6))
            # No manifest, disk writes, or memoization of frame/region pairs.
            self.box_cache = self.box_manifest = None
            if box_pair_mode == 'readwrite':
                if box_cache is None:
                    raise ValueError('box_cache is required in readwrite mode')
                self._initialize_pair_cache(box_cache, box_bins_path)
            return
        if box_cache is None:
            raise ValueError('box_cache is required in cache mode')
        self.box_cache = Path(box_cache)
        self.box_manifest = json.loads((self.box_cache/'manifest.json').read_text())
        policy = self.box_manifest['policy']
        if (policy.get('coordinate_convention') != KRADAR_COORDINATES
                or not self.box_manifest['complete'] or policy['gt_used']
                or self.frame_nums != 4 or self.current_only
                or policy['format'] != 'box_z_pairs_v1'
                or policy['weights'] != [1, 0]
                or policy['y_gates'] != list(expected_y_gates)
                or policy['center_gates'] != list(expected_center_gates)):
            raise ValueError('Expected complete kradar_physical_v2 box cache with matching gates')
        for source, expected in ((self.ann_file, policy['ann_sha256']),
                                 (box_bins_path, policy['bins_sha256'])):
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
                raise ValueError('Box cache annotation/bin fingerprint mismatch')
        if len(self.data_infos) != self.box_manifest['frames']:
            raise ValueError('Box cache frame count mismatch')

    def _attach_temporal_payload(self, example, index):
        if self.box_pair_mode == 'cache':
            return super()._attach_temporal_payload(example, index)
        example = TemporalKRadarDataset._attach_temporal_payload(self, example, index)
        example['box_pairs'] = (self._readwrite_box_pairs(example, index)
                                if self.box_pair_mode == 'readwrite'
                                else self._online_box_pairs(example, index))
        return example

    def _initialize_pair_cache(self, directory, bins_path):
        """Immutable policy; each sample is published independently and atomically."""
        from tools.temporal_vis import p1_reflection_regions, benchmark_region_matching
        from temporal.core import ego_alignment
        root = Path(__file__).resolve().parents[3]
        sources = [Path(__file__), Path(p1_reflection_regions.__file__),
                   Path(benchmark_region_matching.__file__), Path(ego_alignment.__file__),
                   root/'projects/occ_plugin/datasets/temporal_kradar_dataset.py']
        sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        policy = dict(format='box_z_pairs_lazy_v1', gt_used=False,
                      coordinate_convention=KRADAR_COORDINATES,
                      ann_sha256=sha(self.ann_file), bins_sha256=sha(bins_path),
                      y_gates=self.box_y_gates, center_gates=self.box_center_gates,
                      weights=[1, 0], region_config=asdict(self.box_region_config),
                      extrinsic=self.box_extrinsic.tolist(),
                      sources={str(path.relative_to(root)): sha(path) for path in sources})
        self.box_policy_json = json.dumps(policy, sort_keys=True)
        self.box_cache = Path(directory)
        self.box_cache.mkdir(parents=True, exist_ok=True)
        # flock protects initialization across loader workers / distributed ranks.
        with (self.box_cache/'.policy.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.box_cache/'policy.json'
            if (self.box_cache/'manifest.json').exists():
                raise ValueError('Use a separate directory for readwrite and offline box caches')
            if path.exists():
                if json.loads(path.read_text()) != json.loads(self.box_policy_json):
                    raise ValueError('Readwrite box cache policy mismatch; use a new cache directory')
            else:
                self._atomic_pair_write(path, self.box_policy_json.encode())

    @staticmethod
    def _atomic_pair_write(path, content):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.pending-', delete=False) as stream:
                temporary = Path(stream.name)
                if isinstance(content, bytes):
                    stream.write(content)
                else:
                    np.savez_compressed(stream, **content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _readwrite_box_pairs(self, example, index):
        infos = [self.data_infos[i] for i in self.sequence_indices[index]]
        sources = []
        for info in infos:
            path = Path(info.get('sparse_radar_path') or info['radar_path']).resolve()
            stat = path.stat()
            sources.append(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
        identity = dict(scene=str(self._scene_id(infos[-1])),
                        order=int(self._order_in_scene(infos[-1])))
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.box_cache/(key+'.npz')
        inputs = json.dumps(dict(sources=sources,
            valid=np.asarray(example['temporal_valid_mask'], dtype=bool).tolist()), sort_keys=True)
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                if str(archive['policy'].item()) != self.box_policy_json:
                    raise ValueError(f'Readwrite box cache policy mismatch: {path}')
                if str(archive['inputs'].item()) != inputs:
                    raise ValueError(f'Radar input/history changed after caching: {path}')
                pairs = {name: archive[name].copy()
                         for name in ('current', 'history', 'matched', 'inverse')}
            n = len(pairs['current'])
            shapes = dict(current=(n, 6), history=(n, 3, 6), matched=(n, 3), inverse=(3, 4, 4))
            if (any(pairs[k].shape != shape for k, shape in shapes.items())
                    or pairs['matched'].dtype != np.bool_
                    or any(not np.isfinite(value).all() for value in pairs.values())):
                raise ValueError(f'Invalid readwrite box cache payload: {path}')
            return pairs
        pairs = self._online_box_pairs(example, index)
        self._atomic_pair_write(path, dict(pairs, policy=self.box_policy_json, inputs=inputs))
        return pairs

    def _online_box_pairs(self, example, index):
        """Match t-1/t-2/t-3 in current radar coordinates; GT is never used."""
        from temporal.core.ego_alignment import validate_transform
        from tools.temporal_vis.p1_reflection_regions import extract_regions
        from tools.temporal_vis.benchmark_region_matching import aligned_box, select_match
        infos = [self.data_infos[i] for i in self.sequence_indices[index]]
        valid = np.asarray(example['temporal_valid_mask'], dtype=bool)
        if valid.shape != (4,) or not valid[-1]:
            raise ValueError('Invalid online box sequence mask')
        poses = []
        for info in infos:
            if not info.get('ego_pose_valid', False):
                raise ValueError('Online box pairing requires valid ego poses')
            poses.append(validate_transform(np.asarray(info['ego_pose'])))

        def regions(slot):
            radar = example['temporal_sparse_radar'][slot]
            n = 175*250
            power = np.asarray(radar['power_val'])[2, :n]
            az = np.asarray(radar['azimuth_ind'])[:n]
            el = np.asarray(radar['elevation_ind'])[:n]
            if len(power) != n or len(az) != n or len(el) != n:
                raise ValueError('Online box pairing requires 175 x 250 P1 returns')
            coords = np.stack((np.repeat(np.arange(175), 250), az, el), -1)
            result = extract_regions(self.box_geometry.to_xyz(coords), power, self.box_region_config)
            return [r for r in result['regions'] if r['accepted']]

        current_regions = regions(3)
        current = np.array([np.r_[r['center'], r['box_size']] for r in current_regions]).reshape(-1, 6)
        history = np.broadcast_to(current[:, None], (len(current), 3, 6)).copy()
        matched = np.zeros((len(current), 3), bool)
        inverse = np.repeat(np.eye(4)[None], 3, axis=0)
        world_current = poses[-1] @ self.box_extrinsic
        for j in range(3):
            slot = 2-j
            if not valid[slot]:
                continue
            if (self._scene_id(infos[slot]) != self._scene_id(infos[-1]) or
                    self._order_in_scene(infos[-1])-self._order_in_scene(infos[slot]) != j+1):
                raise ValueError('Online box pairing requires consecutive same-scene history')
            relative = np.linalg.inv(world_current) @ (poses[slot] @ self.box_extrinsic)
            inverse[j] = np.linalg.inv(relative)
            candidates = []
            for region in regions(slot):
                center, size = aligned_box(region['center'], region['box_size'], relative)
                candidates.append(dict(id=region['id'], aligned_center=center, aligned_size=size))
            for k, region in enumerate(current_regions):
                chosen = select_match(region, candidates, j+1, 1., 0.,
                    self.box_y_gates, self.box_center_gates, match_space='xy')
                if chosen is not None:
                    history[k, j] = np.r_[chosen['aligned_center'], chosen['aligned_size']]
                    matched[k, j] = True
        return dict(current=current.astype(np.float32), history=history.astype(np.float32),
                    matched=matched, inverse=inverse)
