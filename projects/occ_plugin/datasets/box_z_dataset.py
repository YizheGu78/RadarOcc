"""Attach frozen, GT-free box pairs to the existing temporal dataset."""
import hashlib
import json
from pathlib import Path
import numpy as np
from mmdet.datasets import DATASETS
from .temporal_kradar_dataset import TemporalKRadarDataset


@DATASETS.register_module()
class BoxZTemporalDataset(TemporalKRadarDataset):
    def __init__(self, box_cache, box_bins_path='data/K-Radar-official-meta/resources/info_arr.mat', **kwargs):
        super().__init__(**kwargs)
        self.box_cache = Path(box_cache)
        self.box_manifest = json.loads((self.box_cache/'manifest.json').read_text())
        policy = self.box_manifest['policy']
        if (not self.box_manifest['complete'] or self.frame_nums != 4 or self.current_only
                or policy['format'] != 'box_z_pairs_v1' or policy['gt_used']
                or policy['y_gates'] != [4, 5, 3] or policy['center_gates'] != [4, 6, 8]
                or policy['weights'] != [1, 0]):
            raise ValueError('Wrong/incomplete box Z cache policy')
        if hashlib.sha256(Path(self.ann_file).read_bytes()).hexdigest() != policy['ann_sha256']:
            raise ValueError('Cache annotation fingerprint mismatch')
        if hashlib.sha256(Path(box_bins_path).read_bytes()).hexdigest() != policy['bins_sha256']:
            raise ValueError('Cache physical-bin fingerprint mismatch')
        if len(self.data_infos) != self.box_manifest['frames']:
            raise ValueError('Cache frame count mismatch')

    def _attach_temporal_payload(self, example, index):
        example = super()._attach_temporal_payload(example, index)
        info = self.data_infos[index]
        key = f'{self._scene_id(info)}_{self._order_in_scene(info):05d}'
        record = self.box_manifest['records'][key]
        paths = [str(Path(self.data_infos[i].get('sparse_radar_path') or
                               self.data_infos[i]['radar_path']).resolve()) for i in self.sequence_indices[index]]
        if paths != record['radar_paths']:
            raise ValueError(f'Cached history sequence mismatch: {key}')
        for path in paths:
            stat = Path(path).stat()
            if self.box_manifest['sources'][path] != dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns):
                raise ValueError(f'Radar input changed after caching: {path}')
        with np.load(self.box_cache/record['file'], allow_pickle=False) as z:
            example['box_pairs'] = {k: z[k].copy() for k in ('current', 'history', 'matched', 'inverse')}
        return example
