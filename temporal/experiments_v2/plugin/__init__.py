"""Versioned experiments using the corrected RadarOcc Small projection."""
import hashlib
import json
from pathlib import Path

from mmdet.datasets import DATASETS
from projects.occ_plugin.datasets.box_z_dataset import BoxZTemporalDataset
from projects.occ_plugin.datasets.temporal_kradar_dataset import TemporalKRadarDataset
from temporal.p0_plugin import RadarOccP0Temporal
from projects.unaligned_temporal import RadarOccSmallUnalignedTemporal, UnalignedTemporalKRadarDataset
from temporal.core.ego_alignment import KRADAR_COORDINATES


@DATASETS.register_module()
class BoxZTemporalDatasetV2(BoxZTemporalDataset):
    """Reuse the old payload loader with a strict, versioned cache contract."""
    def __init__(self, box_cache, box_bins_path='data/K-Radar-official-meta/resources/info_arr.mat',
                 expected_y_gates=(4, 5, 3), expected_center_gates=(4, 6, 8), **kwargs):
        TemporalKRadarDataset.__init__(self, **kwargs)
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
