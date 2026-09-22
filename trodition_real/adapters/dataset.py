"""K-Radar adapters shared by training and evaluation."""
from pathlib import Path
import csv
import numpy as np
from .paths import _load_infos, _resolve_rpc_radar, _resolve_gt, _scene_variants
from .pose_reader import RadarOccPoseReader
from trodition_real.evaluation.radarocc_metrics import load_gt_sparse_xyz


def scene_translation(calib_root, scene, z=-.7):
    paths = [Path(calib_root) / s / 'info_calib/calib_radar_lidar.txt' for s in _scene_variants(scene)]
    matches = [p for p in paths if p.is_file()]
    if not matches:
        raise FileNotFoundError(f'No calibration for scene {scene}: {paths}')
    with matches[0].open() as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2 or len(rows[1]) < 3:
        raise ValueError(f'Expected frame difference,x,y in {matches[0]}')
    offset, x, y = map(float, rows[1][:3])
    if not np.isfinite([offset, x, y, z]).all():
        raise ValueError('Nonfinite calibration')
    # K-Radar file translation maps LiDAR -> radar; RPC -> LiDAR is inverse.
    # Frame difference is recorded, NOT applied again to already paired order.
    return np.array([-x, -y, z]), {'path': str(matches[0]), 'frame_difference': offset}


class Dataset:
    def __init__(self, annotation, radar_root, pose_root, calib_root,
                 repo_root='.', gt_root=None, gt_order='xyz', radar_z=-.7,
                 scenes=None, max_frames=None):
        self.annotation = Path(annotation).resolve()
        self.infos_all = _load_infos(self.annotation)
        selected = [i for i in self.infos_all if scenes is None or str(i['scene_token']) in scenes]
        # Scene order is deterministic; do not interleave equal timestamps.
        self.infos = sorted(selected, key=lambda i: (str(i['scene_token']), i['_rpc_sequence_ordinal']))
        if max_frames is not None:
            if max_frames < 1:
                raise ValueError('max_frames must be positive')
            self.infos = self.infos[:max_frames]
        if not self.infos:
            raise ValueError('No frames selected')
        keys = [(str(i['scene_token']), str(i['lidar_token'])) for i in self.infos]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate scene/token entries in annotation')
        self.radar_root = Path(radar_root).resolve()
        self.pose_reader = RadarOccPoseReader(pose_root)
        self.calib_root = Path(calib_root)
        self.repo_root = Path(repo_root).resolve()
        self.gt_root = Path(gt_root).resolve() if gt_root else None
        self.gt_order, self.radar_z = gt_order, radar_z
        self.calibrations = {}

    def __iter__(self):
        for info in self.infos:
            scene, token = str(info['scene_token']), str(info['lidar_token'])
            path = _resolve_rpc_radar(info, self.repo_root, self.radar_root)
            pose, pose_path = self.pose_reader.read(scene, token)
            if scene not in self.calibrations:
                self.calibrations[scene] = scene_translation(self.calib_root, scene, self.radar_z)
            translation, calibration = self.calibrations[scene]
            gt_path = _resolve_gt(info, self.repo_root, self.gt_root)
            # No skipped failures: evaluated frame count must be auditable.
            yield {
                'rpc': np.load(path, allow_pickle=False), 'pose': pose,
                'translation': translation, 'scene': scene, 'token': token,
                'ordinal': info['_rpc_sequence_ordinal'],
                'gt': load_gt_sparse_xyz(gt_path, self.gt_order),
                'rpc_path': str(path), 'gt_path': str(gt_path),
                'pose_path': str(pose_path), 'calibration': calibration,
            }
