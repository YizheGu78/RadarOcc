"""Sequence loading without a pose dependency. No historical GT is loaded."""
import numpy as np
from mmdet.datasets import DATASETS
from projects.occ_plugin.datasets.temporal_kradar_dataset import TemporalKRadarDataset


@DATASETS.register_module()
class UnalignedTemporalKRadarDataset(TemporalKRadarDataset):
    def _validate_temporal_metadata(self):
        seen = set()
        for info in self.data_infos:
            key = (self._scene_id(info), self._order_in_scene(info))
            if key in seen:
                raise ValueError('Duplicate scene/order in temporal annotations: %s' % (key,))
            seen.add(key)

    def get_temporal_info(self, index):
        sequence = self.sequence_indices[index]
        infos = [self.data_infos[i] for i in sequence]
        orders = [self._order_in_scene(info) for info in infos]
        return dict(
            dataset_index=int(index), current_index=int(sequence[-1]),
            sequence_indices=list(sequence), scene_id=self._scene_id(infos[-1]),
            orders_in_scene=orders,
            # Actual order differences remain correct for non-contiguous subsets.
            frame_offsets=np.asarray(orders, dtype=np.int64) - orders[-1],
            valid_mask=self.sequence_valid_masks[index].copy(),
            # Parent payload interface includes this field. These placeholders
            # are never read by UnalignedMemoryTemporal; poses are not required.
            ego_poses=np.tile(np.eye(4, dtype=np.float64), (self.frame_nums, 1, 1)),
            timestamps=[info.get('timestamp_sec') for info in infos],
            radar_paths=[info.get('radar_path', info.get('sparse_radar_path', '')) for info in infos],
            gt_path=infos[-1].get('gt_path', infos[-1].get('occ_path', '')),
        )
