"""Reuse the original detector interface without changing its source or builder."""
from mmdet.models import DETECTORS
from projects.occ_plugin.occupancy.detectors.radarocc_self_small import RadarOcc_small
from temporal.adapters import P1TopKAdapter
from temporal.fusion.unaligned_memory import UnalignedMemoryTemporal


@DETECTORS.register_module()
class RadarOccSmallUnalignedTemporal(RadarOcc_small):
    def __init__(self, temporal_cfg=None, unaligned_temporal=None, **kwargs):
        if temporal_cfg is not None:
            raise ValueError('Use unaligned_temporal, not the aligned temporal_cfg')
        super().__init__(temporal_cfg=None, **kwargs)
        self.temporal_plugin = UnalignedMemoryTemporal(**dict(unaligned_temporal or {}))
        self.temporal_adapter = P1TopKAdapter()

    def init_weights(self):
        super().init_weights()
        self.temporal_plugin.reset_output()
