"""Install a temporal plugin into the existing P0 hook without modifying RadarOcc."""
from mmdet.models import DETECTORS
import torch
from projects.occ_plugin.occupancy.detectors.radarocc_self_small import RadarOcc_small
from temporal.adapters import P1TopKAdapter
from temporal.centerformer.fusion import CenterFormerP0
from temporal.rodnet.fusion import RODNetP0


@DETECTORS.register_module()
class RadarOccP0Temporal(RadarOcc_small):
    def __init__(self, p0_temporal_cfg, **kwargs):
        if kwargs.pop('temporal_cfg', None) is not None:
            raise ValueError('Select only one P0 temporal plugin')
        super().__init__(temporal_cfg=None, **kwargs)
        cfg = dict(p0_temporal_cfg)
        method = cfg.pop('method')
        choices = {'centerformer': CenterFormerP0, 'rodnet_tdc': RODNetP0}
        if method not in choices:
            raise ValueError('Unknown P0 temporal method: '+str(method))
        self.temporal_plugin = choices[method](**cfg)
        self.temporal_adapter = P1TopKAdapter()

    def forward_train(self, *args, **kwargs):
        losses = super().forward_train(*args, **kwargs)
        # Diagnostic keys deliberately contain no 'loss', so they are not optimized.
        device = next(self.temporal_plugin.parameters()).device
        for key, value in self.temporal_plugin.diagnostics.items():
            losses['p0_'+key] = torch.tensor(float(value), device=device)
        return losses
