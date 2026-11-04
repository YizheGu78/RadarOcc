"""Z mapping ablation at user's P0 = existing post-selection P1 entry."""
import json
import uuid
from pathlib import Path
import torch
from mmdet.models import DETECTORS
from mmcv.runner import HOOKS, Hook
from .radarocc_self_small import RadarOcc_small
from temporal.core.ego_alignment import RadarBinGeometry
from temporal.fusion.box_z_p0 import BoxZP0Fusion
from spconv.pytorch.conv import SparseConvolution


@HOOKS.register_module()
class BoxZEpochHook(Hook):
    def __init__(self, dataloader=None):
        # This project's custom hook builder supplies the validation dataloader.
        self.dataloader = dataloader

    def before_train_epoch(self, runner):
        model = getattr(runner.model, 'module', runner.model)
        model._box_z_epoch = runner.epoch+1


@DETECTORS.register_module()
class RadarOccBoxZ(RadarOcc_small):
    def __init__(self, box_z_cfg, **kwargs):
        if kwargs.pop('temporal_cfg', None) is not None:
            raise ValueError('Box Z ablation cannot also enable the old temporal plugin')
        super().__init__(temporal_cfg=None, **kwargs)
        cfg = dict(box_z_cfg)
        bins = RadarBinGeometry.from_file(cfg.pop('bins_path'))
        self.freeze_original = cfg.pop('freeze_original', True)
        self.diagnostic_path = cfg.pop('diagnostic_path', None)
        self.box_z_fusion = BoxZP0Fusion([bins.bins[0][:175], *bins.bins[1:]], **cfg)
        if self.freeze_original:
            self.requires_grad_(False)
            self.box_z_fusion.requires_grad_(True)
        self._prepared_box_z = None
        self._box_pairs = None

    def train(self, mode=True):
        if not mode:
            self._box_z_eval_run = uuid.uuid4().hex
        super().train(mode)
        if self.freeze_original:
            # Frozen BN/dropout behavior is identical in all four experiments.
            for name, child in self.named_children():
                if name != 'box_z_fusion':
                    child.eval()
            self.box_z_fusion.train(mode)
            # spconv's eval kernels bypass autograd. Keep only sparse convolution
            # kernels in training mode; their weights and all BN/dropout stay frozen.
            for module in self.modules():
                if isinstance(module, SparseConvolution):
                    module.train(mode)
        return self

    def _prepare_p1_frame(self, rdr_cube):
        if self._prepared_box_z is not None:
            return self._prepared_box_z
        return super()._prepare_p1_frame(rdr_cube)

    def extract_pts_feat(self, rdr_cube, temporal_rdr_cubes=None, **kwargs):
        if not self.box_z_fusion.enabled:
            return super().extract_pts_feat(rdr_cube)
        if temporal_rdr_cubes is None or len(temporal_rdr_cubes) != 4 or self._box_pairs is None:
            raise ValueError('P0 box fusion requires cached box pairs and four radar frames')
        with torch.no_grad():
            prepared = [RadarOcc_small._prepare_p1_frame(self, frame) for frame in temporal_rdr_cubes]
        features = torch.stack([p[0] for p in prepared])
        coords = torch.stack([p[1][:, [2, 3, 1]].long() for p in prepared])
        device = features.device
        pairs = {key: value[0].to(device=device, dtype=torch.bool if key == 'matched' else torch.float32)
                 for key, value in self._box_pairs.items()}
        with torch.autocast(device_type=device.type, enabled=False):
            fused = self.box_z_fusion(features.float(), coords, pairs)
        self._prepared_box_z = (fused, prepared[-1][1], prepared[-1][2])
        try:
            # The original Range Attention, Sparse Encoder and entire decoder path remain intact.
            return super().extract_pts_feat(rdr_cube)
        finally:
            self._prepared_box_z = None

    def forward_train(self, *args, **kwargs):
        self._box_pairs = kwargs.pop('box_pairs', None)
        losses = super().forward_train(*args, **kwargs)
        # Add regularizers AFTER the parent's occupancy-loss normalization.
        losses.update(self.box_z_fusion.losses)
        device = next(self.box_z_fusion.parameters()).device
        for key, value in self.box_z_fusion.diagnostics.items():
            losses['z_'+key] = torch.tensor(value, device=device, dtype=torch.float32)
        for key, values in self.box_z_fusion.pair_values.items():
            if values:
                losses['z_'+key+'_mean'] = torch.tensor(values, device=device).mean()
        return losses

    def simple_test(self, *args, **kwargs):
        self._box_pairs = kwargs.pop('box_pairs', None)
        result = super().simple_test(*args, **kwargs)
        if self.diagnostic_path and self.box_z_fusion.enabled:
            path = Path(self.diagnostic_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as stream:
                stream.write(json.dumps(dict(mode=self.box_z_fusion.mode,
                    epoch=getattr(self, '_box_z_epoch', 0), eval_run=getattr(self, '_box_z_eval_run', ''),
                    **self.box_z_fusion.diagnostics, pair_values=self.box_z_fusion.pair_values))+'\n')
        return result
