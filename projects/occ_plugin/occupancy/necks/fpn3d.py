# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn
from mmcv.cnn import ConvModule
from mmdet.models import NECKS

import torch.nn.functional as F
import pdb
from contextlib import contextmanager, nullcontext


@contextmanager
def _recompute_batchnorm_buffers(module):
    """Use temporary running buffers during activation recomputation only.

    Training BatchNorm still uses batch statistics and keeps the same graph.
    Replaying a checkpoint must not update inference statistics a second time.
    """
    originals = []
    for layer in module.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm) and layer.track_running_stats:
            for name in ('running_mean', 'running_var', 'num_batches_tracked'):
                original = getattr(layer, name)
                if original is not None:
                    originals.append((layer, name, original))
                    setattr(layer, name, original.clone())
    try:
        yield
    finally:
        for layer, name, original in originals:
            setattr(layer, name, original)

@NECKS.register_module()
class FPN3D(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """
    def __init__(self,
                 in_channels=[80, 160, 320, 640],
                 out_channels=256,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 conv_cfg=dict(type='Conv3d'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False,
                 checkpoint_use_reentrant=True,
                 upsample_cfg=dict(mode='trilinear'),
                 init_cfg=None):
        super(FPN3D, self).__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg
        self.with_cp = with_cp
        self.checkpoint_use_reentrant = checkpoint_use_reentrant
        
        self.num_out = len(self.in_channels)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for i in range(self.num_out):
            # 拓展感受野
            l_conv = nn.Sequential(
                ConvModule(in_channels[i], out_channels, 
                    kernel_size=1, padding=0,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, 
                    act_cfg=act_cfg, bias=False, 
                    inplace=True),
            )
            
            fpn_conv = nn.Sequential(
                ConvModule(out_channels, out_channels, 
                    kernel_size=3, padding=1,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, 
                    act_cfg=act_cfg, bias=False, 
                    inplace=True),
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def _checkpoint_layer(self, module, value):
        if self.checkpoint_use_reentrant:
            def replay(x):
                # Reentrant checkpoint's first forward is under no_grad;
                # its backward replay enables gradients.
                context = _recompute_batchnorm_buffers(module) if torch.is_grad_enabled() else nullcontext()
                with context:
                    return module(x)
            return checkpoint(replay, value, use_reentrant=True)
        return checkpoint(module, value, use_reentrant=False,
            context_fn=lambda: (nullcontext(), _recompute_batchnorm_buffers(module)))

    @auto_fp16()
    def forward(self, inputs):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if self.with_cp:
                # lateral_i = checkpoint(lateral_conv, inputs[i],
                #                        use_reentrant=self.checkpoint_use_reentrant)
                lateral_i = self._checkpoint_layer(lateral_conv, inputs[i])
            else:
                lateral_i = lateral_conv(inputs[i])
            laterals.append(lateral_i)

        # build down-top path
        for i in range(self.num_out - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], 
                    size=prev_shape, align_corners=False, **self.upsample_cfg)
        
        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(self.num_out)
        # ]
        
        outs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            if self.with_cp:
                # out_i = checkpoint(fpn_conv, laterals[i],
                #                    use_reentrant=self.checkpoint_use_reentrant)
                out_i = self._checkpoint_layer(fpn_conv, laterals[i])
            else:
                out_i = fpn_conv(laterals[i])
            outs.append(out_i)
        
        return outs
