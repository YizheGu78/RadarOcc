# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd.function import Function, once_differentiable
from mmcv.utils import ext_loader
# ext_module = ext_loader.load_ext(
#     '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
import sys


# sys.path.append("/root/autodl-tmp/vox/mmcv-main/cntest/build/lib.linux-x86_64-cpython-38/")
# sys.path.append("/home/yml/bevformer/mmdetection3d/deform_3d_cuda/")
# raise NotImplementedError("Use sys.path.append here to modify the path to your .so file")
sys.path.append("/home/user1/projects/RadarOcc/third_party/VoxFormer/deform_attn_3d")



import deform3dattn_custom_cn as ext_module


class MultiScaleDeformableAttn3DCustomFunction_fp16(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """
        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None


class MultiScaleDeformableAttn3DCustomFunction_fp32(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """

        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None




def multi_scale_deformable_attn_3d_pytorch(value, spatial_shapes, sampling_locations,
                                         attention_weights):
    """Differentiable 3D reference: storage [A,R,E], locations [R,A,E].

    Unlike MMCV's 2D fallback, this uses 5D grid_sample and the same half-cell
    and zero-padding convention as the checked-in CUDA sampler.
    """
    if value.ndim != 4 or spatial_shapes.ndim != 2 or spatial_shapes.shape[1] != 3:
        raise ValueError('Expected value [B,N,heads,C] and spatial_shapes [levels,3].')
    batch, keys, heads, channels = value.shape
    shapes = spatial_shapes.tolist()
    sizes = [a*r*e for a,r,e in shapes]
    if min(min(shape) for shape in shapes) <= 0 or sum(sizes) != keys:
        raise ValueError('Value length disagrees with positive 3D spatial shapes.')
    if (sampling_locations.ndim != 6 or sampling_locations.shape[0] != batch or
            sampling_locations.shape[2:4] != (heads, len(shapes)) or
            sampling_locations.shape[-1] != 3 or
            attention_weights.shape != sampling_locations.shape[:-1]):
        raise ValueError('3D value, location and attention-weight shapes disagree.')
    original_dtype = value.dtype
    if original_dtype in (torch.float16, torch.bfloat16):
        value = value.float()
        sampling_locations = sampling_locations.float()
        attention_weights = attention_weights.float()
    queries, points = sampling_locations.shape[1], sampling_locations.shape[4]
    output = value.new_zeros(batch, queries, heads, channels)
    for level, ((a,r,e), values) in enumerate(zip(shapes, value.split(sizes, dim=1))):
        source = values.permute(0,2,3,1).reshape(batch*heads, channels, a,r,e)
        source = source.permute(0,1,4,2,3).contiguous()  # [BH,C,E,A,R]
        grid = sampling_locations[:,:,:,level].permute(0,2,1,3,4)
        grid = (grid*2-1).reshape(batch*heads, queries, points, 1, 3)
        sampled = F.grid_sample(source, grid, mode='bilinear', padding_mode='zeros',
                                align_corners=False)
        sampled = sampled.reshape(batch,heads,channels,queries,points).permute(0,3,1,4,2)
        output = output + (sampled * attention_weights[:,:,:,level,:,None]).sum(dim=3)
    return output.reshape(batch, queries, heads*channels).to(original_dtype)
