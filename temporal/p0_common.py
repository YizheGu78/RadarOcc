"""Shared P0 boundary: calibrated BEV rasterization and raw-descriptor adapter.

P0 is the existing post-selection/pre-Range-Attention hook called P1 in the
original detector. No Doppler channels enter either new temporal branch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core.ego_alignment import RadarBinGeometry, align_to_current

CONTINUOUS = (0, 1, 2, 6, 7)


def sample_volume(field, positions, observation=None, min_weight=1e-4):
    """Trilinear XYZ sampling with missing observations excluded from the mean.

    field [B,C,Z,H,W], positions [B,N,3] in normalized [0,1] XYZ. A real
    observed zero stays in the denominator; an unobserved cell does not.
    """
    grid = (2*positions-1)[:, :, None, None]
    numerator = F.grid_sample(field if observation is None else field*observation,
                              grid, align_corners=False, padding_mode='zeros')
    values = numerator[:, :, :, 0, 0].transpose(1, 2)
    inside = ((positions >= 0) & (positions < 1)).all(-1)
    if observation is None:
        return values, inside
    weight = F.grid_sample(observation, grid, align_corners=False)[:, 0, :, 0, 0]
    good = inside & (weight >= min_weight)
    return values/weight.clamp_min(min_weight)[..., None]*good[..., None], good


def masked_xy_pool(field, mask, scale):
    """Pool XY while preserving every height slice and excluding empty cells."""
    kernel = (1, scale, scale)
    weight = F.avg_pool3d(mask, kernel)
    numerator = F.avg_pool3d(field*mask, kernel)
    valid = (weight > 0).to(field)
    return numerator/weight.clamp_min(1e-4)*valid, valid


class P0TemporalBase(nn.Module):
    def __init__(self, bins_path, frame_nums=4, channels=32, chunk_size=1024,
                 bev_bounds=(-4., -64., 84., 64.), cell_size=.8,
                 max_log_correction=.5, time_scale_frames=3., enabled=True,
                 radar_to_lidar=None, height_bounds=(-4., 8.), height_cell_size=1.):
        super().__init__()
        if frame_nums < 2 or channels < 1 or chunk_size < 1:
            raise ValueError('Need at least two frames and positive channels/chunk_size')
        if len(bev_bounds) != 4 or not all(math.isfinite(x) for x in bev_bounds):
            raise ValueError('bev_bounds must be finite xmin,ymin,xmax,ymax')
        xmin, ymin, xmax, ymax = bev_bounds
        if not (math.isfinite(cell_size) and cell_size > 0 and xmax > xmin and ymax > ymin):
            raise ValueError('Invalid BEV bounds/cell size')
        if max_log_correction <= 0 or time_scale_frames <= 0:
            raise ValueError('Correction/time scale must be positive')
        if (len(height_bounds) != 2 or not all(math.isfinite(x) for x in height_bounds)
                or height_bounds[1] <= height_bounds[0] or not math.isfinite(height_cell_size)
                or height_cell_size <= 0):
            raise ValueError('Invalid physical height bounds/cell size')
        self.geometry = RadarBinGeometry.from_file(bins_path)
        self.frame_nums, self.channels, self.chunk_size = frame_nums, channels, chunk_size
        # Pad XY to multiples of four: all CenterFormer scales then have exactly
        # the same physical extent and normalized reference coordinates.
        self.height = 4*math.ceil((ymax-ymin)/cell_size/4)
        self.width = 4*math.ceil((xmax-xmin)/cell_size/4)
        self.depth = math.ceil((height_bounds[1]-height_bounds[0])/height_cell_size)
        self.register_buffer('origin', torch.tensor([xmin, ymin, height_bounds[0]]))
        self.register_buffer('extent', torch.tensor([self.width*cell_size, self.height*cell_size,
                                                     self.depth*height_cell_size]))
        self.register_buffer('voxel_size', torch.tensor([cell_size, cell_size, height_cell_size]))
        self.cell_size = cell_size
        self.extrinsic = radar_to_lidar
        self.bound, self.time_scale, self.enabled = max_log_correction, time_scale_frames, enabled
        self.encoder = nn.Sequential(nn.Conv2d(5, channels, 1, bias=False), nn.ReLU())
        self.output = nn.Linear(channels, 5)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.diagnostics = {}

    def _rasterize(self, descriptors, xyz, valid):
        """Height-resolved XYZ voxel means; never combine different Z cells.

        Raster in the CURRENT radar coordinate system, once, after full SE(3)
        ego alignment. Missing frames/cells are zero, with a separate mask.
        """
        b, t, n, c = descriptors.shape
        ix = ((xyz-self.origin)/self.voxel_size).floor().long()
        inside = ((ix[..., 0] >= 0) & (ix[..., 0] < self.width) &
                  (ix[..., 1] >= 0) & (ix[..., 1] < self.height) &
                  (ix[..., 2] >= 0) & (ix[..., 2] < self.depth) & valid[..., None])
        size = self.depth*self.height*self.width
        index = (ix[..., 2]*self.height+ix[..., 1])*self.width+ix[..., 0]
        index = index.clamp(0, size-1)
        count = descriptors.new_zeros(b, t, size)
        count.scatter_add_(2, index, inside.to(descriptors))
        field = descriptors.new_zeros(b, t, c, size)
        field.scatter_add_(3, index[:, :, None].expand(-1, -1, c, -1),
                           (descriptors * inside[..., None]).permute(0, 1, 3, 2))
        field = (field/count.clamp_min(1)[:, :, None]).reshape(b*t, c, self.depth, self.height, self.width)
        # Share the original 2D pointwise encoder across height slices.
        field = field.permute(0, 2, 1, 3, 4).reshape(b*t*self.depth, c, self.height, self.width)
        mask = (count > 0).reshape(b, t, 1, self.depth, self.height, self.width).to(descriptors)
        encoded = self.encoder(field).reshape(b, t, self.depth, self.channels, self.height, self.width)
        encoded = encoded.permute(0, 1, 3, 2, 4, 5)
        return encoded * mask, mask

    def fuse(self, maps, masks, query, positions, times, valid):
        raise NotImplementedError

    def forward(self, features, coords, poses=None, frame_offsets=None,
                doppler=None, valid_mask=None, timestamps=None):
        if features.ndim != 4 or features.shape[-1] != 8 or coords.shape != features.shape[:-1]+(3,):
            raise ValueError('P0 requires features [B,T,N,8] and R/A/E coords [B,T,N,3]')
        if not torch.isfinite(features).all():
            raise ValueError('Nonfinite P0 descriptors')
        current = features[:, -1]
        # Connect unused parameters for DDP even at scene starts / disabled runs.
        anchor = sum((p.reshape(-1)[0]*0 for p in self.parameters()), features.new_zeros(()))
        if not self.enabled:
            self.diagnostics = dict(used_history=False, queries_with_history=0)
            return current + anchor, self.diagnostics
        b, t, n, _ = features.shape
        if t != self.frame_nums or n == 0:
            raise ValueError('Temporal frame count disagrees with configuration or empty point set')
        if poses is None or frame_offsets is None or valid_mask is None:
            raise ValueError('P0 temporal fusion requires ego poses, frame offsets and validity mask')
        def cpu(x):
            return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        valid = np.asarray(cpu(valid_mask), dtype=bool)
        dt = np.asarray(cpu(frame_offsets), dtype=np.float64)
        if valid.shape != (b, t) or not valid[:, -1].all():
            raise ValueError('Invalid temporal validity mask')
        if dt.shape != (b, t) or not np.isfinite(dt).all():
            raise ValueError('Invalid frame offsets')
        if (dt[:, -1] != 0).any() or (dt[valid & (np.arange(t)[None] < t-1)] >= 0).any():
            raise ValueError('Only past frames and a current frame at offset zero are accepted')
        if not valid[:, :-1].any():
            self.diagnostics = dict(used_history=False, queries_with_history=0)
            return current + anchor, self.diagnostics
        xyz = self.geometry.to_xyz(cpu(coords))
        xyz, _ = align_to_current(xyz, cpu(poses), self.extrinsic, valid)
        # Do geometry and raw dynamic-range conversion in FP32 even under AMP.
        with torch.autocast(device_type=features.device.type, enabled=False):
            xyz = torch.as_tensor(xyz, device=features.device, dtype=torch.float32)
            raw = features.float()[..., list(CONTINUOUS)]
            descriptors = torch.sign(raw) * torch.log1p(raw.abs()) / 80.
            vt = torch.as_tensor(valid, device=features.device)
            maps, masks = self._rasterize(descriptors, xyz, vt)
            positions = (xyz[:, -1] - self.origin) / self.extent
            times = torch.as_tensor(dt, device=features.device, dtype=torch.float32) / self.time_scale
            mixed, support, info = self.fuse(maps, masks, descriptors[:, -1], positions, times, vt)
            inside = ((positions >= 0) & (positions < 1)).all(-1)
            support = support & inside & vt[:, :-1].any(-1, keepdim=True)
            correction = self.bound * self.output(mixed).tanh() * support[..., None]
            continuous = raw[:, -1] + raw[:, -1] * torch.expm1(correction)
            result = torch.cat((continuous[..., :3], features[:, -1, :, 3:6].float(), continuous[..., 3:]), -1)
            self.diagnostics = dict(used_history=bool(support.any()),
                                    queries_with_history=int(support.sum()),
                                    queries_in_volume=int(inside.sum()),
                                    **info)
        return result.to(features.dtype) + anchor, self.diagnostics
