import torch

from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS

from .radarocc_self_small import RadarOcc_small


@DETECTORS.register_module()
class RadarOcc_small_strict(RadarOcc_small):
    """Official RadarOcc-S execution path.

    This class keeps the existing RTX 5060/FP32-compatible implementation in
    ``RadarOcc_small`` untouched, while restoring the official Small-model
    behavior used by the released code:

    - 800 candidates per range are read, then the top 250 are selected.
    - sparse encoding, occupancy encoding, and occupancy-head inputs follow
      the original AMP/FP16 path.
    - the original loss-normalization logic is retained.
    """

    def cartesian_to_spherical(self, x, y, z):
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.atan2(y, x)
        phi = torch.asin(z / r)
        return r, theta, phi

    @auto_fp16()
    def occ_encoder(self, x):
        x = self.occ_encoder_backbone(x)
        x = self.occ_encoder_neck(x)
        return x

    def extract_pts_feat(self, rdr_cube):
        if self.record_time:
            torch.cuda.synchronize()
            t0 = __import__('time').time()

        power_values = rdr_cube['power_val']
        range_indices = rdr_cube['range_ind']
        elevation_indices = rdr_cube['elevation_ind']
        azimuth_indices = rdr_cube['azimuth_ind']

        dtype = torch.float32
        batch_size = B = 1
        list_sparse_rdr_cubes = []
        list_sp_indices = []

        for batch_idx in range(B):
            # Official RadarOcc-S flow: 800 candidates -> top 250 per range.
            original_k = 250
            n_ranges = 175

            power_val = power_values[batch_idx][:, :original_k * n_ranges]
            elevation_ind = elevation_indices[batch_idx][:original_k * n_ranges]
            azimuth_ind = azimuth_indices[batch_idx][:original_k * n_ranges]
            range_ind = range_indices[batch_idx][:original_k * n_ranges]

            # Keep the original released implementation exactly: the range
            # indices are not used during the local top-k gather below.
            elevation_ind = elevation_ind
            azimuth_ind = azimuth_ind
            range_ind = range_ind

            k = 250
            power_val = power_val[2]
            reshaped_power_vals = power_val.view(n_ranges, original_k)

            values, indices = torch.sort(
                reshaped_power_vals, dim=1, descending=True)
            top_k_values = values[:, :k]
            top_k_indices = indices[:, :k]

            top_k_range_inds = (
                torch.arange(n_ranges).unsqueeze(1).repeat(1, k).cuda())
            top_k_elevation_inds = elevation_ind.view(
                n_ranges, original_k).gather(1, top_k_indices)
            top_k_azimuth_inds = azimuth_ind.view(
                n_ranges, original_k).gather(1, top_k_indices)

            final_range_inds = top_k_range_inds.flatten()
            final_elevation_inds = top_k_elevation_inds.flatten()
            final_azimuth_inds = top_k_azimuth_inds.flatten()

            power_val = rdr_cube['power_val'][batch_idx]
            sparse_rdr_cube = torch.swapaxes(
                power_val, 0, 1)[top_k_indices.flatten(), :]
            list_sparse_rdr_cubes.append(sparse_rdr_cube)

            N, C = sparse_rdr_cube.shape
            batch_indices = torch.full(
                (N, 1), batch_idx, dtype=torch.long).cuda()
            sp_indices = torch.cat(
                (
                    batch_indices,
                    final_elevation_inds.unsqueeze(-1),
                    final_range_inds.unsqueeze(-1),
                    final_azimuth_inds.unsqueeze(-1),
                ),
                dim=-1,
            )
            list_sp_indices.append(sp_indices)

        sparse_rdr_cube_all_batches = torch.cat(
            list_sparse_rdr_cubes, dim=0)
        sp_indices_all_batches = torch.cat(
            list_sp_indices, dim=0).cuda()

        with torch.cuda.amp.autocast():
            pts_enc_feats = self.pts_middle_encoder(
                sparse_rdr_cube_all_batches,
                sp_indices_all_batches,
                batch_size,
            )

        if self.record_time:
            torch.cuda.synchronize()
            t1 = __import__('time').time()
            self.time_stats['sparse_encoder'].append(t1 - t0)

        if self.record_time:
            torch.cuda.synchronize()
            t0 = __import__('time').time()

        pts_feats = pts_enc_feats['pts_feats']
        spherical_feat = pts_enc_feats['x']
        h, w, z = [128, 128, 14]
        spherical_pos_self = self.positional_encoding(
            torch.zeros((batch_size, h, w, z)).to(dtype).cuda()).to(dtype)
        vox_feats_flatten = spherical_feat.flatten(2)
        vox_coords, ref_3d = self.get_ref_3d()

        vox_feats_diff = self.self_transformer.diffuse_vox_features(
            vox_feats_flatten,
            h,
            w,
            ref_3d=ref_3d,
            spatial_shapes=[h, w, z],
            vox_coords=None,
            bev_pos=spherical_pos_self,
            prev_bev=None,
        )

        if self.record_time:
            torch.cuda.synchronize()
            t1 = __import__('time').time()
            self.time_stats['self_attn'].append(t1 - t0)
            t0 = __import__('time').time()

        voxel_feat = vox_feats_diff.reshape(
            (1, self.embed_dims, h, w, z))
        residual = voxel_feat
        voxel_queries = self.cart_voxel_emb.weight.to(dtype)
        bev_pos_self_attn = self.positional_encoding(
            torch.zeros((batch_size, 128, 128, 14)).to(dtype).cuda()
        ).to(dtype)
        spherical_pos = self.positional_encoding(
            torch.zeros((batch_size, h, w, z)).to(dtype).cuda()
        ).to(dtype)
        voxel_queries = voxel_queries.permute(1, 0)
        voxel_queries = voxel_queries.unsqueeze(0)
        sph_ref_3d = self.get_reference_points_spherical(
            device=spherical_feat.device,
            spherical_shape=[h, w, z],
            scale=2,
        )

        vox_feats_diff = self.transformer.get_cart_features(
            voxel_queries,
            voxel_feat,
            128,
            128,
            ref_3d=sph_ref_3d,
            sph_ref_3d=sph_ref_3d,
            spatial_shapes=[h, w, z],
            cart_spatial_shape=[128, 128, 14],
            vox_coords=None,
            bev_pos=bev_pos_self_attn,
            spherical_pos=spherical_pos,
            prev_bev=None,
        )

        vox_feats_diff = vox_feats_diff.reshape(
            128, 128, 14, self.embed_dims)
        voxel_feat = vox_feats_diff.permute(3, 0, 1, 2)
        voxel_feat = voxel_feat.unsqueeze(0) + residual

        if self.record_time:
            torch.cuda.synchronize()
            t1 = __import__('time').time()
            self.time_stats['sph_to_cart'].append(t1 - t0)

        return voxel_feat.half(), pts_feats

    def extract_feat(self, rdr_cube):
        img_voxel_feats = None
        pts_voxel_feats, pts_feats = None, None
        depth, img_feats = None, None

        pts_voxel_feats, pts_feats = self.extract_pts_feat(rdr_cube)

        if self.record_time:
            torch.cuda.synchronize()
            t0 = __import__('time').time()

        if self.occ_fuser is not None:
            voxel_feats = self.occ_fuser(
                img_voxel_feats, pts_voxel_feats)
        else:
            assert (img_voxel_feats is None) or (pts_voxel_feats is None)
            voxel_feats = (
                img_voxel_feats
                if pts_voxel_feats is None
                else pts_voxel_feats
            )

        if self.record_time:
            torch.cuda.synchronize()
            t1 = __import__('time').time()
            self.time_stats['occ_fuser'].append(t1 - t0)

        with torch.cuda.amp.autocast():
            voxel_feats_enc = self.occ_encoder(voxel_feats)

        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]

        if self.record_time:
            torch.cuda.synchronize()
            t2 = __import__('time').time()
            self.time_stats['occ_encoder'].append(t2 - t1)

        return voxel_feats_enc, img_feats, pts_feats, depth

    @auto_fp16()
    def forward_pts_train(
        self,
        voxel_feats,
        gt_occ=None,
        points_occ=None,
        img_metas=None,
        transform=None,
        img_feats=None,
        pts_feats=None,
        visible_mask=None,
    ):
        if self.record_time:
            torch.cuda.synchronize()
            t0 = __import__('time').time()

        for i in range(len(voxel_feats)):
            voxel_feats[i] = voxel_feats[i].half()

        outs = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=None,
            transform=transform,
        )

        if self.record_time:
            torch.cuda.synchronize()
            t1 = __import__('time').time()
            self.time_stats['occ_head'].append(t1 - t0)

        losses = self.pts_bbox_head.loss(
            output_voxels=outs['output_voxels'],
            output_voxels_fine=outs['output_voxels_fine'],
            output_coords_fine=outs['output_coords_fine'],
            target_voxels=gt_occ,
            target_points=points_occ,
            img_metas=img_metas,
            visible_mask=visible_mask,
        )

        if self.record_time:
            torch.cuda.synchronize()
            t2 = __import__('time').time()
            self.time_stats['loss_occ'].append(t2 - t1)

        return losses

    def forward_train(
        self,
        points=None,
        img_metas=None,
        img_inputs=None,
        gt_occ=None,
        points_occ=None,
        visible_mask=None,
        **kwargs,
    ):
        sparse_radar = kwargs['sparse_radar']
        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(
            rdr_cube=sparse_radar)

        losses = dict()

        if self.record_time:
            torch.cuda.synchronize()
            t0 = __import__('time').time()

        if not self.disable_loss_depth and depth is not None:
            losses['loss_depth'] = self.img_view_transformer.get_depth_loss(
                img_inputs[-2], depth)

        if self.record_time:
            torch.cuda.synchronize()
            t1 = __import__('time').time()
            self.time_stats['loss_depth'].append(t1 - t0)

        transform = img_inputs[1:8] if img_inputs is not None else None
        losses_occupancy = self.forward_pts_train(
            voxel_feats,
            gt_occ,
            points_occ,
            img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
            visible_mask=visible_mask,
        )
        losses.update(losses_occupancy)

        if self.loss_norm:
            l1_deatch = 0
            for loss_key in losses.keys():
                if loss_key.startswith('loss'):
                    losses[loss_key] = losses[loss_key] / (
                        losses[loss_key].detach() + 1e-9)

        def logging_latencies():
            avg_time = {
                key: sum(val) / len(val)
                for key, val in self.time_stats.items()
            }
            sum_time = sum(list(avg_time.values()))
            out_res = ''
            for key, val in avg_time.items():
                out_res += '{}: {:.4f}, {:.1f}, '.format(
                    key, val, val / sum_time)
            print(out_res)

        if self.record_time:
            logging_latencies()

        return losses
