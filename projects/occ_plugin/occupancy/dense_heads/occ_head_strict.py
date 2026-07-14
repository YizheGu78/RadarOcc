import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import HEADS

from .lovasz_softmax import lovasz_softmax
from .occ_head_fp16 import OccHead


def geo_scal_loss_strict(
    pred,
    ssc_target,
    ignore_index=255,
    non_empty_idx=0,
):
    """Official geometric scaling loss without compatibility casts/clamps."""
    pred = F.softmax(pred, dim=1)

    empty_probs = pred[:, non_empty_idx]
    nonempty_probs = 1 - empty_probs

    mask = ssc_target != ignore_index
    nonempty_target = ssc_target != non_empty_idx
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    eps = 1e-5
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum() + eps)
    recall = intersection / (nonempty_target.sum() + eps)
    spec = (
        ((1 - nonempty_target) * empty_probs).sum()
        / ((1 - nonempty_target).sum() + eps)
    )

    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


def sem_scal_loss_strict(pred, ssc_target, ignore_index=255):
    """Official semantic scaling loss without compatibility casts/clamps."""
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != ignore_index
    n_classes = pred.shape[1]

    for i in range(n_classes):
        p = pred[:, i]
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0

        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0

            if torch.sum(p) > 0:
                precision = nominator / torch.sum(p)
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision))
                loss_class += loss_precision

            if torch.sum(completion_target) > 0:
                recall = nominator / torch.sum(completion_target)
                loss_recall = F.binary_cross_entropy(
                    recall, torch.ones_like(recall))
                loss_class += loss_recall

            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum(
                    (1 - p) * (1 - completion_target)
                ) / torch.sum(1 - completion_target)
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity))
                loss_class += loss_specificity

            loss += loss_class

    return loss / count


def ce_ssc_loss_strict(
    pred,
    target,
    class_weights=None,
    ignore_index=255,
):
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=ignore_index,
        reduction='mean',
    )
    return criterion(pred, target.long())


@HEADS.register_module()
class OccHeadStrict(OccHead):
    """OccHead using the loss implementation from the official release."""

    def loss_voxel(
        self,
        output_voxels,
        target_voxels,
        tag,
        sigmoid_cube=None,
    ):
        loss_dict = {}

        B, C, H, W, D = output_voxels.shape
        ratio = target_voxels.shape[2] // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(
                B, H, ratio, W, ratio, D, ratio
            ).permute(0, 1, 3, 5, 2, 4, 6).reshape(
                B, H, W, D, ratio**3
            )
            empty_mask = target_voxels.sum(-1) == self.empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space == 0] = -torch.arange(
                len(occ_space[occ_space == 0])
            ).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels < 0] = 255
            target_voxels = target_voxels.long()

        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict[f'loss_voxel_ce_{tag}'] = (
            self.loss_voxel_ce_weight
            * ce_ssc_loss_strict(
                output_voxels,
                target_voxels,
                self.class_weights.type_as(output_voxels),
                ignore_index=255,
            )
        )
        loss_dict[f'loss_voxel_sem_scal_{tag}'] = (
            self.loss_voxel_sem_scal_weight
            * sem_scal_loss_strict(
                output_voxels,
                target_voxels,
                ignore_index=255,
            )
        )
        loss_dict[f'loss_voxel_geo_scal_{tag}'] = (
            self.loss_voxel_geo_scal_weight
            * geo_scal_loss_strict(
                output_voxels,
                target_voxels,
                ignore_index=255,
                non_empty_idx=self.empty_idx,
            )
        )
        loss_dict[f'loss_voxel_lovasz_{tag}'] = (
            self.loss_voxel_lovasz_weight
            * lovasz_softmax(
                torch.softmax(output_voxels, dim=1),
                target_voxels,
                ignore=255,
            )
        )

        return loss_dict

    def loss_point(self, fine_coord, fine_output, target_voxels, tag):
        selected_gt = target_voxels[
            :,
            fine_coord[0, :],
            fine_coord[1, :],
            fine_coord[2, :],
        ].long()[0]

        assert torch.isnan(selected_gt).sum().item() == 0
        assert torch.isnan(fine_output).sum().item() == 0

        loss_dict = {}
        loss_dict[f'loss_voxel_ce_{tag}'] = (
            self.loss_voxel_ce_weight
            * ce_ssc_loss_strict(
                fine_output,
                selected_gt,
                ignore_index=255,
            )
        )
        loss_dict[f'loss_voxel_sem_scal_{tag}'] = (
            self.loss_voxel_sem_scal_weight
            * sem_scal_loss_strict(
                fine_output,
                selected_gt,
                ignore_index=255,
            )
        )
        loss_dict[f'loss_voxel_geo_scal_{tag}'] = (
            self.loss_voxel_geo_scal_weight
            * geo_scal_loss_strict(
                fine_output,
                selected_gt,
                ignore_index=255,
                non_empty_idx=self.empty_idx,
            )
        )
        loss_dict[f'loss_voxel_lovasz_{tag}'] = (
            self.loss_voxel_lovasz_weight
            * lovasz_softmax(
                torch.softmax(fine_output, dim=1),
                selected_gt,
                ignore=255,
            )
        )

        return loss_dict
