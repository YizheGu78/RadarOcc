import torch
import torch.nn as nn


class P1TopKAdapter(nn.Module):
    """
    Adapter for RadarOcc P1: immediately after per-range Top-K selection.

    At the identity stage, P1 coordinates are kept as spherical grid indices
    in (range, azimuth, elevation) order.  They are NOT yet ego-motion warped.
    The next DG-STF stage will convert/align them using the calibrated geometry.

    Inputs contain one already-selected P1 tensor per temporal frame:
        frame_features[t]: [N, C]
        frame_indices[t]:  [N, 4] = [batch, elevation, range, azimuth]

    Output:
        features: [1, T, N, C]
        coords:   [1, T, N, 3] = [range_idx, azimuth_idx, elevation_idx]
        doppler:  [1, T, N, C] (raw P1 descriptor; identity mode ignores it)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        frame_features,
        frame_indices,
    ):
        if len(frame_features) == 0:
            raise ValueError("P1TopKAdapter received no temporal frames.")

        if len(frame_features) != len(frame_indices):
            raise ValueError(
                "frame_features and frame_indices must have the same length."
            )

        reference_shape = frame_features[-1].shape
        reference_index_shape = frame_indices[-1].shape

        for temporal_idx, (features, indices) in enumerate(
            zip(frame_features, frame_indices)
        ):
            if features.shape != reference_shape:
                raise ValueError(
                    "All P1 frame feature tensors must have identical shape; "
                    f"frame {temporal_idx} has {tuple(features.shape)}, "
                    f"current has {tuple(reference_shape)}."
                )
            if indices.shape != reference_index_shape:
                raise ValueError(
                    "All P1 frame index tensors must have identical shape; "
                    f"frame {temporal_idx} has {tuple(indices.shape)}, "
                    f"current has {tuple(reference_index_shape)}."
                )

        # RadarOcc-S currently runs samples_per_gpu=1 and internally hardcodes
        # B=1 in the P1 sparse preparation path.  Keep that invariant explicit
        # for the regression stage instead of silently producing wrong batches.
        batch_ids = frame_indices[-1][:, 0]
        if not torch.all(batch_ids == 0):
            raise ValueError(
                "P1TopKAdapter identity stage currently requires batch size 1."
            )

        features = torch.stack(
            frame_features,
            dim=0,
        ).unsqueeze(0)

        # sp_indices columns are [batch, elevation, range, azimuth].
        # Reorder the non-batch coordinates to a stable semantic convention.
        coords = torch.stack(
            [
                indices[:, [2, 3, 1]].to(
                    dtype=features.dtype
                )
                for indices in frame_indices
            ],
            dim=0,
        ).unsqueeze(0)

        return {
            "features": features,
            "coords": coords,
            # The original 11-D P1 descriptor is preserved wholesale here.
            # Doppler-channel parsing is deliberately deferred until the
            # Doppler-guidance stage, so identity regression cannot alter it.
            "doppler": features,
        }
