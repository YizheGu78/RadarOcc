import torch
import torch.nn as nn


class TemporalPlugin(nn.Module):
    """
    DG-STF temporal plugin.

    The first implementation is intentionally identity-only.  It verifies that
    real T-frame loading, the P1 adapter, and the temporal interface can be
    inserted without changing the RadarOcc baseline.

    Expected canonical interface:
        output, debug = plugin(
            features=[B,T,N,C],
            coords=[B,T,N,3],
            poses=[B,T,4,4],
            frame_offsets=[B,T],
            doppler=[B,T,N,Cd],
            valid_mask=[B,T],
            timestamps=[B,T],
        )
    """

    def __init__(
        self,
        mode="identity",
        position="P1",
    ):
        super().__init__()
        self.mode = str(mode).lower()
        self.position = str(position).upper()

        if self.mode != "identity":
            raise NotImplementedError(
                "Only mode='identity' is implemented in the current "
                "regression stage."
            )

        if self.position != "P1":
            raise NotImplementedError(
                "The current regression stage supports P1 only."
            )

    def forward(
        self,
        features,
        coords,
        poses=None,
        frame_offsets=None,
        doppler=None,
        valid_mask=None,
        timestamps=None,
    ):
        if features.ndim != 4:
            raise ValueError(
                "TemporalPlugin expects features [B,T,N,C], got "
                f"{tuple(features.shape)}"
            )

        if coords.ndim != 4:
            raise ValueError(
                "TemporalPlugin expects coords [B,T,N,3], got "
                f"{tuple(coords.shape)}"
            )

        batch_size, frame_nums, num_points, channels = features.shape

        if coords.shape[:3] != features.shape[:3]:
            raise ValueError(
                "Temporal feature/coordinate shapes disagree: "
                f"features={tuple(features.shape)}, "
                f"coords={tuple(coords.shape)}"
            )

        if valid_mask is not None and torch.is_tensor(valid_mask):
            if valid_mask.shape[-1] != frame_nums:
                raise ValueError(
                    "valid_mask frame dimension does not match T: "
                    f"{tuple(valid_mask.shape)} vs T={frame_nums}"
                )
            if not bool(valid_mask[..., -1].all()):
                raise ValueError(
                    "Current temporal slot must always be valid."
                )

        # Identity guarantee: return the current P1 features byte-for-byte
        # (modulo view/stack semantics), and do not use history in any way.
        output = features[:, -1]

        debug = {
            "mode": self.mode,
            "position": self.position,
            "batch_size": int(batch_size),
            "frame_nums": int(frame_nums),
            "num_points": int(num_points),
            "channels": int(channels),
            "used_history": False,
            "coord_type": "spherical_index_range_azimuth_elevation",
        }

        return output, debug
