import numpy as np
from mmdet.datasets import DATASETS

from .nuscenes_occ_dataset import NuscOCCDataset


@DATASETS.register_module()
class TemporalKRadarDataset(NuscOCCDataset):
    """
    T-frame K-Radar dataset wrapper for DG-STF development.

    Current phase: sequence construction + current-only regression.

    The dataset builds T-frame sequence indices from frame-level metadata, but
    intentionally sends only the current frame through the original RadarOcc
    pipeline. This lets us verify that introducing temporal metadata and
    sequence bookkeeping does not change the established single-frame
    baseline before any temporal fusion module is enabled.

    Required per-frame fields in the temporal PKL:
        scene_id (or scene_token)
        order_in_scene
        ego_pose
        radar_path / sparse_radar_path
        occ_path

    Optional:
        timestamp_sec
        timestamp_valid

    Scene-start policy:
        Missing history slots repeat the earliest available frame only as a
        tensor-shape placeholder and are marked invalid in valid_mask.
        Future temporal fusion MUST ignore slots whose valid_mask is False.
    """

    def __init__(
        self,
        frame_nums=4,
        current_only=True,
        **kwargs,
    ):
        if int(frame_nums) < 1:
            raise ValueError(
                f"frame_nums must be >= 1, got {frame_nums}"
            )

        self.frame_nums = int(frame_nums)
        self.current_only = bool(current_only)

        # Keep all original single-frame initialization behavior unchanged.
        super().__init__(**kwargs)

        self._validate_temporal_metadata()
        (
            self.sequence_indices,
            self.sequence_valid_masks,
            self.sequence_frame_offsets,
        ) = self._build_sequence_indices()

        if not self.current_only:
            raise NotImplementedError(
                "TemporalKRadarDataset currently implements the "
                "current-only regression stage only. Do not enable real "
                "multi-frame loading until the regression check passes."
            )

    @staticmethod
    def _scene_id(info):
        if "scene_id" in info:
            return str(info["scene_id"])
        if "scene_token" in info:
            return str(info["scene_token"])
        raise KeyError(
            "Temporal metadata requires scene_id or scene_token."
        )

    @staticmethod
    def _order_in_scene(info):
        if "order_in_scene" not in info:
            raise KeyError(
                "Temporal metadata requires order_in_scene. "
                "Regenerate the temporal PKL first."
            )
        return int(info["order_in_scene"])

    def _validate_temporal_metadata(self):
        """
        Fail early instead of allowing a subtly wrong temporal sequence.
        """
        required = (
            "order_in_scene",
            "ego_pose",
        )

        seen = set()

        for dataset_idx, info in enumerate(self.data_infos):
            missing = [
                key for key in required
                if key not in info
            ]
            if missing:
                raise KeyError(
                    f"data_infos[{dataset_idx}] is missing temporal "
                    f"fields: {missing}"
                )

            scene_id = self._scene_id(info)
            order = self._order_in_scene(info)
            key = (scene_id, order)

            if key in seen:
                raise ValueError(
                    "Duplicate temporal position detected: "
                    f"scene={scene_id}, order_in_scene={order}"
                )
            seen.add(key)

            ego_pose = np.asarray(info["ego_pose"])
            if ego_pose.shape != (4, 4):
                raise ValueError(
                    f"data_infos[{dataset_idx}] ego_pose shape is "
                    f"{ego_pose.shape}, expected (4, 4)"
                )

    def _build_sequence_indices(self):
        """
        Build all sequence relations once during Dataset initialization.

        Returns:
            sequence_indices:
                list[list[int]], shape conceptually [N, T].
            sequence_valid_masks:
                np.ndarray bool [N, T].
            sequence_frame_offsets:
                np.ndarray int64 [N, T], e.g. [-3, -2, -1, 0].

        Important:
            Sequences never cross scene boundaries.
        """
        scene_to_indices = {}

        for dataset_idx, info in enumerate(self.data_infos):
            scene_id = self._scene_id(info)
            scene_to_indices.setdefault(scene_id, []).append(
                dataset_idx
            )

        # Sort by the explicit temporal order, never by global dataset index.
        for scene_id, indices in scene_to_indices.items():
            indices.sort(
                key=lambda idx: self._order_in_scene(
                    self.data_infos[idx]
                )
            )

        sequence_indices = [
            None for _ in range(len(self.data_infos))
        ]
        sequence_valid_masks = np.zeros(
            (len(self.data_infos), self.frame_nums),
            dtype=bool,
        )
        sequence_frame_offsets = np.tile(
            np.arange(
                -(self.frame_nums - 1),
                1,
                dtype=np.int64,
            )[None, :],
            (len(self.data_infos), 1),
        )

        for scene_id, scene_indices in scene_to_indices.items():
            if not scene_indices:
                continue

            first_idx = scene_indices[0]

            for scene_pos, current_idx in enumerate(scene_indices):
                indices = []
                valid_mask = []

                start_pos = scene_pos - self.frame_nums + 1

                for relative_pos in range(
                    start_pos,
                    scene_pos + 1,
                ):
                    if relative_pos < 0:
                        # Placeholder only. valid_mask=False prevents future
                        # fusion from treating this as genuine history.
                        indices.append(first_idx)
                        valid_mask.append(False)
                    else:
                        indices.append(
                            scene_indices[relative_pos]
                        )
                        valid_mask.append(True)

                if len(indices) != self.frame_nums:
                    raise AssertionError(
                        "Internal sequence length mismatch."
                    )

                # Current frame must always remain the original dataset sample.
                if indices[-1] != current_idx:
                    raise AssertionError(
                        "Current-frame sequence index mismatch: "
                        f"scene={scene_id}, current={current_idx}, "
                        f"sequence={indices}"
                    )

                # Extra hard check: no history slot may cross scenes.
                for hist_idx, is_valid in zip(
                    indices,
                    valid_mask,
                ):
                    if (
                        is_valid
                        and self._scene_id(
                            self.data_infos[hist_idx]
                        ) != scene_id
                    ):
                        raise AssertionError(
                            "Temporal sequence crossed scene boundary."
                        )

                sequence_indices[current_idx] = indices
                sequence_valid_masks[current_idx] = np.asarray(
                    valid_mask,
                    dtype=bool,
                )

        if any(x is None for x in sequence_indices):
            missing = [
                i for i, x in enumerate(sequence_indices)
                if x is None
            ]
            raise RuntimeError(
                "Failed to build temporal sequence for dataset indices: "
                f"{missing[:20]}"
            )

        return (
            sequence_indices,
            sequence_valid_masks,
            sequence_frame_offsets,
        )

    def get_temporal_info(self, index):
        """
        Return sequence bookkeeping without loading Radar/GT tensors.
        Useful for regression/debug scripts.
        """
        sequence = self.sequence_indices[index]
        current_idx = sequence[-1]

        infos = [
            self.data_infos[i]
            for i in sequence
        ]

        return {
            "dataset_index": int(index),
            "current_index": int(current_idx),
            "sequence_indices": [
                int(i) for i in sequence
            ],
            "scene_id": self._scene_id(
                self.data_infos[current_idx]
            ),
            "orders_in_scene": [
                self._order_in_scene(info)
                for info in infos
            ],
            "frame_offsets": (
                self.sequence_frame_offsets[index]
                .astype(np.int64)
                .copy()
            ),
            "valid_mask": (
                self.sequence_valid_masks[index]
                .astype(bool)
                .copy()
            ),
            "ego_poses": np.stack(
                [
                    np.asarray(
                        info["ego_pose"],
                        dtype=np.float64,
                    )
                    for info in infos
                ],
                axis=0,
            ),
            "timestamps": [
                info.get("timestamp_sec", None)
                for info in infos
            ],
            "radar_paths": [
                info.get(
                    "radar_path",
                    info.get("sparse_radar_path", ""),
                )
                for info in infos
            ],
            "gt_path": self.data_infos[current_idx].get(
                "gt_path",
                self.data_infos[current_idx].get(
                    "occ_path",
                    "",
                ),
            ),
        }

    def get_data_info(self, index):
        """
        Current-only regression path.

        Build/use T-frame sequence bookkeeping, but pass the current frame into
        the exact parent RadarOcc get_data_info() implementation.
        """
        sequence = self.sequence_indices[index]
        current_idx = sequence[-1]

        input_dict = super().get_data_info(current_idx)
        temporal_info = self.get_temporal_info(index)

        # These fields are intentionally additional metadata only. The
        # existing pipeline/model consumes the same current-frame keys as
        # before, so checkpoint inference remains unchanged.
        input_dict["temporal_indices"] = temporal_info[
            "sequence_indices"
        ]
        input_dict["frame_offsets"] = temporal_info[
            "frame_offsets"
        ]
        input_dict["temporal_valid_mask"] = temporal_info[
            "valid_mask"
        ]
        input_dict["temporal_ego_poses"] = temporal_info[
            "ego_poses"
        ]
        input_dict["temporal_timestamps"] = temporal_info[
            "timestamps"
        ]

        return input_dict
