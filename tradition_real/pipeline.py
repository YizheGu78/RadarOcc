"""RPC -> pinned OctoMap 3D OcTree occupancy -> unchanged DBSCAN/42D -> RF."""
from collections import deque
from dataclasses import dataclass
import numpy as np
from tradition_real.config import Config
from tradition_real.adapters.octomap_grid import OctomapGrid, transform_points
from tradition_real.semantics.features import proposals


@dataclass
class MappingResult:
    probability: np.ndarray
    observed: np.ndarray
    occupied: np.ndarray
    free: np.ndarray
    bev_probability: np.ndarray
    candidates: list


class Pipeline:
    def __init__(self, config=None):
        self.config = config or Config()
        self.history = deque(maxlen=self.config.temporal_window)
        self.scene = None
        self.last_ordinal = None

    def reset(self):
        self.history.clear()
        self.scene = None
        self.last_ordinal = None

    def map_frame(self, rpc, pose, translation, scene, ordinal):
        rpc = np.asarray(rpc, np.float64)
        if rpc.ndim != 2 or rpc.shape[1] != 11:
            raise ValueError(f'RPC must be [N,11], got {rpc.shape}')
        if not np.all(np.isfinite(rpc)):
            raise ValueError('RPC contains nonfinite measurements')
        pose = np.asarray(pose, np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError('Expected finite 4x4 pose')
        if not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(pose[:3,:3].T @ pose[:3,:3], np.eye(3), atol=.002) or not np.isclose(np.linalg.det(pose[:3,:3]), 1, atol=.002):
            raise ValueError('Pose must be a rigid LiDAR-to-world transform')
        translation = np.asarray(translation, np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError('Expected finite radar-to-LiDAR translation')
        if scene != self.scene:
            self.reset()
            self.scene = scene
        if self.last_ordinal is not None and ordinal <= self.last_ordinal:
            raise ValueError('Frames must be unique and strictly chronological within a scene')
        if self.last_ordinal is not None and ordinal != self.last_ordinal + 1:
            self.history.clear()
        self.last_ordinal = ordinal
        self.history.append((rpc.copy(), pose.copy(), translation.copy(), ordinal))
        cfg = self.config
        mapping = OctomapGrid(cfg)
        all_points, all_returns, all_ages = [], [], []
        # Replay each source observation ONCE in the current reference frame.
        # Rebuild from prior every call; never re-add a window to a persistent map.
        for source_rpc, source_pose, source_translation, source_ordinal in self.history:
            relative = np.linalg.inv(pose) @ source_pose
            points = transform_points(source_rpc[:, :3] + source_translation, relative)
            origin = transform_points(source_translation[None], relative)[0]
            mapping.insert(points, origin)
            all_points.append(points)
            all_returns.append(source_rpc)
            all_ages.append(np.full(len(points), ordinal - source_ordinal, np.int32))
        probability, observed, occupied, free, bev_probability = mapping.export()
        candidates = proposals(probability, occupied, np.concatenate(all_points),
                               np.concatenate(all_returns), np.concatenate(all_ages), cfg)
        return MappingResult(probability, observed, occupied, free,
                             bev_probability, candidates)

    def predict(self, mapping, classifier):
        native = np.full(self.config.shape_xyz, 255, np.uint8)
        native[mapping.free] = 0
        native[mapping.occupied] = 1
        if mapping.candidates:
            features = np.stack([c['features'] for c in mapping.candidates])
            probabilities = classifier.foreground_probabilities(features)
            for candidate, probability in zip(mapping.candidates, probabilities):
                if probability >= self.config.foreground_threshold:
                    native[tuple(candidate['voxels'].T)] = 2
        dense = native.copy()
        dense[native == 255] = 0 if self.config.unknown_export == 'free' else 1
        return dense, native
