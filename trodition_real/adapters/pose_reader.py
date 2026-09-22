from __future__ import annotations

import re
from pathlib import Path

import numpy as np


class RadarOccPoseReader:
    """Resolve RadarOcc ``lidar_ego_pose{i}.npy`` by LiDAR token.

    Pose indexing deliberately follows ``lidar_token`` rather than the RPC
    filename.  For example, token ``3_00000`` uses pose 0 even when its aligned
    radar file is ``rpc_00031.npy``.
    """

    _TOKEN_PATTERN = re.compile(r"(\d+)$")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Pose root not found: {self.root}")
        self._scene_dirs: dict[str, Path] = {}

    @classmethod
    def frame_index(cls, lidar_token: str) -> int:
        match = cls._TOKEN_PATTERN.search(str(lidar_token))
        if match is None:
            raise ValueError(
                f"Cannot extract a pose index from lidar_token={lidar_token!r}."
            )
        return int(match.group(1))

    def pose_path(self, scene: str, frame_index: int) -> Path:
        pose_dir = self._pose_dir(str(scene))
        names = (
            f"lidar_ego_pose{frame_index}.npy",
            f"lidar_ego_pose{frame_index:05d}.npy",
            f"lidar_ego_pose{frame_index:06d}.npy",
        )
        for name in names:
            path = pose_dir / name
            if path.is_file():
                return path.resolve()
        raise FileNotFoundError(
            f"Pose {frame_index} for scene {scene} not found under {pose_dir}."
        )

    def read(self, scene: str, lidar_token: str) -> tuple[np.ndarray, Path]:
        return self.read_index(scene, self.frame_index(lidar_token))

    def read_index(self, scene: str, frame_index: int) -> tuple[np.ndarray, Path]:
        path = self.pose_path(scene, frame_index)
        pose = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        self._validate_pose(pose, path)
        return pose, path

    def try_read_index(
        self, scene: str, frame_index: int
    ) -> tuple[np.ndarray, Path] | None:
        if frame_index < 0:
            return None
        try:
            return self.read_index(scene, frame_index)
        except FileNotFoundError:
            return None

    def _pose_dir(self, scene: str) -> Path:
        cached = self._scene_dirs.get(scene)
        if cached is not None:
            return cached

        candidates = [
            self.root,
            self.root / "pose",
            self.root / scene / "pose",
            self.root / "train" / scene / "pose",
            self.root / "val" / scene / "pose",
            self.root / "test" / scene / "pose",
        ]
        matches = [
            path.resolve()
            for path in candidates
            if path.is_dir()
            and (
                (path / "lidar_ego_pose0.npy").is_file()
                or any(path.glob("lidar_ego_pose*.npy"))
            )
        ]
        matches = list(dict.fromkeys(matches))
        if not matches:
            raise FileNotFoundError(
                f"Cannot find a pose directory for scene {scene} below "
                f"{self.root}. Expected e.g. "
                f"{self.root}/train/{scene}/pose/lidar_ego_pose0.npy."
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous pose directories for scene {scene}: {matches}. "
                "Pass a more specific --pose-root."
            )
        self._scene_dirs[scene] = matches[0]
        return matches[0]

    @staticmethod
    def _validate_pose(pose: np.ndarray, path: Path) -> None:
        if pose.shape != (4, 4):
            raise ValueError(f"Pose must be 4x4, got {pose.shape}: {path}")
        if not np.all(np.isfinite(pose)):
            raise ValueError(f"Pose contains non-finite values: {path}")
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
            raise ValueError(f"Invalid homogeneous pose last row: {path}")
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
            raise ValueError(f"Pose rotation is not orthonormal: {path}")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
            raise ValueError(f"Pose rotation determinant is not +1: {path}")
