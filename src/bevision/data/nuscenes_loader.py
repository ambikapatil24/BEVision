"""nuScenes-backed :class:`~bevision.detection.base.FrameSource`.

The only module that knows the nuScenes devkit exists. Everything downstream consumes
:class:`~bevision.detection.base.Frame`, so the pipeline can be tested without the devkit or
its multi-gigabyte dataset.

> **Not verified in CI.** This adapter needs the nuScenes devkit and the mini split, so it is
> exercised only on a machine with the dataset present. The transforms it produces are
> quaternion rotations converted by :func:`bevision.geometry.quaternion_to_matrix`, which is
> unit-tested independently.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from bevision.config import SURROUND_CAMERAS
from bevision.detection.base import CameraFrame, Frame
from bevision.geometry import quaternion_to_matrix

FloatArray = NDArray[np.floating]


def _rotation(record: dict) -> list[list[float]]:
    """Rotation matrix from a nuScenes record's ``(w, x, y, z)`` quaternion."""
    return quaternion_to_matrix(record["rotation"]).tolist()


def load_lidar_points(path: str, columns: int = 5) -> FloatArray:
    """Read a nuScenes LiDAR sweep (``.pcd.bin``) and return its ``(N, 3)`` xyz points.

    nuScenes stores each point as five float32 values: x, y, z, intensity, ring index. Only
    xyz is kept, since that is all the geometry and the renderers use.

    This is a free function rather than a method because the point cloud is only needed by
    rendering and diagnostics — the detection pipeline hands the *path* to the LiDAR detector
    and never reads the points itself.
    """
    flat = np.fromfile(path, dtype=np.float32)
    if flat.size % columns != 0:
        raise ValueError(f"{path}: {flat.size} floats is not a multiple of {columns} columns")
    points = flat.reshape(-1, columns)
    return np.ascontiguousarray(points[:, :3])


class NuScenesFrameSource:
    """Yields frames for one nuScenes split, scene by scene.

    Consecutive samples of a scene are yielded contiguously, which the pipeline relies on to
    reset the tracker at scene boundaries.

    Args:
        dataroot: Directory containing ``v1.0-mini/``, ``samples/`` and ``sweeps/``.
        version: nuScenes version string, e.g. ``v1.0-mini``.
        split: Split name understood by ``create_splits_scenes``, e.g. ``mini_val``.
        camera_names: Which cameras to load per frame.
        verbose: Passed through to the devkit loader.
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        split: str = "mini_val",
        camera_names: Sequence[str] = SURROUND_CAMERAS,
        verbose: bool = False,
    ) -> None:
        from bevision._deps import require  # noqa: PLC0415

        require("nuscenes", "loading nuScenes data")
        from nuscenes.nuscenes import NuScenes  # lazy: heavy, and optional

        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
        self.dataroot = dataroot
        self.split = split
        self.camera_names = tuple(camera_names)

    # -- helpers ------------------------------------------------------------
    def _sample_tokens(self) -> list[str]:
        """Every sample token in the split, in scene order."""
        from nuscenes.utils.splits import create_splits_scenes

        wanted = set(create_splits_scenes()[self.split])
        tokens: list[str] = []
        for scene in self.nusc.scene:
            if scene["name"] not in wanted:
                continue
            token = scene["first_sample_token"]
            while token:
                tokens.append(token)
                token = self.nusc.get("sample", token)["next"]
        return tokens

    def _camera_frame(self, sample: dict, name: str) -> CameraFrame:
        import os

        sample_data = self.nusc.get("sample_data", sample["data"][name])
        calibration = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        return CameraFrame(
            name=name,
            image_path=os.path.join(self.dataroot, sample_data["filename"]),
            rotation=_rotation(calibration),
            translation=calibration["translation"],
            intrinsic=calibration["camera_intrinsic"],
        )

    def _frame(self, sample_token: str) -> Frame:
        import os

        sample = self.nusc.get("sample", sample_token)
        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_calibration = self.nusc.get(
            "calibrated_sensor", lidar_data["calibrated_sensor_token"]
        )
        ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])

        return Frame(
            sample_token=sample_token,
            scene_token=sample["scene_token"],
            lidar_path=os.path.join(self.dataroot, lidar_data["filename"]),
            lidar_rotation=_rotation(lidar_calibration),
            lidar_translation=lidar_calibration["translation"],
            ego_rotation=_rotation(ego_pose),
            ego_translation=ego_pose["translation"],
            cameras={
                name: self._camera_frame(sample, name)
                for name in self.camera_names
                if name in sample["data"]
            },
        )

    # -- protocol -----------------------------------------------------------
    def frames(self) -> Iterator[Frame]:
        """Yield every frame of the split in order."""
        for token in self._sample_tokens():
            yield self._frame(token)
