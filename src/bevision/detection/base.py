"""Detector interfaces and the frame abstraction the pipeline consumes.

These are :class:`typing.Protocol` definitions rather than ABCs. The concrete detectors wrap
third-party models (Ultralytics, mmdetection3d) whose return types have nothing in common,
so structural typing keeps those dependencies out of the import graph entirely -- the
pipeline is testable against fakes without torch being installed.

``FrameSource`` is deliberately plain data. It is the seam between "read nuScenes" and
"run perception": the real loader is one implementation, a test fixture is another.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bevision.fusion import CameraDetections
from bevision.types import RawDetection3D

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class CameraFrame:
    """Everything needed to use one camera's image in one frame."""

    name: str
    image_path: str
    rotation: FloatArray
    """Camera-to-ego rotation, shape ``(3, 3)``."""

    translation: FloatArray
    """Camera-to-ego translation, shape ``(3,)``."""

    intrinsic: FloatArray
    """3x3 camera intrinsic matrix."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", np.asarray(self.rotation, dtype=float).reshape(3, 3))
        object.__setattr__(
            self, "translation", np.asarray(self.translation, dtype=float).reshape(3)
        )
        object.__setattr__(self, "intrinsic", np.asarray(self.intrinsic, dtype=float).reshape(3, 3))


@dataclass(frozen=True)
class Frame:
    """One synchronised sample: a LiDAR sweep, its calibration, the ego pose, and cameras."""

    sample_token: str
    scene_token: str
    lidar_path: str
    lidar_rotation: FloatArray
    """LiDAR-to-ego rotation, shape ``(3, 3)``."""

    lidar_translation: FloatArray
    """LiDAR-to-ego translation, shape ``(3,)``."""

    ego_rotation: FloatArray
    """Ego-to-global rotation, shape ``(3, 3)``."""

    ego_translation: FloatArray
    """Ego-to-global translation, shape ``(3,)``."""

    cameras: Mapping[str, CameraFrame]

    def __post_init__(self) -> None:
        for name in ("lidar_rotation", "ego_rotation"):
            object.__setattr__(
                self, name, np.asarray(getattr(self, name), dtype=float).reshape(3, 3)
            )
        for name in ("lidar_translation", "ego_translation"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float).reshape(3))


@runtime_checkable
class FrameSource(Protocol):
    """Supplies frames in temporal order, grouped by scene.

    Implementations must yield consecutive samples of one scene contiguously, because the
    pipeline resets the tracker whenever ``scene_token`` changes.
    """

    def frames(self) -> Iterable[Frame]:
        """Yield frames in order."""
        ...


@runtime_checkable
class CameraDetector(Protocol):
    """Runs 2D detection on one camera image."""

    def __call__(self, image_path: str) -> CameraDetections:
        """Return detections for the image at ``image_path``."""
        ...


@runtime_checkable
class LidarDetector(Protocol):
    """Runs 3D detection on one LiDAR sweep."""

    def __call__(self, lidar_path: str) -> list[RawDetection3D]:
        """Return detections in the LiDAR sensor frame."""
        ...


def as_array(values: ArrayLike) -> FloatArray:
    """Small helper so adapters can hand back lists without repeating ``np.asarray``."""
    return np.asarray(values, dtype=float)
