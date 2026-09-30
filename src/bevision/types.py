"""Lightweight value types shared across the perception pipeline.

These are plain containers, deliberately free of framework types: nothing in
:mod:`bevision.types` imports torch, OpenMMLab or the nuScenes devkit, so the
core stays testable without the perception stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class Box3D:
    """A 3D box in the global frame, in nuScenes convention."""

    center: FloatArray
    """Centre position ``(x, y, z)`` in the global frame."""

    size: tuple[float, float, float]
    """``(width, length, height)`` in metres."""

    yaw: float
    """Heading about +z, in the global frame, radians."""

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float).reshape(3)
        object.__setattr__(self, "center", center)

    @property
    def center_xy(self) -> FloatArray:
        """Centre projected onto the ground plane."""
        return self.center[:2]


@dataclass(frozen=True)
class RawDetection3D:
    """A 3D detection straight from the LiDAR detector, in the sensor frame.

    ``label`` is the model's own label id; the pipeline resolves it to a class name using
    the model's declared ordering, so a reordered class list cannot silently mislabel.
    """

    label: int
    score: float
    center: FloatArray
    """Centre in the **sensor** frame, shape ``(3,)``."""

    size: tuple[float, float, float]
    """``(width, length, height)`` in metres."""

    yaw: float
    """Heading in the sensor frame, radians."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float).reshape(3))


@dataclass
class Detection3D:
    """One fused 3D detection for a single frame."""

    label: int
    """Model label id; decode with the model's own class ordering."""

    class_name: str
    """nuScenes class name for this detection, already resolved."""

    score: float
    """Confidence after fusion."""

    center_global: FloatArray
    """Centre position in the global frame, shape ``(3,)``."""

    size: tuple[float, float, float]
    """``(width, length, height)`` in metres."""

    yaw_global: float
    """Heading in the global frame, radians."""

    bbox_2d: tuple[float, float, float, float] | None = None
    """Projected image bbox of the best confirming camera, if any."""

    camera_confirmed: bool = False
    """Whether a camera detection agreed on both position and class."""

    def __post_init__(self) -> None:
        self.center_global = np.asarray(self.center_global, dtype=float).reshape(3)

    @property
    def center_xyz(self) -> FloatArray:
        """Centre as used by association, shape ``(3,)``."""
        return self.center_global


@dataclass
class TrackOutput:
    """A track's emitted box for one frame, in submission-ready form."""

    track_id: int
    class_name: str
    score: float
    box: Box3D
    velocity: tuple[float, float] = (0.0, 0.0)
    """Velocity in the global frame, m/s, shape ``(vx, vy)``."""


@dataclass
class FrameStats:
    """Per-frame pipeline counters, collected for diagnostics and plots."""

    frame_index: int
    lidar_detections: int = 0
    camera_detections: int = 0
    fused_detections: int = 0
    tracks_active: int = 0
    tracks_spawned: int = 0
    matches: int = 0
    extras: dict[str, float] = field(default_factory=dict)
