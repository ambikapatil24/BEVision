"""Detector interfaces and concrete adapters.

The adapters import their third-party dependencies (Ultralytics, mmdetection3d) inside
``__init__`` rather than at module scope, so importing this package never requires them.
"""

from __future__ import annotations

from bevision.detection.base import (
    CameraDetector,
    CameraFrame,
    Frame,
    FrameSource,
    LidarDetector,
)
from bevision.detection.camera import UltralyticsCameraDetector
from bevision.detection.lidar import MMDet3DLidarDetector

__all__ = [
    "CameraDetector",
    "CameraFrame",
    "Frame",
    "FrameSource",
    "LidarDetector",
    "MMDet3DLidarDetector",
    "UltralyticsCameraDetector",
]
