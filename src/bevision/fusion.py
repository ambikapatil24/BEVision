"""LiDAR-anchored sensor fusion.

LiDAR decides *where* an object is and cameras decide *what* it is: a camera can confirm a 3D box
and raise its confidence, and can never introduce, remove or move one. Depth comes from the point
cloud, which is the only sensor here that measures it directly.

Two matching strategies are provided:

* :func:`best_camera_confirmation` -- greedy, one LiDAR box against every camera, keeping the
  highest-confidence confirming camera. A box usually appears in two or three overlapping
  cameras, so several confirmations are typically available.
* :func:`iou_matrix_with_class_gate` / :func:`hungarian_match` -- the global optimum over all
  pairs. The class constraint is folded into the cost matrix rather than applied to the result,
  so the assignment cannot spend a match on a pair that would then be discarded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bevision.geometry import bbox_iou_matrix

FloatArray = NDArray[np.floating]

#: Cost marking a class-mismatched pair as unusable inside a maximisation matrix.
FORBIDDEN_IOU: float = -1.0


@dataclass(frozen=True)
class CameraDetections:
    """The 2D detections produced by one camera in one frame."""

    name: str
    boxes: FloatArray
    """``(N, 4)`` array of ``(x_min, y_min, x_max, y_max)`` pixel boxes."""

    confidences: FloatArray
    """``(N,)`` array of detection confidences."""

    class_names: tuple[str, ...]
    """nuScenes class name per box, already mapped from COCO."""

    def __post_init__(self) -> None:
        boxes = np.asarray(self.boxes, dtype=float).reshape(-1, 4)
        confidences = np.asarray(self.confidences, dtype=float).reshape(-1)
        if boxes.shape[0] != confidences.shape[0] or boxes.shape[0] != len(self.class_names):
            raise ValueError("boxes, confidences and class_names must have the same length")
        object.__setattr__(self, "boxes", boxes)
        object.__setattr__(self, "confidences", confidences)

    @property
    def is_empty(self) -> bool:
        return self.boxes.shape[0] == 0

    @classmethod
    def empty(cls, name: str) -> CameraDetections:
        """A camera that produced no usable detections this frame."""
        return cls(name, np.empty((0, 4)), np.empty((0,)), ())


@dataclass(frozen=True)
class CameraConfirmation:
    """The camera evidence supporting one LiDAR detection."""

    camera_name: str
    confidence: float
    bbox_2d: tuple[float, float, float, float]
    box_index: int


def fuse_confidence(lidar_score: float, camera_confidence: float, weight: float = 0.5) -> float:
    """Blend a LiDAR score with a confirming camera confidence.

    The result is a weighted mean of the two inputs, so it always lies between them: a confirmed
    detection is pulled towards the camera's actual confidence. Agreement between the sensors
    raises the score and disagreement lowers it. ``weight`` is the camera's share and must be in
    [0, 1].
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    return (1.0 - weight) * float(lidar_score) + weight * float(camera_confidence)


def best_camera_confirmation(
    camera_detections: Sequence[CameraDetections],
    projected_bboxes: Sequence[tuple[float, float, float, float] | None],
    detection_class_name: str,
    iou_threshold: float,
) -> CameraConfirmation | None:
    """Find the strongest camera agreement for one LiDAR detection.

    Args:
        camera_detections: One entry per camera.
        projected_bboxes: The detection's 2D bbox in each camera's image, aligned
            with ``camera_detections``. ``None`` means the box is not visible in that
            camera (behind it, or off-image).
        detection_class_name: The LiDAR detection's nuScenes class name.
        iou_threshold: Minimum IoU for a camera box to count as agreement.

    Returns:
        The highest-confidence class-consistent confirmation, or ``None`` if no
        camera agrees.
    """
    if len(camera_detections) != len(projected_bboxes):
        raise ValueError("camera_detections and projected_bboxes must be the same length")

    best: CameraConfirmation | None = None
    for camera, bbox in zip(camera_detections, projected_bboxes, strict=True):
        if bbox is None or camera.is_empty:
            continue

        same_class = np.fromiter(
            (name == detection_class_name for name in camera.class_names),
            dtype=bool,
            count=len(camera.class_names),
        )
        if not same_class.any():
            continue

        candidates = np.flatnonzero(same_class)
        overlaps = bbox_iou_matrix([bbox], camera.boxes[candidates])[0]
        passing = candidates[overlaps > iou_threshold]
        if passing.size == 0:
            continue

        best_local = int(passing[np.argmax(camera.confidences[passing])])
        confidence = float(camera.confidences[best_local])
        if best is None or confidence > best.confidence:
            best = CameraConfirmation(
                camera_name=camera.name,
                confidence=confidence,
                bbox_2d=tuple(float(v) for v in bbox),  # type: ignore[arg-type]
                box_index=best_local,
            )

    return best


def iou_matrix_with_class_gate(
    detection_boxes: ArrayLike,
    detection_class_names: Sequence[str],
    camera: CameraDetections,
) -> FloatArray:
    """IoU matrix between LiDAR-projected boxes and one camera's boxes.

    Class-mismatched pairs are set to :data:`FORBIDDEN_IOU` so a maximising
    assignment can never select them.
    """
    matrix = bbox_iou_matrix(detection_boxes, camera.boxes)
    if matrix.size == 0:
        return matrix

    detection_names = np.asarray(list(detection_class_names))[:, None]
    camera_names = np.asarray(list(camera.class_names))[None, :]
    return np.where(detection_names == camera_names, matrix, FORBIDDEN_IOU)


def hungarian_match(
    iou_matrix: ArrayLike,
    min_iou: float,
) -> list[tuple[int, int]]:
    """Optimal one-to-one pairing maximising total IoU.

    Pairs whose IoU does not exceed ``min_iou`` are dropped, including any pair that
    the assignment had to make from forbidden entries.
    """
    from scipy.optimize import linear_sum_assignment

    matrix = np.asarray(iou_matrix, dtype=float)
    if matrix.size == 0:
        return []

    rows, cols = linear_sum_assignment(-matrix)
    return [
        (int(row), int(col))
        for row, col in zip(rows, cols, strict=True)
        if matrix[row, col] > min_iou
    ]
