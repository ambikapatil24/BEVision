"""LiDAR-anchored sensor fusion: the invariants that keep depth honest.

Three properties are load-bearing, and the sections below are ordered around them:

* **LiDAR anchors, cameras confirm.** A camera raises a 3D box's score and can never create, move
  or delete one, so depth always comes from the point cloud. A box no camera agrees with keeps its
  raw LiDAR score, and a camera that cannot see a box cannot reject it.
* **The class constraint lives inside the assignment objective.** Putting it in the cost matrix
  rather than filtering the result afterwards stops the optimizer spending a match on a pair that
  would then be discarded.
* **Class disagreement blocks confirmation**, so a car cannot be "confirmed" by a pedestrian box
  that happens to overlap it.
"""

from __future__ import annotations

import numpy as np
import pytest

from bevision.fusion import (
    FORBIDDEN_IOU,
    CameraDetections,
    best_camera_confirmation,
    fuse_confidence,
    hungarian_match,
    iou_matrix_with_class_gate,
)


def _camera(name: str, boxes, confidences, classes) -> CameraDetections:
    return CameraDetections(
        name=name,
        boxes=np.asarray(boxes, dtype=float).reshape(-1, 4),
        confidences=np.asarray(confidences, dtype=float),
        class_names=tuple(classes),
    )


# ---------------------------------------------------------------------------
# Why the class gate goes inside the objective
# ---------------------------------------------------------------------------
def test_class_gate_stops_the_optimizer_wasting_a_match() -> None:
    """The reason the constraint is folded into the matrix instead of applied afterwards.

    Box A is a perfect geometric match for YOLO box 1 but the classes differ; box B overlaps
    YOLO box 0 only loosely but matches its class. A greedy IoU-first pass would take A-1 and
    leave B unmatched. Constraining the objective is what makes the optimizer skip the pair it
    would have to throw away.
    """
    detection_boxes = [(0.0, 0.0, 10.0, 10.0), (12.0, 0.0, 22.0, 10.0)]
    detection_classes = ["car", "pedestrian"]
    camera = _camera(
        "CAM_FRONT",
        [[0.0, 0.0, 10.0, 10.0], [14.0, 0.0, 24.0, 10.0]],
        [0.9, 0.9],
        ["pedestrian", "pedestrian"],
    )

    matrix = iou_matrix_with_class_gate(detection_boxes, detection_classes, camera)
    matches = hungarian_match(matrix, min_iou=0.1)

    assert matrix[0, 0] == FORBIDDEN_IOU  # perfect IoU, wrong class
    assert 0 not in [row for row, _ in matches]  # the car gets no match at all
    assert matches == [(1, 1)]


def test_class_gate_sets_mismatched_pairs_forbidden() -> None:
    camera = _camera("CAM_FRONT", [[0.0, 0.0, 10.0, 10.0]], [0.9], ["pedestrian"])

    matrix = iou_matrix_with_class_gate([(0.0, 0.0, 10.0, 10.0)], ["car"], camera)

    assert matrix[0, 0] == FORBIDDEN_IOU


def test_class_gate_preserves_matching_pairs() -> None:
    camera = _camera("CAM_FRONT", [[0.0, 0.0, 10.0, 10.0]], [0.9], ["car"])

    matrix = iou_matrix_with_class_gate([(0.0, 0.0, 10.0, 10.0)], ["car"], camera)

    assert matrix[0, 0] == pytest.approx(1.0)


def test_hungarian_match_drops_pairs_below_the_iou_floor() -> None:
    matrix = np.array([[0.9, 0.0], [0.0, 0.05]])
    assert hungarian_match(matrix, min_iou=0.3) == [(0, 0)]


def test_hungarian_match_on_empty_matrix() -> None:
    assert hungarian_match(np.empty((0, 0)), min_iou=0.3) == []


def test_hungarian_match_is_one_to_one() -> None:
    matrix = np.array([[0.9, 0.8], [0.85, 0.1]])
    matches = hungarian_match(matrix, min_iou=0.3)
    assert len({row for row, _ in matches}) == len(matches)
    assert len({col for _, col in matches}) == len(matches)


# ---------------------------------------------------------------------------
# Best-camera confirmation
# ---------------------------------------------------------------------------
def test_best_confirmation_requires_class_agreement() -> None:
    """A pedestrian box overlapping a car box must not confirm it.

    Position agreement alone is not evidence: the two sensors would be saying the object is in
    the same place and is a different thing.
    """
    detection_bbox = (0.0, 0.0, 100.0, 100.0)
    cameras = [_camera("CAM_FRONT", [[5.0, 5.0, 95.0, 95.0]], [0.95], ["pedestrian"])]

    assert best_camera_confirmation(cameras, [detection_bbox], "car", 0.3) is None


def test_best_confirmation_picks_the_most_confident_agreeing_camera() -> None:
    detection_bbox = (0.0, 0.0, 100.0, 100.0)
    cameras = [
        _camera("CAM_FRONT", [[5.0, 5.0, 95.0, 95.0]], [0.55], ["car"]),
        _camera("CAM_FRONT_LEFT", [[5.0, 5.0, 95.0, 95.0]], [0.88], ["car"]),
    ]

    confirmation = best_camera_confirmation(cameras, [detection_bbox, detection_bbox], "car", 0.3)

    assert confirmation is not None
    assert confirmation.camera_name == "CAM_FRONT_LEFT"
    assert confirmation.confidence == pytest.approx(0.88)


def test_best_confirmation_requires_iou_above_threshold() -> None:
    detection_bbox = (0.0, 0.0, 100.0, 100.0)
    cameras = [_camera("CAM_FRONT", [[90.0, 90.0, 110.0, 110.0]], [0.95], ["car"])]

    assert best_camera_confirmation(cameras, [detection_bbox], "car", 0.3) is None


def test_best_confirmation_skips_boxes_not_visible_in_a_camera() -> None:
    detection_bbox = (0.0, 0.0, 100.0, 100.0)
    cameras = [
        _camera("CAM_BACK", [[5.0, 5.0, 95.0, 95.0]], [0.99], ["car"]),
        _camera("CAM_FRONT", [[5.0, 5.0, 95.0, 95.0]], [0.42], ["car"]),
    ]

    confirmation = best_camera_confirmation(cameras, [None, detection_bbox], "car", 0.3)

    assert confirmation is not None
    assert confirmation.camera_name == "CAM_FRONT"
    assert confirmation.confidence == pytest.approx(0.42)


def test_best_confirmation_returns_none_when_nothing_agrees() -> None:
    assert best_camera_confirmation([], [], "car", 0.3) is None


def test_best_confirmation_handles_empty_cameras() -> None:
    cameras = [CameraDetections.empty("CAM_FRONT")]
    assert best_camera_confirmation(cameras, [(0.0, 0.0, 10.0, 10.0)], "car", 0.3) is None


def test_best_confirmation_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        best_camera_confirmation([CameraDetections.empty("CAM_FRONT")], [], "car", 0.3)


def test_best_confirmation_reports_the_matched_box_index() -> None:
    detection_bbox = (0.0, 0.0, 100.0, 100.0)
    cameras = [
        _camera(
            "CAM_FRONT",
            [[500.0, 500.0, 600.0, 600.0], [5.0, 5.0, 95.0, 95.0]],
            [0.99, 0.44],
            ["car", "car"],
        )
    ]

    confirmation = best_camera_confirmation(cameras, [detection_bbox], "car", 0.3)

    assert confirmation is not None
    assert confirmation.box_index == 1


# ---------------------------------------------------------------------------
# Score blending
# ---------------------------------------------------------------------------
def test_fuse_confidence_uses_the_real_camera_confidence() -> None:
    """The blend must track the camera's actual confidence, not a constant.

    Both inputs here sit below the spawn threshold, so a blend that used a fixed value instead
    of the camera score would lift them over it and quietly change what the threshold means.
    """
    assert fuse_confidence(0.20, 0.30) < 0.30
    assert fuse_confidence(0.90, 0.95) > 0.90


def test_fuse_confidence_is_an_even_blend() -> None:
    assert fuse_confidence(0.20, 0.30) == pytest.approx(0.25)


def test_fuse_confidence_rejects_out_of_range_weight() -> None:
    with pytest.raises(ValueError, match="weight"):
        fuse_confidence(0.5, 0.5, weight=1.5)


# ---------------------------------------------------------------------------
# CameraDetections
# ---------------------------------------------------------------------------
def test_camera_detections_validates_aligned_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        CameraDetections(
            name="CAM_FRONT",
            boxes=np.zeros((2, 4)),
            confidences=np.zeros(2),
            class_names=("car",),
        )


def test_empty_camera_helper() -> None:
    camera = CameraDetections.empty("CAM_BACK")
    assert camera.is_empty
    assert camera.boxes.shape == (0, 4)
