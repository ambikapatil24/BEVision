"""Tests for the mmdet3d box-ordering conversion.

The swap is the one part of the LiDAR adapter that can be wrong *silently*: mis-ordered sizes
leave mAP, AMOTA and every box centre completely untouched and only inflate the evaluator's
scale error. That is exactly what happened, and it cost 0.018-0.023 NDS on every run. So the
conversion is split out of the model wrapper and pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from bevision.detection.lidar import raw_detections_from_tensors
from bevision.geometry import box_corners


def test_size_is_reordered_from_length_first_to_width_first() -> None:
    """mmdet3d gives (dx, dy, dz) = (length, width, height); nuScenes wants (w, l, h)."""
    # a car: mmdet3d order is length 4.6, width 1.9, height 1.6
    boxes = np.array([[10.0, 2.0, 0.5, 4.6, 1.9, 1.6, 0.3]])

    detection = raw_detections_from_tensors(boxes, [0.9], [0])[0]

    assert detection.size == pytest.approx((1.9, 4.6, 1.6))


def test_height_is_not_moved() -> None:
    boxes = np.array([[0.0, 0.0, 0.0, 4.6, 1.9, 1.6, 0.0]])
    detection = raw_detections_from_tensors(boxes, [0.5], [0])[0]
    assert detection.size[2] == pytest.approx(1.6)


def test_centre_xy_and_yaw_are_passed_through_unchanged() -> None:
    boxes = np.array([[10.0, -2.0, 0.5, 4.6, 1.9, 1.6, 0.7]])

    detection = raw_detections_from_tensors(boxes, [0.9], [1])[0]

    assert detection.center[:2] == pytest.approx([10.0, -2.0])  # xy untouched
    assert detection.yaw == pytest.approx(0.7)
    assert detection.label == 1
    assert detection.score == pytest.approx(0.9)


def test_a_vehicle_ends_up_longer_than_it_is_wide() -> None:
    """The regression this test exists for.

    Every nuScenes vehicle has length > width. Before the fix a car came out as 5.5 m wide and
    1.4 m long; after, the ratio is the right way round.
    """
    boxes = np.array([[0.0, 0.0, 0.0, 5.5, 1.4, 1.6, 0.0]])
    width, length, _ = raw_detections_from_tensors(boxes, [0.9], [0])[0].size

    assert length > width, "car came out wider than it is long -- the swap is missing"


def test_multiple_boxes_are_converted_in_order() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 0.0, 4.5, 1.8, 1.5, 0.0],
            [5.0, 1.0, 0.0, 10.2, 2.9, 3.6, 0.1],
        ]
    )
    detections = raw_detections_from_tensors(boxes, [0.9, 0.4], [0, 1])

    assert [d.label for d in detections] == [0, 1]
    assert detections[0].size == pytest.approx((1.8, 4.5, 1.5))
    assert detections[1].size == pytest.approx((2.9, 10.2, 3.6))


def test_9_column_tensor_is_accepted_and_velocity_dropped() -> None:
    """mmdet3d emits (x,y,z,dx,dy,dz,yaw,vx,vy) in some versions.

    A real PointPillars run produced a (457, 9) tensor, and assuming 7 columns raised
    ValueError: cannot reshape array of size 4113 into shape (7) -- 4113 = 457 x 9.
    """
    nine = np.array([[10.0, 2.0, 0.5, 4.6, 1.9, 1.6, 0.3, 1.5, -0.2]])

    detection = raw_detections_from_tensors(nine, [0.9], [0])[0]

    assert detection.size == pytest.approx((1.9, 4.6, 1.6))
    assert detection.center[:2] == pytest.approx([10.0, 2.0])
    assert detection.yaw == pytest.approx(0.3)


def test_7_and_9_column_tensors_agree_on_the_shared_columns() -> None:
    seven = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.7]])
    nine = np.hstack([seven, np.array([[9.0, 9.0]])])

    from_seven = raw_detections_from_tensors(seven, [0.5], [0])[0]
    from_nine = raw_detections_from_tensors(nine, [0.5], [0])[0]

    assert from_seven.center == pytest.approx(from_nine.center)
    assert from_seven.size == pytest.approx(from_nine.size)
    assert from_seven.yaw == pytest.approx(from_nine.yaw)


def test_z_is_lifted_to_the_box_centre() -> None:
    """mmdet3d anchors LiDAR boxes at their bottom face, so tensor[2] is the BOTTOM.

    Against a real PointPillars run, comparing my corners with mmdet3d's showed a
    disagreement of exactly h/2 in z for all eight boxes checked (0.809 vs h/2 = 0.810;
    0.876 vs 0.875). Reading z as the centre shifted every projected box ~67 px down a
    900 px image and produced zero camera confirmations.
    """
    boxes = np.array([[10.0, 2.0, 0.5, 4.6, 1.9, 1.6, 0.3]])

    detection = raw_detections_from_tensors(boxes, [0.9], [0])[0]

    assert detection.center[2] == pytest.approx(0.5 + 1.6 / 2.0)  # 1.3
    assert detection.center[:2] == pytest.approx([10.0, 2.0])  # xy untouched


def test_corners_span_the_box_bottom_to_top() -> None:
    """The invariant that makes the lift correct: z in [bottom, bottom + height]."""
    height, bottom = 1.6, 0.5
    boxes = np.array([[0.0, 0.0, bottom, 4.6, 1.9, height, 0.0]])

    detection = raw_detections_from_tensors(boxes, [0.9], [0])[0]
    corners = box_corners(detection.center, detection.size, detection.yaw)

    assert corners[:, 2].min() == pytest.approx(bottom)
    assert corners[:, 2].max() == pytest.approx(bottom + height)


def test_centre_height_scales_with_box_height() -> None:
    """The lift is h/2, so a taller box is lifted further -- the relation the evidence showed."""
    tall = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0]])
    short = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0]])

    assert raw_detections_from_tensors(tall, [1.0], [0])[0].center[2] == pytest.approx(1.0)
    assert raw_detections_from_tensors(short, [1.0], [0])[0].center[2] == pytest.approx(0.25)


def test_many_boxes_matching_the_real_smoke_run_shape() -> None:
    """457 boxes x 9 columns is the shape the failing Kaggle run actually produced."""
    boxes = np.zeros((457, 9))
    boxes[:, 3:6] = (4.0, 2.0, 1.5)
    detections = raw_detections_from_tensors(boxes, np.full(457, 0.5), np.zeros(457))
    assert len(detections) == 457


def test_empty_input_returns_no_detections() -> None:
    assert raw_detections_from_tensors(np.empty((0, 7)), np.empty((0,)), np.empty((0,))) == []
    assert raw_detections_from_tensors(np.empty((0, 9)), np.empty((0,)), np.empty((0,))) == []


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        raw_detections_from_tensors(np.zeros((2, 7)), [0.5], [0, 1])


def test_wrong_column_count_is_rejected_with_the_shape_named() -> None:
    with pytest.raises(ValueError, match=r"got shape \(2, 6\)"):
        raw_detections_from_tensors(np.zeros((2, 6)), [0.5, 0.5], [0, 1])
    with pytest.raises(ValueError, match=r"got shape \(6,\)"):
        raw_detections_from_tensors(np.zeros(6), [0.5], [0])
