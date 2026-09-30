"""Tests for the vectorised geometry core.

The projection tests are the load-bearing ones: they pin the exact pixel
bounding box for a synthetic box, and verify that boxes at or behind the camera
plane are rejected rather than mirrored through the focal plane.
"""

from __future__ import annotations

import numpy as np
import pytest

from bevision.geometry import (
    bbox_inside_image,
    bbox_iou,
    bbox_iou_matrix,
    box_corners,
    pairwise_distance_matrix,
    project_box_to_image,
    quaternion_to_matrix,
    rigid_inverse,
    sensor_to_global,
    to_global_yaw,
    transform_points,
    view_points,
    wrap_angle,
    yaw_matrix,
)

IDENTITY_3 = np.eye(3)
ZERO_3 = np.zeros(3)

# fx = fy = 1000, principal point (800, 450) -- the centre of a 1600x900 image.
INTRINSIC = np.array([[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------
def test_quaternion_identity_is_identity_matrix() -> None:
    assert np.allclose(quaternion_to_matrix([1.0, 0.0, 0.0, 0.0]), np.eye(3))


def test_quaternion_sixty_degree_yaw_about_z() -> None:
    half = np.pi / 12.0  # half of a 30-degree angle
    q = [np.cos(half), 0.0, 0.0, np.sin(half)]
    expected = np.array(
        [
            [np.cos(np.pi / 6), -np.sin(np.pi / 6), 0.0],
            [np.sin(np.pi / 6), np.cos(np.pi / 6), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(quaternion_to_matrix(q), expected, atol=1e-12)


def test_quaternion_is_normalised() -> None:
    assert np.allclose(quaternion_to_matrix([2.0, 0.0, 0.0, 0.0]), np.eye(3))


def test_quaternion_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="wxyz"):
        quaternion_to_matrix([1.0, 0.0, 0.0])


def test_yaw_matrix_matches_quaternion_path() -> None:
    yaw = 0.7
    half = yaw / 2.0
    q = [np.cos(half), 0.0, 0.0, np.sin(half)]
    assert np.allclose(yaw_matrix(yaw), quaternion_to_matrix(q), atol=1e-12)


def test_rigid_inverse_round_trip() -> None:
    rotation = yaw_matrix(0.9)
    translation = np.array([3.0, -4.0, 1.5])
    rotation_inv, translation_inv = rigid_inverse(rotation, translation)

    points = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 0.0]])
    round_tripped = transform_points(
        transform_points(points, rotation, translation), rotation_inv, translation_inv
    )
    assert np.allclose(round_tripped, points, atol=1e-12)


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------
def test_box_corners_are_centred_and_correctly_extented() -> None:
    center = np.array([5.0, -2.0, 1.0])
    width, length, height = 2.0, 4.0, 1.5
    corners = box_corners(center, (width, length, height))

    assert corners.shape == (8, 3)
    assert np.allclose(corners.mean(axis=0), center)
    # length spans x, width spans y, height spans z
    assert np.isclose(corners[:, 0].min(), 5.0 - length / 2)
    assert np.isclose(corners[:, 0].max(), 5.0 + length / 2)
    assert np.isclose(corners[:, 1].min(), -2.0 - width / 2)
    assert np.isclose(corners[:, 1].max(), -2.0 + width / 2)
    assert np.isclose(corners[:, 2].min(), 1.0 - height / 2)
    assert np.isclose(corners[:, 2].max(), 1.0 + height / 2)


def test_box_corners_yaw_swaps_extents() -> None:
    """A 90-degree yaw must swap the length and width extents."""
    corners = box_corners(np.zeros(3), (2.0, 4.0, 1.0), yaw=np.pi / 2)
    assert np.isclose(corners[:, 0].min(), -1.0)  # width now spans x
    assert np.isclose(corners[:, 0].max(), 1.0)
    assert np.isclose(corners[:, 1].min(), -2.0)  # length now spans y
    assert np.isclose(corners[:, 1].max(), 2.0)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def test_view_points_normalises_by_depth() -> None:
    points = np.array([[1.0], [0.0], [2.0]])  # x=1, z=2
    projected = view_points(points, INTRINSIC)
    assert np.isclose(projected[0, 0], 1000.0 * 1.0 / 2.0 + 800.0)
    assert np.isclose(projected[1, 0], 450.0)


def test_view_points_accepts_both_orientations() -> None:
    points = np.array([[1.0, 0.0, 2.0], [3.0, 0.0, 2.0]])
    assert np.allclose(view_points(points, INTRINSIC), view_points(points.T, INTRINSIC))


def test_project_box_ahead_returns_exact_pixel_bounds() -> None:
    """Box 10 m ahead, 4 m long, 2 m wide, 2 m tall, camera-frame == sensor-frame."""
    corners = box_corners(np.array([0.0, 0.0, 10.0]), (2.0, 4.0, 2.0))

    bbox = project_box_to_image(corners, IDENTITY_3, ZERO_3, IDENTITY_3, ZERO_3, INTRINSIC)

    assert bbox is not None
    # u = 1000x/z + 800, v = 1000y/z + 450, extremes taken over all eight corners
    assert bbox == pytest.approx((577.7778, 338.8889, 1022.2222, 561.1111), abs=1e-3)


def test_project_box_behind_camera_returns_none() -> None:
    """A box behind the camera must be rejected before perspective division."""
    corners = box_corners(np.array([0.0, 0.0, -10.0]), (2.0, 4.0, 2.0))
    assert project_box_to_image(corners, IDENTITY_3, ZERO_3, IDENTITY_3, ZERO_3, INTRINSIC) is None


def test_project_box_straddling_the_image_plane_returns_none() -> None:
    """A box the camera sits inside has corners at z <= 0 and must be rejected.

    Without the depth guard this box would project to a plausible-looking bbox
    inside the image, because corners with negative z are mirrored.
    """
    corners = box_corners(np.zeros(3), (2.0, 4.0, 2.0))
    assert corners[:, 2].min() < 0.0
    assert project_box_to_image(corners, IDENTITY_3, ZERO_3, IDENTITY_3, ZERO_3, INTRINSIC) is None


def test_min_depth_parameter_is_honoured() -> None:
    """A box just at the plane passes only when the guard is relaxed."""
    corners = box_corners(np.array([0.0, 0.0, 1.0]), (2.0, 4.0, 2.0))
    assert corners[:, 2].min() == pytest.approx(0.0)
    assert project_box_to_image(corners, IDENTITY_3, ZERO_3, IDENTITY_3, ZERO_3, INTRINSIC) is None
    assert (
        project_box_to_image(
            corners, IDENTITY_3, ZERO_3, IDENTITY_3, ZERO_3, INTRINSIC, min_depth_m=-1e-9
        )
        is not None
    )


def test_projection_respects_sensor_to_camera_offset() -> None:
    """Translating the camera off the box's axis must shift the bbox in the image."""
    corners = box_corners(np.array([0.0, 0.0, 10.0]), (2.0, 4.0, 2.0))

    centred = project_box_to_image(corners, IDENTITY_3, ZERO_3, IDENTITY_3, ZERO_3, INTRINSIC)
    shifted = project_box_to_image(
        corners, IDENTITY_3, ZERO_3, IDENTITY_3, np.array([0.0, 5.0, 0.0]), INTRINSIC
    )

    assert centred is not None
    assert shifted is not None
    # The camera moved +5 m in y, so the box sits at y = -5 in the camera frame.
    # Image coordinates increase downward, so a smaller box y shifts to smaller v.
    assert shifted[1] < centred[1]
    assert shifted[3] < centred[3]


def test_projection_rejects_non_3x3_intrinsic() -> None:
    with pytest.raises(ValueError, match="3x3"):
        view_points(np.zeros((3, 1)), np.eye(4))


def test_bbox_inside_image() -> None:
    assert bbox_inside_image((10.0, 10.0, 20.0, 20.0), (1600, 900))
    assert not bbox_inside_image((-100.0, 10.0, 0.0, 20.0), (1600, 900))
    assert not bbox_inside_image((10.0, 10.0, 20.0, -5.0), (1600, 900))


# ---------------------------------------------------------------------------
# Overlap and distance
# ---------------------------------------------------------------------------
def _reference_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Scalar IoU, written independently of bbox_iou_matrix."""
    ix = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    iy = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    inter = ix * iy
    union = (
        (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        - inter
    )
    return inter / union if union > 0 else 0.0


def test_bbox_iou_matrix_matches_scalar_reference() -> None:
    boxes_a = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0], [20.0, 20.0, 30.0, 30.0]])
    boxes_b = np.array([[2.0, 2.0, 12.0, 12.0], [50.0, 50.0, 60.0, 60.0]])

    matrix = bbox_iou_matrix(boxes_a, boxes_b)

    assert matrix.shape == (3, 2)
    for i in range(3):
        for j in range(2):
            assert matrix[i, j] == pytest.approx(_reference_iou(boxes_a[i], boxes_b[j]))


def test_bbox_iou_of_identical_boxes_is_one() -> None:
    assert bbox_iou([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]) == pytest.approx(1.0)


def test_bbox_iou_of_disjoint_boxes_is_zero() -> None:
    assert bbox_iou([0.0, 0.0, 1.0, 1.0], [5.0, 5.0, 6.0, 6.0]) == 0.0


def test_bbox_iou_matrix_handles_empty_inputs() -> None:
    assert bbox_iou_matrix(np.empty((0, 4)), np.empty((0, 4))).shape == (0, 0)
    assert bbox_iou_matrix(np.zeros((2, 4)), np.empty((0, 4))).shape == (2, 0)


def test_pairwise_distance_matrix() -> None:
    a = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0]])
    matrix = pairwise_distance_matrix(a, b)
    assert matrix.shape == (2, 1)
    assert matrix[0, 0] == pytest.approx(0.0)
    assert matrix[1, 0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def test_to_global_yaw_composes_sensor_and_ego_rotation() -> None:
    sensor = yaw_matrix(np.pi / 2)  # sensor turned 90 degrees inside the ego frame
    assert to_global_yaw(0.0, sensor, IDENTITY_3) == pytest.approx(np.pi / 2)
    assert to_global_yaw(0.0, IDENTITY_3, sensor) == pytest.approx(np.pi / 2)


def test_to_global_yaw_wraps_into_minus_pi_pi() -> None:
    result = to_global_yaw(np.pi * 0.9, yaw_matrix(np.pi * 0.9), IDENTITY_3)
    assert -np.pi <= result < np.pi


def test_wrap_angle() -> None:
    assert wrap_angle(0.0) == pytest.approx(0.0)
    assert wrap_angle(np.pi / 2) == pytest.approx(np.pi / 2)
    # The range is [-pi, pi), so a half turn from either direction lands on -pi.
    assert wrap_angle(3 * np.pi) == pytest.approx(-np.pi)
    assert wrap_angle(-3 * np.pi) == pytest.approx(-np.pi)


def test_wrap_angle_always_in_half_open_range() -> None:
    for raw in np.linspace(-20.0, 20.0, 401):
        assert -np.pi <= wrap_angle(float(raw)) < np.pi


def test_sensor_to_global_matches_manual_composition() -> None:
    sensor_rotation = yaw_matrix(0.3)
    sensor_translation = np.array([1.0, 0.0, 0.8])
    ego_rotation = yaw_matrix(1.1)
    ego_translation = np.array([100.0, -50.0, 2.0])

    points = np.array([[1.0, 2.0, 3.0]])
    in_ego = transform_points(points, sensor_rotation, sensor_translation)
    expected = transform_points(in_ego, ego_rotation, ego_translation)

    assert np.allclose(
        sensor_to_global(
            points, sensor_rotation, sensor_translation, ego_rotation, ego_translation
        ),
        expected,
    )
