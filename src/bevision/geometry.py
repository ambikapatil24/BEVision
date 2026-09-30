"""Vectorised 3D geometry: rigid transforms, box corners, projection, overlap.

This module deliberately depends on nothing but numpy. The original notebook used
``pyquaternion.Quaternion.rotate`` per corner, which is a Python-level method call
eight times per box per camera; here every operation is a batched matrix product.

Frame conventions
-----------------
* A sensor/ego pose is a rotation matrix ``R`` (3, 3) and a translation ``t``
  (3,), mapping points from that frame into its parent frame: ``p_parent = R @ p + t``.
* A nuScenes box is ``(center, size, yaw)`` with ``size = (width, length, height)``.
  The box's local x-axis is its *length* and its local y-axis is its *width*, and
  ``yaw`` rotates about +z.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------
def quaternion_to_matrix(quaternion: ArrayLike) -> FloatArray:
    """Convert a ``(w, x, y, z)`` quaternion to a 3x3 rotation matrix.

    Accepts a single quaternion or a batch of shape ``(..., 4)``.
    """
    q = np.asarray(quaternion, dtype=float)
    if q.shape[-1] != 4:
        raise ValueError(f"expected a (..., 4) wxyz quaternion, got shape {q.shape}")
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = (q[..., i] for i in range(4))
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def yaw_matrix(yaw: ArrayLike) -> FloatArray:
    """Rotation matrix (or batch of them) for a rotation about +z by ``yaw``."""
    yaw = np.asarray(yaw, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    zeros, ones = np.zeros_like(c), np.ones_like(c)
    return np.stack(
        [
            np.stack([c, -s, zeros], -1),
            np.stack([s, c, zeros], -1),
            np.stack([zeros, zeros, ones], -1),
        ],
        axis=-2,
    )


def rigid_inverse(rotation: ArrayLike, translation: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Invert a rigid transform ``p -> R @ p + t`` without a general inverse."""
    r = np.asarray(rotation, dtype=float)
    t = np.asarray(translation, dtype=float)
    r_inv = np.swapaxes(r, -1, -2)
    return r_inv, -np.einsum("...ij,...j->...i", r_inv, t)


def transform_points(
    points: ArrayLike,
    rotation: ArrayLike,
    translation: ArrayLike,
) -> FloatArray:
    """Apply ``p -> R @ p + t`` to a point or a batch of points."""
    p = np.asarray(points, dtype=float)
    r = np.asarray(rotation, dtype=float)
    t = np.asarray(translation, dtype=float)
    return np.einsum("...ij,...j->...i", r, p) + t


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------
def box_corners(
    center: ArrayLike,
    size: ArrayLike,
    yaw: ArrayLike | float = 0.0,
) -> FloatArray:
    """Return the eight corners of a 3D box in its parent frame, shape ``(8, 3)``.

    ``size`` is ``(width, length, height)``, matching nuScenes: the local x-axis
    spans the length, the local y-axis spans the width, and the z-axis the height.
    """
    cx, cy, cz = (float(v) for v in np.asarray(center, dtype=float).reshape(3))
    width, length, height = (float(v) for v in np.asarray(size, dtype=float).reshape(3))

    half_l, half_w, half_h = length / 2.0, width / 2.0, height / 2.0
    local = np.array(
        [
            [+half_l, +half_w, +half_h],
            [+half_l, -half_w, +half_h],
            [-half_l, -half_w, +half_h],
            [-half_l, +half_w, +half_h],
            [+half_l, +half_w, -half_h],
            [+half_l, -half_w, -half_h],
            [-half_l, -half_w, -half_h],
            [-half_l, +half_w, -half_h],
        ],
        dtype=float,
    )
    return local @ yaw_matrix(yaw).T + np.array([cx, cy, cz])


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def view_points(
    points: ArrayLike,
    intrinsic: ArrayLike,
    *,
    normalize: bool = True,
) -> FloatArray:
    """Project 3D camera-frame points to 2D with a 3x3 intrinsic matrix.

    Mirrors ``nuscenes.utils.geometry_utils.view_points``. With
    ``normalize=True`` the result is divided by depth, so callers must have
    already rejected points at or behind the image plane.
    """
    p = np.asarray(points, dtype=float)
    if p.shape[0] != 3 and p.shape[-1] == 3:
        p = p.T
    if p.shape[0] != 3:
        raise ValueError(f"expected (3, N) or (N, 3) points, got shape {p.shape}")

    k = np.asarray(intrinsic, dtype=float)
    if k.shape != (3, 3):
        raise ValueError(f"expected a 3x3 intrinsic matrix, got shape {k.shape}")

    projected = k @ p
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            projected = projected / projected[2:3, :]
    return projected


def project_box_to_image(
    corners_sensor: ArrayLike,
    sensor_rotation: ArrayLike,
    sensor_translation: ArrayLike,
    camera_rotation: ArrayLike,
    camera_translation: ArrayLike,
    camera_intrinsic: ArrayLike,
    *,
    min_depth_m: float = 0.0,
) -> tuple[float, float, float, float] | None:
    """Project a box given by its 8 sensor-frame corners into a camera image.

    Returns an ``(x_min, y_min, x_max, y_max)`` pixel bounding box, or ``None`` if
    the box is not in front of the camera.

    The depth check is not optional bookkeeping: ``view_points`` divides by depth,
    so a corner behind the image plane (z <= 0) is mirrored through the focal
    plane and lands inside the image as if the camera were looking backwards.
    Such a box would then produce spurious IoU matches, so it is rejected first.
    """
    corners_sensor = np.asarray(corners_sensor, dtype=float).reshape(-1, 3)
    corners_ego = transform_points(corners_sensor, sensor_rotation, sensor_translation)

    cam_rot_inv, cam_trans_inv = rigid_inverse(camera_rotation, camera_translation)
    corners_camera = corners_ego @ np.swapaxes(cam_rot_inv, -1, -2) + cam_trans_inv

    if corners_camera[:, 2].min() <= min_depth_m:
        return None

    points_2d = view_points(corners_camera.T, camera_intrinsic, normalize=True)[:2, :].T
    return (
        float(points_2d[:, 0].min()),
        float(points_2d[:, 1].min()),
        float(points_2d[:, 0].max()),
        float(points_2d[:, 1].max()),
    )


def bbox_inside_image(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> bool:
    """Whether a 2D bbox overlaps the image rectangle at all."""
    width, height = image_size
    x_min, y_min, x_max, y_max = bbox
    return x_max > 0 and x_min < width and y_max > 0 and y_min < height


# ---------------------------------------------------------------------------
# Overlap and distance
# ---------------------------------------------------------------------------
def bbox_iou_matrix(boxes_a: ArrayLike, boxes_b: ArrayLike) -> FloatArray:
    """Pairwise IoU between two sets of ``(x_min, y_min, x_max, y_max)`` boxes.

    Returns a ``(len(boxes_a), len(boxes_b))`` array of IoUs in [0, 1].
    """
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)

    x_min = np.maximum(a[:, None, 0], b[None, :, 0])
    y_min = np.maximum(a[:, None, 1], b[None, :, 1])
    x_max = np.minimum(a[:, None, 2], b[None, :, 2])
    y_max = np.minimum(a[:, None, 3], b[None, :, 3])

    intersection = np.clip(x_max - x_min, 0.0, None) * np.clip(y_max - y_min, 0.0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection

    return np.where(union > 0.0, intersection / np.where(union > 0.0, union, 1.0), 0.0)


def bbox_iou(box_a: ArrayLike, box_b: ArrayLike) -> float:
    """IoU of two ``(x_min, y_min, x_max, y_max)`` boxes."""
    return float(bbox_iou_matrix(box_a, box_b)[0, 0])


def pairwise_distance_matrix(points_a: ArrayLike, points_b: ArrayLike) -> FloatArray:
    """Euclidean distance between every pair of 3D points."""
    a = np.asarray(points_a, dtype=float).reshape(-1, 3)
    b = np.asarray(points_b, dtype=float).reshape(-1, 3)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


# ---------------------------------------------------------------------------
# Orientation between frames
# ---------------------------------------------------------------------------
def compose_frames(outer_rotation: ArrayLike, inner_rotation: ArrayLike) -> FloatArray:
    """Rotation of ``inner`` expressed inside ``outer``: ``R_outer @ R_inner``."""
    return np.asarray(outer_rotation, dtype=float) @ np.asarray(inner_rotation, dtype=float)


def yaw_of(rotation: ArrayLike) -> float:
    """Extract the yaw (rotation about +z) of a rotation matrix."""
    r = np.asarray(rotation, dtype=float)
    return float(np.arctan2(r[1, 0], r[0, 0]))


def wrap_angle(angle: float) -> float:
    """Wrap an angle into ``[-pi, pi)``."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def to_global_yaw(
    yaw_sensor: float,
    sensor_rotation: ArrayLike,
    ego_rotation: ArrayLike,
) -> float:
    """Express a sensor-frame yaw in the global frame."""
    yaw_offset = yaw_of(compose_frames(ego_rotation, sensor_rotation))
    return wrap_angle(yaw_sensor + yaw_offset)


def sensor_to_global(
    points: ArrayLike,
    sensor_rotation: ArrayLike,
    sensor_translation: ArrayLike,
    ego_rotation: ArrayLike,
    ego_translation: ArrayLike,
) -> FloatArray:
    """Lift points from the sensor frame to the global frame via the ego pose."""
    in_ego = transform_points(points, sensor_rotation, sensor_translation)
    return transform_points(in_ego, ego_rotation, ego_translation)


def global_to_ego(
    points: ArrayLike,
    ego_rotation: ArrayLike,
    ego_translation: ArrayLike,
) -> FloatArray:
    """Drop global-frame points into the ego frame (car at the origin, +x forward)."""
    rotation_inv, translation_inv = rigid_inverse(ego_rotation, ego_translation)
    return transform_points(points, rotation_inv, translation_inv)
