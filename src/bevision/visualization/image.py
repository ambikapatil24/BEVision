"""Draw 2D detections onto a camera image, with no image library required.

The original notebook used OpenCV to draw its fusion overlays. Rectangles on a numpy array
are four slice assignments, so this module needs only numpy — which keeps the visualisation
path importable and testable on a machine with no OpenCV, and removes cv2 from the render
dependency surface entirely.

Colour convention, kept from the original figures so they stay comparable:

* **green** — camera-only detection (YOLO found it, LiDAR did not)
* **red** — LiDAR-only, projected into the image (no camera agreement)
* **yellow** — fused: both sensors agree on position and class
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bevision.fusion import CameraDetections

UInt8Image = NDArray[np.uint8]

#: RGB triples, matching the original overlays.
BOX_COLOURS: dict[str, tuple[int, int, int]] = {
    "camera": (0, 255, 0),  # green
    "lidar": (255, 0, 0),  # red
    "fused": (255, 255, 0),  # yellow
}


def draw_box(
    image: UInt8Image,
    bbox: ArrayLike,
    colour: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> UInt8Image:
    """Draw one ``(x_min, y_min, x_max, y_max)`` rectangle in place and return the image.

    The box is clipped to the image, so a partially off-screen detection draws its visible
    part rather than raising or wrapping around.
    """
    if thickness < 1:
        raise ValueError("thickness must be at least 1")

    height, width = image.shape[:2]
    x_min, y_min, x_max, y_max = (int(round(float(v))) for v in np.asarray(bbox).reshape(4))

    # Reject boxes that miss the image entirely. Clamping alone would collapse such a box
    # onto the border and draw a stray pixel in the corner.
    if x_max < 0 or y_max < 0 or x_min > width - 1 or y_min > height - 1:
        return image

    x_min = max(0, min(x_min, width - 1))
    x_max = max(0, min(x_max, width - 1))
    y_min = max(0, min(y_min, height - 1))
    y_max = max(0, min(y_max, height - 1))
    if x_max < x_min or y_max < y_min:
        return image

    colour_array = np.asarray(colour, dtype=np.uint8)
    for offset in range(thickness):
        top, bottom = y_min + offset, y_max - offset
        left, right = x_min + offset, x_max - offset
        if top > bottom or left > right:
            break
        image[top, left : right + 1] = colour_array
        image[bottom, left : right + 1] = colour_array
        image[top : bottom + 1, left] = colour_array
        image[top : bottom + 1, right] = colour_array
    return image


def draw_boxes(
    image: UInt8Image,
    boxes: Sequence[ArrayLike],
    colour: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> UInt8Image:
    """Draw several boxes in the same colour."""
    for bbox in boxes:
        draw_box(image, bbox, colour, thickness)
    return image


def overlay_frame(
    image: ArrayLike,
    camera_detections: CameraDetections | None = None,
    projected: Sequence[ArrayLike] = (),
    fused: Sequence[ArrayLike] = (),
    *,
    thickness: int = 2,
) -> UInt8Image:
    """Return a copy of ``image`` with the fusion overlay drawn on it.

    Args:
        image: ``(H, W, 3)`` array, uint8 or convertible.
        camera_detections: YOLO boxes drawn green.
        projected: LiDAR boxes projected into this camera, drawn red.
        fused: The subset of ``projected`` that a camera confirmed, drawn yellow last so it
            sits on top of the red.
        thickness: Line width in pixels.

    Returns:
        A new array; the input is not modified.
    """
    canvas = np.array(image, dtype=np.uint8, copy=True)
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got shape {canvas.shape}")

    if camera_detections is not None and not camera_detections.is_empty:
        draw_boxes(canvas, camera_detections.boxes, BOX_COLOURS["camera"], thickness)
    if projected:
        draw_boxes(canvas, projected, BOX_COLOURS["lidar"], thickness)
    if fused:
        draw_boxes(canvas, fused, BOX_COLOURS["fused"], thickness)

    return canvas


def show_image(ax, image: ArrayLike, title: str | None = None):
    """Display an image on ``ax`` and return it."""
    ax.imshow(np.asarray(image, dtype=np.uint8))
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=12)
    return ax
