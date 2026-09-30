"""Bird's-Eye View and image-overlay rendering.

Rendering is kept separate from perception on purpose: the inputs here are already
global-frame points, detections and tracks, so nothing in this package needs a dataset or a
GPU to produce a figure. Matplotlib is imported inside the functions, so
``import bevision.visualization`` costs nothing until something is actually drawn.
"""

from __future__ import annotations

from bevision.visualization.bev import BevFrame, BevView, render_bev, render_bev_figure
from bevision.visualization.image import (
    BOX_COLOURS,
    draw_box,
    draw_boxes,
    overlay_frame,
    show_image,
)

__all__ = [
    "BOX_COLOURS",
    "BevFrame",
    "BevView",
    "draw_box",
    "draw_boxes",
    "overlay_frame",
    "render_bev",
    "render_bev_figure",
    "show_image",
]
