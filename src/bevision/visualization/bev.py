"""Bird's-Eye View rendering.

Draws the LiDAR point cloud, the current frame's detections and the live tracks around the
ego vehicle. Points are coloured by height, which is what makes it readable: the ground plane
flattens to one colour and everything standing on it separates out.

Frame transforms come from :func:`bevision.geometry.global_to_ego` rather than a second copy, so
the panel and the pipeline share one implementation of the ego-frame conversion. The plotted
frame is the ego frame: the ego vehicle sits at the origin with +x forward.

Nothing here inspects a dataset: the inputs are global-frame points and boxes, so a figure can
be produced from anything that can supply those.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bevision.geometry import global_to_ego
from bevision.types import Detection3D, TrackOutput

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class BevView:
    """Frame extents and styling for a BEV panel. Defaults match the reported figures."""

    x_limits: tuple[float, float] = (-60.0, 60.0)
    y_limits: tuple[float, float] = (-40.0, 80.0)
    point_stride: int = 5
    """Draw every Nth point. A sweep is ~35k points; this is the dial that trades fidelity
    for render time."""
    point_size: float = 0.4
    height_range: tuple[float, float] = (-2.0, 2.0)
    """Colour-scale limits in metres. Clipping is deliberate: one tall pole would otherwise
    wash out the whole cloud."""
    point_alpha: float = 0.35
    detection_colour: str = "red"
    track_colour: str = "blue"


@dataclass(frozen=True)
class BevFrame:
    """Everything needed to draw one BEV panel."""

    frame_index: int
    points_global: ArrayLike
    """``(N, 3)`` points in the global frame."""

    ego_rotation: ArrayLike
    ego_translation: ArrayLike
    detections: Sequence[Detection3D] = field(default_factory=tuple)
    tracks: Sequence[TrackOutput] = field(default_factory=tuple)
    label: str | None = None

    @property
    def title(self) -> str:
        if self.label is not None:
            return self.label
        return f"frame {self.frame_index}: {len(self.tracks)} tracks"


def _ego_frame_points(
    points_global: ArrayLike,
    ego_rotation: ArrayLike,
    ego_translation: ArrayLike,
) -> FloatArray:
    points = np.asarray(points_global, dtype=float).reshape(-1, 3)
    if points.size == 0:
        return points
    return np.asarray(global_to_ego(points, ego_rotation, ego_translation), dtype=float)


def _ego_frame_centres(
    centres: Sequence[ArrayLike],
    ego_rotation: ArrayLike,
    ego_translation: ArrayLike,
) -> FloatArray:
    if not centres:
        return np.empty((0, 3))
    stacked = np.asarray(centres, dtype=float).reshape(-1, 3)
    return np.asarray(global_to_ego(stacked, ego_rotation, ego_translation), dtype=float)


def render_bev(ax, frame: BevFrame, view: BevView | None = None):
    """Draw one BEV panel onto ``ax`` and return it.

    Args:
        ax: A matplotlib ``Axes``.
        frame: The global-frame data to draw.
        view: Extents and styling; defaults to :class:`BevView`.

    Returns:
        The same ``ax``, for chaining.
    """
    view = view or BevView()

    points = _ego_frame_points(frame.points_global, frame.ego_rotation, frame.ego_translation)
    if points.shape[0]:
        sampled = points[:: max(view.point_stride, 1)]
        ax.scatter(
            sampled[:, 0],
            sampled[:, 1],
            s=view.point_size,
            c=sampled[:, 2],
            cmap="viridis",
            vmin=view.height_range[0],
            vmax=view.height_range[1],
            alpha=view.point_alpha,
        )

    detection_centres = _ego_frame_centres(
        [d.center_global for d in frame.detections],
        frame.ego_rotation,
        frame.ego_translation,
    )
    if detection_centres.shape[0]:
        ax.plot(
            detection_centres[:, 0],
            detection_centres[:, 1],
            ".",
            color=view.detection_colour,
            markersize=3,
            label="detections",
        )

    track_centres = _ego_frame_centres(
        [t.box.center for t in frame.tracks],
        frame.ego_rotation,
        frame.ego_translation,
    )
    if track_centres.shape[0]:
        ax.plot(
            track_centres[:, 0],
            track_centres[:, 1],
            "o",
            color=view.track_colour,
            markersize=5,
            fillstyle="none",
            label="tracks",
        )
        for track, centre in zip(frame.tracks, track_centres, strict=True):
            ax.annotate(
                str(track.track_id),
                (centre[0], centre[1]),
                fontsize=7,
                color=view.track_colour,
            )

    # The ego vehicle is the origin by construction, so it needs no pose lookup.
    ax.plot(0, 0, "k^", markersize=14, label="ego")
    ax.set_aspect("equal")
    ax.set_xlim(*view.x_limits)
    ax.set_ylim(*view.y_limits)
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    ax.set_title(frame.title, fontsize=10)
    ax.grid(alpha=0.3)
    return ax


def render_bev_figure(
    frames: Sequence[BevFrame],
    *,
    view: BevView | None = None,
    columns: int = 2,
    panel_size: tuple[float, float] = (8.0, 7.0),
    suptitle: str | None = None,
):
    """Draw several BEV panels into one figure.

    Returns ``(figure, axes)``. The default of two columns reproduces the 2x2 layout of the
    four-moment figures in the README.
    """
    from bevision._deps import require

    require("matplotlib", "rendering figures")
    import matplotlib.pyplot as plt

    if not frames:
        raise ValueError("frames must not be empty")
    if columns < 1:
        raise ValueError("columns must be at least 1")

    rows = -(-len(frames) // columns)  # ceiling division
    figure, axes = plt.subplots(
        rows, columns, figsize=(panel_size[0] * columns, panel_size[1] * rows), squeeze=False
    )
    flat = axes.ravel()
    for ax, frame in zip(flat, frames, strict=False):
        render_bev(ax, frame, view)
    for ax in flat[len(frames) :]:
        ax.axis("off")

    if suptitle is not None:
        figure.suptitle(suptitle)
    return figure, axes
