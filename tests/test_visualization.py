"""Tests for the BEV and image-overlay renderers.

Rendering is worth testing because it is where a refactor silently lies: a BEV panel that
draws the point cloud in the *sensor* frame instead of the ego frame still looks like a BEV
panel. So the transform is asserted numerically, not just "something was plotted".

matplotlib runs headless via Agg, so these need no display.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from bevision.fusion import CameraDetections  # noqa: E402
from bevision.geometry import yaw_matrix  # noqa: E402
from bevision.types import Box3D, Detection3D, TrackOutput  # noqa: E402
from bevision.visualization import (  # noqa: E402
    BOX_COLOURS,
    BevFrame,
    BevView,
    draw_box,
    draw_boxes,
    overlay_frame,
    render_bev,
    render_bev_figure,
    show_image,
)

IDENTITY = np.eye(3)
ORIGIN = np.zeros(3)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _detection(centre=(0.0, 0.0, 0.0), class_name: str = "car") -> Detection3D:
    return Detection3D(
        label=0,
        class_name=class_name,
        score=0.8,
        center_global=np.array(centre, dtype=float),
        size=(2.0, 4.0, 1.5),
        yaw_global=0.0,
    )


def _track(track_id: int = 7, centre=(0.0, 0.0, 0.0), class_name: str = "car") -> TrackOutput:
    return TrackOutput(
        track_id=track_id,
        class_name=class_name,
        score=0.8,
        box=Box3D(center=np.array(centre, dtype=float), size=(2.0, 4.0, 1.5), yaw=0.0),
        velocity=(0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# BEV
# ---------------------------------------------------------------------------
def test_points_are_drawn_in_the_ego_frame_not_the_global_frame() -> None:
    """An ego at x=100 must place a point at global x=105 at panel x=5."""
    frame = BevFrame(
        frame_index=0,
        points_global=np.array([[105.0, 0.0, 0.0], [106.0, 2.0, 0.5]]),
        ego_rotation=IDENTITY,
        ego_translation=np.array([100.0, 0.0, 0.0]),
    )
    _, ax = plt.subplots()
    render_bev(ax, frame, BevView(point_stride=1))

    offsets = ax.collections[0].get_offsets()
    assert np.allclose(offsets[0], [5.0, 0.0])
    assert np.allclose(offsets[1], [6.0, 2.0])


def test_ego_rotation_is_applied_to_points() -> None:
    """A 90-degree ego yaw must rotate the cloud, not just translate it."""
    frame = BevFrame(
        frame_index=0,
        points_global=np.array([[10.0, 0.0, 0.0]]),
        ego_rotation=yaw_matrix(np.pi / 2),
        ego_translation=ORIGIN,
    )
    _, ax = plt.subplots()
    render_bev(ax, frame, BevView(point_stride=1))

    # global (10, 0) seen from an ego yawed +90 deg lands at ego (0, -10)
    assert np.allclose(ax.collections[0].get_offsets()[0], [0.0, -10.0], atol=1e-9)


def test_point_stride_subsamples_the_cloud() -> None:
    points = np.zeros((100, 3))
    frame = BevFrame(0, points, IDENTITY, ORIGIN)

    _, ax = plt.subplots()
    render_bev(ax, frame, BevView(point_stride=10))
    assert ax.collections[0].get_offsets().shape[0] == 10

    _, ax2 = plt.subplots()
    render_bev(ax2, frame, BevView(point_stride=1))
    assert ax2.collections[0].get_offsets().shape[0] == 100


def test_stride_of_zero_is_treated_as_one() -> None:
    """Guards the `[:: max(stride, 1)]` slice against a divide-by-zero style blowup."""
    frame = BevFrame(0, np.zeros((5, 3)), IDENTITY, ORIGIN)
    _, ax = plt.subplots()
    render_bev(ax, frame, BevView(point_stride=0))
    assert ax.collections[0].get_offsets().shape[0] == 5


def test_detections_and_tracks_are_drawn_with_their_own_colours() -> None:
    frame = BevFrame(
        frame_index=0,
        points_global=np.zeros((4, 3)),
        ego_rotation=IDENTITY,
        ego_translation=ORIGIN,
        detections=[_detection((5.0, 1.0, 0.0))],
        tracks=[_track(3, (8.0, -2.0, 0.0))],
    )
    _, ax = plt.subplots()
    render_bev(ax, frame)

    colours = [line.get_color() for line in ax.lines]
    assert "red" in colours  # detections
    assert "blue" in colours  # tracks


def test_track_ids_are_annotated() -> None:
    frame = BevFrame(
        0, np.zeros((4, 3)), IDENTITY, ORIGIN, tracks=[_track(12), _track(34)], detections=[]
    )
    _, ax = plt.subplots()
    render_bev(ax, frame)

    assert {text.get_text() for text in ax.texts} == {"12", "34"}


def test_track_positions_come_from_the_box_centre() -> None:
    frame = BevFrame(
        0,
        np.zeros((4, 3)),
        IDENTITY,
        np.array([50.0, 0.0, 0.0]),
        tracks=[_track(1, (55.0, 3.0, 0.0))],
    )
    _, ax = plt.subplots()
    render_bev(ax, frame)

    track_line = next(line for line in ax.lines if line.get_color() == "blue")
    assert np.allclose(track_line.get_xdata(), [5.0])
    assert np.allclose(track_line.get_ydata(), [3.0])


def test_ego_marker_is_at_the_origin() -> None:
    frame = BevFrame(0, np.zeros((4, 3)), IDENTITY, np.array([999.0, 999.0, 0.0]))
    _, ax = plt.subplots()
    render_bev(ax, frame)

    ego_line = next(line for line in ax.lines if line.get_marker() == "^")
    assert np.allclose([ego_line.get_xdata()[0], ego_line.get_ydata()[0]], [0.0, 0.0])


def test_axes_limits_aspect_and_labels_are_set() -> None:
    view = BevView(x_limits=(-10.0, 10.0), y_limits=(-5.0, 25.0))
    _, ax = plt.subplots()
    render_bev(ax, BevFrame(0, np.zeros((4, 3)), IDENTITY, ORIGIN), view)

    assert ax.get_xlim() == (-10.0, 10.0)
    assert ax.get_ylim() == (-5.0, 25.0)
    assert ax.get_aspect() == 1.0
    assert "forward" in ax.get_xlabel()


def test_title_defaults_to_the_track_count() -> None:
    _, ax = plt.subplots()
    render_bev(ax, BevFrame(4, np.zeros((4, 3)), IDENTITY, ORIGIN, tracks=[_track()]))
    assert ax.get_title() == "frame 4: 1 tracks"


def test_explicit_label_overrides_the_title() -> None:
    frame = BevFrame(4, np.zeros((4, 3)), IDENTITY, ORIGIN, label="scene-0916 / frame 4")
    _, ax = plt.subplots()
    render_bev(ax, frame)
    assert ax.get_title() == "scene-0916 / frame 4"


def test_empty_inputs_render_without_error() -> None:
    """A frame with no points and no objects must still produce a usable panel."""
    _, ax = plt.subplots()
    render_bev(ax, BevFrame(0, np.empty((0, 3)), IDENTITY, ORIGIN))

    assert len(ax.collections) == 0  # no cloud drawn
    # only the ego marker
    assert len(ax.lines) == 1
    assert ax.get_title() == "frame 0: 0 tracks"


def test_render_bev_returns_the_axes() -> None:
    _, ax = plt.subplots()
    assert render_bev(ax, BevFrame(0, np.zeros((2, 3)), IDENTITY, ORIGIN)) is ax


def test_render_bev_figure_lays_out_four_panels() -> None:
    frames = [BevFrame(i, np.zeros((10, 3)), IDENTITY, ORIGIN) for i in range(4)]
    figure, axes = render_bev_figure(frames, suptitle="four moments")

    assert figure.get_suptitle() == "four moments"
    assert axes.shape == (2, 2)


def test_render_bev_figure_blanks_unused_panels() -> None:
    frames = [BevFrame(i, np.zeros((4, 3)), IDENTITY, ORIGIN) for i in range(3)]
    _, axes = render_bev_figure(frames)

    assert axes.shape == (2, 2)
    assert not axes.ravel()[3].axison  # the fourth panel is switched off


def test_render_bev_figure_respects_columns() -> None:
    frames = [BevFrame(i, np.zeros((4, 3)), IDENTITY, ORIGIN) for i in range(5)]
    _, axes = render_bev_figure(frames, columns=3)
    assert axes.shape == (2, 3)


def test_render_bev_figure_rejects_empty_and_bad_columns() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        render_bev_figure([])
    with pytest.raises(ValueError, match="columns"):
        render_bev_figure([BevFrame(0, np.zeros((2, 3)), IDENTITY, ORIGIN)], columns=0)


# ---------------------------------------------------------------------------
# Image overlay
# ---------------------------------------------------------------------------
def _blank(height: int = 100, width: int = 200):
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_draw_box_strokes_all_four_edges() -> None:
    image = _blank()
    draw_box(image, (10, 20, 60, 70), (255, 0, 0), thickness=1)

    assert tuple(image[20, 35]) == (255, 0, 0)  # top edge
    assert tuple(image[70, 35]) == (255, 0, 0)  # bottom edge
    assert tuple(image[45, 10]) == (255, 0, 0)  # left edge
    assert tuple(image[45, 60]) == (255, 0, 0)  # right edge
    assert tuple(image[45, 35]) == (0, 0, 0)  # interior untouched


def test_draw_box_clips_a_partially_visible_box() -> None:
    image = _blank()
    draw_box(image, (-50, -50, 30, 30), (0, 255, 0), thickness=1)

    assert tuple(image[0, 0]) == (0, 255, 0)
    assert tuple(image[30, 30]) == (0, 255, 0)


def test_draw_box_ignores_a_box_entirely_outside_the_image() -> None:
    """Clamping alone would collapse it onto the border and leave a stray pixel."""
    image = _blank()
    draw_box(image, (-100, -100, -50, -50), (0, 255, 0))

    assert not image.any()
    image2 = _blank()
    draw_box(image2, (500, 500, 600, 600), (0, 255, 0))
    assert not image2.any()


def test_draw_box_thickness_widens_the_stroke() -> None:
    thin, thick = _blank(), _blank()
    draw_box(thin, (10, 10, 50, 50), (255, 255, 255), thickness=1)
    draw_box(thick, (10, 10, 50, 50), (255, 255, 255), thickness=3)

    assert thick.sum() > thin.sum()


def test_draw_box_rejects_zero_thickness() -> None:
    with pytest.raises(ValueError, match="thickness"):
        draw_box(_blank(), (1, 1, 5, 5), thickness=0)


def test_draw_boxes_draws_every_box() -> None:
    image = _blank()
    draw_boxes(image, [(1, 1, 5, 5), (100, 50, 120, 80)], (255, 255, 255))
    assert tuple(image[1, 3]) == (255, 255, 255)
    assert tuple(image[50, 110]) == (255, 255, 255)


def test_overlay_frame_uses_the_documented_colours() -> None:
    camera = CameraDetections(
        name="CAM_FRONT",
        boxes=np.array([[1.0, 1.0, 9.0, 9.0]]),
        confidences=np.array([0.9]),
        class_names=("car",),
    )
    overlaid = overlay_frame(
        _blank(), camera, projected=[(20, 20, 40, 40)], fused=[(20, 20, 40, 40)]
    )

    assert tuple(overlaid[1, 5]) == BOX_COLOURS["camera"]
    # fused is drawn last, so it wins where both overlap
    assert tuple(overlaid[20, 30]) == BOX_COLOURS["fused"]
    assert tuple(overlaid[40, 30]) == BOX_COLOURS["fused"]


def test_overlay_frame_does_not_modify_the_input() -> None:
    original = _blank()
    before = original.copy()
    overlay_frame(original, projected=[(10, 10, 50, 50)])
    assert np.array_equal(original, before)


def test_overlay_frame_with_no_objects_returns_an_equal_copy() -> None:
    original = _blank()
    result = overlay_frame(original)
    assert np.array_equal(result, original)
    assert result is not original


def test_overlay_frame_skips_an_empty_camera_detection() -> None:
    empty = CameraDetections.empty("CAM_FRONT")
    result = overlay_frame(_blank(), empty)
    assert not result.any()


def test_overlay_frame_rejects_a_wrong_image_shape() -> None:
    with pytest.raises(ValueError, match="H, W, 3"):
        overlay_frame(np.zeros((10, 10), dtype=np.uint8))


def test_show_image_sets_the_title_and_hides_the_axes() -> None:
    _, ax = plt.subplots()
    show_image(ax, _blank(), title="frame 0")
    assert ax.get_title() == "frame 0"
    assert not ax.axison
