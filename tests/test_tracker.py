"""Tests for association and track lifecycle.

The centrepiece is ``test_class_aware_matching_prevents_relabelling``, which reproduces
the defect that cost the original pipeline 3.2x AMOTA: a car track being updated by a
nearer truck detection and silently becoming a truck.
"""

from __future__ import annotations

import numpy as np
import pytest

from bevision.config import PerceptionConfig
from bevision.tracking.tracker import Tracker, associate
from bevision.types import Detection3D

CAR, TRUCK, PEDESTRIAN = 0, 1, 7


def _detection(
    label: int,
    name: str,
    center: tuple[float, float, float],
    *,
    score: float = 0.9,
    camera_confirmed: bool = True,
) -> Detection3D:
    return Detection3D(
        label=label,
        class_name=name,
        score=score,
        center_global=np.array(center, dtype=float),
        size=(2.0, 4.0, 1.5),
        yaw_global=0.0,
        camera_confirmed=camera_confirmed,
    )


def _config(**overrides) -> PerceptionConfig:
    return PerceptionConfig(**overrides)


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------
def test_associate_matches_nearest_within_the_gate() -> None:
    matched, unmatched_tracks, unmatched_detections = associate(
        [[0.0, 0.0, 0.0]], [CAR], [[1.0, 0.0, 0.0]], [CAR], max_distance_m=3.0
    )
    assert matched == [(0, 0)]
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_associate_rejects_matches_beyond_the_gate() -> None:
    matched, unmatched_tracks, unmatched_detections = associate(
        [[0.0, 0.0, 0.0]], [CAR], [[10.0, 0.0, 0.0]], [CAR], max_distance_m=3.0
    )
    assert matched == []
    assert unmatched_tracks == [0]
    assert unmatched_detections == [0]


def test_associate_handles_empty_inputs() -> None:
    assert associate([], [], [[1.0, 0.0, 0.0]], [CAR], 3.0) == ([], [], [0])
    assert associate([[1.0, 0.0, 0.0]], [CAR], [], [], 3.0) == ([], [0], [])


def test_class_aware_association_forbids_cross_class_pairs() -> None:
    matched, unmatched_tracks, _ = associate(
        [[0.0, 0.0, 0.0]],
        [CAR],
        [[0.5, 0.0, 0.0]],
        [TRUCK],
        max_distance_m=3.0,
        class_aware=True,
    )
    assert matched == []
    assert unmatched_tracks == [0]


def test_class_blind_association_allows_cross_class_pairs() -> None:
    matched, _, _ = associate(
        [[0.0, 0.0, 0.0]],
        [CAR],
        [[0.5, 0.0, 0.0]],
        [TRUCK],
        max_distance_m=3.0,
        class_aware=False,
    )
    assert matched == [(0, 0)]


def test_class_aware_association_survives_a_track_with_no_same_class_candidate() -> None:
    """A finite penalty is required: an inf cost matrix is infeasible for scipy.

    Track 0 is a car with no car detection to pair with, so it must take a forbidden
    entry rather than crash the assignment, and end up unmatched.
    """
    matched, unmatched_tracks, unmatched_detections = associate(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],  # car, truck -- both near the detection
        [CAR, TRUCK],
        [[0.6, 0.0, 0.0]],
        [TRUCK],
        max_distance_m=3.0,
        class_aware=True,
    )
    assert matched == [(1, 0)]
    assert unmatched_tracks == [0]
    assert unmatched_detections == []


def test_association_is_one_to_one() -> None:
    matched, _, _ = associate(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [CAR, CAR],
        [[0.4, 0.0, 0.0]],
        [CAR],
        max_distance_m=3.0,
    )
    assert len({row for row, _ in matched}) == len(matched)
    assert len({col for _, col in matched}) == len(matched)


# ---------------------------------------------------------------------------
# The relabelling bug
# ---------------------------------------------------------------------------
def test_class_aware_matching_prevents_relabelling() -> None:
    """A car track must not adopt a nearer truck detection's class."""
    tracker = Tracker(_config(class_aware_track_matching=True))
    tracker.step([_detection(CAR, "car", (0.0, 0.0, 0.0))])

    tracks, _ = tracker.step(
        [
            _detection(TRUCK, "truck", (0.5, 0.0, 0.0)),  # nearer, wrong class
            _detection(CAR, "car", (2.0, 0.0, 0.0)),  # farther, right class
        ]
    )

    car_tracks = [t for t in tracks if t.track_id == 0]
    assert len(car_tracks) == 1
    assert car_tracks[0].class_name == "car"


def test_class_blind_matching_relabels_a_track() -> None:
    """Documents the original behaviour, so the fix stays measurable.

    With no class constraint the optimizer takes the nearest detection, so the car
    track becomes a truck. Because nuScenes tracking scores class by class, that
    single update turns one ground-truth car into both a false negative and a phantom
    truck -- the mechanism behind the 1061 -> 546 false-positive drop.
    """
    tracker = Tracker(_config(class_aware_track_matching=False))
    tracker.step([_detection(CAR, "car", (0.0, 0.0, 0.0))])

    tracks, _ = tracker.step(
        [
            _detection(TRUCK, "truck", (0.5, 0.0, 0.0)),
            _detection(CAR, "car", (2.0, 0.0, 0.0)),
        ]
    )

    relabelled = [t for t in tracks if t.track_id == 0]
    assert len(relabelled) == 1
    assert relabelled[0].class_name == "truck"


# ---------------------------------------------------------------------------
# Spawn policy
# ---------------------------------------------------------------------------
def test_low_score_detections_cannot_spawn() -> None:
    tracker = Tracker(_config(spawn_score_threshold=0.3))
    tracks, stats = tracker.step([_detection(CAR, "car", (0.0, 0.0, 0.0), score=0.2)])
    assert tracks == []
    assert stats.tracks_spawned == 0


def test_camera_unconfirmed_detections_cannot_spawn_by_default() -> None:
    tracker = Tracker(_config())
    tracks, _ = tracker.step([_detection(CAR, "car", (0.0, 0.0, 0.0), camera_confirmed=False)])
    assert tracks == []


def test_relaxing_camera_confirmation_allows_lidar_only_spawn() -> None:
    tracker = Tracker(_config(spawn_requires_camera_confirmation=False))
    tracks, _ = tracker.step([_detection(CAR, "car", (0.0, 0.0, 0.0), camera_confirmed=False)])
    assert len(tracks) == 1


def test_matched_detections_do_not_spawn_duplicate_tracks() -> None:
    tracker = Tracker(_config())
    tracker.step([_detection(CAR, "car", (0.0, 0.0, 0.0))])
    tracks, stats = tracker.step([_detection(CAR, "car", (0.4, 0.0, 0.0))])
    assert len(tracks) == 1
    assert stats.tracks_spawned == 0


def test_track_ids_are_never_reused() -> None:
    tracker = Tracker(_config())
    tracker.step([_detection(CAR, "car", (0.0, 0.0, 0.0))])
    tracker.step([_detection(CAR, "car", (100.0, 0.0, 0.0))])
    assert tracker.next_track_id == 2


# ---------------------------------------------------------------------------
# Track lifetime
# ---------------------------------------------------------------------------
def test_track_coasts_for_max_age_frames_then_dies() -> None:
    """A track with no further detections is emitted for exactly ``track_max_age`` frames."""
    tracker = Tracker(_config(track_max_age=5))
    frames_alive = 0
    if tracker.step([_detection(CAR, "car", (0.0, 0.0, 0.0))])[0]:
        frames_alive += 1
    for _ in range(10):
        tracks, _ = tracker.step([])
        if tracks:
            frames_alive += 1
    assert frames_alive == 5


def test_shorter_max_age_drops_the_track_sooner() -> None:
    tracker = Tracker(_config(track_max_age=2))
    frames_alive = int(bool(tracker.step([_detection(CAR, "car", (0.0, 0.0, 0.0))])[0]))
    for _ in range(10):
        tracks, _ = tracker.step([])
        frames_alive += int(bool(tracks))
    assert frames_alive == 2


def test_persistently_detected_track_survives_indefinitely() -> None:
    tracker = Tracker(_config(track_max_age=1))
    for step in range(30):
        tracks, _ = tracker.step([_detection(CAR, "car", (step * 0.5, 0.0, 0.0))])
        assert len(tracks) == 1


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
def test_reset_clears_tracks_and_identity_counter() -> None:
    tracker = Tracker(_config())
    tracker.step([_detection(CAR, "car", (0.0, 0.0, 0.0))])
    assert tracker.next_track_id == 1

    tracker.reset()

    assert tracker.tracks == []
    assert tracker.next_track_id == 0
    tracks, _ = tracker.step([_detection(CAR, "car", (0.0, 0.0, 0.0))])
    assert tracks[0].track_id == 0


def test_stats_reported_per_frame() -> None:
    tracker = Tracker(_config())
    _, stats = tracker.step(
        [
            _detection(CAR, "car", (0.0, 0.0, 0.0)),
            _detection(CAR, "car", (50.0, 0.0, 0.0)),
        ]
    )
    assert stats.frame_index == 0
    assert stats.lidar_detections == 2
    assert stats.tracks_spawned == 2
    assert stats.tracks_active == 2
    assert stats.matches == 0


def test_track_age_and_hits_advance() -> None:
    tracker = Tracker(_config())
    tracker.step([_detection(CAR, "car", (0.0, 0.0, 0.0))])
    track = tracker.tracks[0]
    assert (track.age, track.hits, track.time_since_update) == (1, 1, 0)

    tracker.step([_detection(CAR, "car", (0.2, 0.0, 0.0))])
    track = tracker.tracks[0]
    assert (track.age, track.hits, track.time_since_update) == (2, 2, 0)

    tracker.step([])
    track = tracker.tracks[0]
    assert (track.age, track.hits, track.time_since_update) == (3, 2, 1)


def test_filter_position_is_used_for_association() -> None:
    """The token emitted box follows the filter, not the raw detection."""
    tracker = Tracker(_config())
    tracker.step([_detection(CAR, "car", (0.0, 0.0, 0.0))])
    tracker.step([_detection(CAR, "car", (1.0, 0.0, 0.0))])
    position = tracker.tracks[0].position
    assert 0.0 < position[0] < 1.0
    assert position[0] == pytest.approx(tracker.tracks[0].filter.position[0])
