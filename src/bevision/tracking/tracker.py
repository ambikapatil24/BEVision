"""Track management and frame-to-frame association.

Association is Hungarian matching on 3D centre distance, optionally constrained to same-class
pairs. The constraint matters because the emitted class comes from the matched detection: if a
track could be updated by a detection of a different class, that update would relabel the track.
nuScenes scores tracking per class, so a relabelled track is simultaneously a false negative for
its previous class and a false positive for its new one. ``config.class_aware_track_matching``
selects between the two behaviours.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bevision.config import PerceptionConfig
from bevision.geometry import pairwise_distance_matrix
from bevision.tracking.kalman import ConstantVelocityKalmanFilter
from bevision.types import Detection3D, FrameStats

IntArray = NDArray[np.intp]

#: Cost assigned to class-mismatched pairs so the optimizer never spends a match on
#: them. Finite rather than ``inf`` because ``linear_sum_assignment`` raises on an
#: infeasible ``inf`` cost matrix -- for example when a track has no detection of its
#: own class to pair with, which happens routinely with sparse classes.
FORBIDDEN_PAIR_COST: float = 1e6


def associate(
    track_positions: ArrayLike,
    track_labels: ArrayLike,
    detection_positions: ArrayLike,
    detection_labels: ArrayLike,
    max_distance_m: float,
    *,
    class_aware: bool = True,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match tracks to detections by 3D centre distance.

    Returns ``(matched_pairs, unmatched_track_indices, unmatched_detection_indices)``
    where ``matched_pairs`` holds ``(track_index, detection_index)`` tuples.
    """
    from scipy.optimize import linear_sum_assignment

    n_tracks = len(track_positions)
    n_detections = len(detection_positions)
    if n_tracks == 0 or n_detections == 0:
        return [], list(range(n_tracks)), list(range(n_detections))

    cost = pairwise_distance_matrix(track_positions, detection_positions)

    if class_aware:
        track_labels = np.asarray(track_labels)[:, None]
        detection_labels = np.asarray(detection_labels)[None, :]
        cost = np.where(track_labels == detection_labels, cost, FORBIDDEN_PAIR_COST)

    rows, cols = linear_sum_assignment(cost)

    matched: list[tuple[int, int]] = []
    unmatched_tracks = list(range(n_tracks))
    unmatched_detections = list(range(n_detections))
    for row, col in zip(rows, cols, strict=True):
        if cost[row, col] < max_distance_m:
            matched.append((int(row), int(col)))
            unmatched_tracks.remove(int(row))
            unmatched_detections.remove(int(col))

    return matched, unmatched_tracks, unmatched_detections


class Track3D:
    """A single tracked object, owning its own motion filter and last box."""

    def __init__(
        self,
        track_id: int,
        detection: Detection3D,
        config: PerceptionConfig,
    ) -> None:
        self.track_id = track_id
        self.label = detection.label
        self.filter = ConstantVelocityKalmanFilter(
            detection.center_global,
            config.sample_period_s,
            config.kalman,
        )
        self.last_detection: Detection3D = detection
        self.time_since_update = 0
        self.age = 1
        self.hits = 1

    def predict(self) -> None:
        """Advance the filter one sample and age the track."""
        self.filter.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection3D) -> None:
        """Fold a matched detection into the filter and refresh the emitted box."""
        self.filter.update(detection.center_global)
        self.last_detection = detection
        self.label = detection.label
        self.time_since_update = 0
        self.hits += 1

    @property
    def position(self) -> NDArray[np.floating]:
        """Filtered centre, used for the next association step."""
        return self.filter.position

    @property
    def velocity(self) -> NDArray[np.floating]:
        """Filtered velocity in the global frame."""
        return self.filter.velocity

    @property
    def class_name(self) -> str:
        """Class name carried by the most recent detection."""
        return self.last_detection.class_name

    def is_alive(self, max_age: int) -> bool:
        """Whether the track is still within its coasting budget."""
        return self.time_since_update < max_age


class Tracker:
    """Multi-object tracker over global-frame detections.

    Call :meth:`step` once per sample. Call :meth:`reset` at scene boundaries so
    identities never leak between different drives.
    """

    def __init__(self, config: PerceptionConfig | None = None) -> None:
        self.config = config or PerceptionConfig()
        self.tracks: list[Track3D] = []
        self.next_track_id = 0
        self._frame_index = 0

    def reset(self) -> None:
        """Drop all tracks and restart identity numbering."""
        self.tracks = []
        self.next_track_id = 0
        self._frame_index = 0

    def step(self, detections: Sequence[Detection3D]) -> tuple[list[Track3D], FrameStats]:
        """Advance one sample: predict, associate, update, spawn, prune.

        Returns the surviving tracks (the frame's output) and per-frame counters.
        """
        detections = list(detections)
        config = self.config

        for track in self.tracks:
            track.predict()

        matched, _, unmatched_detections = associate(
            [t.position for t in self.tracks],
            [t.label for t in self.tracks],
            [d.center_global for d in detections],
            [d.label for d in detections],
            config.track_match_distance_m,
            class_aware=config.class_aware_track_matching,
        )

        for track_index, detection_index in matched:
            self.tracks[track_index].update(detections[detection_index])

        spawned = 0
        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            if self._may_spawn(detection):
                self.tracks.append(Track3D(self.next_track_id, detection, config))
                self.next_track_id += 1
                spawned += 1

        self.tracks = [t for t in self.tracks if t.is_alive(config.track_max_age)]

        stats = FrameStats(
            frame_index=self._frame_index,
            lidar_detections=len(detections),
            camera_detections=sum(1 for d in detections if d.camera_confirmed),
            tracks_active=len(self.tracks),
            tracks_spawned=spawned,
            matches=len(matched),
        )
        self._frame_index += 1
        return list(self.tracks), stats

    def _may_spawn(self, detection: Detection3D) -> bool:
        """Whether an unmatched detection is allowed to start a new track.

        The camera-confirmation requirement is configurable because it controls whether
        a detection must be camera-confirmed before it can start a new track.
        """
        if detection.score <= self.config.spawn_score_threshold:
            return False
        return detection.camera_confirmed or not self.config.spawn_requires_camera_confirmation
