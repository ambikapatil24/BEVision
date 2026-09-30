"""Tests for centre-NMS duplicate suppression."""

from __future__ import annotations

import numpy as np
import pytest

from bevision.nms import center_nms, center_nms_indices
from bevision.types import Detection3D


def _detection(
    label: int, name: str, score: float, center: tuple[float, float, float]
) -> Detection3D:
    return Detection3D(
        label=label,
        class_name=name,
        score=score,
        center_global=np.array(center, dtype=float),
        size=(2.0, 4.0, 1.5),
        yaw_global=0.0,
    )


def test_empty_input_returns_no_indices() -> None:
    assert center_nms_indices(np.empty((0, 3)), np.empty((0,)), np.empty((0,)), 2.0).size == 0


def test_keeps_the_highest_scoring_box_in_a_cluster() -> None:
    centers = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.2, 0.1, 0.0]]
    scores = [0.2, 0.9, 0.5]
    labels = [0, 0, 0]

    kept = center_nms_indices(centers, scores, labels, radius_m=2.0)

    assert kept.tolist() == [1]


def test_boxes_further_apart_than_the_radius_are_all_kept() -> None:
    centers = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
    kept = center_nms_indices(centers, [0.9, 0.8, 0.7], [0, 0, 0], radius_m=2.0)
    assert kept.tolist() == [0, 1, 2]


def test_different_classes_co_located_are_both_kept() -> None:
    """Suppression is class-conditional: a car and a pedestrian can share a position."""
    centers = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
    kept = center_nms_indices(centers, [0.9, 0.8], [0, 7], radius_m=2.0)
    assert sorted(kept.tolist()) == [0, 1]


def test_duplicate_chain_is_only_partially_suppressed() -> None:
    """Centre-NMS is not transitive, and this pins that as known behaviour.

    Each box is compared only against boxes that were *kept*, never against ones that
    were suppressed. So a chain spaced just under the radius survives at both ends:
    box 3 sits 1.5 m from the kept box 0, beyond the 1.0 m radius, even though it is
    only 0.5 m from the suppressed box 2.

    Duplicates that matter in practice are near-coincident, so this is acceptable --
    but it is a real limitation and the notebook's implementation behaves identically,
    which is why the replacement stays faithful rather than "improving" it.
    """
    centers = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
    kept = center_nms_indices(centers, [0.9, 0.8, 0.7, 0.6], [0, 0, 0, 0], radius_m=1.0)
    assert kept.tolist() == [0, 3]


def test_near_coincident_duplicates_do_collapse() -> None:
    """The case that actually occurs: several boxes on the same object."""
    centers = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [0.15, -0.1, 0.0]]
    kept = center_nms_indices(centers, [0.9, 0.8, 0.7, 0.6], [0, 0, 0, 0], radius_m=1.0)
    assert kept.tolist() == [0]


def test_z_height_is_ignored() -> None:
    """Suppression works in the ground plane, so differing z must not save a duplicate."""
    centers = [[0.0, 0.0, 0.0], [0.0, 0.0, 50.0]]
    kept = center_nms_indices(centers, [0.9, 0.8], [0, 0], radius_m=2.0)
    assert kept.tolist() == [0]


def test_results_are_ordered_by_descending_score() -> None:
    centers = [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [40.0, 0.0, 0.0]]
    kept = center_nms_indices(centers, [0.3, 0.9, 0.6], [0, 0, 0], radius_m=2.0)
    assert kept.tolist() == [1, 2, 0]


def test_equal_scores_keep_original_order_for_reproducibility() -> None:
    centers = [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]
    kept = center_nms_indices(centers, [0.5, 0.5], [0, 0], radius_m=2.0)
    assert kept.tolist() == [0, 1]


def test_zero_radius_still_removes_exact_duplicates() -> None:
    centers = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    kept = center_nms_indices(centers, [0.9, 0.5], [0, 0], radius_m=0.0)
    assert kept.tolist() == [0]


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        center_nms_indices([[0.0, 0.0, 0.0]], [0.5], [0, 1], radius_m=1.0)


def test_negative_radius_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        center_nms_indices([[0.0, 0.0, 0.0]], [0.5], [0], radius_m=-1.0)


def test_object_level_helper_returns_detections() -> None:
    detections = [
        _detection(0, "car", 0.9, (0.0, 0.0, 0.0)),
        _detection(0, "car", 0.4, (0.3, 0.0, 0.0)),
        _detection(0, "car", 0.7, (30.0, 0.0, 0.0)),
    ]
    kept = center_nms(detections, radius_m=2.0)
    assert [d.score for d in kept] == [0.9, 0.7]


def test_object_level_helper_handles_empty_sequence() -> None:
    assert center_nms([], radius_m=2.0) == []
