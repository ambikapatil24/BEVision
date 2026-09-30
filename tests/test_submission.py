"""Tests for submission construction and validation.

These matter more than they look: the submission schema is where a refactor can silently
change results. One `continue` statement in the original code dropped three classes from the
detection metrics, and nothing failed -- it just scored 0.000 on them. So the schema, the
class sets and the token coverage are all pinned here.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from bevision.classes import TRACKING_CLASS_NAMES
from bevision.config import DetectionClassPolicy
from bevision.evaluation.submission import (
    SUBMISSION_META,
    build_detection_submission,
    build_tracking_submission,
    detection_entry,
    load_submission,
    tracking_entry,
    validate_submission,
    write_submission,
    yaw_to_quaternion,
)
from bevision.types import Box3D, TrackOutput


def _track(
    track_id: int = 0,
    class_name: str = "car",
    *,
    score: float = 0.8,
    center=(1.0, 2.0, 3.0),
    size=(2.0, 4.0, 1.5),
    yaw: float = 0.0,
    velocity=(0.5, -0.25),
) -> TrackOutput:
    return TrackOutput(
        track_id=track_id,
        class_name=class_name,
        score=score,
        box=Box3D(center=np.array(center, dtype=float), size=size, yaw=yaw),
        velocity=velocity,
    )


# ---------------------------------------------------------------------------
# Quaternion
# ---------------------------------------------------------------------------
def test_yaw_to_quaternion_for_zero_yaw() -> None:
    assert yaw_to_quaternion(0.0) == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_yaw_to_quaternion_is_a_unit_quaternion() -> None:
    for yaw in np.linspace(-math.pi, math.pi, 33):
        assert np.linalg.norm(yaw_to_quaternion(float(yaw))) == pytest.approx(1.0)


def test_yaw_to_quaternion_rotates_about_z() -> None:
    assert yaw_to_quaternion(math.pi) == pytest.approx((0.0, 0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Per-box entries
# ---------------------------------------------------------------------------
def test_detection_entry_has_the_required_schema() -> None:
    entry = detection_entry("tok", _track(class_name="truck", yaw=0.3))
    assert set(entry) == {
        "sample_token",
        "translation",
        "size",
        "rotation",
        "velocity",
        "detection_name",
        "attribute_name",
        "detection_score",
    }
    assert entry["detection_name"] == "truck"
    assert entry["attribute_name"] == "vehicle.moving"
    assert entry["sample_token"] == "tok"
    assert len(entry["rotation"]) == 4


def test_detection_entry_rejects_an_unknown_class() -> None:
    with pytest.raises(ValueError, match="unknown nuScenes class"):
        detection_entry("tok", _track(class_name="spaceship"))


def test_tracking_entry_uses_tracking_keys() -> None:
    entry = tracking_entry("tok", _track(class_name="bus", track_id=7))
    assert entry["tracking_id"] == "7"
    assert entry["tracking_name"] == "bus"
    assert entry["tracking_score"] == pytest.approx(0.8)
    assert "detection_name" not in entry


def test_tracking_id_is_a_string_as_the_schema_requires() -> None:
    assert isinstance(tracking_entry("tok", _track(track_id=42))["tracking_id"], str)


def test_tracking_entry_rejects_an_untracked_class() -> None:
    """nuScenes tracking scores seven classes; a barrier box cannot be submitted."""
    with pytest.raises(ValueError, match="not tracked"):
        tracking_entry("tok", _track(class_name="barrier"))


# ---------------------------------------------------------------------------
# Submissions
# ---------------------------------------------------------------------------
def test_every_sample_token_appears_even_with_no_tracks() -> None:
    tokens = ["a", "b", "c"]
    results = build_detection_submission({"b": [_track()]}, tokens)
    assert list(results) == tokens
    assert results["a"] == []
    assert results["c"] == []
    assert len(results["b"]) == 1


def test_detection_class_policy_all_keeps_ten_classes() -> None:
    tracks = [_track(class_name=name) for name in ("car", "barrier", "traffic_cone")]
    results = build_detection_submission({"t": tracks}, ["t"], DetectionClassPolicy.ALL)
    assert [e["detection_name"] for e in results["t"]] == ["car", "barrier", "traffic_cone"]


def test_detection_class_policy_tracking_restricts_to_seven() -> None:
    tracks = [_track(class_name=name) for name in ("car", "barrier", "traffic_cone", "bus")]
    results = build_detection_submission({"t": tracks}, ["t"], DetectionClassPolicy.TRACKING)
    assert [e["detection_name"] for e in results["t"]] == ["car", "bus"]


def test_default_policy_does_not_drop_three_classes() -> None:
    """The defect this module exists to prevent.

    With the two policies conflated, `barrier`, `traffic_cone` and `construction_vehicle`
    were silently absent from the detection submission, so they scored AP 0.000 and capped
    mAP. The default must submit all ten.
    """
    every_class = [
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    ]
    tracks = [_track(class_name=name) for name in every_class]
    results = build_detection_submission({"t": tracks}, ["t"])

    submitted = {e["detection_name"] for e in results["t"]}
    assert submitted == set(every_class)


def test_tracking_submission_filters_silently() -> None:
    """Unlike the per-box builder, a tracker may legitimately emit untracked classes."""
    tracks = [_track(class_name="car"), _track(class_name="barrier", track_id=1)]
    results = build_tracking_submission({"t": tracks}, ["t"])
    assert [e["tracking_name"] for e in results["t"]] == ["car"]


def test_tracking_submission_covers_every_token() -> None:
    results = build_tracking_submission({}, ["a", "b"])
    assert results == {"a": [], "b": []}


# ---------------------------------------------------------------------------
# Round trip and validation
# ---------------------------------------------------------------------------
def test_write_then_load_round_trip(tmp_path) -> None:
    tracks = {"t": [_track(class_name="pedestrian")]}
    detection = build_detection_submission(tracks, ["t"])
    path = write_submission(tmp_path / "nested" / "det.json", detection)

    meta, loaded = load_submission(path)
    assert meta == SUBMISSION_META
    assert loaded["t"][0]["detection_name"] == "pedestrian"
    assert loaded["t"][0]["translation"] == [1.0, 2.0, 3.0]


def test_write_submission_creates_parent_directories(tmp_path) -> None:
    path = write_submission(tmp_path / "a" / "b" / "c.json", {"t": []})
    assert path.exists()


def test_written_file_is_valid_json_with_both_top_level_keys(tmp_path) -> None:
    path = write_submission(tmp_path / "det.json", {"t": []})
    blob = json.loads(path.read_text())
    assert set(blob) == {"meta", "results"}


def test_submission_meta_claims_camera_and_lidar() -> None:
    """The original submission declared use_camera=False for a camera-fusion system."""
    assert SUBMISSION_META["use_camera"] is True
    assert SUBMISSION_META["use_lidar"] is True
    assert SUBMISSION_META["use_radar"] is False


def test_load_submission_rejects_a_non_submission_file(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"results": {}}')
    with pytest.raises(ValueError, match="not a nuScenes submission"):
        load_submission(path)


# ---------------------------------------------------------------------------
# validate_submission
# ---------------------------------------------------------------------------
def test_valid_detection_submission_reports_no_problems() -> None:
    tracks = {name: [_track(class_name=name)] for name in ("car", "truck")}
    tokens = list(tracks)
    assert validate_submission(build_detection_submission(tracks, tokens), tokens) == []


def test_valid_tracking_submission_reports_no_problems() -> None:
    tracks = {"a": [_track(class_name="car")], "b": []}
    tokens = ["a", "b"]
    assert (
        validate_submission(build_tracking_submission(tracks, tokens), tokens, schema="tracking")
        == []
    )


def test_validation_catches_a_missing_token() -> None:
    problems = validate_submission({"a": []}, ["a", "b"])
    assert any("absent" in p for p in problems)


def test_validation_catches_a_wrong_class_for_the_schema() -> None:
    """A barrier box is legal for detection but not for tracking."""
    results = {"t": [dict(tracking_entry("t", _track(class_name="car")))]}
    results["t"][0]["tracking_name"] = "barrier"
    problems = validate_submission(results, ["t"], schema="tracking")
    assert any("invalid tracking_name" in p for p in problems)


def test_validation_reports_a_detection_entry_as_missing_tracking_keys() -> None:
    """Schema mismatches surface as missing keys, which is the first thing checked."""
    tracks = {"t": [_track(class_name="car")]}
    detection = build_detection_submission(tracks, ["t"])
    problems = validate_submission(detection, ["t"], schema="tracking")
    assert any("missing keys" in p and "tracking_id" in p for p in problems)


def test_validation_catches_a_mismatched_sample_token() -> None:
    tracks = {"t": [_track()]}
    results = build_detection_submission(tracks, ["t"])
    results["t"][0]["sample_token"] = "other"
    problems = validate_submission(results, ["t"])
    assert any("sample_token is" in p for p in problems)


def test_validation_catches_a_missing_key() -> None:
    tracks = {"t": [_track()]}
    results = build_detection_submission(tracks, ["t"])
    del results["t"][0]["size"]
    problems = validate_submission(results, ["t"])
    assert any("missing keys" in p for p in problems)


def test_validation_catches_a_bad_quaternion_length() -> None:
    tracks = {"t": [_track()]}
    results = build_detection_submission(tracks, ["t"])
    results["t"][0]["rotation"] = [1.0, 0.0, 0.0]
    problems = validate_submission(results, ["t"])
    assert any("quaternion" in p for p in problems)


def test_validation_catches_an_out_of_range_score() -> None:
    tracks = {"t": [_track(score=1.4)]}
    results = build_detection_submission(tracks, ["t"])
    problems = validate_submission(results, ["t"])
    assert any("outside [0, 1]" in p for p in problems)


def test_validation_rejects_an_unknown_schema() -> None:
    with pytest.raises(ValueError, match="schema must be"):
        validate_submission({}, [], schema="segmentation")


def test_validation_accepts_every_tracked_class() -> None:
    tracks = {"t": [_track(class_name=name) for name in sorted(TRACKING_CLASS_NAMES)]}
    problems = validate_submission(
        build_tracking_submission(tracks, ["t"]), ["t"], schema="tracking"
    )
    assert problems == []
