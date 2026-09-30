"""Build and validate nuScenes submission files.

The two submissions look similar but are not interchangeable:

* **Detection** is scored over all ten classes and keys each box by ``detection_name``.
* **Tracking** is scored over seven classes and keys each box by ``tracking_id``.

Treating them as the same thing silently deletes three classes from the detection metrics,
which is one of the defects catalogued in docs/REFACTORING.md. They are built by separate
functions here, and the class sets are read from :mod:`bevision.classes` rather than
duplicated.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from bevision.classes import (
    DEFAULT_ATTRIBUTE_BY_CLASS,
    NUSCENES_CLASS_NAMES,
    TRACKING_CLASS_NAMES,
)
from bevision.config import DetectionClassPolicy
from bevision.types import TrackOutput

#: Which sensors the pipeline actually used. Reported to the evaluator as metadata; it does
#: not change the computed metrics, but it should describe the system truthfully.
SUBMISSION_META: dict[str, bool] = {
    "use_camera": True,
    "use_lidar": True,
    "use_radar": False,
    "use_map": False,
    "use_external": False,
}

REQUIRED_DETECTION_KEYS = (
    "sample_token",
    "translation",
    "size",
    "rotation",
    "velocity",
    "detection_name",
    "attribute_name",
    "detection_score",
)

REQUIRED_TRACKING_KEYS = (
    "sample_token",
    "translation",
    "size",
    "rotation",
    "velocity",
    "tracking_id",
    "tracking_name",
    "tracking_score",
)


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """Convert a yaw about +z into a ``(w, x, y, z)`` quaternion, as nuScenes expects."""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _common_fields(sample_token: str, track: TrackOutput) -> dict[str, object]:
    return {
        "sample_token": sample_token,
        "translation": [float(v) for v in track.box.center],
        "size": [float(v) for v in track.box.size],
        "rotation": list(yaw_to_quaternion(track.box.yaw)),
        "velocity": [float(track.velocity[0]), float(track.velocity[1])],
    }


def detection_entry(sample_token: str, track: TrackOutput) -> dict[str, object]:
    """One detection-schema box."""
    if track.class_name not in DEFAULT_ATTRIBUTE_BY_CLASS:
        raise ValueError(f"unknown nuScenes class {track.class_name!r}")
    return {
        **_common_fields(sample_token, track),
        "detection_name": track.class_name,
        "attribute_name": DEFAULT_ATTRIBUTE_BY_CLASS[track.class_name],
        "detection_score": float(track.score),
    }


def tracking_entry(sample_token: str, track: TrackOutput) -> dict[str, object]:
    """One tracking-schema box."""
    if track.class_name not in TRACKING_CLASS_NAMES:
        raise ValueError(
            f"{track.class_name!r} is not tracked by nuScenes tracking "
            f"(valid: {', '.join(sorted(TRACKING_CLASS_NAMES))})"
        )
    return {
        **_common_fields(sample_token, track),
        "tracking_id": str(track.track_id),
        "tracking_name": track.class_name,
        "tracking_score": float(track.score),
    }


def build_detection_submission(
    per_sample_tracks: Mapping[str, Sequence[TrackOutput]],
    sample_tokens: Sequence[str],
    class_policy: DetectionClassPolicy = DetectionClassPolicy.ALL,
) -> dict[str, list[dict[str, object]]]:
    """Build the detection results dict, keyed by sample token.

    Every token in ``sample_tokens`` appears in the output, empty list included. The
    evaluator requires the submission to cover the whole split, so a missing token is an
    error; pre-filling means a frame with no detections is explicitly empty rather than
    absent.
    """
    only_tracked_classes = class_policy is DetectionClassPolicy.TRACKING
    results: dict[str, list[dict[str, object]]] = {}

    for token in sample_tokens:
        entries: list[dict[str, object]] = []
        for track in per_sample_tracks.get(token, ()):
            if only_tracked_classes and track.class_name not in TRACKING_CLASS_NAMES:
                continue
            entries.append(detection_entry(token, track))
        results[token] = entries

    return results


def build_tracking_submission(
    per_sample_tracks: Mapping[str, Sequence[TrackOutput]],
    sample_tokens: Sequence[str],
) -> dict[str, list[dict[str, object]]]:
    """Build the tracking results dict, keyed by sample token.

    Classes outside nuScenes' seven tracking classes are dropped here rather than raising,
    because a tracker legitimately may emit them and the submission simply cannot carry
    them.
    """
    results: dict[str, list[dict[str, object]]] = {}
    for token in sample_tokens:
        results[token] = [
            tracking_entry(token, track)
            for track in per_sample_tracks.get(token, ())
            if track.class_name in TRACKING_CLASS_NAMES
        ]
    return results


def write_submission(
    path: str | Path,
    results: Mapping[str, Sequence[Mapping[str, object]]],
    meta: Mapping[str, bool] | None = None,
) -> Path:
    """Write a submission file in nuScenes format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": dict(meta if meta is not None else SUBMISSION_META),
        "results": {token: list(entries) for token, entries in results.items()},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def load_submission(path: str | Path) -> tuple[dict[str, bool], dict[str, list[dict]]]:
    """Read a submission file back, returning ``(meta, results)``."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if "meta" not in payload or "results" not in payload:
        raise ValueError(f"{path} is not a nuScenes submission (needs 'meta' and 'results')")
    return payload["meta"], payload["results"]


def validate_submission(
    results: Mapping[str, Sequence[Mapping[str, object]]],
    sample_tokens: Sequence[str],
    *,
    schema: str = "detection",
) -> list[str]:
    """Check a submission against the parts of the nuScenes schema that fail loudly.

    Returns a list of human-readable problems; an empty list means the submission passed.
    Deliberately returns rather than raises so callers can report every issue at once
    instead of fixing them one exception at a time.
    """
    if schema not in {"detection", "tracking"}:
        raise ValueError("schema must be 'detection' or 'tracking'")

    required = REQUIRED_DETECTION_KEYS if schema == "detection" else REQUIRED_TRACKING_KEYS
    name_key = "detection_name" if schema == "detection" else "tracking_name"
    score_key = "detection_score" if schema == "detection" else "tracking_score"
    valid_names = set(NUSCENES_CLASS_NAMES if schema == "detection" else TRACKING_CLASS_NAMES)

    problems: list[str] = []

    missing_tokens = [t for t in sample_tokens if t not in results]
    if missing_tokens:
        problems.append(
            f"{len(missing_tokens)} of {len(sample_tokens)} sample tokens are absent "
            f"(first: {missing_tokens[0]})"
        )

    for token in sample_tokens:
        for index, entry in enumerate(results.get(token, ())):
            missing_keys = [key for key in required if key not in entry]
            if missing_keys:
                problems.append(f"{token}[{index}]: missing keys {missing_keys}")
                continue

            if entry["sample_token"] != token:
                problems.append(f"{token}[{index}]: sample_token is {entry['sample_token']!r}")
            if entry[name_key] not in valid_names:
                problems.append(f"{token}[{index}]: invalid {name_key} {entry[name_key]!r}")
            if len(entry["translation"]) != 3 or len(entry["size"]) != 3:
                problems.append(f"{token}[{index}]: translation/size must have 3 elements")
            if len(entry["rotation"]) != 4:
                problems.append(f"{token}[{index}]: rotation must be a 4-element quaternion")
            if len(entry["velocity"]) != 2:
                problems.append(f"{token}[{index}]: velocity must have 2 elements")
            if not 0.0 <= float(entry[score_key]) <= 1.0:
                problems.append(f"{token}[{index}]: {score_key} outside [0, 1]")
            if schema == "tracking" and not isinstance(entry["tracking_id"], str):
                problems.append(f"{token}[{index}]: tracking_id must be a string")

    return problems
