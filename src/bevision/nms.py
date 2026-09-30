"""Duplicate suppression for LiDAR detections in the ground plane.

Centre-NMS rather than IoU-NMS because 3D IoU is expensive and, for co-located
duplicates of the same class, the box centres are already within a metre or two
of each other. Merging them before fusion matters: a duplicate pair that reaches
the tracker becomes two tracks competing for the same object, which shows up as
identity switches rather than as an obvious error.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bevision.types import Detection3D

IntArray = NDArray[np.intp]
FloatArray = NDArray[np.floating]


def center_nms_indices(
    centers: ArrayLike,
    scores: ArrayLike,
    labels: ArrayLike,
    radius_m: float,
) -> IntArray:
    """Return the indices of detections to keep, highest score first.

    A detection is suppressed when an already-kept detection shares its label and
    its xy-centre lies within ``radius_m``. Iterating in descending score order
    means the most confident box in each cluster wins.

    Ties in score keep their original relative order, so the result is stable and
    reproducible across runs.
    """
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)[:, :2]
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels)

    if centers.shape[0] != scores.shape[0] or scores.shape[0] != labels.shape[0]:
        raise ValueError("centers, scores and labels must have the same length")
    if radius_m < 0:
        raise ValueError("radius_m must be non-negative")
    if scores.size == 0:
        return np.empty(0, dtype=np.intp)

    order = np.argsort(-scores, kind="stable")

    kept: list[int] = []
    kept_centers: list[FloatArray] = []
    kept_labels: list[object] = []

    for index in order:
        if kept_centers:
            same_class = np.fromiter(
                (label == labels[index] for label in kept_labels),
                dtype=bool,
                count=len(kept_labels),
            )
            if same_class.any():
                deltas = np.asarray(kept_centers)[same_class] - centers[index]
                if np.any(np.linalg.norm(deltas, axis=1) <= radius_m):
                    continue
        kept.append(int(index))
        kept_centers.append(centers[index])
        kept_labels.append(labels[index])

    return np.asarray(kept, dtype=np.intp)


def center_nms(detections: Sequence[Detection3D], radius_m: float) -> list[Detection3D]:
    """Apply :func:`center_nms_indices` to a sequence of detections."""
    items = list(detections)
    if not items:
        return []
    centers = np.asarray([d.center_global for d in items], dtype=float)
    scores = np.asarray([d.score for d in items], dtype=float)
    labels = np.asarray([d.label for d in items])
    return [items[i] for i in center_nms_indices(centers, scores, labels, radius_m)]
