"""mmdetection3d PointPillars adapter: one LiDAR sweep in, 3D boxes in the sensor frame out.

> **Not verified in CI.** Needs ``mmdetection3d`` and a checkpoint, which cannot be installed
> portably (see ``scripts/install_mmdet3d.sh``). The box-ordering conversion below is a plain
> function and *is* unit-tested.

**Box ordering.** ``LiDARInstance3DBoxes.tensor`` stores ``(x, y, z, dx, dy, dz, yaw)`` where
``dx`` is the extent along the box's own x-axis — the **length**. nuScenes' submission schema
expects ``size = (width, length, height)``. The two differ by a swap, so the size field is
reordered; see :func:`raw_detections_from_tensors`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from bevision.types import RawDetection3D


def raw_detections_from_tensors(
    boxes: ArrayLike,
    scores: ArrayLike,
    labels: ArrayLike,
) -> list[RawDetection3D]:
    """Convert mmdet3d's box tensor into :class:`RawDetection3D` objects.

    The size field is reordered from mmdet3d's ``(dx, dy, dz)`` = ``(length, width, height)``
    into nuScenes' ``(width, length, height)``. Split out from the model wrapper so the size
    ordering can be unit-tested without mmdetection3d installed.

    **Box origin.** mmdet3d anchors LiDAR boxes at their **bottom face**
    (``origin = (0.5, 0.5, 0)``), so ``tensor[:, 2]`` is the bottom of the box, not its centre.
    The geometric centre is therefore ``z + dz/2``, and that is what this function returns.
    Reading the bottom face as the centre is invisible in the centres' xy and in the box size,
    but it shifts every projected box vertically by half its height, which is enough to push
    projected boxes clear of the camera detections entirely.

    **Tensor layout.** mmdet3d emits either 7 or 9 columns depending on version and config::

        7:  (x, y, z, dx, dy, dz, yaw)
        9:  (x, y, z, dx, dy, dz, yaw, vx, vy)   <- velocity appended

    Both share the leading seven, so the trailing two are dropped and the remaining tensor must
    be ``(N, 7)``; any other column count raises.
    """
    boxes = np.asarray(boxes, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] not in (7, 9):
        raise ValueError(
            f"expected an (N, 7) or (N, 9) mmdet3d box tensor, got shape {boxes.shape}"
        )
    boxes = boxes[:, :7]

    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if not (boxes.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError(
            f"boxes, scores and labels must have the same length, got "
            f"{boxes.shape[0]}, {scores.shape[0]}, {labels.shape[0]}"
        )

    return [
        RawDetection3D(
            label=int(label),
            score=float(score),
            # z is the bottom face; lift by half the height to get the centre
            center=np.array([box[0], box[1], box[2] + box[5] / 2.0], dtype=float),
            size=(float(box[4]), float(box[3]), float(box[5])),
            yaw=float(box[6]),
        )
        for box, score, label in zip(boxes, scores, labels, strict=True)
    ]


class MMDet3DLidarDetector:
    """Wraps an mmdetection3d model as a :class:`~bevision.detection.base.LidarDetector`.

    Args:
        config_path: Model config, e.g. ``pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py``.
        checkpoint_path: Matching ``.pth`` checkpoint.
        device: Torch device string.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda:0",
    ) -> None:
        from bevision._deps import require  # noqa: PLC0415

        require("mmdet3d", "running 3D LiDAR detection")
        from mmdet3d.apis import init_model  # lazy: heavy, and optional

        self.model = init_model(config_path, checkpoint_path, device=device)
        self.class_names = self._read_class_names()

    def _read_class_names(self) -> tuple[str, ...]:
        """Read the model's declared class ordering, validated against nuScenes.

        A silently reordered class list decodes every label to the wrong name, and nothing
        crashes -- so :func:`bevision.classes.resolve_class_names` is used to fail loudly
        instead.
        """
        from bevision.classes import resolve_class_names

        declared = self.model.dataset_meta.get("classes")
        if declared is None:
            raise RuntimeError("model has no dataset_meta['classes']; cannot label detections")
        return resolve_class_names(declared)

    def __call__(self, lidar_path: str) -> list[RawDetection3D]:
        """Run inference on one sweep and return boxes in the LiDAR sensor frame."""
        from mmdet3d.apis import inference_detector

        result, _ = inference_detector(self.model, lidar_path)
        instances = result.pred_instances_3d
        return raw_detections_from_tensors(
            instances.bboxes_3d.tensor.cpu().numpy(),
            instances.scores_3d.cpu().numpy(),
            instances.labels_3d.cpu().numpy(),
        )
