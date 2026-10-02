"""Ultralytics YOLO adapter: one camera image in, nuScenes-classed 2D boxes out.

> **Not verified in CI.** Needs the ``ultralytics`` extra and a weights file, so it is
> exercised only where those are present. The COCO → nuScenes mapping it relies on is
> unit-tested in ``tests/test_classes_and_config.py``.
"""

from __future__ import annotations

import numpy as np

from bevision.classes import coco_id_to_nuscenes_name
from bevision.fusion import CameraDetections

#: Ultralytics' own default. Stated explicitly rather than inherited, because the reported
#: metrics depend on it and any change to it would move them.
DEFAULT_CONFIDENCE_THRESHOLD = 0.25


class UltralyticsCameraDetector:
    """Wraps an Ultralytics YOLO model as a :class:`~bevision.detection.base.CameraDetector`.

    Args:
        weights: Path to the ``.pt`` weights (``yolov8n.pt`` for the reported results).
        camera_name: Name recorded on the returned detections.
        confidence_threshold: Passed to the model; see
            :data:`DEFAULT_CONFIDENCE_THRESHOLD`.
        device: Torch device string, or ``None`` to let Ultralytics choose.
        verbose: Ultralytics logging.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        camera_name: str = "CAM_FRONT",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        from bevision._deps import require  # noqa: PLC0415

        require("ultralytics", "running 2D camera detection")
        from ultralytics import YOLO  # lazy: heavy, and optional

        self.model = YOLO(weights)
        self.camera_name = camera_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.verbose = verbose

    def __call__(self, image_path: str) -> CameraDetections:
        """Detect in one image and keep only the classes nuScenes can represent."""
        kwargs: dict[str, object] = {
            "conf": self.confidence_threshold,
            "verbose": self.verbose,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        result = self.model(image_path, **kwargs)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return CameraDetections.empty(self.camera_name)

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        coco_ids = result.boxes.cls.cpu().numpy().astype(int)

        # COCO has 80 classes; nuScenes can represent six of them. Anything else is dropped
        # rather than guessed at, because a wrong class would let the wrong sensor
        # "confirm" a detection.
        names = [coco_id_to_nuscenes_name(int(c)) for c in coco_ids]
        keep = np.array([name is not None for name in names], dtype=bool)

        return CameraDetections(
            name=self.camera_name,
            boxes=boxes[keep],
            confidences=confidences[keep],
            class_names=tuple(name for name in names if name is not None),
        )
