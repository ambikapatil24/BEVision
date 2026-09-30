"""BEVision -- camera + LiDAR fusion perception for autonomous driving.

Importing this package requires only numpy. The heavy dependencies (torch,
OpenMMLab, the nuScenes devkit) are imported lazily by the submodules that need
them, so the geometry, class-mapping and configuration cores are usable and
testable anywhere.

Typical use::

    from bevision import PerceptionConfig, box_corners, project_box_to_image

    config = PerceptionConfig()          # six-camera, class-aware defaults
    baseline = config.replace(cameras=("CAM_FRONT",))
"""

from __future__ import annotations

from bevision.classes import (
    CLASSES_WITHOUT_COCO_COUNTERPART,
    COCO_ID_TO_NUSCENES_NAME,
    NUSCENES_CLASS_NAMES,
    TRACKING_CLASS_NAMES,
    coco_id_to_nuscenes_name,
    resolve_class_names,
)
from bevision.config import (
    FRONT_CAMERA_ONLY,
    SURROUND_CAMERAS,
    DetectionClassPolicy,
    KalmanConfig,
    PerceptionConfig,
)
from bevision.detection.base import CameraDetector, CameraFrame, Frame, FrameSource, LidarDetector
from bevision.evaluation.submission import (
    build_detection_submission,
    build_tracking_submission,
    validate_submission,
    write_submission,
)
from bevision.fusion import (
    CameraConfirmation,
    CameraDetections,
    best_camera_confirmation,
    fuse_confidence,
    hungarian_match,
    iou_matrix_with_class_gate,
)
from bevision.geometry import (
    bbox_inside_image,
    bbox_iou,
    bbox_iou_matrix,
    box_corners,
    pairwise_distance_matrix,
    project_box_to_image,
    quaternion_to_matrix,
    rigid_inverse,
    sensor_to_global,
    to_global_yaw,
    transform_points,
    view_points,
)
from bevision.nms import center_nms, center_nms_indices
from bevision.pipeline import (
    PerceptionPipeline,
    PipelineResult,
    run_pipeline,
    summarise,
)
from bevision.tracking import ConstantVelocityKalmanFilter, Track3D, Tracker
from bevision.tracking.tracker import associate
from bevision.types import Box3D, Detection3D, FrameStats, RawDetection3D, TrackOutput
from bevision.visualization.bev import BevFrame, BevView, render_bev, render_bev_figure
from bevision.visualization.image import (
    BOX_COLOURS,
    draw_box,
    draw_boxes,
    overlay_frame,
    show_image,
)

__version__ = "0.1.0"

__all__ = [
    "BOX_COLOURS",
    "CLASSES_WITHOUT_COCO_COUNTERPART",
    "COCO_ID_TO_NUSCENES_NAME",
    "FRONT_CAMERA_ONLY",
    "NUSCENES_CLASS_NAMES",
    "SURROUND_CAMERAS",
    "TRACKING_CLASS_NAMES",
    "BevFrame",
    "BevView",
    "Box3D",
    "CameraConfirmation",
    "CameraDetections",
    "CameraDetector",
    "CameraFrame",
    "ConstantVelocityKalmanFilter",
    "Detection3D",
    "DetectionClassPolicy",
    "Frame",
    "FrameSource",
    "FrameStats",
    "KalmanConfig",
    "LidarDetector",
    "PerceptionConfig",
    "PerceptionPipeline",
    "PipelineResult",
    "RawDetection3D",
    "Track3D",
    "TrackOutput",
    "Tracker",
    "__version__",
    "associate",
    "bbox_inside_image",
    "bbox_iou",
    "bbox_iou_matrix",
    "best_camera_confirmation",
    "box_corners",
    "build_detection_submission",
    "build_tracking_submission",
    "center_nms",
    "center_nms_indices",
    "coco_id_to_nuscenes_name",
    "draw_box",
    "draw_boxes",
    "fuse_confidence",
    "hungarian_match",
    "iou_matrix_with_class_gate",
    "overlay_frame",
    "pairwise_distance_matrix",
    "project_box_to_image",
    "quaternion_to_matrix",
    "render_bev",
    "render_bev_figure",
    "resolve_class_names",
    "rigid_inverse",
    "run_pipeline",
    "sensor_to_global",
    "show_image",
    "summarise",
    "to_global_yaw",
    "transform_points",
    "validate_submission",
    "view_points",
    "write_submission",
]
