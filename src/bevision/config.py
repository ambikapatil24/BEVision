"""Centralised configuration for the BEVision perception pipeline.

Every threshold, policy switch and frame convention lives here, so that each experimental axis is
a named field rather than a constant buried in a loop.

The two axes that matter most:

* ``cameras`` -- which cameras the 2D detector runs on. One tuple selects either the
  single-camera configuration or the six-camera surround one.
* ``class_aware_track_matching`` -- whether track<->detection association is constrained to
  same-class pairs. ``False`` selects the class-blind matcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Canonical nuScenes surround-camera ordering. Matches the dataroot layout and
# the order used in the official devkit's sample_data records.
SURROUND_CAMERAS: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
)

# The single-camera baseline configuration from the README results table.
FRONT_CAMERA_ONLY: tuple[str, ...] = ("CAM_FRONT",)

# nuScenes sample period. Keyframes are 2 Hz, so consecutive keyframe samples are
# 0.5 s apart; this is the dt the constant-velocity model is built on.
NUSCENES_SAMPLE_PERIOD_S: float = 0.5


class DetectionClassPolicy(str, Enum):
    """Which classes may appear in the *detection* submission.

    nuScenes scores detection over all ten classes but tracking over only seven, so the two
    submissions need different class sets. Conflating them silently drops three classes from the
    detection metrics.
    """

    ALL = "all"
    TRACKING = "tracking"


@dataclass(frozen=True)
class KalmanConfig:
    """Noise parameters for the constant-velocity 3D Kalman filter.

    The defaults match the reference ``filterpy`` configuration the committed results were
    produced with.
    """

    #: Initial state covariance as a multiple of the identity: ``P = initial_covariance · I``.
    initial_covariance: float = 5.0
    measurement_noise: float = 1.0
    process_noise: float = 0.01


@dataclass(frozen=True)
class PerceptionConfig:
    """All tunable parameters for detection -> fusion -> tracking -> submission."""

    # -- score gates --------------------------------------------------------
    #: Detections below this LiDAR score are discarded outright.
    lidar_score_floor: float = 0.15
    #: A detection must exceed this score to start a new track. Detections
    #: between the floor and this threshold may only *update* existing tracks.
    spawn_score_threshold: float = 0.30

    # -- fusion -------------------------------------------------------------
    #: Minimum 2D IoU for a LiDAR box to be considered confirmed by a camera box.
    fusion_iou_threshold: float = 0.30

    # -- association --------------------------------------------------------
    #: Maximum 3D centre distance (metres) for a track<->detection match.
    track_match_distance_m: float = 3.0
    #: Constrain track<->detection association to same-class pairs. ``False`` selects
    #: class-blind association, the baseline used by the reproduction check.
    class_aware_track_matching: bool = True

    # -- track birth --------------------------------------------------------
    #: Require camera confirmation before a detection may start a track. With
    #: ``True``, classes absent from COCO can never spawn -- see README limits.
    spawn_requires_camera_confirmation: bool = True

    # -- duplicate suppression ---------------------------------------------
    #: Same-class detections within this radius (metres, xy-plane) are merged.
    nms_radius_m: float = 2.0

    # -- track lifetime -----------------------------------------------------
    #: A track is dropped once it has missed this many consecutive frames.
    #: It therefore still emits coasting predictions for ``track_max_age - 1``.
    track_max_age: int = 5

    # -- kinematics ---------------------------------------------------------
    sample_period_s: float = NUSCENES_SAMPLE_PERIOD_S

    # -- camera selection ---------------------------------------------------
    #: Cameras the 2D detector runs on. ``FRONT_CAMERA_ONLY`` gives the baseline.
    cameras: tuple[str, ...] = SURROUND_CAMERAS

    # -- image bounds -------------------------------------------------------
    #: (width, height) of a nuScenes camera image, in pixels.
    image_size: tuple[int, int] = (1600, 900)

    # -- projection ---------------------------------------------------------
    #: Minimum box depth (metres) in the camera frame. A box whose nearest corner
    #: is at or behind this depth is dropped before projection, because
    #: perspective division by z <= 0 mirrors the box through the focal plane.
    min_box_depth_m: float = 0.0

    # -- submission ---------------------------------------------------------
    detection_class_policy: DetectionClassPolicy = DetectionClassPolicy.ALL

    # -- filter -------------------------------------------------------------
    kalman: KalmanConfig = field(default_factory=KalmanConfig)

    @property
    def image_width(self) -> int:
        return self.image_size[0]

    @property
    def image_height(self) -> int:
        return self.image_size[1]

    def replace(self, **changes: object) -> PerceptionConfig:
        """Return a copy with ``changes`` applied (frozen dataclass convenience)."""
        from dataclasses import replace as _replace

        return _replace(self, **changes)


def front_camera_class_aware() -> PerceptionConfig:
    """Run A: single front camera, class-consistent association."""
    return PerceptionConfig(cameras=FRONT_CAMERA_ONLY)


def front_camera_class_blind() -> PerceptionConfig:
    """Run A0: single front camera. Reference class-blind baseline."""
    return PerceptionConfig(
        cameras=FRONT_CAMERA_ONLY,
        class_aware_track_matching=False,
    )


def surround_class_blind() -> PerceptionConfig:
    """Run B0: six-camera class-blind baseline used for the reproduction check."""
    return PerceptionConfig(class_aware_track_matching=False)


def surround_class_aware() -> PerceptionConfig:
    """Run B: six cameras and class-consistent association."""
    return PerceptionConfig()


def surround_lidar_spawn() -> PerceptionConfig:
    """Run C: class-aware, and LiDAR-only detections may start tracks."""
    return PerceptionConfig(spawn_requires_camera_confirmation=False)
