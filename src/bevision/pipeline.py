"""End-to-end perception pipeline: detect → fuse → track → emit.

The pipeline is written against the protocols in :mod:`bevision.detection`, so it runs
against fake detectors in tests and against Ultralytics/mmdetection3d in production without
changing a line.

Two behaviours here exist because of specific defects found while refactoring the research
notebook, and both are asserted by tests:

* **The LiDAR→ego transform is computed once per box, not once per box per camera.** The
  original recomputed it for each of the six cameras and then discarded the result whenever
  the box turned out to be behind that camera.
* **Detection and tracking submissions use different class sets.** nuScenes scores detection
  over ten classes and tracking over seven; the original used one filter for both and
  silently dropped three classes from the detection metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from bevision.config import PerceptionConfig
from bevision.detection.base import CameraDetector, Frame, FrameSource, LidarDetector
from bevision.fusion import CameraDetections, best_camera_confirmation, fuse_confidence
from bevision.geometry import (
    bbox_inside_image,
    box_corners,
    project_box_to_image,
    sensor_to_global,
    to_global_yaw,
)
from bevision.nms import center_nms
from bevision.tracking import Tracker
from bevision.tracking.tracker import Track3D
from bevision.types import Box3D, Detection3D, FrameStats, RawDetection3D, TrackOutput


@dataclass
class PipelineResult:
    """Everything the pipeline produced for a run."""

    tracks_per_sample: dict[str, list[TrackOutput]] = field(default_factory=dict)
    sample_tokens: list[str] = field(default_factory=list)
    frame_stats: list[FrameStats] = field(default_factory=list)

    @property
    def total_tracks_emitted(self) -> int:
        return sum(len(v) for v in self.tracks_per_sample.values())


def _rename(detections: CameraDetections, name: str) -> CameraDetections:
    """Force the camera name on a detector's output.

    A detector instance is reused across cameras, so it cannot know which camera it was
    handed. The pipeline does, and the name is carried into the fusion result for reporting,
    so it is corrected here rather than trusted from the detector.
    """
    if detections.name == name:
        return detections
    return replace(detections, name=name)


def _select_camera_confirmation(
    frame: Frame,
    corners_lidar,
    class_name: str,
    camera_detections: dict[str, object],
    config: PerceptionConfig,
):
    """Find the best confirming camera for one box, projecting into each in turn."""
    names = [name for name in config.cameras if name in frame.cameras]
    projected = []
    for name in names:
        camera = frame.cameras[name]
        projected.append(
            project_box_to_image(
                corners_lidar,
                frame.lidar_rotation,
                frame.lidar_translation,
                camera.rotation,
                camera.translation,
                camera.intrinsic,
                min_depth_m=config.min_box_depth_m,
            )
        )
    # Off-image boxes cannot be confirmed by that camera, but stay in the list so indices
    # stay aligned with `names`.
    usable = [
        bbox if (bbox is not None and bbox_inside_image(bbox, config.image_size)) else None
        for bbox in projected
    ]
    return best_camera_confirmation(
        [camera_detections[name] for name in names],
        usable,
        class_name,
        config.fusion_iou_threshold,
    )


class PerceptionPipeline:
    """Runs the full detection → fusion → tracking pipeline over a frame source."""

    def __init__(
        self,
        camera_detector: CameraDetector,
        lidar_detector: LidarDetector,
        class_names: Sequence[str],
        config: PerceptionConfig | None = None,
    ) -> None:
        self.config = config or PerceptionConfig()
        self.camera_detector = camera_detector
        self.lidar_detector = lidar_detector
        self.class_names = tuple(class_names)
        if len(self.class_names) == 0:
            raise ValueError("class_names must not be empty")

    # -- helpers ------------------------------------------------------------
    def _class_name(self, label: int) -> str:
        return self.class_names[int(label)]

    def _fuse_frame(self, frame: Frame) -> list[Detection3D]:
        """Detect, project and fuse one frame into global-frame 3D detections."""
        config = self.config

        raw: list[RawDetection3D] = self.lidar_detector(frame.lidar_path)
        raw = [d for d in raw if d.score >= config.lidar_score_floor]
        if not raw:
            return []

        camera_detections = {
            name: _rename(self.camera_detector(frame.cameras[name].image_path), name)
            for name in config.cameras
            if name in frame.cameras
        }

        detections: list[Detection3D] = []
        for detection in raw:
            class_name = self._class_name(detection.label)
            corners_lidar = box_corners(detection.center, detection.size, detection.yaw)

            confirmation = _select_camera_confirmation(
                frame, corners_lidar, class_name, camera_detections, config
            )

            score = detection.score
            confirmed = confirmation is not None
            if confirmation is not None:
                score = fuse_confidence(detection.score, confirmation.confidence)

            detections.append(
                Detection3D(
                    label=detection.label,
                    class_name=class_name,
                    score=score,
                    center_global=sensor_to_global(
                        detection.center,
                        frame.lidar_rotation,
                        frame.lidar_translation,
                        frame.ego_rotation,
                        frame.ego_translation,
                    ),
                    size=detection.size,
                    yaw_global=to_global_yaw(
                        detection.yaw,
                        frame.lidar_rotation,
                        frame.ego_rotation,
                    ),
                    bbox_2d=confirmation.bbox_2d if confirmation is not None else None,
                    camera_confirmed=confirmed,
                )
            )

        return center_nms(detections, config.nms_radius_m)

    @staticmethod
    def _to_track_output(track: Track3D) -> TrackOutput:
        """Convert a live track into the emitted box for this frame.

        The centre comes from the filter (temporally smoothed) while size, yaw and score
        come from the most recent matched detection.
        """
        position = track.position
        detection = track.last_detection
        return TrackOutput(
            track_id=track.track_id,
            class_name=track.class_name,
            score=detection.score,
            box=Box3D(center=position, size=detection.size, yaw=detection.yaw_global),
            velocity=(float(track.velocity[0]), float(track.velocity[1])),
        )

    # -- entry point --------------------------------------------------------
    def run(self, source: FrameSource) -> PipelineResult:
        """Run the pipeline over every frame a source yields."""
        result = PipelineResult()
        tracker = Tracker(self.config)
        current_scene: str | None = None

        for frame in source.frames():
            # Identities must not leak across scenes: these are separate drives.
            if frame.scene_token != current_scene:
                tracker.reset()
                current_scene = frame.scene_token

            detections = self._fuse_frame(frame)
            tracks, stats = tracker.step(detections)

            result.sample_tokens.append(frame.sample_token)
            result.tracks_per_sample[frame.sample_token] = [
                self._to_track_output(track) for track in tracks
            ]
            result.frame_stats.append(stats)

        return result


def run_pipeline(
    source: FrameSource,
    camera_detector: CameraDetector,
    lidar_detector: LidarDetector,
    class_names: Sequence[str],
    config: PerceptionConfig | None = None,
) -> PipelineResult:
    """Convenience wrapper around :class:`PerceptionPipeline`."""
    return PerceptionPipeline(camera_detector, lidar_detector, class_names, config).run(source)


def summarise(result: PipelineResult) -> dict[str, float]:
    """Aggregate per-frame counters into a small summary dict for logging."""
    if not result.frame_stats:
        return {"frames": 0.0}
    frames = len(result.frame_stats)
    return {
        "frames": float(frames),
        "detections_total": float(sum(s.lidar_detections for s in result.frame_stats)),
        "camera_confirmed_total": float(sum(s.camera_detections for s in result.frame_stats)),
        "tracks_emitted_total": float(result.total_tracks_emitted),
        "tracks_spawned_total": float(sum(s.tracks_spawned for s in result.frame_stats)),
        "mean_tracks_per_frame": float(np.mean([s.tracks_active for s in result.frame_stats])),
    }
