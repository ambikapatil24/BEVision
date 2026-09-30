"""Tests for the end-to-end pipeline, driven by fake detectors and fake frames.

Because the pipeline is written against protocols, it can be exercised here with no torch,
no OpenMMLab and no nuScenes dataset: a fake LiDAR detector returns boxes, a fake camera
detector returns 2D boxes, and the real fusion/tracking/geometry code runs unchanged. That
is the payoff of keeping the heavy imports behind adapters.
"""

from __future__ import annotations

import numpy as np
import pytest

from bevision.classes import NUSCENES_CLASS_NAMES
from bevision.config import FRONT_CAMERA_ONLY, PerceptionConfig
from bevision.detection.base import CameraFrame, Frame
from bevision.fusion import CameraDetections
from bevision.pipeline import PerceptionPipeline, run_pipeline, summarise
from bevision.types import RawDetection3D

INTRINSIC = np.array([[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]])

# The pixel bbox of a 2 m x 4 m x 2 m box 10 m ahead, from tests/test_geometry.py.
BOX_10M_AHEAD_BBOX = (577.7778, 338.8889, 1022.2222, 561.1111)

CAR, TRUCK = 0, 1


def _frame(token: str = "sample-0", scene: str = "scene-1", cameras=("CAM_FRONT",)) -> Frame:
    """Identity LiDAR/ego/camera poses, so sensor frame == global frame."""
    camera_frames = {
        name: CameraFrame(
            name=name,
            image_path=f"{name}.jpg",
            rotation=np.eye(3),
            translation=np.zeros(3),
            intrinsic=INTRINSIC,
        )
        for name in cameras
    }
    return Frame(
        sample_token=token,
        scene_token=scene,
        lidar_path="lidar.bin",
        lidar_rotation=np.eye(3),
        lidar_translation=np.zeros(3),
        ego_rotation=np.eye(3),
        ego_translation=np.zeros(3),
        cameras=camera_frames,
    )


class FakeSource:
    def __init__(self, frames: list[Frame]) -> None:
        self._frames = frames

    def frames(self):
        return iter(self._frames)


class FakeLidar:
    """Returns a fixed set of boxes regardless of input."""

    def __init__(self, detections: list[RawDetection3D] | None = None) -> None:
        self._detections = detections if detections is not None else []
        self.calls = 0

    def __call__(self, lidar_path: str) -> list[RawDetection3D]:
        self.calls += 1
        return list(self._detections)

    @property
    def class_names(self) -> tuple[str, ...]:
        return NUSCENES_CLASS_NAMES


class FakeCamera:
    """Returns a fixed set of 2D boxes for whichever image it is handed."""

    def __init__(self, boxes=(), confidences=(), class_names=()) -> None:
        self._detections = CameraDetections(
            name="unset",
            boxes=np.asarray(boxes, dtype=float).reshape(-1, 4),
            confidences=np.asarray(confidences, dtype=float).reshape(-1),
            class_names=tuple(class_names),
        )
        self.calls = 0

    def __call__(self, image_path: str) -> CameraDetections:
        self.calls += 1
        return self._detections


def _box_ahead(label: int = CAR, *, score: float = 0.4, center=(0.0, 0.0, 10.0)) -> RawDetection3D:
    return RawDetection3D(
        label=label,
        score=score,
        center=np.array(center, dtype=float),
        size=(2.0, 4.0, 2.0),
        yaw=0.0,
    )


def _pipeline(lidar_detections, camera_detections, config=None) -> PerceptionPipeline:
    return PerceptionPipeline(
        camera_detector=FakeCamera(**camera_detections),
        lidar_detector=FakeLidar(lidar_detections),
        class_names=NUSCENES_CLASS_NAMES,
        config=config or PerceptionConfig(cameras=FRONT_CAMERA_ONLY),
    )


def _agreeing_camera(class_name: str = "car", confidence: float = 0.9) -> dict:
    return {
        "boxes": [BOX_10M_AHEAD_BBOX],
        "confidences": [confidence],
        "class_names": [class_name],
    }


# ---------------------------------------------------------------------------
# Fusion through the pipeline
# ---------------------------------------------------------------------------
def test_camera_agreement_confirms_and_boosts_a_detection() -> None:
    pipeline = _pipeline([_box_ahead(score=0.4)], _agreeing_camera())

    detections = pipeline._fuse_frame(_frame())

    assert len(detections) == 1
    assert detections[0].camera_confirmed is True
    # fuse_confidence(0.4, 0.9) with the default 0.5 weight
    assert detections[0].score == pytest.approx(0.5 * 0.4 + 0.5 * 0.9)


def test_class_mismatch_prevents_confirmation() -> None:
    """A YOLO car box must not confirm a LiDAR truck."""
    pipeline = _pipeline([_box_ahead(label=TRUCK)], _agreeing_camera(class_name="car"))

    detections = pipeline._fuse_frame(_frame())

    assert detections[0].camera_confirmed is False
    assert detections[0].score == pytest.approx(0.4)  # unboosted


def test_box_behind_the_camera_is_rejected_before_projection() -> None:
    pipeline = _pipeline([_box_ahead(center=(0.0, 0.0, -10.0))], _agreeing_camera())

    detections = pipeline._fuse_frame(_frame())

    # Still detected -- LiDAR saw it -- but unconfirmable, so no camera boost.
    assert len(detections) == 1
    assert detections[0].camera_confirmed is False


def test_score_floor_discards_low_confidence_detections() -> None:
    config = PerceptionConfig(cameras=FRONT_CAMERA_ONLY, lidar_score_floor=0.5)
    pipeline = _pipeline([_box_ahead(score=0.2)], _agreeing_camera(), config=config)

    assert pipeline._fuse_frame(_frame()) == []


def test_duplicate_boxes_are_merged_by_centre_nms() -> None:
    duplicates = [
        _box_ahead(score=0.9),
        _box_ahead(score=0.5, center=(0.3, 0.0, 10.0)),
    ]
    pipeline = _pipeline(duplicates, _agreeing_camera())

    detections = pipeline._fuse_frame(_frame())

    assert len(detections) == 1


def test_class_name_is_resolved_from_the_model_ordering() -> None:
    pipeline = _pipeline([_box_ahead(label=CAR)], _agreeing_camera())
    assert pipeline._fuse_frame(_frame())[0].class_name == "car"


def test_no_lidar_detections_skips_camera_inference() -> None:
    """A frame with nothing to fuse should not pay for six YOLO passes."""
    camera = FakeCamera(**_agreeing_camera())
    pipeline = PerceptionPipeline(
        camera, FakeLidar([]), NUSCENES_CLASS_NAMES, PerceptionConfig(cameras=FRONT_CAMERA_ONLY)
    )

    assert pipeline._fuse_frame(_frame()) == []
    assert camera.calls == 0


# ---------------------------------------------------------------------------
# Tracking through the pipeline
# ---------------------------------------------------------------------------
def test_run_produces_one_entry_per_sample() -> None:
    pipeline = _pipeline([_box_ahead()], _agreeing_camera())
    result = pipeline.run(FakeSource([_frame("s0"), _frame("s1"), _frame("s2")]))

    assert result.sample_tokens == ["s0", "s1", "s2"]
    assert set(result.tracks_per_sample) == {"s0", "s1", "s2"}
    assert len(result.frame_stats) == 3


def test_cache_confirmed_detection_spawns_a_track_and_persists() -> None:
    pipeline = _pipeline([_box_ahead()], _agreeing_camera())
    result = pipeline.run(FakeSource([_frame("s0"), _frame("s1")]))

    assert result.tracks_per_sample["s0"][0].track_id == 0
    assert result.tracks_per_sample["s1"][0].track_id == 0  # same identity maintained


def test_unconfirmed_detection_does_not_spawn_by_default() -> None:
    pipeline = _pipeline([_box_ahead()], {"boxes": [], "confidences": [], "class_names": []})
    result = pipeline.run(FakeSource([_frame("s0")]))

    assert result.tracks_per_sample["s0"] == []


def test_lidar_only_spawn_config_creates_a_track() -> None:
    config = PerceptionConfig(cameras=FRONT_CAMERA_ONLY, spawn_requires_camera_confirmation=False)
    pipeline = _pipeline(
        [_box_ahead()], {"boxes": [], "confidences": [], "class_names": []}, config=config
    )
    result = pipeline.run(FakeSource([_frame("s0")]))

    assert len(result.tracks_per_sample["s0"]) == 1


def test_tracker_resets_at_a_scene_boundary() -> None:
    """Identities must not leak between drives."""
    pipeline = _pipeline([_box_ahead()], _agreeing_camera())
    result = pipeline.run(
        FakeSource(
            [
                _frame("a0", scene="scene-1"),
                _frame("a1", scene="scene-1"),
                _frame("b0", scene="scene-2"),
            ]
        )
    )

    assert result.tracks_per_sample["a1"][0].track_id == 0
    # the second scene restarts numbering
    assert result.tracks_per_sample["b0"][0].track_id == 0


def test_track_output_uses_the_filter_not_the_raw_detection() -> None:
    pipeline = _pipeline([_box_ahead()], _agreeing_camera())
    result = pipeline.run(FakeSource([_frame("s0"), _frame("s1")]))

    centre = result.tracks_per_sample["s1"][0].box.center
    assert centre[2] == pytest.approx(10.0)
    assert len(result.tracks_per_sample["s1"][0].velocity) == 2


def test_moving_target_produces_velocity() -> None:
    frames = [_frame(f"s{i}") for i in range(6)]
    boxes_per_frame = [[_box_ahead(center=(0.5 * i, 0.0, 10.0))] for i in range(6)]

    class PerFrameLidar:
        class_names = NUSCENES_CLASS_NAMES

        def __init__(self):
            self.index = -1

        def __call__(self, lidar_path):
            self.index += 1
            return boxes_per_frame[self.index]

    pipeline = PerceptionPipeline(
        FakeCamera(**_agreeing_camera()),
        PerFrameLidar(),
        NUSCENES_CLASS_NAMES,
        PerceptionConfig(cameras=FRONT_CAMERA_ONLY),
    )
    result = pipeline.run(FakeSource(frames))

    velocity = result.tracks_per_sample["s5"][0].velocity
    assert velocity[0] > 0.0  # moving in +x, which is 1 m/s at 0.5 s per sample


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def test_summarise_reports_frame_and_track_totals() -> None:
    pipeline = _pipeline([_box_ahead()], _agreeing_camera())
    result = pipeline.run(FakeSource([_frame("s0"), _frame("s1")]))

    summary = summarise(result)
    assert summary["frames"] == 2.0
    assert summary["tracks_spawned_total"] == 1.0
    assert summary["tracks_emitted_total"] == 2.0


def test_summarise_handles_an_empty_run() -> None:
    pipeline = PerceptionPipeline(FakeCamera(), FakeLidar(), NUSCENES_CLASS_NAMES)
    assert summarise(pipeline.run(FakeSource([]))) == {"frames": 0.0}


def test_run_pipeline_convenience_wrapper() -> None:
    result = run_pipeline(
        FakeSource([_frame("s0")]),
        FakeCamera(**_agreeing_camera()),
        FakeLidar([_box_ahead()]),
        NUSCENES_CLASS_NAMES,
        PerceptionConfig(cameras=FRONT_CAMERA_ONLY),
    )
    assert result.total_tracks_emitted == 1


def test_empty_class_names_is_rejected() -> None:
    with pytest.raises(ValueError, match="class_names"):
        PerceptionPipeline(FakeCamera(), FakeLidar(), [])


def test_surround_config_runs_camera_detector_once_per_camera() -> None:
    camera = FakeCamera(**_agreeing_camera())
    pipeline = PerceptionPipeline(
        camera, FakeLidar([_box_ahead()]), NUSCENES_CLASS_NAMES, PerceptionConfig()
    )
    pipeline._fuse_frame(_frame(cameras=("CAM_FRONT", "CAM_BACK", "CAM_FRONT_LEFT")))

    assert camera.calls == 3
