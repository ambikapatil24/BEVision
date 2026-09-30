"""Tests for the COCO <-> nuScenes class interface and the config surface."""

from __future__ import annotations

import dataclasses

import pytest

from bevision.classes import (
    CLASSES_WITHOUT_COCO_COUNTERPART,
    DEFAULT_ATTRIBUTE_BY_CLASS,
    NUSCENES_CLASS_NAMES,
    TRACKING_CLASS_NAMES,
    class_name_for_label,
    coco_id_to_nuscenes_name,
    has_coco_counterpart,
    resolve_class_names,
)
from bevision.config import (
    FRONT_CAMERA_ONLY,
    SURROUND_CAMERAS,
    DetectionClassPolicy,
    PerceptionConfig,
    front_camera_class_aware,
    front_camera_class_blind,
    surround_class_aware,
    surround_class_blind,
    surround_lidar_spawn,
)


# ---------------------------------------------------------------------------
# Class spaces
# ---------------------------------------------------------------------------
def test_nuscenes_class_list_is_the_ten_scored_classes() -> None:
    assert len(NUSCENES_CLASS_NAMES) == 10
    assert len(set(NUSCENES_CLASS_NAMES)) == 10
    assert NUSCENES_CLASS_NAMES[:4] == ("car", "truck", "trailer", "bus")
    assert NUSCENES_CLASS_NAMES[-2:] == ("traffic_cone", "barrier")


def test_tracking_scores_seven_classes() -> None:
    assert len(TRACKING_CLASS_NAMES) == 7
    assert set(TRACKING_CLASS_NAMES) <= set(NUSCENES_CLASS_NAMES)
    # tracking omits the three classes nuScenes never tracks
    assert set(NUSCENES_CLASS_NAMES) - set(TRACKING_CLASS_NAMES) == {
        "construction_vehicle",
        "traffic_cone",
        "barrier",
    }


def test_coco_mapping_covers_the_six_fusable_classes() -> None:
    assert coco_id_to_nuscenes_name(0) == "pedestrian"
    assert coco_id_to_nuscenes_name(1) == "bicycle"
    assert coco_id_to_nuscenes_name(2) == "car"
    assert coco_id_to_nuscenes_name(3) == "motorcycle"
    assert coco_id_to_nuscenes_name(5) == "bus"
    assert coco_id_to_nuscenes_name(7) == "truck"


def test_coco_mapping_returns_none_for_other_coco_classes() -> None:
    assert coco_id_to_nuscenes_name(4) is None  # aeroplane
    assert coco_id_to_nuscenes_name(56) is None  # chair
    assert coco_id_to_nuscenes_name(999) is None


def test_classes_without_coco_counterpart_are_the_four_unreachable_ones() -> None:
    assert {
        "trailer",
        "construction_vehicle",
        "traffic_cone",
        "barrier",
    } == CLASSES_WITHOUT_COCO_COUNTERPART


def test_has_coco_counterpart() -> None:
    assert has_coco_counterpart("car")
    assert has_coco_counterpart("pedestrian")
    assert not has_coco_counterpart("barrier")
    assert not has_coco_counterpart("traffic_cone")


def test_every_class_has_a_submission_attribute() -> None:
    assert set(DEFAULT_ATTRIBUTE_BY_CLASS) == set(NUSCENES_CLASS_NAMES)
    assert DEFAULT_ATTRIBUTE_BY_CLASS["barrier"] == ""
    assert DEFAULT_ATTRIBUTE_BY_CLASS["traffic_cone"] == ""
    assert DEFAULT_ATTRIBUTE_BY_CLASS["car"].startswith("vehicle.")


def test_resolve_class_names_accepts_a_permutation() -> None:
    shuffled = tuple(reversed(NUSCENES_CLASS_NAMES))
    assert resolve_class_names(shuffled) == shuffled


def test_resolve_class_names_rejects_a_wrong_class_set() -> None:
    with pytest.raises(ValueError, match="missing"):
        resolve_class_names(list(NUSCENES_CLASS_NAMES[:-1]))


def test_resolve_class_names_rejects_extra_classes() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        resolve_class_names([*NUSCENES_CLASS_NAMES, "spaceship"])


def test_class_name_for_label_uses_the_models_own_ordering() -> None:
    assert class_name_for_label(0, NUSCENES_CLASS_NAMES) == "car"
    assert class_name_for_label(7, NUSCENES_CLASS_NAMES) == "pedestrian"
    assert class_name_for_label(0, tuple(reversed(NUSCENES_CLASS_NAMES))) == "barrier"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_default_config_is_six_camera_class_aware() -> None:
    config = PerceptionConfig()
    assert config.cameras == SURROUND_CAMERAS
    assert len(config.cameras) == 6
    assert config.class_aware_track_matching is True
    assert config.spawn_requires_camera_confirmation is True


def test_default_thresholds_match_the_original_pipeline() -> None:
    """These are the numbers the reproduction check depends on."""
    config = PerceptionConfig()
    assert config.lidar_score_floor == 0.15
    assert config.spawn_score_threshold == 0.30
    assert config.fusion_iou_threshold == 0.30
    assert config.track_match_distance_m == 3.0
    assert config.nms_radius_m == 2.0
    assert config.track_max_age == 5
    assert config.sample_period_s == 0.5
    assert config.image_size == (1600, 900)
    assert config.image_width == 1600
    assert config.image_height == 900


def test_config_is_frozen() -> None:
    config = PerceptionConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.lidar_score_floor = 0.5  # type: ignore[misc]


def test_config_replace_is_non_mutating() -> None:
    base = PerceptionConfig()
    modified = base.replace(cameras=FRONT_CAMERA_ONLY)

    assert modified.cameras == FRONT_CAMERA_ONLY
    assert base.cameras == SURROUND_CAMERAS
    assert modified is not base


def test_kalman_config_defaults_reproduce_the_filterpy_setup() -> None:
    kalman = PerceptionConfig().kalman
    assert kalman.initial_covariance == 5.0
    assert kalman.measurement_noise == 1.0
    assert kalman.process_noise == 0.01


def test_default_detection_class_policy_scores_all_ten_classes() -> None:
    assert PerceptionConfig().detection_class_policy is DetectionClassPolicy.ALL


def test_ablation_presets_differ_in_exactly_one_axis() -> None:
    """The README ablation is only meaningful if each preset moves one knob."""
    surround_aware = surround_class_aware()
    surround_blind = surround_class_blind()

    diffs = {
        f.name
        for f in dataclasses.fields(PerceptionConfig)
        if getattr(surround_aware, f.name) != getattr(surround_blind, f.name)
    }
    assert diffs == {"class_aware_track_matching"}

    front_aware = front_camera_class_aware()
    front_diffs = {
        f.name
        for f in dataclasses.fields(PerceptionConfig)
        if getattr(front_aware, f.name) != getattr(surround_aware, f.name)
    }
    assert front_diffs == {"cameras"}


def test_front_camera_presets_use_one_camera() -> None:
    assert front_camera_class_aware().cameras == ("CAM_FRONT",)
    assert front_camera_class_blind().cameras == ("CAM_FRONT",)
    assert front_camera_class_blind().class_aware_track_matching is False


def test_lidar_spawn_preset_relaxes_only_the_spawn_gate() -> None:
    preset = surround_lidar_spawn()
    base = surround_class_aware()

    assert preset.spawn_requires_camera_confirmation is False
    assert preset.class_aware_track_matching is True
    assert preset.cameras == base.cameras
