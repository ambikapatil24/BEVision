"""Tests for the CLI surface.

The important one is ``test_entry_points_resolve``: ``pyproject.toml`` declares
``bevision-run`` and ``bevision-eval``, and an entry point that points at a missing module
installs happily and only fails when a user finally runs it. That happened once already.
"""

from __future__ import annotations

import subprocess

import pytest

from bevision import cli
from bevision.cli import build_run_parser, config_from_args
from bevision.config import FRONT_CAMERA_ONLY, SURROUND_CAMERAS, DetectionClassPolicy


def _parse(*argv: str):
    return build_run_parser().parse_args(["--dataroot", "/data", "--run-tag", "T", *argv])


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_defaults_are_the_surround_class_aware_configuration() -> None:
    config = config_from_args(_parse())
    assert config.cameras == SURROUND_CAMERAS
    assert config.class_aware_track_matching is True
    assert config.spawn_requires_camera_confirmation is True
    assert config.detection_class_policy is DetectionClassPolicy.ALL


def test_dataroot_and_run_tag_are_required() -> None:
    parser = build_run_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataroot", "/data"])


# ---------------------------------------------------------------------------
# The ablation axes, driven from the command line
# ---------------------------------------------------------------------------
def test_front_camera_flag_selects_the_baseline() -> None:
    assert config_from_args(_parse("--cameras", "front")).cameras == FRONT_CAMERA_ONLY


def test_class_blind_flag_reproduces_the_original_matcher() -> None:
    config = config_from_args(_parse("--association", "class-blind"))
    assert config.class_aware_track_matching is False


def test_lidar_allowed_spawn_flag() -> None:
    config = config_from_args(_parse("--spawn", "lidar-allowed"))
    assert config.spawn_requires_camera_confirmation is False


def test_detection_classes_can_be_restricted_without_touching_tracking() -> None:
    config = config_from_args(_parse("--detection-classes", "tracking"))
    assert config.detection_class_policy is DetectionClassPolicy.TRACKING


def test_run_b_configuration_is_reachable_from_the_cli() -> None:
    """The configuration behind the headline result.

    Six cameras, class-aware association, camera-gated spawning, all ten detection classes.
    """
    config = config_from_args(_parse())
    assert config.cameras == SURROUND_CAMERAS
    assert config.class_aware_track_matching is True
    assert config.spawn_requires_camera_confirmation is True
    assert config.detection_class_policy is DetectionClassPolicy.ALL


def test_run_b0_reproduction_configuration_is_reachable() -> None:
    config = config_from_args(
        _parse("--association", "class-blind", "--detection-classes", "tracking")
    )
    assert config.cameras == SURROUND_CAMERAS
    assert config.class_aware_track_matching is False
    assert config.detection_class_policy is DetectionClassPolicy.TRACKING


def test_run_a0_configuration_is_reachable() -> None:
    config = config_from_args(_parse("--cameras", "front", "--association", "class-blind"))
    assert config.cameras == FRONT_CAMERA_ONLY
    assert config.class_aware_track_matching is False


def test_invalid_choice_is_rejected() -> None:
    with pytest.raises(SystemExit):
        _parse("--cameras", "backwards")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def test_entry_points_resolve() -> None:
    """pyproject declares these; a missing module would only fail at call time."""
    assert callable(cli.pipeline_main)
    assert callable(cli.eval_main)


def test_declared_entry_points_match_the_installed_metadata() -> None:
    """Guard against the declaration and the code drifting apart."""
    import importlib.metadata

    try:
        entry_points = importlib.metadata.entry_points(group="console_scripts")
    except TypeError:  # Python 3.10 signature
        entry_points = importlib.metadata.entry_points()["console_scripts"]

    declared = {
        point.name: point.value
        for point in entry_points
        if point.name in {"bevision-run", "bevision-eval"}
    }
    if not declared:
        pytest.skip("bevision is not installed; entry points are not registered")
    assert declared == {
        "bevision-run": "bevision.cli:pipeline_main",
        "bevision-eval": "bevision.cli:eval_main",
    }


def test_console_scripts_run_help_without_the_perception_stack() -> None:
    """The real user path: invoking the installed scripts must not need torch or data."""
    for prog in ("bevision-run", "bevision-eval"):
        completed = subprocess.run([prog, "--help"], capture_output=True, text=True, timeout=180)
        assert completed.returncode == 0, completed.stderr
        assert "usage" in completed.stdout.lower()
