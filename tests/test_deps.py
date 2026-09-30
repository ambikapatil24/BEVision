"""Tests for the actionable missing-dependency messages.

This is what a first-time user meets on a fresh machine, and the design cost of lazy imports is
that a missing module otherwise surfaces as a bare ``ModuleNotFoundError`` naming nothing to
install. The messages are user-facing text, so they are pinned.
"""

from __future__ import annotations

import pytest

from bevision._deps import dependency_hint, missing, report_missing, require


# ---------------------------------------------------------------------------
# Hints name the right extra
# ---------------------------------------------------------------------------
def test_devkit_points_at_the_nuscenes_extra() -> None:
    assert dependency_hint("nuscenes") == 'pip install -e ".[nuscenes]"'


def test_ultralytics_points_at_the_camera_extra() -> None:
    assert dependency_hint("ultralytics") == 'pip install -e ".[camera]"'


def test_matplotlib_points_at_the_viz_extra() -> None:
    assert dependency_hint("matplotlib") == 'pip install -e ".[viz]"'


def test_mmdet3d_points_at_the_install_script_not_pip() -> None:
    """mmdetection3d is not on PyPI, so a pip hint would be wrong and unhelpful."""
    hint = dependency_hint("mmdet3d")
    assert "install_mmdet3d.sh" in hint
    assert not hint.startswith("pip install -e")


def test_mmcv_also_points_at_the_script() -> None:
    assert "install_mmdet3d.sh" in dependency_hint("mmcv")


def test_unknown_module_falls_back_to_a_plain_pip_hint() -> None:
    assert dependency_hint("some_unknown_pkg") == "pip install some_unknown_pkg"


# ---------------------------------------------------------------------------
# require()
# ---------------------------------------------------------------------------
def test_require_passes_for_an_installed_module() -> None:
    require("numpy", "doing arithmetic")  # must not raise


def test_require_names_the_module_the_purpose_and_the_fix() -> None:
    with pytest.raises(ModuleNotFoundError) as excinfo:
        require("definitely_not_installed_xyz", "loading nuScenes data")

    message = str(excinfo.value)
    assert "definitely_not_installed_xyz" in message
    assert "loading nuScenes data" in message
    assert "pip install" in message
    assert "environment problem, not a bug" in message


def test_require_sets_the_module_name_on_the_exception() -> None:
    with pytest.raises(ModuleNotFoundError) as excinfo:
        require("definitely_not_installed_xyz", "doing something")
    assert excinfo.value.name == "definitely_not_installed_xyz"


# ---------------------------------------------------------------------------
# missing() / report_missing()
# ---------------------------------------------------------------------------
def test_missing_returns_only_the_absent_ones() -> None:
    absent = missing(["numpy", "definitely_not_installed_xyz", "scipy"])
    assert absent == ["definitely_not_installed_xyz"]


def test_report_missing_is_silent_when_everything_is_present() -> None:
    report_missing(["numpy", "scipy"], "the core")  # must not raise


def test_report_missing_lists_every_gap_at_once() -> None:
    """The point: one failed run, not one per package."""
    with pytest.raises(ModuleNotFoundError) as excinfo:
        report_missing(
            ["totally_absent_one", "totally_absent_two", "numpy"],
            "the perception pipeline",
        )

    message = str(excinfo.value)
    assert "the perception pipeline" in message
    assert "totally_absent_one" in message
    assert "totally_absent_two" in message
    assert "numpy" not in message  # present, so not reported
    assert "2 package(s)" in message
    assert "docs/SETUP.md" in message


def test_cli_reports_the_whole_stack_before_touching_the_dataset() -> None:
    """pipeline_main must fail on dependencies, not deep inside dataset loading."""
    import bevision._deps as deps

    called: list[list[str]] = []

    def fake_report(modules, purpose):
        called.append(list(modules))
        raise SystemExit(0)

    original = deps.report_missing
    deps.report_missing = fake_report
    try:
        from bevision.cli import pipeline_main

        with pytest.raises(SystemExit):
            pipeline_main(["--dataroot", "/nowhere", "--run-tag", "t"])
    finally:
        deps.report_missing = original

    assert called == [["nuscenes", "ultralytics", "mmdet3d"]]
