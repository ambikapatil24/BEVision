"""Friendly errors for the optional heavy dependencies.

The core package deliberately does not depend on torch, OpenMMLab, the nuScenes devkit or
matplotlib -- they are imported lazily, inside the functions that need them, so `import
bevision` works on a laptop with none of them installed.

The cost of that design is that a missing dependency surfaces as a bare
``ModuleNotFoundError`` at the point of use, which does not tell you what to install. This
module turns that into an actionable message naming the exact extra. It is what a first-time
user actually needs: the failure mode is normal, but it should be self-explanatory.
"""

from __future__ import annotations

from importlib.util import find_spec

#: Which installable extra provides each heavy module. Mirrors pyproject.toml.
_EXTRA_BY_MODULE: dict[str, str] = {
    "nuscenes": "nuscenes",
    "pandas": "nuscenes",
    "matplotlib": "viz",
    "ultralytics": "camera",
    "torch": "camera",
    "mmdet3d": "lidar",
    "mmcv": "lidar",
    "mmdet": "lidar",
    "mmengine": "lidar",
}

#: Modules that cannot be installed from PyPI at all and need scripts/install_mmdet3d.sh.
_NEEDS_SCRIPT = frozenset({"mmdet3d", "mmcv"})


def dependency_hint(module: str) -> str:
    """A copy-pasteable install command for ``module``, or a generic hint if unknown."""
    if module in _NEEDS_SCRIPT:
        return (
            f"bash scripts/install_mmdet3d.sh   # {module} has no portable PyPI wheel,\n"
            f"                                   #  see docs/SETUP.md section 2"
        )
    extra = _EXTRA_BY_MODULE.get(module)
    if extra:
        return f'pip install -e ".[{extra}]"'
    return f"pip install {module}"


def require(module: str, purpose: str) -> None:
    """Raise an actionable :class:`ModuleNotFoundError` if ``module`` is not importable.

    Called before the actual import so the message can name the feature and the fix.

    Args:
        module: The importable module name, e.g. ``"nuscenes"``.
        purpose: What needed it, phrased as a gerund, e.g. ``"loading nuScenes data"``.
    """
    if find_spec(module) is not None:
        return

    raise ModuleNotFoundError(
        f"{purpose} requires '{module}', which is not installed.\n"
        f"  Install it with:  {dependency_hint(module)}\n"
        f"  ('{module}' is an optional dependency: the core package works without it,\n"
        f"   so this is an environment problem, not a bug.)",
        name=module,
    )


def missing(modules: list[str]) -> list[str]:
    """Which of ``modules`` are not importable in this environment."""
    return [name for name in modules if find_spec(name) is None]


def report_missing(modules: list[str], purpose: str) -> None:
    """Check a whole stack at once, listing every gap and its fix in one message.

    Checking individually means a user on a fresh environment discovers one missing package
    per attempt -- install the devkit, re-run, hit Ultralytics, re-run, hit mmdetection3d.
    The whole list is known up front, so it is reported up front.
    """
    absent = missing(modules)
    if not absent:
        return

    lines = [f"{purpose} needs {len(absent)} package(s) that are not installed:"]
    lines += [f"  {name:<12s} -> {dependency_hint(name)}" for name in absent]
    lines.append("  See docs/SETUP.md section 2 for the full environment.")
    raise ModuleNotFoundError("\n".join(lines))
