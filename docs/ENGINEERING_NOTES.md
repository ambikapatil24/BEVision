# Engineering notes

Meaningful changes to the package, and the current verification status. Newest first.

## Package data loader restored

The `.gitignore` rule `data/` was unanchored, so git matched it at every directory depth and it
excluded the `src/bevision/data/` package — `nuscenes_loader.py` and its `__init__.py` — from the
repository. `cli.pipeline_main` imports `NuScenesFrameSource` from that package, so `bevision-run`
failed at startup on a fresh clone. The rule is now anchored (`/data/`) and both files are tracked.

## Test dependencies declared

`pip install -e ".[dev]"` did not install `matplotlib` or `filterpy`, so the documented test
command did not exercise the suite:

- `tests/test_visualization.py` imports matplotlib at module scope, so collection aborted and
  `pytest` ran **zero** tests.
- `tests/test_kalman.py` guards its filterpy equivalence check with `pytest.importorskip`, so the
  check silently skipped.

Both are now in the `dev` extra. Neither is a runtime dependency of the package.

## Ignore rules anchored

The dataset, build and environment rules are anchored to the repository root — `/samples/`,
`/sweeps/`, `/maps/`, `/data/`, `/datasets/`, `/checkpoints/`, `/build/`, `/dist/`, `/.eggs/`,
`/venv/`, `/env/`, `/htmlcov/` — so a future source package with one of those names cannot be
silently excluded. Root-level ignore behaviour is unchanged.

Patterns that legitimately occur at any depth (`__pycache__/`, `*.egg-info/`, `.venv/`,
`.ipynb_checkpoints/`, the tool caches) are intentionally left unanchored, as are the
`v1.0-*/` nuScenes dataset directory names.

## Verification status

| Check | Command | Result |
|---|---|---|
| Test suite | `pytest` | 229 passed, 0 skipped |
| Result artifacts | `python scripts/check_results.py` | exit 0 — 5 runs consistent |
| CLI entry points | `bevision-run --help`, `bevision-eval --help` | exit 0 |

Verified on Python 3.12 with only the core and `dev` dependencies installed — no GPU, no CUDA and
no nuScenes dataset.

**Not verified in this repository:** the GPU perception stack (`torch`, `mmcv`, `mmdet`,
`mmdetection3d`) and anything that needs the nuScenes dataset or the model checkpoints.
`requirements-cuda.txt` and `scripts/install_mmdet3d.sh` record the target environment; see
`docs/SETUP.md` section 2.
