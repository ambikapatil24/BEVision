# Engineering notes

Meaningful changes to the package, and the current verification status. Newest first.

## Tracking evaluator config loader fixed

`bevision-eval --metrics tracking` failed with "AssertionError: Requested unknown configuration
tracking_nips_2019". `evaluation.runner._config_factory` preferred
`nuscenes.eval.detection.config.config_factory`: that try-branch always succeeds, and the detection
factory resolves only `detection_*` names against `nuscenes/eval/detection/configs/`, so any tracking
name raised.

It now uses `nuscenes.eval.common.config.config_factory`, which dispatches on the config name's prefix
and reads `nuscenes/eval/<task>/configs/` — the same factory the devkit's own
`nuscenes/eval/tracking/evaluate.py` imports. The detection path was unaffected either way.

## Tracking-evaluation dependency declared

`bevision-eval --metrics tracking` failed with `ModuleNotFoundError: No module named 'motmetrics'`.
The devkit's tracking evaluator imports `motmetrics` (`nuscenes/eval/tracking/mot.py`), but the devkit
stopped installing it by default in Dec 2020, and `nuscenes-devkit` 1.2.0 declares no optional extras
at all — so no plain `pip install nuscenes-devkit` can pull it in.

`motmetrics>=1.4.0` is now in the `nuscenes` and `all` extras. The lower bound matters: motmetrics
1.1.x imports `Iterable` from `collections`, which Python 3.10 removed, so the commonly-suggested
`motmetrics==1.1.3` pin fails on this project's Python 3.12 runtime.

Confirmed end-to-end on Kaggle (Tesla T4): with this and the config-loader fix in place,
`bevision-eval --metrics tracking` completes and writes its summary.

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
| Full pipeline, both evaluators | `bevision-run` + `bevision-eval` on Kaggle (T4), nuScenes mini | completed |

The first three rows were verified on Python 3.12 with only the core and `dev` dependencies installed —
no GPU, no CUDA, no dataset. The fourth needs the GPU stack and the dataset, so it ran on Kaggle.

A fresh end-to-end `B_6cam` run there reproduces the committed artifacts within inference variance:
AMOTA 0.343 against 0.3466, MOTA 0.325 against 0.3303, recall 0.514 against 0.5023, and every per-class
ground-truth count identical (car 2188, pedestrian 1088, motorcycle 224, truck 95, bicycle 41, bus 33).
The submitted box count differs (4465 against 4434), so the small deltas are fresh-inference variance
rather than disagreement.

**Not run in CI.** Nothing here executes on a schedule; every row above was run by hand.
