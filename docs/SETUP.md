# Setup

Section 1 is the developer/test environment and is verified in this repository. Sections 2 and 3
cover the GPU perception stack and the external dataset; both need a GPU and a multi-GB download, so
they were exercised on Kaggle (Tesla T4, nuScenes mini) rather than in CI. Nothing in this repository
runs on a schedule — see the notes in each section for what was and was not checked.

## 1. Developer environment (verified)

Needs Python 3.10–3.12 and nothing else. This is enough to import the package and run the test
suite.

```bash
git clone https://github.com/ambikapatil24/BEVision.git
cd BEVision
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Verified result: **229 passed, 0 skipped**, in a few seconds with no GPU, no CUDA and no
nuScenes dataset.

What the two relevant extras install:

| Extra | Contents | Purpose |
|---|---|---|
| (core) | `numpy`, `scipy` | Runtime dependencies of the package itself |
| `dev` | `pytest`, `ruff`, `matplotlib`, `filterpy` | Test and lint tooling |

`matplotlib` and `filterpy` are test-only but required: `tests/test_visualization.py` imports
matplotlib at collection time, so without it `pytest` aborts with a collection error rather
than failing a test, and `tests/test_kalman.py` skips its filterpy equivalence check when
filterpy is absent. Neither is a runtime dependency of the package.

### Library only

```bash
pip install -e .            # numpy + scipy
pip install -e ".[viz]"     # + matplotlib, to render BEV panels and box overlays
```

### Offline or restricted environments

`pip install -e .` needs network access, because pip downloads `setuptools` and `wheel` into a
throwaway build environment. Where that is blocked (for example a Kaggle notebook with Internet
off, or an air-gapped machine), the package is a plain `src/` layout, so putting `src` on the
import path is enough:

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m bevision.cli --help
```

`python -m bevision.cli` uses the entry point without the console script that `pip install`
would create. `--no-build-isolation` also works if the environment already has a recent
`setuptools`.

## 2. GPU perception stack

Required to actually run the pipeline. **Python 3.11 or 3.12 only** — mmcv 2.2.0 ships no cp313
wheel, and Python 3.13 removed `distutils`, which its build toolchain needs.

```bash
pip install -e ".[camera,nuscenes]"
bash scripts/install_mmdet3d.sh
```

These components are not installable from PyPI as a resolvable set: torch wheels are
CUDA-specific, mmcv wheels are compiled against both a torch version and a CUDA version, and
mmdetection3d is not on PyPI at all. `scripts/install_mmdet3d.sh` pins one combination and
enforces an install *order*, because each step fails in a confusing way if run out of sequence:

1. **setuptools first** — older versions break the mmcv wheel build.
2. **torch pinned** — the mmcv wheel is compiled against exactly one torch version.
3. **mmcv from the prebuilt index matching (cu118, torch2.2.0)** — building from source needs a
   full CUDA toolchain.
4. **mmdet with `--no-deps`** — its declared numpy constraint pulls numpy 2.x, which
   mmdetection3d cannot use; its real runtime dependencies are installed explicitly instead.
5. **mmdetection3d from source.**
6. **Relax the mmcv version gates.** `mmdet` and `mmdet3d` each assert at import that the
   installed mmcv is strictly *less than* a declared maximum. With mmcv 2.2.0 installed and
   2.2.0 declared, `2.2.0 < 2.2.0` is False and the import crashes. The script patches the
   installed files, located with `importlib.util.find_spec` (not by importing — importing is the
   thing that crashes) so it works on any Python version and site-packages layout.
7. **numpy < 2.0 last, force-reinstall** — torch, mmcv and mmdet each try to upgrade it.

> ⚠️ **Step 6 edits `site-packages`.** A runtime restart wipes it. Re-run the script after any
> restart, or `import mmdet` will fail again.

Verify the stack imports:

```bash
python -c "import mmcv, mmdet, mmdet3d, torch; \
print(mmcv.__version__, mmdet.__version__, mmdet3d.__version__, torch.__version__, torch.cuda.is_available())"
```

The expected shape of the output is four version strings followed by `True`/`False` for CUDA.
The versions this project was built against are listed in `requirements-cuda.txt` and under "Stack"
in the top-level README. The stack has been exercised end-to-end on Kaggle against nuScenes mini —
`bevision-run` and both `bevision-eval` metrics complete there, and a fresh run reproduces the
committed `results/` numbers within inference variance. It does not run in CI, so treat the version
list as the combination used to produce `results/`.

`requirements-cuda.txt` lists the same stack for manual installation.

## 3. Data and weights

None of this is in the repository.

**nuScenes mini** (~4 GB) from [nuscenes.org](https://www.nuscenes.org/nuscenes#download).
Unpack so that you have:

```
/path/to/v1.0-mini/
├── v1.0-mini/          # the metadata tables
├── samples/
└── sweeps/
```

Pass the *parent* directory as `--dataroot`.

**Weights.**

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt

wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/\
hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth

git clone --depth 1 https://github.com/open-mmlab/mmdetection3d.git   # provides the config
```

## 4. Running the pipeline

```bash
bevision-run \
  --dataroot /path/to/v1.0-mini \
  --run-tag B_6cam \
  --yolo-weights /path/to/yolov8n.pt \
  --pointpillars-config mmdetection3d/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py \
  --pointpillars-checkpoint /path/to/hv_pointpillars_..._2fa62f3d.pth

bevision-eval --dataroot /path/to/v1.0-mini --run-tag B_6cam
```

`bevision-run` validates both submissions against the nuScenes schema before exiting and returns
non-zero if anything is malformed. Use `--max-frames 5` for a smoke test that takes a minute
rather than an hour.

Each run writes `results_detection_<tag>.json` and `results_tracking_<tag>.json` into
`--output-dir` (default `results/`). `bevision-eval` reads them and writes the evaluator's
output for that tag.

## 5. Reproducing the results table

The committed artifacts in `results/` are the evidence behind every number in the README. They
can be re-checked without a GPU or the dataset:

```bash
python scripts/check_results.py
```

The script uses only the standard library. It re-derives each run's mAP / NDS / AMOTA / MOTA /
recall from that run's evaluator output and fails if the run's `headline_metrics.json` disagrees.
It also asserts the reproduction property on `B0_repro` (mAP 0.1913, AMOTA 0.1071); see
`results/README.md` for what that does and does not establish.

## 6. Troubleshooting

| Symptom | Cause |
|---|---|
| `AssertionError: mmcv version ... is not compatible` on `import mmdet` | Step 6 of the install script was wiped by a runtime restart. Re-run `scripts/install_mmdet3d.sh`. |
| `AttributeError: module 'numpy' has no attribute ...` from mmdet3d | numpy 2.x got installed. `pip install "numpy<2.0.0" --force-reinstall`. |
| `ModuleNotFoundError: bevision.cli` | The package is not installed in the active environment: `pip install -e .`, or use `PYTHONPATH=src python -m bevision.cli`. |
| `ModuleNotFoundError: No module named 'nuscenes' / 'ultralytics' / 'mmdet3d'` | Not a bug — those are optional. `bevision-run` lists every missing one and its install command. See section 2. |
| `FileNotFoundError` for `samples/...` | `--dataroot` points at `v1.0-mini/v1.0-mini` instead of its parent. |
| `linear_sum_assignment` raises `ValueError` about an infeasible matrix | A cost matrix was built with `np.inf` instead of the finite `FORBIDDEN_PAIR_COST`. |
| `TrackingEval` constructor `TypeError` | The devkit has changed its signature across releases; `cli._run_tracking_eval` passes only the parameters this version declares, so report the version if it still fails. |
| `pytest` aborts with `ModuleNotFoundError: No module named 'matplotlib'` | The `dev` extra was not installed: `pip install -e ".[dev]"`. |
| `bevision-eval` fails with `ModuleNotFoundError: No module named 'motmetrics'` | The devkit's tracking evaluator needs `motmetrics`, which the devkit has not installed by default since Dec 2020 (`nuscenes-devkit` 1.2.0 declares no extras at all). `pip install "motmetrics>=1.4.0"` — **not** the often-suggested `motmetrics==1.1.3`, which imports `Iterable` from `collections` and fails on Python 3.10+. |
