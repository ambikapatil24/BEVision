# Setup — environment, data, and pushing to GitHub

Everything here has been run; where a step is environment-specific it says so.

---

## 1. Quick start (core only, no GPU)

The geometry, fusion, tracking, NMS, submission and pipeline logic need only numpy and scipy.
This is enough to run the test suite — 229 tests, roughly 4 seconds.

```bash
git clone https://github.com/<you>/BEVision.git
cd BEVision
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Expected: `210 passed`.

### If `pip install` fails with "Installing build dependencies ... error"

Installing needs network access, because pip downloads `setuptools` and `wheel` into a
throwaway build environment. **Kaggle notebooks have Internet off by default** (Settings →
Internet → On; requires a phone-verified account), and some offline environments block it too.

If you cannot enable it, skip installing entirely — the package is a plain `src/` layout, so
putting `src` on the import path is enough:

```bash
PYTHONPATH=src python -m pytest -q          # instead of: pip install -e . && pytest
PYTHONPATH=src python -m bevision.cli --help
```

`python -m bevision.cli` runs the pipeline entry point without needing the console script
that `pip install` would create. Add `--no-build-isolation` to `pip install` if your
environment already has a recent `setuptools`.

If you only want the library:

```bash
pip install -e .            # numpy + scipy
pip install -e ".[viz]"     # + matplotlib, to render BEV panels and box overlays
```

---

## 2. Full perception stack (GPU + CUDA)

Needed to actually run the pipeline. **Python 3.11 or 3.12 only** — mmcv 2.2.0 ships no cp313
wheel, and Python 3.13 removed `distutils`, which its build toolchain needs.

```bash
pip install -e ".[camera,nuscenes,viz]"
bash scripts/install_mmdet3d.sh
```

`scripts/install_mmdet3d.sh` pins one working combination (torch 2.2.0 + mmcv 2.2.0, CUDA
11.8, verified on a T4) and enforces an install *order*, because each step fails in a
confusing way if run out of sequence:

1. **setuptools first** — older versions break the mmcv wheel build.
2. **torch pinned** — the mmcv wheel is compiled against exactly one torch version.
3. **mmcv from the prebuilt index matching (cu118, torch2.2.0)** — building from source needs
   a full CUDA toolchain.
4. **mmdet with `--no-deps`** — its declared numpy constraint pulls numpy 2.x, which
   mmdetection3d cannot use. Its real runtime deps are installed explicitly instead.
5. **mmdetection3d from source** — it is not on PyPI.
6. **Relax the mmcv version gates.** `mmdet` and `mmdet3d` both assert at import that the
   installed mmcv is strictly *less than* a declared maximum. With mmcv 2.2.0 installed and
   2.2.0 declared, `2.2.0 < 2.2.0` is False and the import crashes. The script patches the
   real installed files, located with `importlib.util.find_spec` (not by importing — importing
   is the thing that crashes) so it works on any Python version and any site-packages layout.
7. **numpy < 2.0 last, force-reinstall** — torch, mmcv and mmdet each try to upgrade it.

> ⚠️ **Step 6 edits `site-packages`.** A runtime restart wipes it. Re-run the script after any
> restart, or `import mmdet` will fail again.

Verify:

```bash
python -c "import mmcv, mmdet, mmdet3d, torch; \
print(mmcv.__version__, mmdet.__version__, mmdet3d.__version__, torch.__version__, torch.cuda.is_available())"
```

Expected roughly: `2.2.0 3.x 1.x 2.2.0+cu118 True`.

---

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

> On Kaggle, nuScenes mini is available as a dataset and the working directory is
> `/kaggle/working`. On Colab you will want the data on Drive — reading 4 GB of LiDAR from a
> Colab-local disk is not reliable.

---

## 4. Running it

```bash
bevision-run \
  --dataroot /path/to/v1.0-mini \
  --run-tag B_6cam \
  --yolo-weights /path/to/yolov8n.pt \
  --pointpillars-config mmdetection3d/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py \
  --pointpillars-checkpoint /path/to/hv_pointpillars_..._2fa62f3d.pth

bevision-eval --dataroot /path/to/v1.0-mini --run-tag B_6cam
```

`bevision-run` validates the submission it wrote against the nuScenes schema and exits
non-zero on any problem. Use `--max-frames 5` for a smoke test that takes a minute rather
than an hour.

Reproducing the full ablation — each is one command, differing only in flags:

```bash
D=/path/to/v1.0-mini
bevision-run --dataroot $D --run-tag B_6cam                                    # headline
bevision-run --dataroot $D --run-tag B0_repro  --association class-blind --detection-classes tracking
bevision-run --dataroot $D --run-tag A0_front_cam_class_blind --cameras front --association class-blind
bevision-run --dataroot $D --run-tag A_front_cam --cameras front
bevision-run --dataroot $D --run-tag C_6cam_lidar_spawn --spawn lidar-allowed

for tag in B_6cam B0_repro A0_front_cam_class_blind A_front_cam C_6cam_lidar_spawn; do
  bevision-eval --dataroot $D --run-tag $tag
done

python scripts/check_results.py     # re-derive the README table from the artifacts
```

> **`B0_repro` must reproduce mAP 0.1913 and AMOTA 0.1071 exactly.** If it does not, something
> changed in the detection, fusion or tracking path and no other number should be trusted until
> it is found. `check_results.py` enforces this. NDS is *reported* rather than asserted, because
> the original shipped the transposed-box bug and the package fixes it.

---

## 5. Pushing to GitHub

```bash
cd BEVision
git init -b main

# Personalise before the first commit.
#   LICENSE            -> replace "Your Name"
#   pyproject.toml     -> replace authors = [{ name = ..., email = ... }] and the URLs

git add .
git status                     # review what is about to be committed

# Guard against the single most common mistake in this kind of project: committing data.
find . -type f -size +10M -not -path "./.git/*"     # expect: nothing
du -sh .gitignore results src tests

git commit -m "BEVision: camera + LiDAR fusion perception on nuScenes

YOLOv8 (6-camera) + PointPillars -> LiDAR-anchored fusion -> class-consistent
Kalman tracking -> BEV, scored with the official nuScenes evaluator.

Refactored from a research notebook. Four findings changed the reported results:
the front-camera baseline was never measured (the real gain is x2.83 mAP, not
x7.3); class-blind track association was relabelling tracks and manufacturing
phantom boxes (fixing it: AMOTA x3.24, FP -49%); four classes were structurally
unscorable; and one filter served both the 10-class detection and 7-class
tracking submissions.

B0_repro reproduces the original results exactly, which is what isolates the
tracking fix as a single-variable change."

git remote add origin https://github.com/<you>/BEVision.git
git push -u origin main
```

`git status` should show **no** `samples/`, `sweeps/`, `*.pth`, `*.pt` or `data/`. `.gitignore`
covers all of them, but look — a 4 GB push fails slowly and unpleasantly.

The committed `results/*/results_*.json` files are the submission outputs, ~9 MB total. They
are intentional: they are the only artifact that lets someone re-run the evaluator and get the
published numbers back without a GPU.

### Adding the two missing artifacts

`A_front_cam` and `C_6cam_lidar_spawn` are cited in the README with no artifact present, and
`check_results.py` warns about them every run. To close it:

1. Re-run those two configs (commands above).
2. Copy the four files per run into `results/<tag>/` — `results_detection.json`,
   `results_tracking.json`, `eval_detection_metrics_summary.json`,
   `eval_tracking_metrics_summary.json`.
3. `python scripts/check_results.py` — the warning should disappear and the counts should match
   the README.

### Suggested repository settings

- **Description:** `Camera + LiDAR fusion perception on nuScenes — YOLOv8 + PointPillars → fusion → Kalman tracking, scored with the official evaluator`
- **Topics:** `autonomous-driving`, `sensor-fusion`, `3d-object-detection`, `multi-object-tracking`, `nuscenes`, `bird-eye-view`, `pytorch`
- **License:** MIT (already in `LICENSE`)
- Consider enabling Actions to run `pytest` + `ruff` on push. The core suite needs no GPU, so
  it will pass on a free runner.

---

## 6. Troubleshooting

| Symptom | Cause |
|---|---|
| `AssertionError: mmcv version ... is not compatible` on `import mmdet` | Step 6 of the install script was wiped by a runtime restart. Re-run `scripts/install_mmdet3d.sh`. |
| `AttributeError: module 'numpy' has no attribute ...` from mmdet3d | numpy 2.x got installed. `pip install "numpy<2.0.0" --force-reinstall`. |
| `ModuleNotFoundError: bevision.cli` | The package is not installed in the active environment: `pip install -e .`, or use `PYTHONPATH=src python -m bevision.cli`. |
| `ModuleNotFoundError: No module named 'nuscenes' / 'ultralytics' / 'mmdet3d'` | Not a bug — those are optional. `bevision-run` now lists every missing one and its install command. See section 2. |
| `FileNotFoundError` for `samples/...` | `--dataroot` points at `v1.0-mini/v1.0-mini` instead of its parent. |
| `linear_sum_assignment` raises `ValueError` about an infeasible matrix | A cost matrix was built with `np.inf` instead of the finite `FORBIDDEN_PAIR_COST`. |
| `TrackingEval` constructor `TypeError` | The devkit has changed its signature across releases; `cli._run_tracking_eval` passes only the parameters this version declares, so report the version if it still fails. |
