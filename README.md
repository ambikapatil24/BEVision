# BEVision — Camera + LiDAR Fusion Perception for Autonomous Driving

Multi-sensor 3D perception on **nuScenes `mini_val`**: YOLOv8n 2D detection on six surround cameras,
PointPillars 3D detection on `LIDAR_TOP`, LiDAR-anchored projection fusion, 3D multi-object tracking
with a constant-velocity Kalman filter in the global frame, and Bird's-Eye View rendering — scored
with the official `nuscenes-devkit` evaluator (mAP / NDS / AMOTA / MOTA).

Objective: a pipeline over real autonomous-driving data whose every reported number is reproducible
from committed artifacts, and whose detectors, fusion, tracking and evaluation can each be replaced
without touching the others.

## Pipeline

```
   6 × CAMERA (YOLOv8n)                    LIDAR_TOP (PointPillars)
        │ 2D boxes + COCO class ids              │ 3D boxes (x,y,z,w,l,h,yaw)
        ▼                                        ▼
   FUSION — LiDAR is the anchor
     3D box → every camera; class-consistent 2D IoU match; score = 0.5·lidar + 0.5·camera
        ▼
   TRACKING — constant-velocity Kalman filter, global frame
     predict → class-consistent Hungarian match → update → spawn (camera-gated) → prune
        ▼
     BEV render   │   submission JSONs → official nuScenes evaluator
```

## Components

**Cameras — YOLOv8n** (`detection/camera.py`). COCO-trained, so only six of its 80 classes map onto
nuScenes (`pedestrian`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`); the rest are dropped rather
than guessed at, because a mislabelled camera box would let the wrong sensor confirm a LiDAR
detection. The mapping is in `classes.py`.

**LiDAR — PointPillars** (`detection/lidar.py`). Two tensor conventions are converted explicitly
because both fail silently: `LiDARInstance3DBoxes.tensor` stores size as `(length, width, height)`
where nuScenes expects `(width, length, height)`, and mmdet3d anchors boxes at their bottom face
(`origin = (0.5, 0.5, 0)`), so the centre is `z + dz/2`, not `z`. The conversion is a standalone
function, unit-tested without mmdetection3d installed.

**Fusion** (`fusion.py`). LiDAR decides *where*, cameras decide *what*. Each 3D box is projected into
every camera and matched against same-class 2D boxes by IoU; the best confirming camera raises the
detection's score. A camera can never create, move or delete a 3D box, so depth always comes from the
point cloud. Boxes behind the image plane are rejected before perspective division, otherwise they
are mirrored through the focal plane into spurious matches.

**Tracking** (`tracking/`). Constant-velocity 3D Kalman filter over `[x, y, z, vx, vy, vz]`, position
observations only, Joseph form for the covariance. Association is
`scipy.optimize.linear_sum_assignment` on 3D centre distance, constrained to same-class pairs — the
constraint goes *into* the cost matrix, not onto its result, so no match is spent on a pair that
would then be discarded. Mismatched pairs take a finite penalty (`FORBIDDEN_PAIR_COST = 1e6`) rather
than `inf`, because `linear_sum_assignment` raises on an infeasible `inf` matrix. Positions are lifted
to world coordinates via `ego_pose`, and the tracker resets at `scene_token` boundaries so identities
never leak between drives.

**Geometry and BEV** (`geometry.py`, `visualization/`). Batched rigid transforms, quaternion→matrix,
box corners, 3×3 intrinsic projection, pairwise IoU and centre distances — numpy only, no per-corner
Python calls. Global-frame points, detections and tracks are drawn on any matplotlib `Axes`; the 2D
overlay is pure numpy. Neither needs a dataset, a GPU or a live model.

**Evaluation** (`evaluation/`). `submission.py` builds and validates both submission formats before the
CLI exits — detection over ten nuScenes classes, tracking over seven — and `runner.py` reads headline
numbers from the evaluator's own `metrics_summary.json` rather than retyping them.

## Results

`mini_val`, 81 samples / 2 scenes, from the official `nuscenes-devkit` evaluator. Each run is a config
preset in `bevision.config`; every figure below is re-derived from the committed artifacts by
`scripts/check_results.py`.

| Run | Cameras | Class-consistent association | mAP | NDS | AMOTA | MOTA | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `A0_front_cam_class_blind` | 1 (front) | no | 0.0676 | 0.1138 | 0.0791 | 0.0884 | 0.1767 |
| `A_front_cam` | 1 (front) | yes | 0.0765 | 0.1181 | 0.1397 | 0.1531 | 0.1793 |
| `B0_repro` | 6 | no | 0.1913 | 0.2019 | 0.1071 | 0.0922 | 0.4172 |
| **`B_6cam`** | 6 | **yes** | **0.2121** | **0.2000** | **0.3466** | **0.3303** | **0.5023** |
| `C_6cam_lidar_spawn` | 6 | yes + LiDAR-only spawn | 0.2133 | 0.2129 | 0.3404 | 0.3027 | 0.5628 |

| Comparison (recomputed from the artifacts) | mAP | NDS | AMOTA | MOTA | Recall |
|---|---:|---:|---:|---:|---:|
| Camera count (`A0` → `B0`, association held class-blind) | ×2.83 | ×1.77 | ×1.35 | ×1.04 | ×2.36 |
| Class-consistent association (`B0` → `B`, six cameras both sides) | ×1.11 | ×0.99 | ×3.24 | ×3.58 | ×1.20 |
| End to end (`A0` → `B`) | ×3.14 | ×1.76 | ×4.38 | ×3.74 | ×2.84 |

`B0_repro` → `B_6cam` also moves TP 1581 → 1718, FP 1061 → 546 (−48.5%) and IDS 118 → 68 (−42.4%).
`B0_repro` runs with `class_aware_track_matching=False`, and `scripts/check_results.py` asserts it
reproduces mAP 0.1913 and AMOTA 0.1071 — the two metrics computed from box centres alone. This
provides the controlled comparison between class-blind and class-consistent association.
`C_6cam_lidar_spawn` increases recall (0.5023 → 0.5628) but reduces AMOTA and MOTA relative to
`B_6cam`, so `B_6cam` is the reported configuration. Per-class AP/AMOTA and per-run tracking detail
are in [results/README.md](results/README.md).

## Repository structure

```
src/bevision/     config.py classes.py types.py geometry.py fusion.py nms.py pipeline.py cli.py
                  tracking/    (kalman.py, tracker.py)
                  detection/   (base.py protocols, camera.py YOLO, lidar.py PointPillars)
                  data/        (nuscenes_loader.py — NuScenesFrameSource)
                  evaluation/  (submission.py build/validate, runner.py official eval)
                  visualization/ (bev.py, image.py)
tests/            229 passing tests; no GPU or dataset required
results/          one directory per run: submissions + evaluator output
scripts/          check_results.py, install_mmdet3d.sh
docs/             ARCHITECTURE.md, ENGINEERING_NOTES.md, SETUP.md
```

The core depends on **numpy and scipy only**. Torch, OpenMMLab and the nuScenes devkit are imported
lazily inside the adapters that need them, so `import bevision` succeeds without the perception stack
and the logic is testable anywhere.

## Setup

Core package and test suite — no GPU, no dataset:

```bash
git clone https://github.com/ambikapatil24/BEVision.git && cd BEVision
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                                 # 229 passed
```

The perception stack is CUDA-specific and installed separately. **Python 3.11 or 3.12 only** — mmcv
2.2.0 ships no cp313 wheel, and Python 3.13 removed `distutils`, which its build toolchain needs.

```bash
pip install -e ".[camera,nuscenes]"
bash scripts/install_mmdet3d.sh
```

Data and weights are external, not in the repository: nuScenes mini (~4 GB) from
[nuscenes.org](https://www.nuscenes.org/nuscenes#download), `yolov8n.pt` (auto-downloaded by
Ultralytics), and the PointPillars checkpoint
`hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth` from the mmdetection3d model
zoo. [docs/SETUP.md](docs/SETUP.md) covers the install order `install_mmdet3d.sh` enforces and the
failure modes behind it.

## Running

```bash
bevision-run  --dataroot /path/to/v1.0-mini --run-tag B_6cam \
              --yolo-weights /path/to/yolov8n.pt \
              --pointpillars-config mmdetection3d/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py \
              --pointpillars-checkpoint /path/to/hv_pointpillars_..._2fa62f3d.pth
bevision-eval --dataroot /path/to/v1.0-mini --run-tag B_6cam
```

`bevision-run` validates both submissions against the nuScenes schema before exiting and returns
non-zero on any problem. `--max-frames 5` gives a smoke test that takes a minute rather than an hour.
Each row of the results table is one command, differing only in flags: `--cameras front`,
`--association class-blind`, `--spawn lidar-allowed`, `--detection-classes tracking` — so `B0_repro`
is `--association class-blind --detection-classes tracking` and `A0` is `--cameras front --association
class-blind`.

The pipeline is usable as a library against any data source, because it talks to detectors through the
protocols in `detection/base.py` (`CameraDetector`, `LidarDetector`, `FrameSource`).
`PerceptionPipeline(camera_detector, lidar_detector, class_names, config).run(frame_source)` where the
detectors are any callable and `frame_source` is anything with `.frames() -> Iterable[Frame]`.

## Tests

`pytest` → **229 passed, 0 skipped**, in a few seconds, with no GPU, no CUDA and no nuScenes dataset.
The heavy imports sit behind the adapters, so `tests/test_pipeline.py` drives the real fusion,
tracking, NMS and geometry code with fake detectors against fake frames, and `tests/test_kalman.py`
asserts the filter matches `filterpy.kalman` step by step, covariance included. `visualization` is
tested headless through matplotlib's Agg backend, with the BEV panel's ego-frame transform asserted
numerically.

## Limitations

- **`mini_val` only** — 81 samples, 2 scenes. `bus` has 33 ground-truth objects and `truck` 95, so
  their per-class AMOTA is highly variant-sensitive and is not comparable to published `val` numbers.
  Treat the per-class columns as directional.
- **Pretrained detectors, no fine-tuning.** The largest available lever.
- **`trailer`, `construction_vehicle`, `traffic_cone` and `barrier` score AP 0.000** in every run.
  They cannot be camera-confirmed, because YOLOv8's COCO classes have no mapping for them, and mAP
  averages over all ten classes. Under the default camera-gated spawn policy they cannot start new
  tracks; `C_6cam_lidar_spawn` disables that spawn gate and is the explicit experiment testing
  LiDAR-only spawning. A per-class spawn threshold is the next step for these four classes.
- **The detection submission is built from tracker state**, so its positions are Kalman-filtered and an
  object must persist long enough to become a track. Legitimate temporal aggregation, but these are
  not raw-detector numbers.
- **Centre-NMS is not transitive.** `nms.center_nms` compares each candidate only against already-kept
  boxes, so a duplicate chain spaced just under the radius can survive at both ends. Near-coincident
  duplicates, the case that matters, do collapse.
- The reported runs were produced on Kaggle with a Tesla T4; the dev/test path in this repository was
  verified on CPU.

## License

MIT — see [LICENSE](LICENSE).
