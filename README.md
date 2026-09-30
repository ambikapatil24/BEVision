# BEVision — Camera + LiDAR Fusion Perception for Autonomous Driving

Multi-sensor 3D perception on **nuScenes `mini_val`**: YOLOv8 (6-camera 2D detection) +
PointPillars (3D LiDAR detection) → LiDAR-anchored projection fusion → global-frame Kalman
tracking → Bird's-Eye View, scored with the **official nuScenes evaluator** (mAP / NDS /
AMOTA / MOTA).

**Headline:** six-camera surround fusion improves **mAP ×2.83** over a single front camera
with the association policy held fixed. Making track association class-consistent cut
**false positives by 49%** and raised **AMOTA ×3.24**. End to end, the original single-camera
pipeline → the final six-camera pipeline is **mAP ×3.14 / AMOTA ×4.38**.

Every figure above is measured between runs whose evaluation artifacts are committed in
`results/`. Nothing here depends on a run you cannot inspect.

---

## Results

All rows are `mini_val` (81 samples / 2 scenes), produced by the official
`nuscenes-devkit` evaluator. The table is machine-checkable against the artifacts:

```bash
python scripts/check_results.py
```

> **A box-ordering bug was found and fixed after the ablation.** mmdet3d's `tensor[3:6]` is
> `(length, width, height)` while nuScenes expects `(width, length, height)`, so every submitted
> box had width and length transposed. Re-evaluating the committed submissions with **only**
> `size` corrected left **mAP and mATE bit-identical across all five runs** — proof it is a pure
> size error, since nuScenes matches boxes by centre distance and never consults size — while
> raising NDS by 0.010–0.023. The corrected values are in the tables below, the fix is in
> `detection/lidar.py`, and the full account is
> [Finding 5](docs/REFACTORING.md#finding-5--every-submitted-box-had-its-width-and-length-transposed).
>
> The consequence worth flagging: `B0_repro`'s NDS (0.2019) is marginally **higher** than
> `B_6cam`'s (0.2000), so **NDS does not support the class-consistency fix.** That fix wins on
> mAP ×1.11, AMOTA ×3.24, recall and false positives; the extra tracks it spawns simply carry
> more marginally-localised matches. Reported as measured rather than dropped.

### The three runs that carry the argument

These are the only runs cited for a claim, and all three have their submission JSONs and
eval summaries committed under `results/`.

| Run | Cameras | Class-consistent association | mAP | NDS | AMOTA | MOTA | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `A0_front_cam_class_blind` | 1 (front) | no | 0.0676 | 0.1138 | 0.0791 | 0.0884 | 0.1767 |
| `B0_repro` | 6 | no | 0.1913 | 0.2019 | 0.1071 | 0.0922 | 0.4172 |
| **`B_6cam`** | 6 | **yes** | **0.2121** | **0.2000** | **0.3466** | **0.3303** | **0.5023** |

`B0_repro` is the load-bearing row: it runs the refactored pipeline with
`class_aware_track_matching=False` and **must** reproduce the original research results. It does
on **mAP 0.1913 and AMOTA 0.1071**, exactly. Its NDS is higher than the original's (0.2019 vs
0.1795) because the original shipped the transposed-box bug described above and we fixed it —
which is why `scripts/check_results.py` asserts the reproduction property on the two
size-independent metrics and reports NDS rather than asserting it.

### Comparison 1 — camera count (`A0` → `B0`, association held at the original class-blind matcher)

This is the honest replacement for the original project's headline, which claimed the same
comparison at **×7.3**:

| | mAP | NDS | AMOTA | MOTA | Recall |
|---|---:|---:|---:|---:|---:|
| Gain | **×2.83** | ×1.77 | ×1.35 | ×1.04 | **×2.36** |

### Comparison 2 — class-consistent association (`B0` → `B`, six cameras on both sides)

One config flag:

| | mAP | NDS | AMOTA | MOTA | Recall | FP | IDS | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gain | ×1.11 | **×0.99** | **×3.24** | **×3.58** | ×1.20 | **−49%** | **−42%** | +8.7% |

False positives falling *while* recall rises is the signature of a correctness fix rather
than a threshold trade-off: loosening a gate would move recall up and FP up together. NDS is
the exception — it slips 0.9%, because the extra tracks bring more marginally-localised
matches. The win here is in mAP and AMOTA, and saying so is more useful than hiding it.

### Combined — original single-camera pipeline (`A0`) → final pipeline (`B`)

| | mAP | NDS | AMOTA | MOTA | Recall |
|---|---:|---:|---:|---:|---:|
| Gain | **×3.14** | ×1.76 | **×4.38** | ×3.74 | **×2.84** |

### The remaining two runs

Both are committed under `results/` like the others.

| Run | Cameras | Class-consistent association | mAP | NDS | AMOTA | MOTA | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `A_front_cam` | 1 (front) | yes | 0.0765 | 0.1181 | 0.1397 | 0.1531 | 0.1793 |
| `C_6cam_lidar_spawn` | 6 | yes + LiDAR-only spawn | 0.2133 | 0.2129 | 0.3404 | 0.3027 | 0.5628 |

`A` corroborates the camera-count result at the corrected association policy — mAP ×2.77,
consistent with the ×2.83 measured at the original policy. `C` shows that relaxing the
camera-confirmation spawn gate raises recall 0.5023 → 0.5628 while *lowering* AMOTA and MOTA:
a measured trade-off, which is why `B_6cam` is the reported configuration rather than `C`.

### Per-class (`B_6cam`)

| Class | AP | AMOTA | | Class | AP | AMOTA |
|---|---:|---:|---|---|---:|---:|
| car | 0.511 | 0.406 | | motorcycle | 0.000 | 0.000 |
| truck | 0.631 | **0.822** | | bicycle | 0.030 | 0.000 |
| bus | 0.608 | 0.513 | | trailer | 0.000 | — |
| pedestrian | 0.341 | 0.338 | | construction_vehicle | 0.000 | — |
| | | | | traffic_cone | 0.000 | — |
| | | | | barrier | 0.000 | — |

Tracking detail (`B_6cam`): TP 1718 · FP 546 · FN 1883 · IDS 68 · MT 49 · ML 94.

---

## What the refactor found

Cleaning up the research notebook was not cosmetic — four findings changed the project's
claims.

**1. The advertised baseline was wrong.** The original write-up claimed a front-camera
baseline of mAP 0.027 / AMOTA 0.019 and a **×7.3** improvement. No run in the project ever
produced those numbers. The measured baseline is **0.0765 / 0.1397**, so the honest gain is
**×2.8**. Reporting ×7.3 is the kind of claim a reviewer checks against the artifact and
finds nothing behind.

**2. Track association was class-blind, and it was manufacturing phantom boxes.** The
matcher assigned tracks to detections purely by 3D distance, and the emitted class came from
the matched detection — so a car track updated by a nearer truck detection *became* a truck.
nuScenes tracking scores class by class, so each such update produced a false negative for
`car` and a false positive for `truck`. `truck` had 329 FP against only 95 ground-truth
objects; that ratio is the fingerprint. Constraining the match to same-class pairs (with the
constraint inside the cost matrix, not filtered afterwards) removed them: FP 1061 → 546,
IDS 118 → 68, AMOTA 0.107 → 0.347. A unit test reproduces both behaviours.

**3. Four classes were structurally impossible to score.** The pipeline spawned tracks only
from camera-confirmed detections, but YOLO predicts COCO classes, which have no counterpart
for `trailer`, `construction_vehicle`, `traffic_cone` or `barrier`. Those four could never
spawn a track, so they were never submitted, so they scored AP 0.000 — and because mAP
averages over all ten classes, they capped it. Measured detection counts above the 0.30
spawn threshold across the whole split: **trailer 0 of 681, barrier 2 of 3188, traffic_cone
9 of 3012, construction_vehicle 4 of 1201.** PointPillars detects these classes fine but at
low confidence, and the spawn gate discards them before they can be scored.

**4. The detection submission was silently restricted to tracking's class set.** One
`continue` statement served both submissions; nuScenes scores detection over ten classes but
tracking over seven, so three classes were dropped from the detection metrics as a side
effect. The two policies are now separate config fields (`detection_class_policy`).

Two smaller correctness fixes: the projection step had no depth guard, so boxes behind the
camera were mirrored through the focal plane into plausible-looking in-image boxes; and the
fused confidence blended against a hard-coded `1.0` instead of the camera's actual
confidence, which added a constant boost and destroyed the signal.

---

## Pipeline

```
   6 × CAMERA (YOLOv8n)                   LiDAR_TOP (PointPillars)
          │                                       │
          │ 2D boxes + COCO class ids             │ 3D boxes (x,y,z,l,w,h,yaw)
          ▼                                       ▼
   ┌───────────────────────────────────────────────────────────┐
   │  FUSION — LiDAR is the anchor                             │
   │  3D box → every camera; class-consistent IoU match;       │
   │  score = 0.5·lidar + 0.5·camera                           │
   └───────────────────────────┬───────────────────────────────┘
                               ▼
   ┌───────────────────────────────────────────────────────────┐
   │  TRACKING — Kalman (constant velocity, global frame)      │
   │  predict → class-consistent Hungarian match → update      │
   │  → spawn (camera-gated) → prune (max_age)                 │
   └───────────────────────────┬───────────────────────────────┘
                               ▼
          BEV render  │  submission JSONs → official nuScenes eval
```

| Module | Responsibility |
|---|---|
| `geometry` | Batched rigid transforms, box corners, projection, IoU, yaw |
| `classes` | COCO ↔ nuScenes class spaces, tracking class set, submission attributes |
| `fusion` | Best-camera confirmation; class-gated Hungarian assignment |
| `nms` | Class-conditional centre-NMS duplicate suppression |
| `tracking.kalman` | Constant-velocity 3D filter, pure numpy |
| `tracking.tracker` | Association and track lifecycle |
| `config` | Every threshold and policy switch, plus the run presets |

The geometry core depends on **numpy only** — no torch, no OpenMMLab, no pyquaternion — so
it is unit-testable anywhere. `import bevision` succeeds on a machine with none of the
perception stack installed.

---

## Design decisions

1. **LiDAR-anchored fusion.** Cameras confirm and boost confidence; they never create, move
   or delete a 3D box. Depth comes from the point cloud.
2. **Class constraint inside the assignment matrix.** Putting the constraint *into* the
   objective stops the optimizer spending a match on a pair that will be discarded later.
   *An optimizer can only respect a constraint that is part of its objective.*
3. **Global-frame tracking.** Positions are lifted to world coordinates via `ego_pose`, so
   objects stay tracked as the ego vehicle drives past them.
4. **Camera-gated track birth.** LiDAR-only detections may update existing tracks but not
   start new ones by default — fewer ghost tracks. This is also the mechanism behind
   finding 3, and it is now a config flag precisely so that cost is visible.
5. **Reproduction as a test.** The refactored pipeline is run with
   `class_aware_track_matching=False` and must reproduce the original results. It does — mAP
   0.1913 and AMOTA 0.1071 exactly, which is what makes the single-variable comparison in the
   results table trustworthy. (NDS deliberately differs: the original shipped the transposed-box
   bug and we fixed it.)

---

## Install

The geometry, fusion and tracking cores need only numpy and scipy:

```bash
git clone https://github.com/<you>/BEVision.git
cd BEVision
pip install -e ".[dev]"
pytest
```

The perception stack is CUDA-specific and installed separately:

```bash
pip install -e ".[camera,nuscenes]"
bash scripts/install_mmdet3d.sh          # torch + mmcv + mmdet3d, pinned to one set
```

> **Python ≤ 3.12.** mmcv 2.2.0 ships no cp313 wheels and Python 3.13 removed `distutils`,
> which its build toolchain needs. Use a 3.11 or 3.12 runtime.

**Data and weights are external** (not in this repo): nuScenes mini (~4 GB) from
[nuscenes.org](https://www.nuscenes.org/nuscenes#download), `yolov8n.pt` (auto-downloaded by
Ultralytics) and the PointPillars checkpoint
`hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth` from the
mmdetection3d model zoo.

---

## Usage

```python
from bevision import PerceptionConfig, box_corners, project_box_to_image

config = PerceptionConfig()  # six-camera, class-aware
baseline = config.replace(cameras=("CAM_FRONT",))  # the §results baseline
```

Every run in the results table is a preset in `bevision.config`:

```python
from bevision.config import (
    front_camera_class_aware,  # A
    front_camera_class_blind,  # A0
    surround_class_blind,  # B0  — the reproduction check
    surround_class_aware,  # B
    surround_lidar_spawn,  # C
)
```

```bash
bevision-run  --dataroot /data/nuscenes-mini --run-tag B_6cam
bevision-eval --dataroot /data/nuscenes-mini --run-tag B_6cam
```

Every run in the results table is one command. The ablation is driven by three flags, which
is what makes the table reproducible rather than a set of remembered notebook states:

| Run | Command |
|---|---|
| `B_6cam` | `bevision-run --dataroot $D --run-tag B_6cam` |
| `B0_repro` | `… --run-tag B0_repro --association class-blind --detection-classes tracking` |
| `A0_front_cam_class_blind` | `… --run-tag A0 --cameras front --association class-blind` |
| `A_front_cam` | `… --run-tag A --cameras front` |
| `C_6cam_lidar_spawn` | `… --run-tag C --spawn lidar-allowed` |

`bevision-run` also validates the submission it just wrote against the nuScenes schema
before exiting, and returns non-zero if anything is malformed — three classes were once
silently missing from the detection submission, which no test caught at the time.

Using the pipeline as a library, against your own data source:

```python
from bevision import PerceptionPipeline, PerceptionConfig

pipeline = PerceptionPipeline(
    camera_detector=my_camera_detector,  # any callable: image path -> CameraDetections
    lidar_detector=my_lidar_detector,  # any callable: sweep path -> list[RawDetection3D]
    class_names=("car", "truck", ...),  # the detector's own ordering
    config=PerceptionConfig(),
)
result = pipeline.run(my_frame_source)  # anything with .frames() -> Iterable[Frame]
```

Rendering a BEV panel needs `pip install -e ".[viz]"`. The renderers take global-frame data
and any matplotlib `Axes`, so they need neither a dataset nor a GPU:

```python
from bevision import BevFrame, render_bev_figure

frames = [
    BevFrame(
        frame_index=i,
        points_global=points,  # (N, 3), global frame
        ego_rotation=ego_rotation,
        ego_translation=ego_translation,
        detections=detections,  # Sequence[Detection3D]
        tracks=tracks,  # Sequence[TrackOutput]
    )
    for i, (points, ego_rotation, ego_translation, detections, tracks) in enumerate(moments)
]
figure, axes = render_bev_figure(frames, suptitle="four moments")  # 2x2, like the figures above
```

The 2D overlay is pure numpy — no OpenCV — and returns a new array rather than mutating its
input:

```python
from bevision import overlay_frame

annotated = overlay_frame(image, camera_detections, projected=lidar_boxes, fused=confirmed_boxes)
# green = camera only, red = LiDAR projected, yellow = fused
```

---

## Limitations

- **`mini_val` only** — 2 scenes. `bus` has 33 ground-truth objects and `truck` 95, so their
  AMOTA figures are extremely variant-sensitive and are **not** comparable to published
  `val` numbers. Treat the per-class columns as directional.
- **Pretrained detectors, no fine-tuning.** The biggest available lever.
- **Detection output is tracker output.** The detection submission is built from surviving
  track state, so its positions are Kalman-filtered and an object must persist long enough to
  become a track. That is legitimate temporal aggregation, but it means the detection numbers
  are not raw-detector numbers, and the README says so rather than leaving it to be found.
- **Centre-NMS is not transitive** — it compares only against kept boxes, so a duplicate
  chain spaced just under the radius can survive at both ends. Near-coincident duplicates,
  the case that matters, do collapse.
- **`C_lidar_spawn` raises recall (0.5023 → 0.5628) but lowers AMOTA and MOTA.** It is a
  measured trade-off, not a win, which is why `B_6cam` is the reported configuration.

### Next step with the clearest payoff

A **per-class spawn threshold**. The diagnostic in finding 3 shows `barrier` and
`traffic_cone` are detected in the thousands but almost entirely below 0.30 — barrier in
particular is normally one of PointPillars' strongest classes. Relaxing the spawn gate for
the four COCO-less classes (and submitting them) should recover mAP that is currently pinned
at 0.000 by construction, at a cost in false positives that is measurable.

---

## Repo layout

```
BEVision/
├── src/bevision/
│   ├── config.py         # all thresholds + the five run presets
│   ├── classes.py        # COCO ↔ nuScenes, tracking class set
│   ├── geometry.py       # batched transforms, projection, IoU (numpy only)
│   ├── fusion.py         # best-camera confirmation, class-gated Hungarian
│   ├── nms.py            # class-conditional centre-NMS
│   ├── tracking/         # kalman.py, tracker.py
│   ├── detection/        # base.py (protocols), camera.py (YOLO), lidar.py (PointPillars)
│   ├── data/             # nuscenes_loader.py
│   ├── evaluation/       # submission.py (build/validate), runner.py (official eval)
│   ├── visualization/    # bev.py (BEV panels), image.py (2D box overlays)
│   ├── pipeline.py       # detect → fuse → track → emit, over a FrameSource
│   ├── cli.py            # bevision-run / bevision-eval
│   └── types.py
├── tests/                # 229 tests
├── results/              # one dir per run: submission JSONs + eval summaries (the evidence)
│   └── README.md         # provenance, and which artifacts are still outstanding
├── scripts/
│   ├── check_results.py      # re-derives the results table from the artifacts
│   └── install_mmdet3d.sh
└── docs/
    ├── ARCHITECTURE.md       # layering, data flow, interfaces, invariants
    ├── REFACTORING.md        # notebook → package, and every behaviour change
    ├── SETUP.md              # environment, data, and pushing to GitHub
    ├── INTERVIEW_NOTES.md    # talking points, expected questions, what not to claim
    └── BLOG_OUTLINE.md       # write-up outline
```

### What is verified where

The package is split so that nearly all of the logic is testable without a GPU, and the
third-party boundary is exercised separately against the real models:

| Module | Verified by | Needs |
|---|---|---|
| `config`, `classes`, `geometry`, `fusion`, `nms`, `tracking`, `evaluation.submission`, `pipeline` | 229 unit tests, incl. the pipeline driven by **fake detectors** | numpy, scipy |
| `visualization` | unit tests, headless via matplotlib's Agg backend — the BEV panel asserts the ego-frame transform numerically, not just that something was plotted | matplotlib |
| `data.nuscenes_loader`, `detection.camera`, `detection.lidar`, `pipeline` end to end | **smoke-tested on Kaggle against the real PointPillars checkpoint and YOLOv8n weights** | GPU + dataset |
| `cli.eval_main` | **not yet run** — the official-evaluator wiring | GPU + dataset |

The end-to-end smoke run (2 frames of `mini_val`, 6 cameras) reported 92 LiDAR detections,
18 camera confirmations, 13 spawned tracks and 21 emitted tracks, and wrote both submission
files. It also found **two real defects that no unit test could have**, because the unit tests
inject objects rather than a live model's tensors:

1. `bboxes_3d.tensor` is `(N, 9)` — velocity columns appended — where the code assumed `(N, 7)`.
2. mmdet3d anchors LiDAR boxes at their **bottom face**, so `tensor[:, 2]` is the bottom, not the
   centre. Reading it as the centre shifted every projected box ~67 px down a 900 px image,
   against YOLO boxes only ~46 px tall — so nothing overlapped, zero tracks spawned, and the
   pipeline wrote *empty* submissions while running perfectly cleanly.

Both are fixed, with regression tests pinned to the exact shapes and offsets observed. See
[Finding 5](docs/REFACTORING.md).

That boundary is deliberate: the pipeline talks to detectors through the protocols in
`detection/base.py`, so `tests/test_pipeline.py` runs the real fusion, tracking and geometry
code against fakes and needs no torch at all.

## Stack

`python 3.12` · `torch 2.2.0+cu118` · `mmcv 2.2.0` · `mmdet 3.x` · `mmdetection3d 1.4.0` ·
`ultralytics (YOLOv8n)` · `nuscenes-devkit` · `numpy` · `scipy`

## Documentation

| | |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layering, data flow, the three protocols, invariants |
| [docs/REFACTORING.md](docs/REFACTORING.md) | Notebook → package: every finding, every behaviour change, and what was deliberately left alone |
| [docs/SETUP.md](docs/SETUP.md) | Environment, data, weights, running it, and the GitHub walkthrough |
| [docs/INTERVIEW_NOTES.md](docs/INTERVIEW_NOTES.md) | Talking points, expected questions, what not to claim |
| [docs/BLOG_OUTLINE.md](docs/BLOG_OUTLINE.md) | Write-up outline |
| [results/README.md](results/README.md) | Provenance for every number, and which artifacts are outstanding |

## License

MIT — see [LICENSE](LICENSE).
