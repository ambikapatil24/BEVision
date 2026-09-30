# Refactoring notes — from a 16.6 MB notebook to a package

This document records what changed in the move from the original research notebook to
`src/bevision/`, and — more importantly — **which of those changes altered the results.**
A refactor that quietly moves the numbers is not a refactor, it is a rewrite with plausible
deniability.

---

## Part 1 — What the refactor found

The original project's headline was:

> mAP 0.027 → **0.199 (×7.3)** · AMOTA 0.019 → **0.115 (×6)** · NDS 0.072 → 0.183

Three of those six numbers did not survive contact with the artifacts. The four findings
below are the reason the rewrite was worth doing.

### Finding 1 — The baseline was never measured

No run in the project produced **0.027 / 0.072 / 0.019**. Those figures appear in exactly one
place: a markdown cell. Every pipeline that emits a submission hard-codes all six cameras:

```python
camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
```

There was no front-camera path to run. The measured baseline is **mAP 0.0765 / AMOTA 0.1397**
— the claimed numbers were roughly 2.8× too pessimistic on mAP and 7.4× on AMOTA. The honest
gain is **×2.83**, not ×7.3.

**Fix.** `cameras` is a `PerceptionConfig` field, so the baseline is one flag
(`--cameras front`) and `scripts/check_results.py` re-derives the table from committed
evaluator output.

### Finding 2 — Class-blind association was manufacturing phantom boxes

This is the one that mattered. Association was Hungarian matching on 3D distance with no
class constraint, and the emitted class came from the *matched detection*:

```python
for t_idx, d_idx in matched:
    tr.update(lidar_dets[d_idx]['center_global'][:3])
    track_box[tr.track_id] = lidar_dets[d_idx]     # ← relabels the track
```

So a car track updated by a nearer truck detection **became** a truck. nuScenes tracking
scores class by class, so every such update produced a false negative for `car` *and* a false
positive for `truck`.

The fingerprint was already visible: `truck` had **329 false positives against 95
ground-truth objects**. Cars have 2188 ground-truth objects, so a few percent of car tracks
mislabelled as `truck` accounts for exactly that. The original README attributed truck's
0.00 AMOTA to mis-localisation; it was mislabelling.

**Fix.** The class constraint goes *inside* the cost matrix rather than being filtered
afterwards — an optimizer can only respect a constraint that is part of its objective:

```python
cost = np.where(track_labels == detection_labels, cost, FORBIDDEN_PAIR_COST)
```

`FORBIDDEN_PAIR_COST` is finite (`1e6`) rather than `np.inf` on purpose:
`scipy.optimize.linear_sum_assignment` raises `ValueError` on an infeasible `inf` matrix,
which happens as soon as one track has no same-class detection to pair with — routine for
sparse classes.

**Measured effect** (`B0_repro` → `B_6cam`, everything else identical):

| | mAP | NDS | AMOTA | MOTA | Recall | FP | IDS | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | ×1.11 | **×0.99** | **×3.24** | **×3.58** | ×1.20 | **−49%** | **−42%** | +8.7% |

False positives falling *while* recall rises is the signature of a correctness fix. Loosening
a threshold moves recall and FP together.

There is a second effect worth naming: with the constraint in place, detections that were
previously absorbed by a wrong-class track land in the unassigned set and spawn their own
tracks, so recall rises as well.

### Finding 3 — Four classes were impossible to score, by construction

Tracks could only be born from **camera-confirmed** detections, but YOLO predicts COCO
classes, which have no counterpart for `trailer`, `construction_vehicle`, `traffic_cone` or
`barrier`:

```python
if d['score'] > score_high and d['fused']:   # camera-confirmed spawn only
```

Those four classes could never confirm, never spawn, never be submitted — so they scored
AP 0.000, and since mAP averages over all ten classes, they capped it.

My first hypothesis was that the spawn *policy* was the blocker. **Run C falsified it**:
setting `SPAWN_REQUIRES_CAMERA=False` still left all four at 0.000. The actual blocker turned
out to be the score gate. Counting raw PointPillars detections above the 0.30 spawn threshold
across all 81 samples:

| Class | Above 0.30 | Total detections |
|---|---:|---:|
| `trailer` | **0** | 681 |
| `barrier` | **2** | 3188 |
| `construction_vehicle` | **4** | 1201 |
| `traffic_cone` | **9** | 3012 |

PointPillars sees these classes fine — `barrier` is normally one of its strongest — but at
low confidence, and the spawn gate discards them before they can be ranked by the evaluator.
nuScenes AP integrates precision over recall from the score ranking, so low-confidence boxes
still earn AP; throwing them away earns nothing.

**Status: diagnosed, not fixed.** The fix is a per-class spawn threshold, which is listed as
the next step in the README. Claiming a fix without measuring it would repeat exactly the
mistake in Finding 1.

### Finding 4 — One filter served two different class sets

```python
if name not in TRACKING_CLASSES: continue    # comment says "tracking scores only 7 classes"
...
dets.append({...})     # detection submission
trks.append({...})     # tracking submission
```

The `continue` sits above **both** appends. nuScenes scores detection over ten classes and
tracking over seven, so `construction_vehicle`, `traffic_cone` and `barrier` were dropped from
the detection submission as a side effect of a rule that only applies to tracking.

**Fix.** `DetectionClassPolicy` is a separate config field, and `evaluation/submission.py`
builds the two submissions with two functions and two class sets. A test asserts that the
default detection submission carries all ten classes.

### Finding 5 — Every submitted box had its width and length transposed

Found last, after the ablation, while checking a suspicion about the mmdet3d tensor layout.

`LiDARInstance3DBoxes.tensor` stores `(x, y, z, dx, dy, dz, yaw)` where `dx` is the extent along
the box's own x-axis — the **length**. nuScenes' submission schema expects
`size = (width, length, height)`. The original code passed `tensor[3:6]` straight through:

```python
'size': [float(v) for v in t[3:6]],
```

so width and length were swapped in every box ever submitted.

**How it was confirmed.** Reading nuScenes ground truth and the matching predictions for the same
frame, all six matched vehicle boxes had `dx > dy`, while every ground-truth vehicle has
`length > width`. Interpreted as `(w, l, h)` those predictions were cars *5.5 m wide and 1.4 m
long* — geometrically absurd. Interpreted as `(l, w, h)` they were ordinary vehicles. Summed size
error fell 4.2× under the swap hypothesis.

**Then confirmed end-to-end**, which is the part that matters. The committed submissions were
re-evaluated with *only* `size` corrected. Across all five runs:

| | mAP | mATE | mASE | NDS |
|---|---|---|---|---|
| change | **+0.0000** (5/5) | **+0.0000** (5/5) | −0.10 to −0.23 | **+0.010 to +0.023** |

mAP and mATE being *bit-identical* is the proof: nuScenes matches detection and tracking boxes by
**centre distance** and never consults size. A pure size error therefore cannot touch them, and
it did not. Only `mASE` — and through it NDS — moved.

**Fix.** `size=(box[4], box[3], box[5])`, in `detection/lidar.py`. The conversion is a standalone
function so the ordering, which is the one thing here that fails *silently*, is unit-tested
without mmdetection3d installed.

**The honest caveat about the first diagnostic.** The initial check printed *"no swap needed"* and
that verdict was wrong. It summed absolute size differences across all classes, where both
hypotheses are equally wrong because PointPillars' size regression is genuinely poor (mASE 0.84),
and 15 of 50 pairs were exact ties. It also weighted classes that cannot answer the question:
pedestrians are near-square (`0.78 × 0.77`), and nuScenes barriers are *wider than they are long*
(`1.98 × 0.70`), which inverts the expected direction. The decisive test is the **shape** of
elongated objects, where width and length cannot be confused.

### Finding 5b — and the z was the box bottom, not its centre

The same misread tensor produced a second error. mmdet3d anchors LiDAR boxes at their **bottom
face** (`origin = (0.5, 0.5, 0)`), so `tensor[2]` is the bottom. The original code took it as the
centre:

```python
center = np.array([t[0].item(), t[1].item(), t[2].item()])
```

**How it was caught.** Comparing my `box_corners()` against mmdet3d's own `bboxes_3d.corners` on
a real PointPillars run, the disagreement was *exactly* `h/2` for all eight boxes checked:

| class | h | h/2 | observed max corner diff |
|---|---|---|---|
| pedestrian | 1.75 | 0.875 | 0.875805 |
| pedestrian | 1.87 | 0.935 | 0.936135 |
| car | 1.62 | 0.810 | 0.808891 |
| car | 1.75 | 0.875 | 0.873277 |

x/y corners matched exactly (a rotation error would have shown a difference of metres, not
`h/2`), so only the z origin was wrong.

**Two consequences, in opposite directions.**

In the *pipeline* it was fatal. A 0.85 m lift error moves a projected box ~67 px down a 900 px
image; against YOLO boxes only 46 px tall that is zero overlap, so **every** camera confirmation
failed, no track could spawn, and the smoke run wrote **empty submissions** — the pipeline ran
cleanly and produced nothing. Three of the four previously-untested modules were verified in the
process, and this was the one real defect the end-to-end run found.

In the *reported metrics* it changes nothing, and the numbers say so themselves:

- **mATE is xy-only** — the devkit's `center_distance` uses `translation[:2]`. That is the only
  way a car can report ATE 0.349 while its z is 0.84 m off.
- **mASE aligns centres before the 3D IoU**, so a pure translation error cannot leak into scale
  error. Car ASE of 0.338 matches a centre-aligned computation from the size error alone.

So this fix repairs the pipeline without disturbing any published number — no re-evaluation, no
artifact churn. It also confirms the same error is in the committed submissions: across classes,
mean z is flat at ~0.1 m (ground level) regardless of height, where a centre convention would put
buses ~1 m above motorcycles.

### Two smaller correctness fixes

- **Projection had no depth guard.** `view_points(..., normalize=True)` divides by depth, so a
  corner behind the camera (z ≤ 0) is mirrored through the focal plane and lands *inside* the
  image. A car behind the ego vehicle produced a plausible-looking 2D box and spurious IoU
  matches. `project_box_to_image` now rejects the box before dividing, and the behaviour is
  pinned by three tests.
- **The fused confidence blended against a constant.** The original computed
  `0.5 * lidar + 0.5 * 1.0`, i.e. it added a fixed boost to every match regardless of how
  confident the camera was. Two uncertain detections (0.20 and 0.30) scored **0.60** instead
  of 0.25, which also meant the spawn threshold no longer meant what it claimed.

---

## Part 2 — What deliberately did **not** change

Faithfulness was a requirement, not a nicety: the refactored pipeline must be able to
reproduce the original results exactly, or the ablation above would be measuring the refactor
rather than the fix.

`B0_repro` is that check. `run_tag=B0_repro` sets `class_aware_track_matching=False` and
`detection_class_policy=tracking` — the original behaviour — and must reproduce the original
on the metrics the box-order fix cannot touch: **mAP 0.1913 and AMOTA 0.1071**, exactly, because
both are computed from centre distance and never from box size. `scripts/check_results.py`
asserts those two and reports NDS rather than asserting it — NDS moved, for the good reason in
Finding 5, and pretending otherwise would be the same category of error as the original ×7.3.

Preserved deliberately:

| Behaviour | Why it looks like a bug but was kept |
|---|---|
| Centre-NMS is not transitive | It compares only against *kept* boxes, so a chain spaced just under the radius survives at both ends. The original does exactly this; "fixing" it would move results for no measured gain. Documented as a limitation instead. |
| Detection output is built from tracker state | Positions are Kalman-filtered and an object must persist to become a track. Legitimate temporal aggregation, but it means the detection numbers are not raw-detector numbers. Now stated in the README. |
| Tracks coast for `max_age - 1` frames after their last detection | An `FN`/`FP` trade-off, now a single config field rather than two different constants in two different cells. |

Note what is **not** in that list any more: the box-ordering question started here as an
honest "flagged, not guessed", and became Finding 5 once the data settled it. A flagged
suspicion is only respectable while it is still open.

That last one is a refactor finding in its own right: **the notebook's two pipelines disagreed
about track lifetime.** The single-scene demo used `time_since_update < 10`; the pipeline that
produced the published numbers used `< 5`. Same tracker class, same thresholds, different
answer to "how long does a lost object survive" — and nothing recorded which was intended.

---

## Part 3 — Code-level changes

### Duplication

Projection, fusion and the frame loop existed in **three divergent implementations** (a demo
cell, a 39-frame statistics cell, and the final pipeline). Divergent rather than merely
duplicated: the demo used greedy matching, the statistics cell used Hungarian matching, and
the final pipeline used best-of-six-cameras with a different spawn rule. Their outputs were
not comparable to each other. All three now call one implementation in `fusion.py`.

The projection helper was also copy-pasted inline in three more places, which is how the
depth-check bug survived.

### Performance

The original used `pyquaternion.Quaternion.rotate` on each of a box's eight corners, inside a
Python list comprehension, inside a per-box loop — and then recomputed the identical
LiDAR→ego transform once per camera, discarding it whenever the box turned out to be behind
that camera. With ~50 boxes, 6 cameras and 81 frames that is roughly 24,000 redundant
transformations per run.

Everything is now batched matrix products:

```python
corners_ego = corners_sensor @ sensor_rotation.T + sensor_translation
```

The same treatment applies to the IoU matrix (was a double Python loop calling a scalar
function), the track↔detection distance matrix (was a nested comprehension computing ~3,600
3-vectors per frame), and centre-NMS (was a per-detection Python scan).

### Dependency surface

The core depends on **numpy and scipy only**. `pyquaternion` is gone — a quaternion is four
numbers and a 15-line conversion — and `filterpy` was replaced by
`tracking/kalman.py`. That replacement is not taken on trust: `tests/test_kalman.py` runs the
numpy filter and filterpy side by side on identical measurements and asserts the state *and*
covariance agree to `1e-9` at every step. filterpy's defaults were probed first
(`P = Q = R = F = I`), which is what makes the notebook's `P *= 5.0` mean `P = 5·I`.

More importantly, **`import bevision` loads none of torch, OpenMMLab, the nuScenes devkit,
OpenCV or Ultralytics.** The perception stack sits behind adapters, so the geometry, fusion,
tracking, submission and pipeline logic are all testable on a laptop — which is the only
reason 229 tests can exist for a system whose real input is 4 GB of LiDAR.

### Configuration

Every threshold lived as a module-level global (`score_low`, `iou_thr`, `dist_thr`,
`image_width`) or was hard-coded mid-loop (`0.5` for dt, `1600`/`900` for image bounds,
`radius=2.0`). They are now fields on a frozen `PerceptionConfig`, and each of the five runs
in the results table is a named preset. The ablation is a set of configs rather than a set of
remembered notebook states.

---

## Part 4 — What to do differently

1. **A claimed number needs a committed artifact.** Every figure in the README now comes from
   a `metrics_summary.json` in `results/`, and `scripts/check_results.py` fails if the table
   and the artifacts disagree. It is genuinely wired to fail: corrupting one value makes it
   exit non-zero.
2. **A reproduction check first, then fixes.** Having `B0_repro` reproduce the original
   exactly is what turns "AMOTA went up" into "AMOTA went up *because of this one flag*".
3. **State the trade-off, don't hide it.** Run C raises recall 0.5023 → 0.5628 and *lowers*
   AMOTA and MOTA. It is reported as a measured trade-off that was not adopted, which is a
   better answer than a selectively quoted win.
4. **Distinguish "quoted" from "relied on".** Two runs in the README have no committed
   artifact. They are marked as such and no claim depends on them, rather than being dropped
   or quietly trusted.
