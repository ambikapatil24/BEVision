# `results/` — the evidence behind every number in the top-level README

Each subdirectory is one ablation run and holds exactly seven files:

| File | What it is |
|---|---|
| `results_detection.json` | Detection submission in **official nuScenes format** — what was fed to the evaluator |
| `results_tracking.json` | Tracking submission, same format |
| `eval_detection_metrics_summary.json` | Output of `nuscenes.eval.detection.evaluate.DetectionEval` |
| `eval_tracking_metrics_summary.json` | Output of `nuscenes.eval.tracking.evaluate.TrackingEval` |
| `eval_detection_metrics_details.json` | The detection evaluator's **raw curve data** — see below |
| `eval_tracking_metrics_details.json` | Same for tracking |
| `headline_metrics.json` | The headline numbers, **extracted from the two summaries above rather than typed by hand** |

Submission JSONs are committed on purpose. They are the only artifact that lets someone
re-run the evaluator and get these exact numbers back, and re-deriving them needs a GPU and
a multi-GB dataset. Rendered `plots/*.pdf` are excluded — they are drawn from
`metrics_details.json`, which *is* committed, so they regenerate without re-running anything.

### About `metrics_details.json`, and what is **not** verified

These two files are the devkit's own output, committed unmodified. Each is keyed
`"<class>:<distance_threshold>"` — ten classes × four thresholds — and each node holds the
`recall`, `precision`, `confidence` and `<metric>_err` arrays (101 sampled points) that the PR
and TP curves are drawn from. They are strictly more information than the summary.

**What I checked:** the structure (40 nodes per run), and that the per-class APs quoted in the
README can be *approached* by re-integrating the PR arrays.

**What I could not do:** reproduce nuScenes' AP exactly. Raw VOC-style integration
(`Σ Δrecall · precision` over the monotone envelope) lands on average **+0.069** too high;
clipping the integral at the config's `min_recall = 0.1` lands **0.022** off on average but
*undershoots* cars and trucks. So the devkit's precise `min_recall`/`min_precision` handling has
not been reproduced here, and **no claim in the README rests on this reconstruction** — the
numbers quoted are read from `metrics_summary.json`, which is the evaluator's own aggregate.

Anyone wanting to close that gap can run `DetectionMetrics._compute_ap` from the devkit against
these arrays directly.

## Box-order correction — applied

Every committed detection submission originally carried `size = (length, width, height)` where
nuScenes expects `(width, length, height)` — mmdet3d's tensor order passed straight through by
the original code. The fix is in `detection/lidar.py`; the full account is
[Finding 5](../docs/REFACTORING.md#finding-5--every-submitted-box-had-its-width-and-length-transposed).

Applying it needed no GPU, because `size` is pure post-processing on the submission: each one was
corrected and re-evaluated. The check that this was safe is that **mAP and mATE came back
bit-identical (+0.0000) on all five runs** — nuScenes matches detection and tracking boxes by
centre distance and never consults size, so a pure size error cannot move them. The correction was
also applied to `results_tracking.json` for consistency; nuScenes' tracking evaluator is likewise
size-independent, so those numbers are unchanged.

| Run | mAP | NDS before | NDS after | mASE before | mASE after |
|---|---:|---:|---:|---:|---:|
| `A0_front_cam_class_blind` | 0.0676 | 0.1035 | **0.1138** | 0.8931 | 0.7909 |
| `A_front_cam` | 0.0765 | 0.1079 | **0.1181** | 0.8927 | 0.7903 |
| `B0_repro` | 0.1913 | 0.1795 | **0.2019** | 0.8407 | 0.6168 |
| `B_6cam` | 0.2121 | 0.1824 | **0.2000** | 0.8562 | 0.6806 |
| `C_6cam_lidar_spawn` | 0.2133 | 0.1903 | **0.2129** | 0.8415 | 0.6152 |

Before writing anything, the corrected submissions were verified to be **identical to the
originals in every field except `size`** — same tokens, same box counts, same scores, same
everything — with `size[0]` and `size[1]` exchanged. Any other difference would have meant the two
runs were not comparable, and the correction would have been refused.

mAP, AMOTA, MOTA and recall are unaffected, so the ablation in the top-level README is unchanged.

## Runs present

All five runs cited in the top-level README have their artifacts committed here.

| Directory | Cameras | Class-consistent association | mAP | NDS | AMOTA | MOTA | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `A0_front_cam_class_blind` | 1 (front) | no | 0.0676 | 0.1138 | 0.0791 | 0.0884 | 0.1767 |
| `A_front_cam` | 1 (front) | yes | 0.0765 | 0.1181 | 0.1397 | 0.1531 | 0.1793 |
| `B0_repro` | 6 | no | 0.1913 | 0.2019 | 0.1071 | 0.0922 | 0.4172 |
| `B_6cam` | 6 | yes | 0.2121 | 0.2000 | 0.3466 | 0.3303 | 0.5023 |
| `C_6cam_lidar_spawn` | 6 | yes + LiDAR-only spawn | 0.2133 | 0.2129 | 0.3404 | 0.3027 | 0.5628 |

`B0_repro` is the **reproduction check**. It runs the refactored pipeline with
`class_aware_track_matching=False` and must match the original research notebook's results on the
metrics the box-order fix cannot touch — **mAP 0.1913 and AMOTA 0.1071**, exactly. That is what
licenses the claim that the `B0_repro` → `B_6cam` delta is caused by the class constraint and
nothing else, and it is why every other number in this repo can be trusted.

> `B0_repro`'s **NDS is expected to exceed** the original's (0.2019 vs 0.1795), because the
> original shipped the transposed-box bug and this package fixes it. `scripts/check_results.py`
> therefore asserts the reproduction property on mAP and AMOTA — the two size-independent
> metrics — and reports NDS rather than asserting it.

## ⚠️ Runs the README cites but which are NOT yet committed

*(Resolved — `A_front_cam` and `C_6cam_lidar_spawn` were uploaded and are now staged above, and
`check_results.py` reports all five runs consistent. This heading is kept only so the history is
legible: it is the section that used to warn that two cited rows had no evidence behind them.)*

That warning existed because the original project shipped a headline claim of "mAP ×7.3" with
**no run anywhere** that produced the baseline it was measured against — the measured baseline
turned out to be ×2.83. An unverifiable number is worse than a smaller one, which is why the gap
was flagged in the README rather than quietly left out.

> This warning exists because the original project shipped a headline claim of "mAP ×7.3"
> with **no run anywhere** that produced the baseline it was measured against. The measured
> baseline turned out to be ×2.8. An unverifiable number is worse than a smaller one.

## How these were produced

Kaggle, Tesla T4, Python 3.12, CUDA 11.8. Environment per `requirements-cuda.txt` and
`scripts/install_mmdet3d.sh`. Evaluated on `mini_val` (81 samples, 2 scenes).

Each run differs from the others only in `PerceptionConfig` fields:

```python
from bevision.config import (
    front_camera_class_blind,  # A0
    surround_class_blind,  # B0
    surround_class_aware,  # B
)
```

## Caveats when reading these numbers

- **`mini_val` is 2 scenes.** `bus` has 33 ground-truth objects and `truck` 95, so their
  per-class AMOTA values are highly variant-sensitive. They are not comparable to published
  `val` results and should be read as directional.
- **The detection submission is built from tracker state**, so its positions are
  Kalman-filtered and an object must persist long enough to become a track before it can be
  scored. These are not raw-detector numbers.
- **`trailer`, `construction_vehicle`, `traffic_cone` and `barrier` score AP 0.000 by
  construction** in every run here — they cannot be camera-confirmed (COCO has no such
  classes) and their LiDAR detections sit below the 0.30 spawn threshold. Because mAP
  averages over all ten classes, they cap the achievable score. See the top-level README.
