# `results/` — committed evaluation artifacts

Each subdirectory is one ablation run and holds the same seven files.

| File | What it is |
|---|---|
| `results_detection.json` | Detection submission in official nuScenes format — what was fed to the evaluator |
| `results_tracking.json` | Tracking submission, same format |
| `eval_detection_metrics_summary.json` | Output of `nuscenes.eval.detection.evaluate.DetectionEval` |
| `eval_tracking_metrics_summary.json` | Output of `nuscenes.eval.tracking.evaluate.TrackingEval` |
| `eval_detection_metrics_details.json` | The detection evaluator's raw curve data |
| `eval_tracking_metrics_details.json` | Same for tracking |
| `headline_metrics.json` | The headline numbers, extracted from the two summaries rather than typed by hand |

The submission JSONs are committed deliberately: they are the only artifact that lets someone
re-run the evaluator and get these exact numbers back, and regenerating them needs a GPU and the
dataset. Rendered `plots/*.pdf` are excluded — they are drawn from `metrics_details.json`, which
*is* committed, so they regenerate without re-running anything.

## Artifact structure

Both submissions are `{"meta": {...}, "results": {<sample_token>: [...]}}` with 81 samples each,
covering the whole `mini_val` split. `meta` records which sensors were used
(`use_camera`/`use_lidar` true, `use_radar`/`use_map`/`use_external` false).

Detection entries carry
`sample_token, translation, size, rotation, velocity, detection_name, attribute_name, detection_score`.
Tracking entries carry the same geometry fields keyed by
`tracking_id, tracking_name, tracking_score`.

`eval_detection_metrics_details.json` is keyed `"<class>:<distance_threshold>"` — 10 classes × 4
thresholds = 40 nodes — and each node holds `recall`, `precision`, `confidence` and the five
`*_err` arrays, 101 sampled points each. `eval_tracking_metrics_details.json` is keyed by the 7
tracking class names, and each node holds its arrays over the evaluator's 40 recall thresholds.

## Runs

| Directory | Cameras | Class-consistent association | mAP | NDS | AMOTA | MOTA | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `A0_front_cam_class_blind` | 1 (front) | no | 0.0676 | 0.1138 | 0.0791 | 0.0884 | 0.1767 |
| `A_front_cam` | 1 (front) | yes | 0.0765 | 0.1181 | 0.1397 | 0.1531 | 0.1793 |
| `B0_repro` | 6 | no | 0.1913 | 0.2019 | 0.1071 | 0.0922 | 0.4172 |
| `B_6cam` | 6 | yes | 0.2121 | 0.2000 | 0.3466 | 0.3303 | 0.5023 |
| `C_6cam_lidar_spawn` | 6 | yes + LiDAR-only spawn | 0.2133 | 0.2129 | 0.3404 | 0.3027 | 0.5628 |

Per-class for `B_6cam` — `AP` averaged over the evaluator's four distance thresholds, read from
`eval_detection_metrics_summary.json` and `eval_tracking_metrics_summary.json`:

| Class | AP | AMOTA | | Class | AP | AMOTA |
|---|---:|---:|---|---|---:|---:|
| car | 0.511 | 0.406 | | motorcycle | 0.000 | 0.000 |
| truck | 0.631 | 0.822 | | bicycle | 0.030 | 0.000 |
| bus | 0.608 | 0.513 | | trailer | 0.000 | — |
| pedestrian | 0.341 | 0.338 | | construction_vehicle | 0.000 | — |
| | | | | traffic_cone | 0.000 | — |
| | | | | barrier | 0.000 | — |

Tracking detail (`B_6cam`): TP 1718 · FP 546 · FN 1883 · IDS 68 · MT 49 · ML 94.

## How these are validated

```bash
python scripts/check_results.py
```

The script uses only the standard library. For each run it re-derives mAP / NDS / AMOTA / MOTA /
recall from that run's evaluator summary and fails if the run's `headline_metrics.json` disagrees.
If any run cited in the top-level README has no artifact directory, it reports that too.

## The reproduction check

`B0_repro` runs the pipeline with `class_aware_track_matching=False` and
`detection_class_policy=tracking`. `scripts/check_results.py` asserts that its **mAP 0.1913 and
AMOTA 0.1071** match, exactly. Those two metrics are computed from box centres alone, so they are
the metrics a change to box *size* handling cannot move. That is what isolates the association
change as the single variable between `B0_repro` and `B_6cam`.

NDS is reported for `B0_repro` but deliberately **not** asserted, because the box-order correction
described below moves it.

## Box-order correction

The committed submissions store `size` as `(width, length, height)`, which is what nuScenes
expects. mmdet3d's `LiDARInstance3DBoxes.tensor` stores `(length, width, height)`, and
`detection/lidar.py` performs that reordering; the `headline_metrics.json` files carry
`box_order_corrected: true` to record that it was applied. The metric values quoted in the table
above are the post-correction values.

**Pre-correction values are not reproducible from this repository.** No committed artifact
records them, and re-evaluating with the size field swapped requires the nuScenes devkit, the
dataset, and the uncorrected submissions. Earlier revisions of this document quoted
pre-correction NDS and mASE figures; those numbers have been removed rather than restated,
because they cannot be checked here.

For reference, the post-correction values that *are* backed by the committed evaluator output:

| Run | NDS | mASE |
|---|---:|---:|
| `A0_front_cam_class_blind` | 0.1138 | 0.7909 |
| `A_front_cam` | 0.1181 | 0.7903 |
| `B0_repro` | 0.2019 | 0.6168 |
| `B_6cam` | 0.2000 | 0.6806 |
| `C_6cam_lidar_spawn` | 0.2129 | 0.6152 |

mAP, AMOTA, MOTA and recall are unaffected by a size-only correction, so the ablation in the
top-level README does not depend on it.

## How these were produced

Each run differs from the others only in `PerceptionConfig` fields, driven by CLI flags:

```bash
bevision-run --dataroot $D --run-tag B_6cam                                    # headline
bevision-run --dataroot $D --run-tag B0_repro  --association class-blind --detection-classes tracking
bevision-run --dataroot $D --run-tag A0_front_cam_class_blind --cameras front --association class-blind
bevision-run --dataroot $D --run-tag A_front_cam --cameras front
bevision-run --dataroot $D --run-tag C_6cam_lidar_spawn --spawn lidar-allowed
```

The artifacts themselves record the eval set (`mini_val`), the sample count (81) and the scene
count (2). The hardware and library versions used to produce them are **not** recorded in the
artifacts; `requirements-cuda.txt` lists the stack this project targets.

## Caveats when reading these numbers

- **`mini_val` is 2 scenes.** `bus` has 33 ground-truth objects and `truck` 95, so their per-class
  AMOTA is highly variant-sensitive. These are not comparable to published `val` results and
  should be read as directional.
- **The detection submission is built from tracker state**, so its positions are Kalman-filtered
  and an object must persist long enough to become a track before it can be scored. These are not
  raw-detector numbers.
- **`trailer`, `construction_vehicle`, `traffic_cone` and `barrier` score AP 0.000 in every run
  here.** They have no COCO counterpart, so with camera-gated spawning they can never form a
  track, and mAP averages over all ten classes. See the top-level README.
