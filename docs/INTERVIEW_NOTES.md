# Interview talking points

The raw material for questions about this project. The short version of the strategy: **the
interesting part is not the mAP, it is that you found out your own numbers were wrong and said
so.** Lead with that. A senior interviewer has seen a hundred "I got 0.2 mAP with YOLO +
PointPillars" projects; they have not seen many candidates volunteer that their baseline was
fabricated.

---

## The 60-second version

> "BEVision is a multi-sensor perception stack on nuScenes — YOLOv8 on six cameras, PointPillars
> on LiDAR, LiDAR-anchored projection fusion, and 3D multi-object tracking in the global frame.
> The interesting part came out of refactoring it. The original notebook claimed a ×7.3 mAP
> improvement over a front-camera baseline, and when I moved it to a package and actually ran
> the baseline, no run in the project had ever produced those numbers. The real gain is ×2.83.
> Second, track association was matching purely on 3D distance with no class constraint, and
> because the emitted class came from the matched detection, a car track updated by a nearer
> truck detection *became* a truck. nuScenes scores tracking per class, so that manufactured
> false positives for one class and false negatives for another. Fixing it took AMOTA from 0.107
> to 0.347 and halved false positives. The way I know that's real and not a threshold artefact
> is that false positives fell *while* recall rose — those move together if you're cheating."

That is a complete answer. It has a result, a mechanism, and a falsification test.

## The 5-minute version

Add, in rough order of how much they will care:

1. **The reproduction check.** "Before fixing anything I added a config flag that restores the
   old behaviour, and required it to reproduce the original numbers. It does — mAP 0.1913 and
   AMOTA 0.1071 exactly, which are the two size-independent metrics. That's what makes the
   ablated comparison a single-variable experiment instead of a story."
2. **The class constraint goes in the objective, not after it.** "I fold it into the cost
   matrix rather than filtering matches afterwards, because an optimizer can only respect a
   constraint that's part of its objective. Filtering after lets the assignment spend a slot on
   a pair you'll then discard, starving a valid pair. The penalty is a finite 1e6 rather than
   inf because `linear_sum_assignment` raises on an infeasible inf matrix, which happens as soon
   as one track has no same-class detection."
3. **The four classes that couldn't be scored.** "Four of the ten nuScenes classes could never
   spawn a track: they need camera confirmation, and YOLO has no COCO class for them. So they
   scored AP 0 and, because mAP averages over ten classes, they capped it. I hypothesised the
   spawn *policy* was the blocker, ran an experiment, and was wrong — it's the score threshold.
   Across the whole split, `trailer` has **zero** PointPillars detections above 0.30 and
   `barrier` has two out of 3188. I know the diagnosis and haven't claimed the fix, because
   claiming an unmeasured fix is the mistake that started all this."
4. **A negative result I kept.** "Relaxing the spawn gate raises recall 0.50 → 0.56 but lowers
   AMOTA and MOTA. I report it as a measured trade-off and don't use that config."
5. **A bug that no metric could have caught.** "Every submitted box had width and length
   transposed — mmdet3d's tensor is `(length, width, height)` and nuScenes wants
   `(width, length, height)`. The tell was that correcting it left mAP and mATE *bit-identical*
   across all five runs while raising NDS. That bit-identical part isn't luck: nuScenes matches
   boxes by centre distance and never looks at size, so a pure size error physically cannot move
   mAP — and that's exactly why nothing had caught it in months of running the pipeline."

---

## Questions you should expect

**"Why is your mAP only 0.21?"**
Two honest reasons. Pretrained detectors, no fine-tuning — fine-tuning PointPillars is the
biggest available lever. And four of ten classes score exactly 0 by construction, which caps
the mean; if `barrier` and `traffic_cone` scored the way the reference PointPillars checkpoint
does, mAP goes up materially with no model change. Don't apologise for `mini_val` without
saying it's 2 scenes — and note that `bus` has 33 ground-truth objects, so its per-class AMOTA
is not comparable to published numbers.

**"How do you know your fusion helps?"**
The cross-check is the interesting one: front-camera-only *with* correct association
(AMOTA 0.1397) beats six cameras *with* the class-blind matcher (0.1071). So association
quality was worth more to tracking than five extra cameras were. That's a genuinely useful
result, and it's the kind of thing that only falls out of a clean ablation.

**"What does the camera actually contribute?"**
Class evidence and confirmation, not geometry. LiDAR is the anchor — depth comes from the
point cloud. A camera can raise a detection's confidence; it can never create, move or delete
a 3D box. That's a deliberate constraint: cameras are much better at *what* than *where*.

**"How is this tested with no GPU?"**
The pipeline talks to detectors through protocols in `detection/base.py`. So `test_pipeline.py`
runs the real fusion, tracking, NMS and geometry code against fake detectors and fake frames,
with no torch installed. 229 tests, ~4 seconds. The end-to-end path is additionally smoke-tested
against the real PointPillars checkpoint and YOLOv8n weights, so the only module never executed
is `cli.eval_main`.

**"You replaced filterpy with your own Kalman filter — how do you know it's right?"**
`test_kalman.py` runs both side by side on identical measurement sequences and asserts state
*and* covariance agree to 1e-9 at every step. I probed filterpy's defaults first — they're
`P = Q = R = F = I` — which matters because the notebook does `P *= 5.0` on top, so the initial
covariance is `5·I`. Guessing that would have silently changed the filter gains.

**"What's the biggest weakness?"**
The detectors are pretrained and the benchmark is two scenes. Beyond that: detection output is
built from tracker state, so positions are Kalman-filtered and an object has to persist to be
scored — legitimate temporal aggregation, but it means those aren't raw-detector numbers, and
I say so in the README rather than letting someone discover it.

**"Anything you'd do differently?"** — *Have an answer ready; this is a favourite.*
Commit the artifacts before trusting the numbers. That's the actual lesson here: the
original project's most-quoted figure had no run behind it. Every number in this README now
comes from a `metrics_summary.json` in `results/`, and `scripts/check_results.py` fails if the
table and the artifacts disagree.

---

## Things to be careful about

- **Don't quote ×7.3, or 0.199, or 0.115.** Those were the original claims and they are not
  backed by any run. If asked about them, say they were wrong and what the measured numbers
  are — that's a better moment than the correction looks.
- **Don't present `trailer`/`barrier`/`traffic_cone`/`construction_vehicle` as "the detector
  can't do those."** It can; the pipeline throws their detections away. Say that.
- **Don't claim the ×3.24 AMOTA as a general result.** It is one class-mean over two scenes
  where `truck` has 95 ground-truth objects. It's a real fix with a real effect size; it is not
  a benchmark comparison.
- **Two README rows have no committed artifact** (`A_front_cam`, `C_6cam_lidar_spawn`). If
  they come up: "those are quoted, not relied on — no claim in the README depends on them, and
  the checker warns about them on every run." Then fix it by re-running them.
- **The recall/FP claim is the one to lean on.** FP −49% *and* recall +20% is hard to get by
  accident.

## Numbers worth having memorised

| | Value |
|---|---|
| Headline (`B_6cam`) | mAP **0.2121** · NDS **0.2000** · AMOTA **0.3466** · MOTA **0.3303** · recall **0.5023** |
| Reproduction check (`B0_repro`) | mAP **0.1913** · AMOTA **0.1071** (NDS 0.2019 — expected to exceed the original's 0.1795, see below) |
| Camera count (`A0` → `B0`) | mAP **×2.83**, recall ×2.36 |
| Class-consistent association (`B0` → `B`) | AMOTA **×3.24**, MOTA ×3.58, FP **−49%**, IDS **−42%**, TP +8.7% |
| End to end (`A0` → `B`) | mAP **×3.14**, AMOTA **×4.38** |
| `B_6cam` tracking detail | TP 1718 · FP 546 · FN 1883 · IDS 68 · MT 49 · ML 94 |
| The four unscorable classes | above-0.30 detections: trailer **0**/681, barrier **2**/3188, traffic_cone **9**/3012, construction_vehicle **4**/1201 |
| Box-order fix | NDS +0.010 to +0.023 on every run, **mAP and mATE unchanged (+0.0000)** — a pure size error |
| Tests | 223, ~4 s, no GPU |

Everything above is traceable to a file in `results/`.
