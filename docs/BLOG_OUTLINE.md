# Blog post outline

For Medium / Substack. Working title and several alternates. The hook is the correction, not
the result — "I found three of my own numbers were wrong" travels much further than "I built a
perception stack".

---

## Title options

1. **"Three of my own metrics were wrong: refactoring a perception project into something I can defend"**
2. "My baseline was never measured"
3. "A one-line bug worth 3.2× AMOTA — and how I proved it"
4. "Why `False positives fell 49% *and* recall rose` is the only number I trust"

Option 1 is the strongest for a general ML audience. Option 3 is the strongest for a
perception-specific one.

## Subtitle

> I refactored a notebook into a package, added a reproduction check, and discovered my headline
> number was inflated by 2.6×.

**Estimated read: 9–12 minutes.** Target audience: ML engineers, perception/autonomy-adjacent,
senior enough to have been burned by this themselves.

---

## Section 1 — The claim (200 words)

Open with the original headline verbatim:

> mAP 0.027 → 0.199 (**7.3×**) · AMOTA 0.019 → 0.115 (6×)

State plainly what the project was: YOLOv8 on six cameras + PointPillars on LiDAR, projection
fusion, Kalman tracking, official nuScenes evaluation. Then the turn:

> The refactor took two weeks. The interesting part happened in the first afternoon, when I
> tried to reproduce the baseline.

**Do not** open with architecture or tech stack. Lead with the failure.

## Section 2 — Finding 1: there was no baseline (400 words)

The concrete evidence: `0.027` appears in exactly one place — a markdown cell — and every
pipeline path hard-codes all six cameras. There was no front-camera configuration to run.

The measured baseline is 0.0765 mAP / 0.1397 AMOTA. So:
- claimed mAP gain ×7.3 → **measured ×2.83**
- claimed AMOTA gain ×6.0 → **measured ×2.5** (at fixed association policy)

Button: *a number you can't regenerate isn't a result, it's a memory.*

Include the short code block showing the hard-coded camera list.

## Section 3 — Finding 2: the bug that was worth 3.2× AMOTA (700 words — the centrepiece)

This is the section the post lives or dies on. Structure:

1. **The code.** Show the matched-pair update:
   ```python
   track_box[tr.track_id] = lidar_dets[d_idx]     # relabels the track
   ```
   Combined with distance-only matching, a car track updated by a nearer truck detection
   *becomes* a truck.

2. **Why it's worse than it looks.** nuScenes tracking is evaluated **per class**. So one bad
   update is simultaneously a false negative for `car` and a false positive for `truck`.

3. **The clue that was already there.** `truck`: 329 false positives against 95 ground-truth
   objects — a 3.5:1 ratio. And 2188 ground-truth cars, so a few percent of car tracks
   relabelled accounts for it exactly. *The original write-up blamed mis-localisation. It was
   mislabelling.*

4. **The fix, and where the constraint goes.** Fold it into the cost matrix, not a filter
   afterwards. Include the "an optimizer can only respect a constraint that is part of its
   objective" line. Mention the finite-penalty detail — `linear_sum_assignment` raises on an
   infeasible `inf` matrix, which happens routinely.

5. **The result, and the falsification test.**

   | | AMOTA | MOTA | FP | IDS | Recall | TP |
   |---|---:|---:|---:|---:|---:|---:|
   | ×3.24 | ×3.58 | **−49%** | **−42%** | +20% | +8.7% |

   Then the key paragraph: **false positives fell while recall rose.** Loosen any threshold and
   recall and FP move *together*. Falling FP with rising recall is what a correctness fix looks
   like. *If you take one idea from this post, take that one.*

6. **The counter-intuitive corollary.** Front-camera-only *with* the fix (AMOTA 0.1397) beats
   six cameras *without* it (0.1071). Association quality was worth more to tracking than five
   extra cameras. That only became visible because of the ablation.

## Section 4 — Finding 3: four classes that could never score (450 words)

Four of ten nuScenes classes could never spawn a track, because spawning required camera
confirmation and YOLO has no COCO class for `trailer`, `construction_vehicle`, `traffic_cone` or
`barrier`. So they scored AP 0.000 — and mAP averages over all ten.

Then **the part that makes this a good story**: my first hypothesis was wrong. I guessed the
spawn *policy* was to blame, ran the experiment, and the four classes stayed at zero. The real
cause was the score gate. The diagnostic table:

| Class | Detections above 0.30 | Total |
|---|---:|---:|
| `trailer` | 0 | 681 |
| `barrier` | 2 | 3188 |
| `construction_vehicle` | 4 | 1201 |
| `traffic_cone` | 9 | 3012 |

PointPillars sees these fine — `barrier` is normally one of its best classes — but at low
confidence, and the gate discards them before the evaluator can rank them.

**And then the discipline:** this is diagnosed, not fixed. Say why. *"Claiming a fix I hadn't
measured is exactly the mistake I started this post with."*

## Section 5 — Finding 4, briefly (150 words)

One `continue` served both submissions. nuScenes scores detection over ten classes and tracking
over seven, so three classes were silently dropped from the detection metrics. One-line bug,
three classes, and the comment above it said "tracking scores only 7 classes" — which was true
and applied to the wrong statement.

## Section 6 — How I made the fixes trustworthy (400 words)

The methodological section. Three practices, each with the mechanism not just the principle:

1. **Reproduction check before fixing.** A config flag restoring the old behaviour, which must
   reproduce the original to four decimals. It does. That's what converts "AMOTA went up" into
   "AMOTA went up because of this one flag". Wire it into CI so it can't regress.
2. **Every number ships with its artifact.** Submission JSONs + evaluator `metrics_summary.json`
   committed, with a script that re-derives the README table and **fails on drift** — demonstrate
   that it genuinely fails by corrupting a value.
3. **Distinguish "quoted" from "relied on".** Two runs are cited without artifacts; they're
   labelled as such and no claim depends on them. *Dropping them would hide information;
   relying on them would repeat the original sin.*

## Section 7 — What I'd do differently (200 words)

- Commit artifacts before believing numbers.
- Write the baseline before the improvement. If you can't run it, you can't claim it.
- Keep one implementation. The projection code existed in three divergent copies, which is how
  a depth-check bug survived — and the three copies weren't even comparable to each other.
- Report negative results. The spawn-gate experiment raises recall and *lowers* AMOTA. Publishing
  only the win would be dishonest, and the trade-off is more interesting anyway.

## Section 8 — Takeaway (150 words)

> The result I'm most confident in isn't the mAP. It's the FP −49% *with* recall +20%, because
> I know what mechanism produces it and I know no threshold tweak can. The rest of the numbers
> are more modest than the ones I started with, and they're the ones I can defend.

Close with a link to the repo and an explicit invitation to check the two rows that are quoted
rather than relied on.

---

## Practical notes

- **Figures to produce** (all regenerable from `results/`):
  1. The ablation table — the anchor image.
  2. A `truck` FP/TP before-after bar chart. This is the single most convincing visual — 329 FP
     against 95 GT collapsing.
  3. The score-band histogram for the four unscorable classes.
  4. Annotated camera image + BEV, for the "here's what it does" moment.
- **Where to publish:** Medium's Towards Data Science or Towards AI for reach; Substack if
  building a personal audience. Cross-post to r/computervision and LinkedIn.
- **What to cut for a shorter version:** Sections 5 and 7. Never cut Section 6 — the
  methodology *is* the post.
- **Companion repo docs:** link `docs/REFACTORING.md` for the full catalogue of behaviour
  changes; the post is the narrative, the doc is the reference.
