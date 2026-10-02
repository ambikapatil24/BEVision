#!/usr/bin/env python
"""Re-derive the results table from the committed artifacts and check it is consistent.

Run from the repo root:

    python scripts/check_results.py

Three things are checked, and the exit code is non-zero if any of them fails:

1. Every run directory carries the full artifact set.
2. The derived metrics agree with that run's ``headline_metrics.json`` -- two committed files
   produced independently of each other, so agreement is evidence rather than tautology.
3. The derived metrics agree with the numbers published in the README.

Uses only the standard library, so it runs anywhere the perception stack cannot be installed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

REQUIRED_FILES = (
    "results_detection.json",
    "results_tracking.json",
    "eval_detection_metrics_summary.json",
    "eval_tracking_metrics_summary.json",
    "headline_metrics.json",
)

COLUMNS = ("mAP", "NDS", "AMOTA", "MOTA", "recall")

#: The numbers published in the README's results tables, keyed by run directory and rounded to the
#: four decimals this script reports. Held here so the README cannot drift away from the committed
#: artifacts without the check failing; the keys double as the set of runs the README cites.
PUBLISHED: dict[str, dict[str, float]] = {
    "A0_front_cam_class_blind": {
        "mAP": 0.0676,
        "NDS": 0.1138,
        "AMOTA": 0.0791,
        "MOTA": 0.0884,
        "recall": 0.1767,
    },
    "A_front_cam": {
        "mAP": 0.0765,
        "NDS": 0.1181,
        "AMOTA": 0.1397,
        "MOTA": 0.1531,
        "recall": 0.1793,
    },
    "B0_repro": {
        "mAP": 0.1913,
        "NDS": 0.2019,
        "AMOTA": 0.1071,
        "MOTA": 0.0922,
        "recall": 0.4172,
    },
    "B_6cam": {
        "mAP": 0.2121,
        "NDS": 0.2000,
        "AMOTA": 0.3466,
        "MOTA": 0.3303,
        "recall": 0.5023,
    },
    "C_6cam_lidar_spawn": {
        "mAP": 0.2133,
        "NDS": 0.2129,
        "AMOTA": 0.3404,
        "MOTA": 0.3027,
        "recall": 0.5628,
    },
}


def load_run(run_dir: Path) -> tuple[dict, dict, dict]:
    with open(run_dir / "eval_detection_metrics_summary.json") as handle:
        detection = json.load(handle)
    with open(run_dir / "eval_tracking_metrics_summary.json") as handle:
        tracking = json.load(handle)
    with open(run_dir / "headline_metrics.json") as handle:
        headline = json.load(handle)
    return detection, tracking, headline


def main() -> int:
    if not RESULTS.is_dir():
        print(f"no results directory at {RESULTS}", file=sys.stderr)
        return 1

    run_dirs = sorted(d for d in RESULTS.iterdir() if d.is_dir())
    if not run_dirs:
        print("no run directories found", file=sys.stderr)
        return 1

    problems: list[str] = []
    unpublished: list[str] = []
    rows: list[tuple[str, dict]] = []

    for run_dir in run_dirs:
        missing = [name for name in REQUIRED_FILES if not (run_dir / name).exists()]
        if missing:
            problems.append(f"{run_dir.name}: missing {', '.join(missing)}")
            continue

        try:
            detection, tracking, headline = load_run(run_dir)
        except (json.JSONDecodeError, KeyError) as error:
            problems.append(f"{run_dir.name}: unreadable artifact ({error})")
            continue

        derived = {
            "mAP": round(detection["mean_ap"], 4),
            "NDS": round(detection["nd_score"], 4),
            "AMOTA": round(tracking["amota"], 4),
            "MOTA": round(tracking["mota"], 4),
            "recall": round(tracking["recall"], 4),
        }

        for column in COLUMNS:
            if headline.get(column) != derived[column]:
                problems.append(
                    f"{run_dir.name}: headline_metrics.json {column}={headline.get(column)} "
                    f"but the eval summary says {derived[column]}"
                )

        published = PUBLISHED.get(run_dir.name)
        if published is None:
            unpublished.append(run_dir.name)
        else:
            for column in COLUMNS:
                if derived[column] != published[column]:
                    problems.append(
                        f"{run_dir.name}: the README publishes {column}={published[column]} "
                        f"but the artifacts derive {derived[column]}"
                    )

        rows.append((run_dir.name, derived))

    # The reproduction check. B0_repro must land on the metrics a change to box *size* handling
    # cannot move: mAP and AMOTA are both computed from centre distance, never from box size, so
    # they must stay bit-identical. NDS is reported but deliberately NOT asserted, because the
    # committed submissions had their size field corrected to nuScenes' (width, length, height)
    # order, which moves mASE and therefore NDS.
    repro = next((row for name, row in rows if name == "B0_repro"), None)
    if repro is not None:
        expected = {"mAP": 0.1913, "AMOTA": 0.1071}
        for column, value in expected.items():
            if repro[column] != value:
                problems.append(
                    f"B0_repro no longer reproduces the expected values on {column}: "
                    f"{repro[column]}, expected {value}"
                )
        print(
            f"\nB0_repro reproduction: mAP={repro['mAP']} AMOTA={repro['AMOTA']} "
            f"(NDS={repro['NDS']}, reported not asserted -- see results/README.md)"
        )

    name_width = max((len(name) for name, _ in rows), default=8) + 2
    header = f"{'run':<{name_width}}" + "".join(f"{column:>9}" for column in COLUMNS)
    print(header)
    print("-" * len(header))
    for name, derived in rows:
        print(f"{name:<{name_width}}" + "".join(f"{derived[c]:>9.4f}" for c in COLUMNS))

    missing_runs = sorted(set(PUBLISHED) - {name for name, _ in rows})
    if missing_runs:
        print(
            "\nNOTE: run(s) the README cites with no artifact in results/: "
            + ", ".join(missing_runs)
        )
    if unpublished:
        print(
            "\nNOTE: run(s) in results/ with no published numbers recorded, so only the headline "
            "cross-check applied: " + ", ".join(sorted(unpublished))
        )

    if problems:
        print("\nFAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(f"\nOK: {len(rows)} run(s) consistent with their artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
