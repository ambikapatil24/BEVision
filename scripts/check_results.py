#!/usr/bin/env python
"""Re-derive the results table from the committed artifacts and check it is consistent.

Run from the repo root:

    python scripts/check_results.py

Exit code is non-zero if any committed artifact's numbers disagree with its
``headline_metrics.json``, so a table that drifts from the evidence fails loudly instead
of being noticed a year later. Uses only the standard library, so it runs anywhere the
perception stack cannot be installed.
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

        rows.append((run_dir.name, derived))

    # The reproduction check. B0 must equal the original notebook's results on the metrics the
    # box-order fix cannot touch: mAP and AMOTA are both computed from centre distance, never
    # from box size, so they must stay bit-identical. NDS is reported but deliberately NOT
    # asserted -- the original passed swapped box sizes straight through, and we fixed that
    # (docs/REFACTORING.md, Finding 5), so its NDS is expected to move.
    repro = next((row for name, row in rows if name == "B0_repro"), None)
    if repro is not None:
        expected = {"mAP": 0.1913, "AMOTA": 0.1071}
        for column, value in expected.items():
            if repro[column] != value:
                problems.append(
                    f"B0_repro no longer reproduces the original pipeline on {column}: "
                    f"{repro[column]}, expected {value}"
                )
        print(
            f"\nB0_repro reproduction: mAP={repro['mAP']} AMOTA={repro['AMOTA']} "
            f"(NDS={repro['NDS']}, reported not asserted -- see REFACTORING.md Finding 5)"
        )

    name_width = max((len(name) for name, _ in rows), default=8) + 2
    header = f"{'run':<{name_width}}" + "".join(f"{column:>9}" for column in COLUMNS)
    print(header)
    print("-" * len(header))
    for name, derived in rows:
        print(f"{name:<{name_width}}" + "".join(f"{derived[c]:>9.4f}" for c in COLUMNS))

    expected_runs = {
        "A0_front_cam_class_blind",
        "A_front_cam",
        "B0_repro",
        "B_6cam",
        "C_6cam_lidar_spawn",
    }
    missing_runs = sorted(expected_runs - {name for name, _ in rows})
    if missing_runs:
        print(
            "\nNOTE: run(s) the README cites with no artifact in results/: "
            + ", ".join(missing_runs)
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
