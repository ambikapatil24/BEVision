"""Official nuScenes evaluator configuration, and headline extraction from its output.

The devkit has moved ``config_factory`` between modules across releases, and has shipped
both a ``DetectionEval`` and a ``TrackingEval`` under slightly different constructor
signatures. The compatibility handling lives here so the rest of the package is not
sprinkled with version checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: The configs the reported results were produced with. Both are the devkit defaults.
DETECTION_CONFIG_NAME = "detection_cvpr_2019"
TRACKING_CONFIG_NAME = "tracking_nips_2019"

#: The devkit's detection config omits this on some versions, and evaluating without it
#: raises. 500 is the value the config sets when present.
DEFAULT_MAX_BOXES_PER_SAMPLE = 500


def _config_factory():
    """Import ``config_factory`` from wherever this devkit version keeps it."""
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:  # older/newer layout
        from nuscenes.eval.common.config import config_factory
    return config_factory


def detection_config() -> Any:
    """The detection evaluation config used for every reported number."""
    config = _config_factory()(DETECTION_CONFIG_NAME)
    if not hasattr(config, "max_boxes_per_sample"):
        config.max_boxes_per_sample = DEFAULT_MAX_BOXES_PER_SAMPLE
    return config


def tracking_config() -> Any:
    """The tracking evaluation config used for every reported number."""
    return _config_factory()(TRACKING_CONFIG_NAME)


def read_metrics_summary(output_dir: str | Path) -> dict[str, Any]:
    """Load an evaluator's ``metrics_summary.json``.

    Args:
        output_dir: The directory passed to the evaluator as ``output_dir``.

    Raises:
        FileNotFoundError: if the evaluator has not been run for that directory.
    """
    path = Path(output_dir) / "metrics_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no metrics_summary.json in {output_dir} -- run the evaluator first"
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def headline_metrics(
    detection_eval_dir: str | Path,
    tracking_eval_dir: str | Path,
) -> dict[str, float]:
    """Extract the summary numbers from two evaluator output directories.

    Reading these from the evaluator's own files, rather than retyping them, is what keeps
    a results table from drifting away from its evidence.
    """
    detection = read_metrics_summary(detection_eval_dir)
    tracking = read_metrics_summary(tracking_eval_dir)
    return {
        "mAP": round(float(detection["mean_ap"]), 4),
        "NDS": round(float(detection["nd_score"]), 4),
        "AMOTA": round(float(tracking["amota"]), 4),
        "MOTA": round(float(tracking["mota"]), 4),
        "recall": round(float(tracking["recall"]), 4),
        "TP": float(tracking["tp"]),
        "FP": float(tracking["fp"]),
        "FN": float(tracking["fn"]),
        "IDS": float(tracking["ids"]),
        "MT": float(tracking["mt"]),
        "ML": float(tracking["ml"]),
    }
