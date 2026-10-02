"""Command-line entry points.

``bevision-run`` runs the perception pipeline and writes submission files.
``bevision-eval`` scores those submissions with the official nuScenes evaluator.

Both wire up the heavy adapters lazily, so ``--help`` works on a machine with none of the
perception stack installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bevision.config import (
    FRONT_CAMERA_ONLY,
    SURROUND_CAMERAS,
    DetectionClassPolicy,
    PerceptionConfig,
)
from bevision.pipeline import run_pipeline, summarise

DEFAULT_POINTPILLARS_CONFIG = (
    "mmdetection3d/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py"
)
DEFAULT_POINTPILLARS_CHECKPOINT = (
    "hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth"
)


def build_run_parser() -> argparse.ArgumentParser:
    """Parser for ``bevision-run``. Exposed so tests can assert on the CLI surface."""
    parser = argparse.ArgumentParser(
        prog="bevision-run",
        description="Run the BEVision perception pipeline and write nuScenes submissions.",
    )
    parser.add_argument("--dataroot", required=True, help="nuScenes root directory")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--split", default="mini_val")
    parser.add_argument("--run-tag", required=True, help="names the output files")
    parser.add_argument("--output-dir", default="results")

    parser.add_argument("--yolo-weights", default="yolov8n.pt")
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--pointpillars-config", default=DEFAULT_POINTPILLARS_CONFIG)
    parser.add_argument("--pointpillars-checkpoint", default=DEFAULT_POINTPILLARS_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument(
        "--cameras",
        choices=("surround", "front"),
        default="surround",
        help="'front' reproduces the single-camera baseline",
    )
    parser.add_argument(
        "--association",
        choices=("class-aware", "class-blind"),
        default="class-aware",
        help="'class-blind' selects class-blind association (the reproduction-check baseline)",
    )
    parser.add_argument(
        "--spawn",
        choices=("camera-confirmed", "lidar-allowed"),
        default="camera-confirmed",
        help="whether LiDAR-only detections may start tracks",
    )
    parser.add_argument("--detection-classes", choices=("all", "tracking"), default="all")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="limit frames, for smoke tests"
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PerceptionConfig:
    """Translate parsed arguments into a :class:`PerceptionConfig`."""
    return PerceptionConfig(
        cameras=SURROUND_CAMERAS if args.cameras == "surround" else FRONT_CAMERA_ONLY,
        class_aware_track_matching=args.association == "class-aware",
        spawn_requires_camera_confirmation=args.spawn == "camera-confirmed",
        detection_class_policy=DetectionClassPolicy(args.detection_classes),
    )


def pipeline_main(argv: list[str] | None = None) -> int:
    """``bevision-run``: detect, fuse, track, and write submissions."""
    args = build_run_parser().parse_args(argv)
    config = config_from_args(args)

    # Check the whole stack up front. One missing package per attempt would mean three
    # failed runs on a fresh environment instead of one clear message.
    from bevision._deps import report_missing

    report_missing(["nuscenes", "ultralytics", "mmdet3d"], "the perception pipeline")

    # Imported here so that --help and argument errors do not need the perception stack.
    from bevision.data import NuScenesFrameSource
    from bevision.detection.camera import UltralyticsCameraDetector
    from bevision.detection.lidar import MMDet3DLidarDetector
    from bevision.evaluation import (
        build_detection_submission,
        build_tracking_submission,
        validate_submission,
        write_submission,
    )

    source = NuScenesFrameSource(
        dataroot=args.dataroot,
        version=args.version,
        split=args.split,
        camera_names=config.cameras,
    )
    frames = source.frames()
    if args.max_frames is not None:
        frames = _take(frames, args.max_frames)

    lidar_detector = MMDet3DLidarDetector(
        args.pointpillars_config, args.pointpillars_checkpoint, device=args.device
    )
    camera_detector = UltralyticsCameraDetector(
        weights=args.yolo_weights,
        confidence_threshold=args.yolo_confidence,
        device=args.device,
    )

    print(
        f"run '{args.run_tag}': {len(config.cameras)} camera(s), "
        f"class_aware={config.class_aware_track_matching}, "
        f"spawn_requires_camera={config.spawn_requires_camera_confirmation}"
    )

    result = run_pipeline(
        _FrameIterable(frames),
        camera_detector,
        lidar_detector,
        lidar_detector.class_names,
        config,
    )

    output_dir = Path(args.output_dir)
    detection_results = build_detection_submission(
        result.tracks_per_sample, result.sample_tokens, config.detection_class_policy
    )
    tracking_results = build_tracking_submission(result.tracks_per_sample, result.sample_tokens)

    detection_path = output_dir / f"results_detection_{args.run_tag}.json"
    tracking_path = output_dir / f"results_tracking_{args.run_tag}.json"
    write_submission(detection_path, detection_results)
    write_submission(tracking_path, tracking_results)

    for label, results, schema in (
        ("detection", detection_results, "detection"),
        ("tracking", tracking_results, "tracking"),
    ):
        problems = validate_submission(results, result.sample_tokens, schema=schema)
        if problems:
            print(f"submission validation FAILED ({label}):", file=sys.stderr)
            for problem in problems[:10]:
                print(f"  - {problem}", file=sys.stderr)
            return 1

    print(json.dumps(summarise(result), indent=2))
    print(f"wrote {detection_path}")
    print(f"wrote {tracking_path}")
    return 0


def eval_main(argv: list[str] | None = None) -> int:
    """``bevision-eval``: score submissions with the official nuScenes evaluator."""
    parser = argparse.ArgumentParser(
        prog="bevision-eval",
        description="Score BEVision submissions with the official nuScenes evaluator.",
    )
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--split", default="mini_val")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--metrics", choices=("detection", "tracking", "both"), default="both")
    args = parser.parse_args(argv)

    from bevision._deps import require

    require("nuscenes", "scoring with the official nuScenes evaluator")
    from nuscenes.nuscenes import NuScenes  # lazy: heavy, and optional

    output_dir = Path(args.output_dir)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    if args.metrics in {"detection", "both"}:
        _run_detection_eval(
            nusc,
            output_dir / f"results_detection_{args.run_tag}.json",
            output_dir / f"eval_detection_{args.run_tag}",
            args.split,
        )
    if args.metrics in {"tracking", "both"}:
        _run_tracking_eval(
            nusc,
            output_dir / f"results_tracking_{args.run_tag}.json",
            output_dir / f"eval_tracking_{args.run_tag}",
            args.split,
            args.version,
            args.dataroot,
        )
    return 0


def _run_detection_eval(nusc, result_path: Path, output_dir: Path, split: str) -> None:
    from nuscenes.eval.detection.evaluate import DetectionEval

    from bevision.evaluation.runner import detection_config

    DetectionEval(
        nusc,
        config=detection_config(),
        result_path=str(result_path),
        eval_set=split,
        output_dir=str(output_dir),
        verbose=True,
    ).main()


def _run_tracking_eval(
    nusc, result_path: Path, output_dir: Path, split: str, version: str, dataroot: str
) -> None:
    import inspect

    from nuscenes.eval.tracking import evaluate as tracking_evaluate

    from bevision.evaluation.runner import tracking_config

    eval_class = (
        getattr(tracking_evaluate, "TrackingEval", None) or tracking_evaluate.TrackingEvaluation
    )
    # The devkit has shipped both a `config` and a `nusc` constructor parameter across
    # versions, so pass whatever this installed version actually declares.
    parameters = inspect.signature(eval_class.__init__).parameters
    kwargs: dict[str, object] = {
        "result_path": str(result_path),
        "eval_set": split,
        "output_dir": str(output_dir),
        "verbose": True,
    }
    if "config" in parameters:
        kwargs["config"] = tracking_config()
    if "nusc" in parameters:
        kwargs["nusc"] = nusc
    if "nusc_version" in parameters:
        kwargs["nusc_version"] = version
    if "nusc_dataroot" in parameters:
        kwargs["nusc_dataroot"] = dataroot
    eval_class(**kwargs).main()


def _take(iterable, count: int):
    """Yield at most ``count`` items from an iterator (for smoke tests)."""
    for index, item in enumerate(iterable):
        if index >= count:
            return
        yield item


class _FrameIterable:
    """Wrap a one-shot generator so :class:`FrameSource` can be satisfied by a callable."""

    def __init__(self, frames) -> None:
        self._frames = frames

    def frames(self):
        return self._frames


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pipeline_main())
