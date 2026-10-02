# Architecture

## Layering

The package is layered so that the numerically interesting code has no dependency on the
perception stack. Nothing in the centre of this diagram imports torch, OpenMMLab or the
nuScenes devkit.

```
                     ┌──────────────────────────────────────┐
   command line ───► │  cli.py                              │
                     │  bevision-run / bevision-eval        │
                     └──────────────┬───────────────────────┘
                                    │
                     ┌──────────────▼───────────────────────┐
                     │  pipeline.py                         │
                     │  detect → fuse → track → emit        │
                     └──┬───────────────────┬───────────────┘
                        │                   │
        ┌───────────────▼──────┐   ┌────────▼─────────────────┐
        │  detection/base.py   │   │  fusion.py  nms.py       │
        │  Protocols:          │   │  tracking/               │
        │   CameraDetector     │   │  geometry.py             │
        │   LidarDetector      │   │  config.py  classes.py   │
        │   FrameSource        │   └────────┬─────────────────┘
        └───────────┬──────────┘            │
                    │                       ▼
     ┌──────────────▼────────────┐   ┌──────────────────────┐
     │ detection/camera.py       │   │ evaluation/          │
     │ detection/lidar.py        │   │  submission.py       │
     │ data/nuscenes_loader.py   │   │  runner.py           │
     │  ── the only modules      │   └──────────────────────┘
     │     that touch torch /    │
     │     mmdet3d / the devkit  │
     └───────────────────────────┘
```

The heavy adapters import their dependencies **inside `__init__`**, not at module scope, so
importing `bevision` never pulls in torch, OpenMMLab, the nuScenes devkit or Ultralytics. The
test suite depends on this: it exercises the fusion, tracking, NMS and geometry code with no GPU
and no dataset. It is checkable directly:

```bash
python -c "import sys, bevision; print([m for m in ('torch','mmdet3d','mmcv','nuscenes','cv2','ultralytics','pandas') if m in sys.modules])"
# -> []
```

## Data flow

```
nuScenes metadata ──► FrameSource.frames() ──► Frame
                                                 │
   ┌─────────────────────────────────────────────┘
   │  Frame = lidar sweep path + LiDAR→ego pose + ego→global pose + cameras
   ▼
LidarDetector(lidar_path) ──► [RawDetection3D]        # LiDAR sensor frame
CameraDetector(image_path) ──► CameraDetections       # per camera, nuScenes class names
   │
   ▼  per LiDAR box, per camera: project the box's 8 corners, class-consistent IoU match
   │  → CameraConfirmation (best camera) or None
   │  → score = 0.5·lidar + 0.5·camera when confirmed
   │  → lift centre and yaw to the global frame
   ▼
center_nms(radius) ──► [Detection3D]                  # global frame, deduplicated
   │
   ▼  Tracker.step(): predict → class-consistent Hungarian match → update → spawn → prune
   ▼
[Track3D] ──► [TrackOutput]                           # centre from the filter, size/yaw/score
   │                                                  # from the last matched detection
   ▼
build_detection_submission / build_tracking_submission ──► JSON ──► official evaluator
```

Two details in that chain are deliberate and load-bearing.

**LiDAR is the anchor.** Cameras confirm and boost; they never create, move or delete a 3D box.
Depth comes from the point cloud, which is the only sensor here that measures it. Classes without
a COCO counterpart cannot be camera-confirmed. With the default camera-gated spawn policy they
cannot start new tracks, which is why four nuScenes classes score AP 0.000 — a consequence of the
design, documented as a limitation.

**The centre comes from the filter, the box shape from the detection.** `_to_track_output`
takes the position from the Kalman state and size/yaw/score from the most recent match. An
unmatched track therefore competes for association using a coasting prediction while still
reporting its last observed box.

## Module responsibilities

| Module | Owns | Depends on |
|---|---|---|
| `config` | Every threshold and policy switch; the five run presets | — |
| `types` | `Box3D`, `RawDetection3D`, `Detection3D`, `TrackOutput`, `FrameStats` | numpy |
| `geometry` | Rigid transforms, quaternions, box corners, projection, IoU, yaw, distances | numpy |
| `classes` | COCO ↔ nuScenes mapping, tracking class set, submission attributes, model class-order validation | — |
| `nms` | Class-conditional centre-NMS | numpy, `types` |
| `fusion` | Best-camera confirmation, class-gated IoU matrix, Hungarian assignment | numpy, scipy, `geometry` |
| `tracking.kalman` | Constant-velocity 3D filter | numpy |
| `tracking.tracker` | Association, track lifecycle, spawn policy | numpy, scipy |
| `detection.base` | Protocols and `Frame` | numpy |
| `pipeline` | The frame loop | all of the above |
| `evaluation.submission` | nuScenes submission schema, build and validate | stdlib |
| `evaluation.runner` | Official evaluator configs, headline extraction | devkit |
| `cli` | Argument parsing, adapter wiring, entry points | `pipeline`, adapters |
| `detection.camera`, `detection.lidar`, `data.nuscenes_loader` | Third-party boundaries | torch / mmdet3d / devkit |

## Interfaces

Three protocols do all the work. They are `typing.Protocol` (structural), not ABCs, because the
concrete adapters wrap libraries with incompatible signatures and inheritance would force those
libraries into the import graph.

```python
class LidarDetector(Protocol):
    def __call__(self, lidar_path: str) -> list[RawDetection3D]: ...

class CameraDetector(Protocol):
    def __call__(self, image_path: str) -> CameraDetections: ...

class FrameSource(Protocol):
    def frames(self) -> Iterable[Frame]: ...
```

`FrameSource` also carries a contract that is not expressible in the type: **consecutive
samples of a scene must be yielded contiguously**, because the pipeline resets the tracker
whenever `scene_token` changes. Identity must not leak between drives.

## Invariants worth knowing

1. **Every sample token appears in a submission, including empty ones.** The evaluator rejects a
   submission that does not cover the whole split, so `build_*_submission` pre-fills from the
   token list rather than filling opportunistically.
2. **Detection and tracking use different class sets.** Ten vs seven, read from separate
   constants in `classes.py` by two separate builder functions; conflating them silently drops
   three classes from the detection metrics.
3. **Submissions are validated before the CLI exits.** `validate_submission` returns a list of
   problems rather than raising, so all of them are reported at once.
4. **`scipy.optimize.linear_sum_assignment` is called with a finite penalty, never `np.inf`.**
   An infeasible `inf` cost matrix raises, and it is infeasible as soon as one track has no
   same-class candidate.
5. **A box at or behind the image plane is rejected before perspective division.** Otherwise it
   is mirrored into the image and produces spurious matches.
6. **A detection's class is fixed at fusion.** `Detection3D.class_name` is set when the
   detection is built; `Track3D` copies it from its matched detection rather than re-deriving it,
   and never changes it between updates.

## Extension points

| To do this | Change here |
|---|---|
| Different 2D or 3D detector | Implement the protocol; no pipeline change |
| Radar or a second LiDAR | Add a `Frame` field and a fusion branch |
| Different dataset | Implement `FrameSource`; `pipeline.py` is dataset-agnostic |
| Per-class spawn thresholds | `config.PerceptionConfig` + `Tracker._may_spawn` |
| Different association metric | `tracking.tracker.associate` |
| A new figure or panel style | `visualization/` — `render_bev` takes a `BevFrame` and any matplotlib `Axes` |
