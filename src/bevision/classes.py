"""Class-space bookkeeping for the COCO (camera) <-> nuScenes (LiDAR) interface.

YOLO predicts COCO class ids; PointPillars predicts nuScenes label ids. They are
different label spaces with different cardinality, so every place the two sensors
are compared must translate first. Getting this wrong lets a LiDAR car be
"confirmed" by a YOLO pedestrian.
"""

from __future__ import annotations

from collections.abc import Sequence

# Canonical nuScenes detection class names, in the order the official detection
# config scores them. Order matters: a model's dataset_meta may use the same
# names in a different order, which is validated by resolve_class_names().
NUSCENES_CLASS_NAMES: tuple[str, ...] = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)

#: nuScenes tracking scores exactly these seven classes, not all ten.
TRACKING_CLASS_NAMES: tuple[str, ...] = (
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "pedestrian",
    "trailer",
    "truck",
)

#: YOLO COCO class id -> nuScenes class name, for the classes that map cleanly.
#: COCO ids: 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck.
COCO_ID_TO_NUSCENES_NAME: dict[int, str] = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

#: nuScenes classes with no COCO counterpart. YOLO can never confirm these, so
#: with camera-gated spawning they can never form a track. This is the mechanism
#: behind the four zero-AP classes documented in the README.
CLASSES_WITHOUT_COCO_COUNTERPART: frozenset[str] = frozenset(
    set(NUSCENES_CLASS_NAMES) - set(COCO_ID_TO_NUSCENES_NAME.values())
)

#: Submitted attribute per class. nuScenes expects an attribute name from the
#: class's own attribute vocabulary; static classes take the empty string.
DEFAULT_ATTRIBUTE_BY_CLASS: dict[str, str] = {
    "car": "vehicle.moving",
    "truck": "vehicle.moving",
    "bus": "vehicle.moving",
    "trailer": "vehicle.moving",
    "construction_vehicle": "vehicle.moving",
    "pedestrian": "pedestrian.moving",
    "bicycle": "cycle.with_rider",
    "motorcycle": "cycle.with_rider",
    "barrier": "",
    "traffic_cone": "",
}


def coco_id_to_nuscenes_name(coco_id: int) -> str | None:
    """Map a YOLO COCO class id to its nuScenes class name.

    Returns ``None`` when the class has no nuScenes counterpart, which the caller
    must treat as "not fusable" rather than as an error.
    """
    return COCO_ID_TO_NUSCENES_NAME.get(int(coco_id))


def is_trackable(name: str) -> bool:
    """Whether nuScenes tracking scores this class."""
    return name in TRACKING_CLASS_NAMES


def has_coco_counterpart(name: str) -> bool:
    """Whether a YOLO detection could ever confirm this nuScenes class."""
    return name not in CLASSES_WITHOUT_COCO_COUNTERPART


def resolve_class_names(model_class_names: Sequence[str]) -> tuple[str, ...]:
    """Validate a model's ``dataset_meta['classes']`` against the nuScenes order.

    A silently reordered class list produces label ids that decode to the wrong
    names -- detections stay plausible but every class-level metric is wrong, and
    nothing crashes. So compare loudly instead.

    Raises:
        ValueError: if the model's class list is not a permutation of the ten
            canonical nuScenes classes.
    """
    names = tuple(model_class_names)
    if set(names) != set(NUSCENES_CLASS_NAMES):
        missing = sorted(set(NUSCENES_CLASS_NAMES) - set(names))
        extra = sorted(set(names) - set(NUSCENES_CLASS_NAMES))
        raise ValueError(
            "model class list does not match the nuScenes class set "
            f"(missing={missing}, unexpected={extra})"
        )
    return names


def class_name_for_label(label_id: int, class_names: Sequence[str]) -> str:
    """Decode a model label id into a class name using the model's own ordering."""
    return tuple(class_names)[int(label_id)]
