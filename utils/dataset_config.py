"""Dataset registry for training and regression suites."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from models.model_factory import LARGE_IMAGE_MODELS


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata and paths for a supported dataset."""

    name: str
    num_classes: int
    class_names: List[str]
    data_root: Path
    default_image_size: int
    channels: int = 1
    train_source: str = "mat"
    test_source: str = "mat"
    quick_train_path: str = "data/test_data/test_images.npy"
    quick_test_path: str = "data/test_data/test_images.npy"
    # Per-class subdirectory under <root>/<class>/ (default: color)
    class_layout: Dict[str, str] = field(default_factory=dict)

    def class_variant(self, class_name: str) -> str:
        """Return the image subfolder for a class (e.g. color, without_augmentation)."""
        return self.class_layout.get(class_name, "color")

    def validate(self) -> None:
        """Raise FileNotFoundError if required dataset files are missing."""
        if self.train_source == "mat":
            train_mat = self.data_root.parent / "MNISTtrain.mat"
            test_mat = self.data_root.parent / "MNISTtest.mat"
            if not train_mat.is_file() or not test_mat.is_file():
                raise FileNotFoundError(
                    f"MNIST .mat files not found under {self.data_root.parent}. "
                    "Expected MNISTtrain.mat and MNISTtest.mat."
                )
            return

        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.data_root}")
        missing: List[str] = []
        for class_name in self.class_names:
            variant = self.class_variant(class_name)
            sets_dir = self.data_root / class_name / variant / "sets"
            for split in ("train", "test"):
                split_file = sets_dir / f"{split}.txt"
                if not split_file.is_file():
                    missing.append(str(split_file))
        if missing:
            raise FileNotFoundError(
                f"{self.name} split files missing:\n  " + "\n  ".join(missing[:8])
                + (f"\n  ... and {len(missing) - 8} more" if len(missing) > 8 else "")
            )


RASPBERRY_CLASSES = [
    "healthy",
    "background_without_leaves",
]

RASPBERRY_CLASS_LAYOUT = {
    "healthy": "color",
    "background_without_leaves": "without_augmentation",
}

ORANGE_CLASSES = [
    "huanglongbing_citrus_greening",
    "background_without_leaves",
]

ORANGE_CLASS_LAYOUT = {
    "huanglongbing_citrus_greening": "color",
    "background_without_leaves": "without_augmentation",
}

STRAWBERRY_CLASSES = [
    "early-turning",
    "green",
    "late-turning",
    "red",
    "turning",
    "white",
]

# Flat layout: strawberries/<class>/{images,sets} (no color/ variant subfolder).
STRAWBERRY_CLASS_LAYOUT = {name: "" for name in STRAWBERRY_CLASSES}

PISTACHIO_CLASSES = [
    "kirmizi",
    "siirt",
]

DATASETS: Dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="mnist",
        num_classes=10,
        class_names=[str(i) for i in range(10)],
        data_root=Path("data"),
        default_image_size=28,
        channels=1,
        train_source="mat",
        test_source="mat",
    ),
    "strawberry": DatasetSpec(
        name="strawberry",
        num_classes=len(STRAWBERRY_CLASSES),
        class_names=list(STRAWBERRY_CLASSES),
        data_root=Path("data/Strawberry/strawberries"),
        default_image_size=224,
        channels=1,
        train_source="pv_style",
        test_source="pv_style",
        quick_train_path="data/test_data/strawberry_quick_train",
        quick_test_path="data/test_data/strawberry_quick_test",
        class_layout=dict(STRAWBERRY_CLASS_LAYOUT),
    ),
    "plant_village_raspberry": DatasetSpec(
        name="plant_village_raspberry",
        num_classes=len(RASPBERRY_CLASSES),
        class_names=list(RASPBERRY_CLASSES),
        data_root=Path("data/Plant_Village_Raspberry/raspberries"),
        default_image_size=224,
        channels=1,
        train_source="pv_style",
        test_source="pv_style",
        class_layout=dict(RASPBERRY_CLASS_LAYOUT),
    ),
    "plant_village_orange": DatasetSpec(
        name="plant_village_orange",
        num_classes=len(ORANGE_CLASSES),
        class_names=list(ORANGE_CLASSES),
        data_root=Path("data/Plant_Village_Orange/oranges"),
        default_image_size=224,
        channels=1,
        train_source="pv_style",
        test_source="pv_style",
        class_layout=dict(ORANGE_CLASS_LAYOUT),
    ),
    "pistachio": DatasetSpec(
        name="pistachio",
        num_classes=len(PISTACHIO_CLASSES),
        class_names=list(PISTACHIO_CLASSES),
        data_root=Path("data/Pistachio/pistachios"),
        default_image_size=224,
        channels=1,
        train_source="pv_style",
        test_source="pv_style",
        class_layout={name: "" for name in PISTACHIO_CLASSES},
    ),
}

# Short aliases map to canonical registry keys (for backward compatibility).
DATASET_ALIASES: Dict[str, str] = {
    "raspberry": "plant_village_raspberry",
    "orange": "plant_village_orange",
}


def resolve_dataset_key(name: str) -> str:
    """Normalize a CLI dataset name to a canonical registry key."""
    key = name.lower().strip().replace("-", "_")
    return DATASET_ALIASES.get(key, key)


def get_image_size(model_name: str, dataset_spec: DatasetSpec) -> int:
    """Resolve input spatial size for a model on a given dataset."""
    if dataset_spec.name != "mnist":
        return dataset_spec.default_image_size
    return 224 if model_name in LARGE_IMAGE_MODELS else 28


def get_dataset_spec(name: str, data_root: str | None = None) -> DatasetSpec:
    """Return a dataset spec by name, optionally overriding the data root."""
    key = resolve_dataset_key(name)
    if key not in DATASETS:
        canonical = ", ".join(sorted(DATASETS))
        aliases = ", ".join(f"{alias}→{target}" for alias, target in sorted(DATASET_ALIASES.items()))
        hint = f" Aliases: {aliases}." if aliases else ""
        raise ValueError(f"Unknown dataset '{name}'. Available: {canonical}.{hint}")

    spec = DATASETS[key]
    if data_root:
        return DatasetSpec(
            name=spec.name,
            num_classes=spec.num_classes,
            class_names=list(spec.class_names),
            data_root=Path(data_root),
            default_image_size=spec.default_image_size,
            channels=spec.channels,
            train_source=spec.train_source,
            test_source=spec.test_source,
            quick_train_path=spec.quick_train_path,
            quick_test_path=spec.quick_test_path,
            class_layout=dict(spec.class_layout),
        )
    return spec
