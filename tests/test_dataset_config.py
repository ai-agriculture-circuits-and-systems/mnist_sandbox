"""Tests for dataset registry and PV-style image loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.dataset_config import get_dataset_spec, resolve_dataset_key
from utils.data_loader import DataLoaderFactory, PVStyleImageDataset


def test_plant_village_raspberry_spec_has_two_classes() -> None:
    """Plant Village Raspberry exposes healthy vs background_without_leaves."""
    spec = get_dataset_spec("plant_village_raspberry")
    assert spec.num_classes == 2
    assert spec.name == "plant_village_raspberry"
    assert spec.class_names == ["healthy", "background_without_leaves"]
    assert spec.class_layout["healthy"] == "color"
    assert spec.class_layout["background_without_leaves"] == "without_augmentation"


def test_plant_village_orange_spec_has_two_classes() -> None:
    """Plant Village Orange exposes HLB vs background_without_leaves."""
    spec = get_dataset_spec("plant_village_orange")
    assert spec.num_classes == 2
    assert spec.name == "plant_village_orange"
    assert "huanglongbing_citrus_greening" in spec.class_names
    assert spec.class_layout["huanglongbing_citrus_greening"] == "color"


def test_orange_alias_resolves_to_plant_village_orange() -> None:
    """Short alias 'orange' maps to the canonical dataset key."""
    assert resolve_dataset_key("orange") == "plant_village_orange"
    assert get_dataset_spec("orange").name == "plant_village_orange"


def test_raspberry_alias_resolves_to_plant_village_raspberry() -> None:
    """Short alias 'raspberry' maps to the canonical dataset key."""
    assert resolve_dataset_key("raspberry") == "plant_village_raspberry"
    assert get_dataset_spec("raspberry").name == "plant_village_raspberry"


def test_strawberry_spec_has_six_classes() -> None:
    """Strawberry ripeness dataset exposes six class labels."""
    spec = get_dataset_spec("strawberry")
    assert spec.num_classes == 6
    assert "red" in spec.class_names
    assert "white" in spec.class_names
    assert spec.class_layout["red"] == ""


def test_pistachio_spec_has_two_classes() -> None:
    """Pistachio dataset exposes kirmizi and siirt cultivars."""
    spec = get_dataset_spec("pistachio")
    assert spec.num_classes == 2
    assert spec.class_names == ["kirmizi", "siirt"]
    assert spec.data_root == Path("data/Pistachio/pistachios")


def test_acfr_multifruit_spec_has_three_classes() -> None:
    """ACFR-derived classification exposes almond/apple/mangoe classes."""
    spec = get_dataset_spec("acfr_multifruit")
    assert spec.num_classes == 3
    assert spec.class_names == ["almond", "apple", "mangoe"]
    assert spec.data_root == Path("data/ACFR_Multifruit_Classification/acfr_multifruit")


def test_acfr_alias_resolves_to_acfr_multifruit() -> None:
    """Short alias 'acfr' maps to the canonical dataset key."""
    assert resolve_dataset_key("acfr") == "acfr_multifruit"
    assert get_dataset_spec("acfr").name == "acfr_multifruit"


def test_unknown_dataset_raises() -> None:
    """Invalid dataset names are rejected."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset_spec("not-a-dataset")


@pytest.mark.skipif(
    not Path("data/test_data/test_images.npy").is_file(),
    reason="MNIST quick-test subset not present",
)
def test_mnist_quick_loader_applies_resize() -> None:
    """MNIST .npy quick subset works with torchvision Resize (PIL path)."""
    spec = get_dataset_spec("mnist")
    train_loader, _test_loader = DataLoaderFactory.get_loaders_for_dataset(
        spec=spec,
        batch_size=4,
        num_workers=0,
        image_size=28,
        quick_test=True,
    )
    images, labels = next(iter(train_loader))
    assert images.shape == (4, 1, 28, 28)
    assert labels.shape == (4,)


@pytest.mark.skipif(
    not Path("data/Strawberry/strawberries/red/sets/train.txt").is_file(),
    reason="Strawberry data not downloaded",
)
def test_strawberry_loader_reads_images() -> None:
    """PV-style loader returns tensors when data is present."""
    spec = get_dataset_spec("strawberry")
    spec.validate()
    dataset = PVStyleImageDataset(
        spec.data_root,
        split="train",
        class_names=spec.class_names,
        class_layout=spec.class_layout,
        max_samples=8,
    )
    assert len(dataset) > 0
    image, label = dataset[0]
    assert image.ndim == 3
    assert 0 <= label.item() < spec.num_classes

    train_loader, test_loader = DataLoaderFactory.get_pv_style_loaders(
        spec, batch_size=2, num_workers=0, quick_test=True, quick_max_samples=12
    )
    assert len(train_loader.dataset) <= 12
    assert len(test_loader.dataset) <= 12


@pytest.mark.skipif(
    not Path(
        "data/Plant_Village_Raspberry/raspberries/healthy/color/sets/train.txt"
    ).is_file(),
    reason="Raspberry data not downloaded",
)
def test_plant_village_raspberry_loader_reads_images() -> None:
    """Plant Village Raspberry loads images from color and without_augmentation."""
    spec = get_dataset_spec("plant_village_raspberry")
    spec.validate()
    train_loader, test_loader = DataLoaderFactory.get_pv_style_loaders(
        spec, batch_size=2, num_workers=0, quick_test=True, quick_max_samples=20
    )
    assert len(train_loader.dataset) > 0
    assert len(test_loader.dataset) > 0
    images, labels = next(iter(train_loader))
    assert images.shape[1] == 1
    assert labels.max().item() < spec.num_classes


@pytest.mark.skipif(
    not Path("data/Pistachio/pistachios/kirmizi/sets/train.txt").is_file(),
    reason="Pistachio data not present",
)
def test_pistachio_loader_reads_images() -> None:
    """Pistachio loads kirmizi and siirt cultivar splits."""
    spec = get_dataset_spec("pistachio")
    spec.validate()
    train_loader, test_loader = DataLoaderFactory.get_pv_style_loaders(
        spec, batch_size=2, num_workers=0, quick_test=True, quick_max_samples=20
    )
    assert len(train_loader.dataset) > 0
    assert len(test_loader.dataset) > 0
    images, labels = next(iter(train_loader))
    assert images.shape[1] == 1
    assert labels.max().item() < spec.num_classes


@pytest.mark.skipif(
    not Path(
        "data/ACFR_Multifruit_Classification/acfr_multifruit/apple/sets/train.txt"
    ).is_file(),
    reason="ACFR derived classification data not prepared",
)
def test_acfr_multifruit_loader_reads_images() -> None:
    """ACFR crop dataset loads PV-style class folders and split files."""
    spec = get_dataset_spec("acfr_multifruit")
    spec.validate()
    train_loader, test_loader = DataLoaderFactory.get_pv_style_loaders(
        spec, batch_size=2, num_workers=0, quick_test=True, quick_max_samples=20
    )
    assert len(train_loader.dataset) > 0
    assert len(test_loader.dataset) > 0
    images, labels = next(iter(train_loader))
    assert images.shape[1] == 1
    assert labels.max().item() < spec.num_classes


@pytest.mark.skipif(
    not Path(
        "data/Plant_Village_Orange/oranges/huanglongbing_citrus_greening/color/sets/train.txt"
    ).is_file(),
    reason="Orange data not downloaded",
)
def test_plant_village_orange_loader_reads_images() -> None:
    """Plant Village Orange loads HLB and background_without_leaves splits."""
    spec = get_dataset_spec("plant_village_orange")
    spec.validate()
    train_loader, test_loader = DataLoaderFactory.get_pv_style_loaders(
        spec, batch_size=2, num_workers=0, quick_test=True, quick_max_samples=20
    )
    assert len(train_loader.dataset) > 0
    assert len(test_loader.dataset) > 0
