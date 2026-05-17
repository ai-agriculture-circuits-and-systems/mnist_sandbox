import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.dataset_config import DatasetSpec

_IMAGE_EXTENSIONS = (".JPG", ".jpg", ".jpeg", ".png", ".JPEG")


def resolve_image_path(images_dir: Path, name: str) -> Optional[Path]:
    """Resolve a sets-file entry to an on-disk image path (handles missing extensions)."""
    direct = images_dir / name
    if direct.is_file():
        return direct

    for ext in _IMAGE_EXTENSIONS:
        candidate = images_dir / f"{name}{ext}"
        if candidate.is_file():
            return candidate

    candidates: List[Path] = []
    for path in images_dir.iterdir():
        if not path.is_file():
            continue
        stem = path.stem
        if stem == name or stem == f"{name}_1":
            candidates.append(path)

    if not candidates:
        return None

    for path in candidates:
        if path.stem == name:
            return path
    return candidates[0]


class MNISTDataset(Dataset):
    def __init__(self, data_path, transform=None):
        if data_path.endswith('.mat'):
            data = sio.loadmat(data_path)
            self.images = data['x'].reshape(-1, 28, 28).astype(np.float32) / 255.0
            self.labels = data['y'].ravel()
        else:  # .npy format
            self.images = np.load(data_path).astype(np.float32) / 255.0
            # For test data, we expect labels in a separate file with _labels suffix
            base_path = os.path.splitext(data_path)[0]  # Remove extension
            label_path = f"{base_path}_labels.npy"
            if not os.path.exists(label_path):
                # If _labels suffix doesn't exist, try test_labels.npy in the same directory
                dir_path = os.path.dirname(data_path)
                label_path = os.path.join(dir_path, 'test_labels.npy')
            self.labels = np.load(label_path)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            # Apply transform and ensure we get a tensor with shape [channels, height, width]
            image = self.transform(image)
        else:
            # If no transform, convert to tensor with shape [channels, height, width]
            image = torch.FloatTensor(image).unsqueeze(0)

        return image, torch.LongTensor([label])[0]


class PVStyleImageDataset(Dataset):
    """Plant-Village-style layout: <root>/<class>/<variant>/{images,sets}."""

    def __init__(
        self,
        root: Path,
        split: str,
        class_names: List[str],
        transform=None,
        max_samples: Optional[int] = None,
        seed: int = 42,
        class_layout: Optional[Dict[str, str]] = None,
    ) -> None:
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        layout = class_layout or {}

        for label_idx, class_name in enumerate(class_names):
            variant = layout.get(class_name, "color")
            class_dir = root / class_name / variant
            sets_file = class_dir / "sets" / f"{split}.txt"
            images_dir = class_dir / "images"
            if not sets_file.is_file():
                continue
            for line in sets_file.read_text(encoding="utf-8").splitlines():
                filename = line.strip()
                if not filename:
                    continue
                image_path = resolve_image_path(images_dir, filename)
                if image_path is not None:
                    self.samples.append((image_path, label_idx))

        if max_samples is not None and len(self.samples) > max_samples:
            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, max_samples)

        if not self.samples:
            raise ValueError(f"No images found for split '{split}' under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, torch.tensor(label, dtype=torch.long)


class DataLoaderFactory:
    @staticmethod
    def _build_transform(image_size: int, channels: int = 1):
        steps = [
            transforms.Resize((image_size, image_size)),
        ]
        if channels == 1:
            steps.append(transforms.Grayscale(num_output_channels=1))
        steps.append(transforms.ToTensor())
        return transforms.Compose(steps)

    @staticmethod
    def get_data_loaders(train_path, test_path, batch_size=32, num_workers=4, image_size=224, channels=1):
        default_transform = DataLoaderFactory._build_transform(image_size, channels)

        train_dataset = MNISTDataset(train_path, transform=default_transform)
        test_dataset = MNISTDataset(test_path, transform=default_transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, test_loader

    @staticmethod
    def get_pv_style_loaders(
        spec: DatasetSpec,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        quick_test: bool = False,
        quick_max_samples: int = 60,
        seed: int = 42,
    ):
        """Load train/test splits from Plant-Village-style directory layout."""
        transform = DataLoaderFactory._build_transform(image_size, spec.channels)
        max_samples = quick_max_samples if quick_test else None

        train_dataset = PVStyleImageDataset(
            spec.data_root,
            split="train",
            class_names=spec.class_names,
            transform=transform,
            max_samples=max_samples,
            seed=seed,
            class_layout=spec.class_layout,
        )
        test_dataset = PVStyleImageDataset(
            spec.data_root,
            split="test",
            class_names=spec.class_names,
            transform=transform,
            max_samples=max_samples,
            seed=seed + 1,
            class_layout=spec.class_layout,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, test_loader

    @staticmethod
    def get_loaders_for_dataset(
        spec: DatasetSpec,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        quick_test: bool = False,
        seed: int = 42,
    ):
        """Return train and test loaders for a registered dataset."""
        if spec.train_source == "mat":
            if quick_test:
                return DataLoaderFactory.get_data_loaders(
                    train_path=spec.quick_train_path,
                    test_path=spec.quick_test_path,
                    batch_size=min(batch_size, 32),
                    num_workers=min(num_workers, 2),
                    image_size=image_size,
                    channels=spec.channels,
                )
            return DataLoaderFactory.get_data_loaders(
                train_path=str(spec.data_root / "MNISTtrain.mat"),
                test_path=str(spec.data_root / "MNISTtest.mat"),
                batch_size=batch_size,
                num_workers=num_workers,
                image_size=image_size,
                channels=spec.channels,
            )

        return DataLoaderFactory.get_pv_style_loaders(
            spec=spec,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            quick_test=quick_test,
            seed=seed,
        )
