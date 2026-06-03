#!/usr/bin/env python3
"""Convert ACFR detection annotations into crop-level classification splits."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image

FRUIT_SOURCES: Tuple[Tuple[str, str], ...] = (
    ("apples", "apple"),
    ("mangoes", "mangoe"),
    ("almonds", "almond"),
)


@dataclass(frozen=True)
class CropRecord:
    """Metadata for one saved crop image."""

    class_name: str
    split: str
    filename: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ACFR crop conversion."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate crop-level classification data from ACFR combined COCO "
            "annotations."
        )
    )
    parser.add_argument(
        "--acfr-root",
        type=Path,
        default=Path("data/acfr-multifruit-2016"),
        help="Path to the ACFR multifruit root directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/ACFR_Multifruit_Classification/acfr_multifruit"),
        help="Output root for PV-style classification dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "val", "test"],
        help="COCO split files to convert.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete output-root before conversion.",
    )
    parser.add_argument(
        "--min-box-size",
        type=int,
        default=8,
        help="Skip bounding boxes smaller than this size in pixels.",
    )
    return parser.parse_args()


def _sanitize_class_name(name: str) -> str:
    """Normalize class names for stable directory names."""
    return name.strip().lower().replace(" ", "_")


def _load_json(path: Path) -> Dict[str, object]:
    """Load a JSON file as a dict."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _bbox_to_rect(
    bbox: Sequence[float],
    image_size: Tuple[int, int],
    min_box_size: int,
) -> Tuple[int, int, int, int] | None:
    """Convert COCO bbox `[x, y, w, h]` to a clipped pixel rectangle."""
    if len(bbox) != 4:
        return None
    x, y, width, height = bbox
    if width < min_box_size or height < min_box_size:
        return None

    img_w, img_h = image_size
    left = max(0, int(round(x)))
    top = max(0, int(round(y)))
    right = min(img_w, int(round(x + width)))
    bottom = min(img_h, int(round(y + height)))

    if right - left < min_box_size or bottom - top < min_box_size:
        return None
    return left, top, right, bottom


def _ensure_layout(output_root: Path, class_names: Iterable[str]) -> None:
    """Create PV-style class directories and split folders."""
    for class_name in class_names:
        class_dir = output_root / class_name
        (class_dir / "images").mkdir(parents=True, exist_ok=True)
        (class_dir / "sets").mkdir(parents=True, exist_ok=True)


def _parse_float(value: str | None) -> float | None:
    """Parse a float value from CSV text, returning None for invalid input."""
    if value is None:
        return None
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


def _read_csv_rectangles(
    csv_path: Path,
    image_size: Tuple[int, int],
    min_box_size: int,
) -> List[Tuple[int, int, int, int]]:
    """Read fruit bounding boxes from an ACFR per-image CSV file."""
    if not csv_path.is_file():
        return []

    boxes: List[Tuple[int, int, int, int]] = []
    with csv_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            normalized = {
                key.strip().lower().replace("-", "_"): value for key, value in row.items()
            }
            x = _parse_float(normalized.get("x"))
            y = _parse_float(normalized.get("y"))
            width = (
                _parse_float(normalized.get("width"))
                or _parse_float(normalized.get("w"))
                or _parse_float(normalized.get("dx"))
            )
            height = (
                _parse_float(normalized.get("height"))
                or _parse_float(normalized.get("h"))
                or _parse_float(normalized.get("dy"))
            )

            # Apple annotations are stored as circle center + radius.
            if x is None or y is None:
                center_x = _parse_float(normalized.get("c_x"))
                center_y = _parse_float(normalized.get("c_y"))
                radius = _parse_float(normalized.get("radius")) or _parse_float(
                    normalized.get("r")
                )
                if center_x is not None and center_y is not None and radius is not None:
                    x = center_x - radius
                    y = center_y - radius
                    width = 2 * radius
                    height = 2 * radius

            if x is None or y is None or width is None or height is None:
                continue
            rect = _bbox_to_rect(
                bbox=[x, y, width, height],
                image_size=image_size,
                min_box_size=min_box_size,
            )
            if rect is not None:
                boxes.append(rect)
    return boxes


def convert_split(
    acfr_root: Path,
    output_root: Path,
    split: str,
    min_box_size: int,
) -> List[CropRecord]:
    """Convert one ACFR split COCO file into crop images."""
    _ensure_layout(output_root, [class_name for _, class_name in FRUIT_SOURCES])
    counters: Dict[str, int] = defaultdict(int)
    saved: List[CropRecord] = []
    for fruit_name, class_name in FRUIT_SOURCES:
        annotations_path = (
            acfr_root / "annotations" / f"{fruit_name}_instances_{split}.json"
        )
        if not annotations_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {annotations_path}")

        payload = _load_json(annotations_path)
        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        image_map: Dict[int, Dict[str, object]] = {
            int(image["id"]): image for image in images if "id" in image
        }
        annotations_by_image: Dict[int, List[Dict[str, object]]] = defaultdict(list)
        for annotation in annotations:
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        image_cache: Dict[int, Image.Image] = {}
        for image_id, image_info in image_map.items():
            image_rel = Path(str(image_info["file_name"]))
            image_path = acfr_root / image_rel
            if not image_path.is_file():
                continue

            image = image_cache.get(image_id)
            if image is None:
                image = Image.open(image_path).convert("RGB")
                image_cache[image_id] = image

            rects: List[Tuple[int, int, int, int]] = []
            for annotation in annotations_by_image.get(image_id, []):
                rect = _bbox_to_rect(
                    bbox=annotation.get("bbox", []),
                    image_size=image.size,
                    min_box_size=min_box_size,
                )
                if rect is not None:
                    rects.append(rect)

            if not rects:
                csv_path = acfr_root / fruit_name / "csv" / f"{image_path.stem}.csv"
                rects = _read_csv_rectangles(
                    csv_path=csv_path,
                    image_size=image.size,
                    min_box_size=min_box_size,
                )

            for rect in rects:
                crop = image.crop(rect)
                stem = image_path.stem
                counters[class_name] += 1
                filename = f"{split}_{stem}_{counters[class_name]:06d}.png"
                crop_path = output_root / class_name / "images" / filename
                crop.save(crop_path)
                saved.append(
                    CropRecord(class_name=class_name, split=split, filename=filename)
                )

    return saved


def write_split_lists(output_root: Path, records: Sequence[CropRecord]) -> None:
    """Write PV-style `sets/train.txt` and `sets/test.txt` for each class."""
    grouped: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for record in records:
        grouped[(record.class_name, record.split)].append(record.filename)

    class_names = {record.class_name for record in records}
    for class_name in class_names:
        sets_dir = output_root / class_name / "sets"
        for split in ("train", "test", "val"):
            filenames = sorted(grouped.get((class_name, split), []))
            split_path = sets_dir / f"{split}.txt"
            split_path.write_text("\n".join(filenames), encoding="utf-8")

        train_val = sorted(
            grouped.get((class_name, "train"), []) + grouped.get((class_name, "val"), [])
        )
        (sets_dir / "train_val.txt").write_text("\n".join(train_val), encoding="utf-8")
        all_items = sorted(
            train_val + grouped.get((class_name, "test"), [])
        )
        (sets_dir / "all.txt").write_text("\n".join(all_items), encoding="utf-8")


def main() -> int:
    """Run ACFR detection-to-classification conversion."""
    args = parse_args()
    acfr_root = args.acfr_root.resolve()
    output_root = args.output_root.resolve()

    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    all_records: List[CropRecord] = []
    for split in args.splits:
        records = convert_split(
            acfr_root=acfr_root,
            output_root=output_root,
            split=split,
            min_box_size=args.min_box_size,
        )
        all_records.extend(records)
        print(f"[{split}] saved {len(records)} crops")

    if not all_records:
        raise RuntimeError("No crops were generated. Check annotations and source files.")

    write_split_lists(output_root=output_root, records=all_records)
    class_counts: Dict[str, int] = defaultdict(int)
    for record in all_records:
        class_counts[record.class_name] += 1

    print("Done. Class distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  - {class_name}: {count}")
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
