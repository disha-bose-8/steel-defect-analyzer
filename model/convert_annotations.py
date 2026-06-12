"""
convert_annotations.py — Pascal VOC XML → YOLO TXT format
NEU-DET dataset converter

Reads:  dataset/NEU-DET/train/annotations/*.xml
        dataset/NEU-DET/validation/annotations/*.xml

Writes: dataset/NEU-DET_YOLO/
        ├── images/
        │   ├── train/        (symlinks or copies of original images)
        │   └── val/
        ├── labels/
        │   ├── train/        (YOLO .txt files)
        │   └── val/
        └── data.yaml         (YOLOv8 config)

YOLO label format per line:
    <class_id> <x_center> <y_center> <width> <height>
    (all values normalized 0–1 relative to image size)

Run:
    python convert_annotations.py
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG — adjust paths if your layout differs
# ──────────────────────────────────────────────
SRC_ROOT  = Path("../dataset/NEU-DET")
DST_ROOT  = Path("../dataset/NEU-DET_YOLO")

SPLITS = {
    "train": SRC_ROOT / "train",
    "val":   SRC_ROOT / "validation",
}

# NEU-DET class order — must be consistent across train/val
# YOLOv8 class IDs will be assigned in this order (0-indexed)
CLASS_ORDER = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Image extension used in your dataset
IMG_EXT = ".jpg"   # change to ".png" if needed — script will auto-detect


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def find_image(images_dir: Path, stem: str) -> Path | None:
    """Find the image file for a given annotation stem (tries jpg, png, bmp)."""
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_voc_xml(xml_path: Path):
    """
    Parse a Pascal VOC XML annotation file.

    Returns:
        img_w (int), img_h (int), objects (list of dicts with keys: name, xmin, ymin, xmax, ymax)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))
        objects.append({"name": name, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

    return img_w, img_h, objects


def voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert absolute VOC bbox to normalized YOLO format."""
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center  = (ymin + ymax) / 2.0 / img_h
    width     = (xmax - xmin) / img_w
    height    = (ymax - ymin) / img_h
    return x_center, y_center, width, height


def build_class_map(splits: dict) -> dict:
    """
    Auto-detect all unique class names from XMLs and map to integer IDs.
    Uses CLASS_ORDER if all classes are found there, otherwise appends unknowns.
    """
    found = set()
    for split_name, split_dir in splits.items():
        ann_dir = split_dir / "annotations"
        if not ann_dir.exists():
            continue
        for xml_file in ann_dir.glob("*.xml"):
            tree = ET.parse(xml_file)
            for obj in tree.getroot().findall("object"):
                found.add(obj.find("name").text.strip())

    # Build ordered class list: predefined order first, then any extras
    class_list = [c for c in CLASS_ORDER if c in found]
    extras = sorted(found - set(CLASS_ORDER))
    if extras:
        print(f"  [Warning] Found classes not in CLASS_ORDER: {extras} — appending at end")
        class_list.extend(extras)

    class_map = {name: idx for idx, name in enumerate(class_list)}
    return class_map, class_list


# ──────────────────────────────────────────────
# MAIN CONVERSION
# ──────────────────────────────────────────────
def convert():
    print("=" * 60)
    print("NEU-DET Pascal VOC → YOLO Converter")
    print("=" * 60)

    # Build class map from all annotations
    class_map, class_list = build_class_map(SPLITS)
    print(f"\nDetected {len(class_list)} classes:")
    for idx, name in enumerate(class_list):
        print(f"  {idx}: {name}")

    total_images = 0
    total_boxes  = 0
    skipped      = 0

    for split_name, split_dir in SPLITS.items():
        ann_dir = split_dir / "annotations"
        img_dir = split_dir / "images"

        if not ann_dir.exists():
            print(f"\n[Warning] No annotations directory found at {ann_dir} — skipping {split_name}")
            continue

        # Create output dirs
        out_img_dir   = DST_ROOT / "images" / split_name
        out_label_dir = DST_ROOT / "labels" / split_name
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        xml_files = sorted(ann_dir.glob("*.xml"))
        print(f"\n[{split_name}] Processing {len(xml_files)} annotation files ...")

        split_images = 0
        split_boxes  = 0

        for xml_path in xml_files:
            stem = xml_path.stem  # filename without extension

            # Find corresponding image
            # NEU-DET stores images in class subdirectories: images/crazing/img.jpg
            # Try flat first, then search class subdirs
            img_path = find_image(img_dir, stem)
            if img_path is None:
                # Search inside class subdirectories
                for class_subdir in img_dir.iterdir():
                    if class_subdir.is_dir():
                        img_path = find_image(class_subdir, stem)
                        if img_path is not None:
                            break

            if img_path is None:
                print(f"    [Skip] No image found for {stem}")
                skipped += 1
                continue

            # Parse annotation
            try:
                img_w, img_h, objects = parse_voc_xml(xml_path)
            except Exception as e:
                print(f"    [Error] Could not parse {xml_path.name}: {e}")
                skipped += 1
                continue

            # Skip if image dimensions missing (some NEU XMLs have 0x0)
            if img_w == 0 or img_h == 0:
                # Fall back to reading actual image size
                try:
                    from PIL import Image as PILImage
                    with PILImage.open(img_path) as pil_img:
                        img_w, img_h = pil_img.size
                except Exception:
                    print(f"    [Skip] Zero dimensions and cannot read image: {stem}")
                    skipped += 1
                    continue

            # Write YOLO label file
            label_lines = []
            for obj in objects:
                class_name = obj["name"]
                if class_name not in class_map:
                    print(f"    [Warning] Unknown class '{class_name}' in {xml_path.name} — skipping box")
                    continue

                class_id = class_map[class_name]
                x_c, y_c, w, h = voc_to_yolo(
                    obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"],
                    img_w, img_h
                )

                # Clamp to [0, 1] — some VOC annotations have slight overflows
                x_c = max(0.0, min(1.0, x_c))
                y_c = max(0.0, min(1.0, y_c))
                w   = max(0.0, min(1.0, w))
                h   = max(0.0, min(1.0, h))

                label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                split_boxes += 1

            if not label_lines:
                # No valid boxes — still copy image but write empty label
                pass

            label_path = out_label_dir / f"{stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

            # Copy image to YOLO dataset dir
            dst_img_path = out_img_dir / img_path.name
            if not dst_img_path.exists():
                shutil.copy2(img_path, dst_img_path)

            split_images += 1

        print(f"    Images: {split_images} | Boxes: {split_boxes}")
        total_images += split_images
        total_boxes  += split_boxes

    # ──────────────────────────────────────────────
    # Write data.yaml
    # ──────────────────────────────────────────────
    data_yaml_path = DST_ROOT / "data.yaml"
    yaml_content = f"""# NEU-DET YOLOv8 dataset config
# Auto-generated by convert_annotations.py

path: {DST_ROOT.resolve()}   # absolute path to dataset root
train: images/train
val:   images/val

nc: {len(class_list)}
names: {class_list}
"""
    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)

    print("\n" + "=" * 60)
    print(f"Conversion complete.")
    print(f"  Total images : {total_images}")
    print(f"  Total boxes  : {total_boxes}")
    print(f"  Skipped      : {skipped}")
    print(f"  Output dir   : {DST_ROOT.resolve()}")
    print(f"  data.yaml    : {data_yaml_path.resolve()}")
    print("=" * 60)

    return class_list, str(data_yaml_path.resolve())


if __name__ == "__main__":
    convert()