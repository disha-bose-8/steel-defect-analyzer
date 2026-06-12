"""
train_yolo.py — YOLOv8 Detection Training for NEU-DET
Steel Surface Defect Detector

Prereqs:
    pip install ultralytics

Steps:
    1. Run convert_annotations.py first to produce dataset/NEU-DET_YOLO/ and data.yaml
    2. Run this script: python train_yolo.py

Output:
    artifacts/yolo/train/weights/best.pt   ← use this in predict.py
    artifacts/yolo/train/weights/last.pt
    artifacts/yolo/train/results.png
    artifacts/yolo/train/confusion_matrix.png
    artifacts/yolo/val_predictions/        ← sample inference on val set

Notes:
    - Uses YOLOv8n (nano) by default — fast, low memory, good for NEU-DET image size
    - Swap to yolov8s or yolov8m for better accuracy if you have GPU
    - NEU-DET images are 200×200 grayscale; imgsz=640 upscales for better feature extraction
    - Pretrained COCO weights used → transfer learning, not training from scratch
"""

import os
from pathlib import Path
from convert_annotations import convert   # reuse conversion logic

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
CONFIG = {
    # Model size: n=nano, s=small, m=medium (bigger = more accurate but slower)
    # For CPU-only training, stick with "n"
    # For GPU training, "s" gives noticeably better mAP on NEU-DET
    "model"       : "yolov8n.pt",

    # data.yaml path — auto-set after conversion
    "data"        : None,  # filled in after convert()

    # Training hyperparameters
    "epochs"      : 100,
    "imgsz"       : 640,       # upsample NEU 200x200 → 640x640 for better features
    "batch"       : 16,        # reduce to 8 if OOM on CPU
    "patience"    : 20,        # early stopping if no improvement for 20 epochs

    # Output
    "project"     : "../artifacts/yolo",
    "name"        : "train",
    "save_period" : 10,        # save checkpoint every N epochs

    # Augmentation — YOLOv8 has built-in augmentation, these tune it
    "hsv_h"       : 0.0,       # no hue shift (grayscale images)
    "hsv_s"       : 0.0,       # no saturation shift (grayscale images)
    "hsv_v"       : 0.4,       # value/brightness variation — important for NEU
    "flipud"      : 0.5,       # vertical flip (defects are orientation-agnostic)
    "fliplr"      : 0.5,       # horizontal flip
    "mosaic"      : 0.5,       # mosaic augmentation (mix 4 images)
    "degrees"     : 15.0,      # rotation
    "translate"   : 0.1,
    "scale"       : 0.3,

    # Optimizer
    "optimizer"   : "AdamW",
    "lr0"         : 0.001,
    "lrf"         : 0.01,      # final LR = lr0 * lrf
    "weight_decay": 0.0005,

    # Workers
    "workers"     : 4,
    "device"      : "",        # "" = auto (GPU if available, else CPU)

    # Confidence threshold for saving predictions
    "conf"        : 0.25,
}


# ──────────────────────────────────────────────
# STEP 1: CONVERT ANNOTATIONS
# ──────────────────────────────────────────────
def run_conversion():
    print("\n[Step 1/2] Converting Pascal VOC annotations → YOLO format ...")
    class_list, data_yaml_path = convert()
    return data_yaml_path


# ──────────────────────────────────────────────
# STEP 2: TRAIN YOLOV8
# ──────────────────────────────────────────────
def run_training(data_yaml_path: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics not installed. Run: pip install ultralytics"
        )

    print("\n[Step 2/2] Starting YOLOv8 training ...")
    print(f"  Model    : {CONFIG['model']}")
    print(f"  Data     : {data_yaml_path}")
    print(f"  Epochs   : {CONFIG['epochs']}")
    print(f"  Img size : {CONFIG['imgsz']}")
    print(f"  Batch    : {CONFIG['batch']}")

    # Create output dir
    project_dir = Path(CONFIG["project"])
    project_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained YOLOv8 nano
    model = YOLO(CONFIG["model"])

    # Train
    results = model.train(
        data         = data_yaml_path,
        epochs       = CONFIG["epochs"],
        imgsz        = CONFIG["imgsz"],
        batch        = CONFIG["batch"],
        patience     = CONFIG["patience"],
        project      = str(project_dir.resolve()),
        name         = CONFIG["name"],
        save_period  = CONFIG["save_period"],

        # Augmentation overrides
        hsv_h        = CONFIG["hsv_h"],
        hsv_s        = CONFIG["hsv_s"],
        hsv_v        = CONFIG["hsv_v"],
        flipud       = CONFIG["flipud"],
        fliplr       = CONFIG["fliplr"],
        mosaic       = CONFIG["mosaic"],
        degrees      = CONFIG["degrees"],
        translate    = CONFIG["translate"],
        scale        = CONFIG["scale"],

        # Optimizer
        optimizer    = CONFIG["optimizer"],
        lr0          = CONFIG["lr0"],
        lrf          = CONFIG["lrf"],
        weight_decay = CONFIG["weight_decay"],

        workers      = CONFIG["workers"],
        device       = CONFIG["device"],
        verbose      = True,
    )

    best_pt = Path(CONFIG["project"]) / CONFIG["name"] / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Best weights : {best_pt.resolve()}")
    print(f"  mAP50        : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"  mAP50-95     : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print("=" * 60)

    return str(best_pt.resolve())


# ──────────────────────────────────────────────
# STEP 3: VALIDATE ON VAL SET (save visual predictions)
# ──────────────────────────────────────────────
def run_validation(best_pt_path: str, data_yaml_path: str):
    from ultralytics import YOLO

    print("\n[Step 3/3] Running validation on val set ...")
    model = YOLO(best_pt_path)

    val_results = model.val(
        data    = data_yaml_path,
        imgsz   = CONFIG["imgsz"],
        conf    = CONFIG["conf"],
        project = str(Path(CONFIG["project"]).resolve()),
        name    = "val_predictions",
        save    = True,         # saves annotated images
        plots   = True,
    )

    print("\nPer-class AP50:")
    for cls_name, ap in zip(val_results.names.values(), val_results.box.ap50):
        print(f"  {cls_name:20s}: {ap:.4f}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Convert annotations
    data_yaml_path = run_conversion()

    # 2. Train
    best_pt_path = run_training(data_yaml_path)

    # 3. Validate with visual output
    run_validation(best_pt_path, data_yaml_path)

    print("\nDone. Use the following path in predict.py:")
    print(f"  YOLO_CHECKPOINT_PATH = \"{best_pt_path}\"")