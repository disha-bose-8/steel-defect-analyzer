"""
predict.py — FastAPI Inference Backend
Steel Surface Defect Analyzer

Endpoints:
    POST /predict        → ResNet18 classifier: defect class + confidence (existing)
    POST /detect         → YOLOv8 detector: bounding boxes + class + confidence (NEW)
    GET  /classes        → list of all defect classes
    GET  /health         → health check

Expects:
    artifacts/checkpoints/best_model.pth     (ResNet18, saved by train.py)
    artifacts/yolo/train/weights/best.pt     (YOLOv8, saved by train_yolo.py)

Run:
    uvicorn predict:app --reload --host 0.0.0.0 --port 8000
"""

import io
import os
import base64

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional


# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR              = os.path.dirname(__file__)
CLASSIFIER_CHECKPOINT = os.path.join(BASE_DIR, "..", "artifacts", "checkpoints", "best_model.pth")
YOLO_CHECKPOINT       = os.path.join(BASE_DIR, "..", "artifacts", "yolo", "train", "weights", "best.pt")

IMAGE_SIZE = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORM_MEAN  = [0.406, 0.406, 0.406]
NORM_STD   = [0.170, 0.170, 0.170]

# Color per class for bbox drawing (BGR order doesn't matter here, using RGB)
CLASS_COLORS = {
    "crazing"        : (255, 99,  99),
    "inclusion"      : (99,  200, 99),
    "patches"        : (99,  160, 255),
    "pitted_surface" : (255, 200, 99),
    "rolled-in_scale": (200, 99,  255),
    "scratches"      : (99,  220, 220),
}
DEFAULT_COLOR = (255, 255, 100)

DISPLAY_NAMES = {
    "crazing"        : "Crazing",
    "inclusion"      : "Inclusion",
    "patches"        : "Patches",
    "pitted_surface" : "Pitted Surface",
    "rolled-in_scale": "Rolled-In Scale",
    "scratches"      : "Scratches",
}


# ──────────────────────────────────────────────
# RESNET18 CLASSIFIER (unchanged from v1)
# ──────────────────────────────────────────────
class DefectClassifier:
    def __init__(self):
        self.model       = None
        self.class_names = []
        self.transform   = None
        self._load()

    def _load(self):
        if not os.path.exists(CLASSIFIER_CHECKPOINT):
            raise FileNotFoundError(
                f"Classifier checkpoint not found at {CLASSIFIER_CHECKPOINT}. "
                "Run train.py first."
            )

        checkpoint       = torch.load(CLASSIFIER_CHECKPOINT, map_location=DEVICE)
        self.class_names = checkpoint["class_names"]
        num_classes      = len(self.class_names)

        model            = models.resnet18(weights=None)
        in_features      = model.fc.in_features
        model.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(DEVICE)
        model.eval()
        self.model = model

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])

        print(f"[Classifier] Loaded — {num_classes} classes: {self.class_names}")

    def predict(self, image: Image.Image) -> dict:
        tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze()

        top_idx    = probs.argmax().item()
        confidence = probs[top_idx].item()
        class_name = self.class_names[top_idx]

        all_scores = {
            self.class_names[i]: round(probs[i].item(), 4)
            for i in range(len(self.class_names))
        }

        return {
            "class_name"  : class_name,
            "display_name": DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").title()),
            "confidence"  : round(confidence, 4),
            "all_scores"  : all_scores,
        }


# ──────────────────────────────────────────────
# YOLOV8 DETECTOR (NEW)
# ──────────────────────────────────────────────
class DefectDetector:
    def __init__(self):
        self.model       = None
        self.class_names = []
        self._load()

    def _load(self):
        if not os.path.exists(YOLO_CHECKPOINT):
            print(
                f"[Detector] WARNING: YOLOv8 checkpoint not found at {YOLO_CHECKPOINT}. "
                "Run train_yolo.py to generate it. /detect endpoint will be unavailable."
            )
            return

        try:
            from ultralytics import YOLO
            self.model       = YOLO(YOLO_CHECKPOINT)
            self.class_names = list(self.model.names.values())
            print(f"[Detector] Loaded — {len(self.class_names)} classes: {self.class_names}")
        except ImportError:
            print("[Detector] ultralytics not installed. Run: pip install ultralytics")

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def detect(self, image: Image.Image, conf_threshold: float = 0.25) -> dict:
        """
        Run YOLOv8 inference on a PIL image.

        Returns:
            dict with:
                boxes       — list of detection dicts (class, confidence, bbox pixel coords)
                annotated_image_b64 — base64 PNG with bounding boxes drawn on original image
                image_width, image_height — original image dimensions
        """
        orig_w, orig_h = image.size

        # Run inference — YOLOv8 handles resizing internally
        results = self.model.predict(
            source    = image,
            conf      = conf_threshold,
            verbose   = False,
            save      = False,
        )

        result    = results[0]
        boxes_out = []

        if result.boxes is not None and len(result.boxes) > 0:
            # xyxy = absolute pixel coords in the RESIZED inference image
            # We need to map back to original image size
            inf_h, inf_w = result.orig_shape  # shape YOLOv8 used internally

            for box in result.boxes:
                cls_id     = int(box.cls.item())
                conf       = round(float(box.conf.item()), 4)
                class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"

                # Coords in original image space
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = round(x1 * orig_w / inf_w)
                y1 = round(y1 * orig_h / inf_h)
                x2 = round(x2 * orig_w / inf_w)
                y2 = round(y2 * orig_h / inf_h)

                boxes_out.append({
                    "class_name"  : class_name,
                    "display_name": DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").title()),
                    "confidence"  : conf,
                    "bbox"        : {
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "width" : x2 - x1,
                        "height": y2 - y1,
                    },
                })

        # Draw boxes on a copy of the original image
        annotated = draw_boxes(image.copy(), boxes_out)

        # Encode to base64 PNG
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "boxes"                  : boxes_out,
            "annotated_image_b64"    : img_b64,
            "image_width"            : orig_w,
            "image_height"           : orig_h,
            "num_detections"         : len(boxes_out),
        }


# ──────────────────────────────────────────────
# BBOX DRAWING HELPER
# ──────────────────────────────────────────────
def draw_boxes(image: Image.Image, boxes: list) -> Image.Image:
    """Draw bounding boxes with class labels on a PIL image."""
    draw   = ImageDraw.Draw(image)
    orig_w, orig_h = image.size

    # Scale line width and font size relative to image size
    line_width  = max(2, int(min(orig_w, orig_h) * 0.007))
    font_size   = max(12, int(min(orig_w, orig_h) * 0.05))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for box in boxes:
        class_name = box["class_name"]
        conf       = box["confidence"]
        b          = box["bbox"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
        label = f'{DISPLAY_NAMES.get(class_name, class_name)} {conf*100:.1f}%'

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw label background
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w    = text_bbox[2] - text_bbox[0]
        text_h    = text_bbox[3] - text_bbox[1]
        label_y1  = max(0, y1 - text_h - 6)
        label_y2  = label_y1 + text_h + 6

        draw.rectangle([x1, label_y1, x1 + text_w + 8, label_y2], fill=color)
        draw.text((x1 + 4, label_y1 + 3), label, fill=(0, 0, 0), font=font)

    return image


# ──────────────────────────────────────────────
# RESPONSE SCHEMAS
# ──────────────────────────────────────────────
class PredictionResponse(BaseModel):
    class_name  : str
    display_name: str
    confidence  : float
    all_scores  : dict
    message     : str


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    width : int
    height: int


class Detection(BaseModel):
    class_name   : str
    display_name : str
    confidence   : float
    bbox         : BoundingBox


class DetectionResponse(BaseModel):
    boxes                : List[Detection]
    annotated_image_b64  : str        # base64 PNG with boxes drawn
    image_width          : int
    image_height         : int
    num_detections       : int
    message              : str


class ClassesResponse(BaseModel):
    classes      : List[str]
    display_names: dict


class HealthResponse(BaseModel):
    status            : str
    classifier_loaded : bool
    detector_loaded   : bool
    num_classes       : int
    device            : str


# ──────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────
app = FastAPI(
    title       = "Steel Surface Defect Analyzer",
    description = "ResNet18 classifier + YOLOv8 detection for NEU surface defect analysis",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Load models at startup
classifier = DefectClassifier()
detector   = DefectDetector()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
async def read_image(file: UploadFile) -> Image.Image:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, BMP, or WEBP."
        )
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. File may be corrupted.")


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict_defect(file: UploadFile = File(...)):
    """ResNet18 classification — returns defect class + confidence. No bounding boxes."""
    image = await read_image(file)
    try:
        result = classifier.predict(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    conf_pct = round(result["confidence"] * 100, 1)
    return PredictionResponse(
        class_name   = result["class_name"],
        display_name = result["display_name"],
        confidence   = result["confidence"],
        all_scores   = result["all_scores"],
        message      = f"Detected: {result['display_name']} ({conf_pct}% confidence)",
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect_defects(
    file           : UploadFile = File(...),
    conf_threshold : float      = 0.25,
):
    """
    YOLOv8 detection — returns bounding boxes + annotated image (base64 PNG).

    Query params:
        conf_threshold (float, default 0.25): minimum confidence to include a detection
    """
    if not detector.is_ready:
        raise HTTPException(
            status_code=503,
            detail="YOLOv8 model not loaded. Run train_yolo.py to train the detector first."
        )

    image = await read_image(file)
    try:
        result = detector.detect(image, conf_threshold=conf_threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    n = result["num_detections"]
    if n == 0:
        message = "No defects detected above confidence threshold."
    elif n == 1:
        box = result["boxes"][0]
        message = f"1 defect detected: {box['display_name']} ({box['confidence']*100:.1f}%)"
    else:
        classes_found = list({b["display_name"] for b in result["boxes"]})
        message = f"{n} defects detected: {', '.join(classes_found)}"

    return DetectionResponse(
        boxes               = result["boxes"],
        annotated_image_b64 = result["annotated_image_b64"],
        image_width         = result["image_width"],
        image_height        = result["image_height"],
        num_detections      = n,
        message             = message,
    )


@app.get("/classes", response_model=ClassesResponse)
def get_classes():
    return ClassesResponse(
        classes       = classifier.class_names,
        display_names = {
            cls: DISPLAY_NAMES.get(cls, cls.replace("_", " ").title())
            for cls in classifier.class_names
        }
    )


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status            = "ok",
        classifier_loaded = classifier.model is not None,
        detector_loaded   = detector.is_ready,
        num_classes       = len(classifier.class_names),
        device            = str(DEVICE),
    )


@app.get("/")
def root():
    return {
        "status"     : "ok",
        "endpoints"  : ["/predict", "/detect", "/classes", "/health"],
        "classifier" : classifier.model is not None,
        "detector"   : detector.is_ready,
    }