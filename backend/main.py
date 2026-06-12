"""
main.py — FastAPI Backend
Steel Surface Defect Analyzer v2

Endpoints:
    GET  /                → health check + loaded model status
    POST /predict         → ResNet18 classifier: class + confidence + top3 (existing, unchanged)
    POST /detect          → YOLOv8 detector: bounding boxes + annotated image as base64 PNG (NEW)
    GET  /classes         → all defect classes with descriptions

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import io
import base64
import os

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from typing import List, Optional

from model_loader import load_model, TRANSFORM, DEFECT_DESCRIPTIONS


# ──────────────────────────────────────────────
# YOLO CHECKPOINT PATH
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR        = os.path.dirname(__file__)
YOLO_CHECKPOINT = os.path.join(BASE_DIR, "..", "artifacts", "yolo", "train", "weights", "best.pt")

# Color per class for bbox drawing (RGB)
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
# APP INIT
# ──────────────────────────────────────────────
app = FastAPI(
    title       = "Steel Surface Defect Analyzer",
    description = "ResNet18 classifier + YOLOv8 bounding box detector for NEU surface defects",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "https://defectnet.netlify.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# LOAD MODELS AT STARTUP
# ──────────────────────────────────────────────
# ResNet18 classifier (always loaded)
resnet_model, class_names = load_model()

# YOLOv8 detector (loaded only if best.pt exists — graceful degradation)
yolo_model        = None
yolo_class_names  = []

if os.path.exists(YOLO_CHECKPOINT):
    try:
        from ultralytics import YOLO
        yolo_model       = YOLO(YOLO_CHECKPOINT)
        yolo_class_names = list(yolo_model.names.values())
        print(f"[YOLO] Loaded — {len(yolo_class_names)} classes: {yolo_class_names}")
    except ImportError:
        print("[YOLO] ultralytics not installed. Run: pip install ultralytics")
    except Exception as e:
        print(f"[YOLO] Failed to load: {e}")
else:
    print(f"[YOLO] Checkpoint not found at {YOLO_CHECKPOINT}. "
          "Run train_yolo.py to enable /detect endpoint.")


# ──────────────────────────────────────────────
# SHARED IMAGE READER
# ──────────────────────────────────────────────
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

async def read_upload(file: UploadFile) -> Image.Image:
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. File may be corrupted.")


# ──────────────────────────────────────────────
# BBOX DRAWING HELPER
# ──────────────────────────────────────────────
def draw_boxes(image: Image.Image, boxes: list) -> Image.Image:
    """Draw bounding boxes with class labels on a PIL image. Returns annotated copy."""
    draw   = ImageDraw.Draw(image)
    orig_w, orig_h = image.size

    line_width = max(2, int(min(orig_w, orig_h) * 0.007))
    font_size  = max(12, int(min(orig_w, orig_h) * 0.05))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except Exception:
        font = ImageFont.load_default()

    for box in boxes:
        class_name = box["class_name"]
        conf       = box["confidence"]
        b          = box["bbox"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
        label = f'{DISPLAY_NAMES.get(class_name, class_name)}'

        # Rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Label background + text
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
class TopPrediction(BaseModel):
    defect    : str
    confidence: float

class PredictionResponse(BaseModel):
    predicted_defect: str
    confidence      : float
    description     : str
    top3            : List[TopPrediction]

class BoundingBox(BaseModel):
    x1    : int
    y1    : int
    x2    : int
    y2    : int
    width : int
    height: int

class Detection(BaseModel):
    class_name   : str
    display_name : str
    confidence   : float
    bbox         : BoundingBox

class DetectionResponse(BaseModel):
    boxes               : List[Detection]
    annotated_image_b64 : str   # base64 PNG with bboxes drawn, render as <img src="data:image/png;base64,...">
    image_width         : int
    image_height        : int
    num_detections      : int
    message             : str


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status"           : "ok",
        "classifier_loaded": resnet_model is not None,
        "detector_loaded"  : yolo_model is not None,
        "classes"          : class_names,
        "endpoints"        : ["/predict", "/detect", "/classes"],
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    ResNet18 classification — returns predicted defect class, confidence, and top-3.
    No bounding boxes. Fast, always available.
    """
    image = await read_upload(file)

    tensor = TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        logits = resnet_model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze()

    confidence, predicted_idx = torch.max(probs, dim=0)
    predicted_class           = class_names[predicted_idx.item()]
    confidence_score          = round(confidence.item() * 100, 2)

    top3_probs, top3_indices  = torch.topk(probs, k=min(3, len(class_names)))
    top3 = [
        {
            "defect"    : class_names[i.item()],
            "confidence": round(p.item() * 100, 2),
        }
        for p, i in zip(top3_probs, top3_indices)
    ]

    return {
        "predicted_defect": predicted_class,
        "confidence"      : confidence_score,
        "description"     : DEFECT_DESCRIPTIONS.get(predicted_class, "No description available."),
        "top3"            : top3,
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    file          : UploadFile = File(...),
    conf_threshold: float      = Query(default=0.25, ge=0.01, le=1.0,
                                       description="Minimum confidence threshold (0.01–1.0)"),
):
    """
    YOLOv8 detection — returns bounding boxes for each defect found,
    plus a base64-encoded PNG of the original image with boxes drawn on it.

    Render the annotated image in frontend with:
        <img src={`data:image/png;base64,${response.annotated_image_b64}`} />
    """
    if yolo_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "YOLOv8 model is not loaded. "
                "Run train_yolo.py first to generate artifacts/yolo/train/weights/best.pt, "
                "then restart the server."
            ),
        )

    image = await read_upload(file)
    orig_w, orig_h = image.size

    # Run YOLOv8 inference
    results = yolo_model.predict(
        source  = image,
        conf    = conf_threshold,
        verbose = False,
        save    = False,
    )

    result    = results[0]
    inf_h, inf_w = result.orig_shape   # shape YOLOv8 used internally for inference
    boxes_out = []

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id     = int(box.cls.item())
            conf       = round(float(box.conf.item()), 4)
            class_name = (
                yolo_class_names[cls_id]
                if cls_id < len(yolo_class_names)
                else f"class_{cls_id}"
            )

            # Map inference coords → original image coords
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

    # Draw boxes on original image
    annotated = draw_boxes(image.copy(), boxes_out)

    # Encode to base64 PNG
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    n = len(boxes_out)
    if n == 0:
        message = "No defects detected above confidence threshold."
    elif n == 1:
        b = boxes_out[0]
        message = f"1 defect detected: {b['display_name']}"
    else:
        unique_classes = list({b["display_name"] for b in boxes_out})
        message = f"{n} defects detected: {', '.join(unique_classes)}"

    return DetectionResponse(
        boxes               = boxes_out,
        annotated_image_b64 = img_b64,
        image_width         = orig_w,
        image_height        = orig_h,
        num_detections      = n,
        message             = message,
    )


@app.get("/classes")
def get_classes():
    return {
        "classes": [
            {
                "name"       : c,
                "description": DEFECT_DESCRIPTIONS.get(c, ""),
            }
            for c in class_names
        ]
    }

@app.get("/health")
def health_check():
    return {
        "status"            : "ok",
        "classifier_loaded" : resnet_model is not None,
        "detector_loaded"   : yolo_model is not None,
        "num_classes"       : len(class_names),
        "device"            : str(DEVICE),
    }