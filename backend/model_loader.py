import torch
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
CHECKPOINT_PATH = ARTIFACTS_DIR / "checkpoints" / "best_model.pth"

TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.406, 0.406, 0.406],
        std=[0.170, 0.170, 0.170]
    ),
])

DEFECT_DESCRIPTIONS = {
    "crazing":         "Fine network of surface cracks caused by stress or thermal cycling.",
    "inclusion":       "Foreign material embedded in the steel surface during rolling.",
    "patches":         "Irregular discolored or rough areas on the surface.",
    "pitted_surface":  "Small cavities or pits formed by corrosion or mechanical damage.",
    "rolled-in_scale": "Oxide scale pressed into the surface during hot rolling.",
    "scratches":       "Linear surface marks caused by abrasive contact or handling.",
    "sliver":          "Thin, elongated metal slivers lifted from the surface.",
}


def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    # Checkpoint schema: { epoch, model_state, val_acc, class_names, config }
    class_names = checkpoint["class_names"]
    state_dict  = checkpoint["model_state"]

    num_classes = len(class_names)
    model = models.resnet18(weights=None)

    # train.py used Sequential(Dropout, Linear) — must match exactly
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.BatchNorm1d(in_features) ,
        torch.nn.Dropout(p=0.4) ,
        torch.nn.Linear(in_features , 256) ,
        torch.nn.ReLU() ,
        torch.nn.Dropout(p=0.3) ,
        torch.nn.Linear(256 , num_classes) ,
    )


    model.load_state_dict(state_dict)
    model.eval()

    return model, class_names
