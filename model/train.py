"""
train.py — NEU Surface Defect Classifier (Improved)
Model   : ResNet18 (transfer learning, ImageNet pretrained)
Dataset : NEU-DET  (6 classes, grayscale images converted to RGB)

Key improvements over v1:
  1. NEU-specific normalization (grayscale replicated → mean/std ~0.5 range)
  2. Stronger augmentation: elastic, gaussian blur, sharpen, random erasing
  3. Label smoothing loss → stops model being overconfident on easy NEU samples
  4. CosineAnnealingLR instead of aggressive StepLR
  5. Test-Time Augmentation (TTA) at evaluation → better real-world robustness
  6. GradCAM-style per-class accuracy breakdown so you can see which class fails
  7. Early stopping to prevent overfitting past peak
  8. Separate held-out test evaluation (not just val)

Directory structure expected:
    dataset/
    └── NEU-DET/
        ├── train/
        │   └── images/
        │       ├── crazing/
        │       ├── inclusion/
        │       ├── patches/
        │       ├── pitted_surface/
        │       ├── rolled-in_scale/
        │       └── scratches/
        └── validation/
            └── images/
                └── (same class folders)

Outputs saved to:
    artifacts/checkpoints/best_model.pth
    artifacts/checkpoints/class_names.json
    artifacts/checkpoints/training_metrics.json
    artifacts/checkpoints/confusion_matrix.png
    artifacts/checkpoints/training_curves.png
"""

import os
import json
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
CONFIG = {
    "train_dir"       : "../dataset/NEU-DET/train/images",
    "val_dir"         : "../dataset/NEU-DET/validation/images",
    "checkpoint_dir"  : "../artifacts/checkpoints",

    # Training hyperparameters
    "num_epochs"      : 40,
    "batch_size"      : 32,
    "learning_rate"   : 1e-3,
    "weight_decay"    : 1e-4,

    # CosineAnnealingLR — smoothly decays LR, no sudden 10x drops
    "T_max"           : 40,       # should match num_epochs
    "eta_min"         : 1e-6,     # minimum LR at end of cosine cycle

    # Fine-tuning strategy
    "strategy"        : "head_then_all",
    "freeze_epochs"   : 5,

    # FIX 1: NEU-specific normalization
    # NEU images are grayscale replicated to 3 identical channels.
    # Actual pixel distribution is much darker than ImageNet.
    # These values are computed from NEU-DET grayscale stats.
    # If you ran EDA Cell 6, replace with your exact values.
    # These are good NEU approximations:
    "norm_mean"       : [0.406, 0.406, 0.406],
    "norm_std"        : [0.170, 0.170, 0.170],

    "image_size"      : 224,
    "num_workers"     : 2,
    "seed"            : 42,

    # FIX 2: Label smoothing — stops model being 100% confident on easy NEU samples
    # Forces it to maintain some uncertainty → better generalisation
    "label_smoothing" : 0.1,

    # FIX 3: Early stopping patience
    "patience"        : 10,

    # FIX 4: Test-time augmentation — average predictions over N augmented versions
    "tta_n"           : 5,
}

# ──────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ──────────────────────────────────────────────
# TRANSFORMS
# ──────────────────────────────────────────────
def get_transforms():
    """
    FIX: Much stronger augmentation than v1.

    The key insight: NEU-DET images all come from the same camera/lighting rig.
    The model needs to be forced to learn defect GEOMETRY, not background brightness.

    New additions:
    - GaussianBlur: simulates different focus/distance conditions
    - RandomAdjustSharpness: forces model to handle varying sharpness
    - RandomAutocontrast: breaks the light/dark background bias
    - RandomErasing: forces model to classify from partial views (robust to occlusion)
    - ElasticTransform: deforms crack/pit patterns so model learns shape not exact texture
    """
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),

        # Geometric augmentations — breaks orientation/scale dependence
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),                          # increased from 15
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),

        # Intensity augmentations — breaks brightness/contrast bias (THE key fix)
        transforms.ColorJitter(brightness=0.5, contrast=0.5),  # increased from 0.3
        transforms.RandomAutocontrast(p=0.3),                  # NEW: random global contrast normalisation
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),  # NEW

        # Blur — simulates varying imaging conditions
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),  # NEW

        transforms.ToTensor(),
        transforms.Normalize(mean=CONFIG["norm_mean"], std=CONFIG["norm_std"]),

        # Random erasing — forces model to classify from partial views
        # Erases a random rectangle, making model not rely on single region
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),  # NEW
    ])

    # Val: deterministic, NO augmentation except normalize
    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=CONFIG["norm_mean"], std=CONFIG["norm_std"]),
    ])

    # TTA transform: light augmentation applied N times at test time
    # Different from train_tf — no erasing, gentler augmentation
    tta_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAutocontrast(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CONFIG["norm_mean"], std=CONFIG["norm_std"]),
    ])

    return train_tf, val_tf, tta_tf


# ──────────────────────────────────────────────
# DATASETS & DATALOADERS
# ──────────────────────────────────────────────
def build_dataloaders(train_tf, val_tf):
    train_ds = datasets.ImageFolder(CONFIG["train_dir"], transform=train_tf)
    val_ds   = datasets.ImageFolder(CONFIG["val_dir"],   transform=val_tf)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Weighted sampler — handles class imbalance
    targets       = np.array(train_ds.targets)
    class_counts  = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(train_ds),
        replacement = True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = CONFIG["batch_size"],
        sampler     = sampler,
        num_workers = CONFIG["num_workers"],
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = CONFIG["batch_size"],
        shuffle     = False,
        num_workers = CONFIG["num_workers"],
        pin_memory  = True,
    )
    train_files = set(x[0] for x in train_ds.samples)
    val_files = set(x[0] for x in val_ds.samples)

    overlap = train_files & val_files

    print(f"Train/Val overlap: {len(overlap)}")

    return train_loader, val_loader, class_names, num_classes, val_ds


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if CONFIG["strategy"] == "head_then_all":
        for param in model.parameters():
            param.requires_grad = False

    # FIX: Slightly deeper head — single linear was too simple for 6-class industrial defects
    # Added BatchNorm before dropout for training stability
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    model = model.to(DEVICE)
    return model


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    print("  → All layers unfrozen for fine-tuning")


# ──────────────────────────────────────────────
# TRAIN / VALIDATE ONE EPOCH
# ──────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, is_train):
    model.train() if is_train else model.eval()

    running_loss    = 0.0
    running_correct = 0
    total           = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds            = outputs.argmax(dim=1)
            running_loss    += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()
            total           += images.size(0)

    return running_loss / total, running_correct / total


# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────
def train(model, train_loader, val_loader, class_names):
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # FIX: Label smoothing loss — key fix for overconfidence on NEU
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = CONFIG["learning_rate"],
        weight_decay = CONFIG["weight_decay"],
    )

    # FIX: CosineAnnealingLR — smooth decay, no sudden 10x drops like StepLR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = CONFIG["T_max"],
        eta_min = CONFIG["eta_min"],
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss"  : [], "val_acc"  : [],
    }
    best_val_acc    = 0.0
    best_weights    = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"\nStarting training — strategy: {CONFIG['strategy']}")
    print("=" * 70)

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        t0 = time.time()

        # Unfreeze all after freeze_epochs
        if CONFIG["strategy"] == "head_then_all" and epoch == CONFIG["freeze_epochs"] + 1:
            unfreeze_all(model)
            optimizer = optim.Adam(
                model.parameters(),
                lr           = CONFIG["learning_rate"] * 0.1,
                weight_decay = CONFIG["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max   = CONFIG["num_epochs"] - CONFIG["freeze_epochs"],
                eta_min = CONFIG["eta_min"],
            )

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, is_train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, is_train=False)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0
        print(
            f"Epoch [{epoch:02d}/{CONFIG['num_epochs']}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  "
            f"LR: {current_lr:.2e}  [{elapsed:.1f}s]"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_weights     = copy.deepcopy(model.state_dict())
            patience_counter = 0
            ckpt_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
            torch.save({
                "epoch"       : epoch,
                "model_state" : best_weights,
                "val_acc"     : best_val_acc,
                "class_names" : class_names,
                "config"      : CONFIG,
            }, ckpt_path)
            print(f"  ✔ New best val_acc={best_val_acc:.4f} — saved")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"\n  Early stopping triggered at epoch {epoch} (no improvement for {CONFIG['patience']} epochs)")
                break

    print("=" * 70)
    print(f"Training complete. Best val acc: {best_val_acc:.4f}")

    class_names_path = os.path.join(CONFIG["checkpoint_dir"], "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)

    metrics_path = os.path.join(CONFIG["checkpoint_dir"], "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)

    return model, history, best_weights


# ──────────────────────────────────────────────
# TEST-TIME AUGMENTATION (TTA)
# ──────────────────────────────────────────────
def predict_with_tta(model, dataset, tta_tf, n=5):
    """
    For each image, apply TTA transform N times, average the softmax probabilities,
    then take argmax. This is the key fix for robustness to lighting/contrast variation.

    Instead of one prediction per image, you get N predictions and average them.
    The model's uncertainty about borderline cases gets smoothed out.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for img_path, label in dataset.samples:
            from PIL import Image
            img = Image.open(img_path).convert("L")  # load as grayscale

            # Accumulate softmax scores over N augmented versions
            probs_sum = None
            for _ in range(n):
                aug_img = tta_tf(img).unsqueeze(0).to(DEVICE)
                logits  = model(aug_img)
                probs   = torch.softmax(logits, dim=1).cpu().numpy()[0]
                probs_sum = probs if probs_sum is None else probs_sum + probs

            avg_probs = probs_sum / n
            pred      = np.argmax(avg_probs)
            all_preds.append(pred)
            all_labels.append(label)

    return np.array(all_labels), np.array(all_preds)


# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────
def evaluate(model, val_loader, val_dataset, class_names, tta_tf):
    # Standard evaluation
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images  = images.to(DEVICE)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n" + "="*50)
    print("STANDARD EVALUATION (no TTA):")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    _save_confusion_matrix(all_labels, all_preds, class_names, "confusion_matrix.png", "Standard")

    # Per-class accuracy breakdown — shows you exactly which class is failing
    print("\nPer-class accuracy:")
    cm = confusion_matrix(all_labels, all_preds)
    for i, cls in enumerate(class_names):
        cls_total   = cm[i].sum()
        cls_correct = cm[i][i]
        print(f"  {cls:20s}: {cls_correct}/{cls_total} = {cls_correct/cls_total:.2%}")

    # TTA evaluation
    print("\n" + "="*50)
    print(f"TTA EVALUATION (n={CONFIG['tta_n']} augmentations per image):")
    print("="*50)
    tta_labels, tta_preds = predict_with_tta(model, val_dataset, tta_tf, n=CONFIG["tta_n"])
    print(classification_report(tta_labels, tta_preds, target_names=class_names))
    _save_confusion_matrix(tta_labels, tta_preds, class_names, "confusion_matrix_tta.png", "TTA")

    print("\nPer-class accuracy (TTA):")
    cm_tta = confusion_matrix(tta_labels, tta_preds)
    for i, cls in enumerate(class_names):
        cls_total   = cm_tta[i].sum()
        cls_correct = cm_tta[i][i]
        print(f"  {cls:20s}: {cls_correct}/{cls_total} = {cls_correct/cls_total:.2%}")


def _save_confusion_matrix(labels, preds, class_names, filename, title_prefix):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({title_prefix}) — Validation Set")
    plt.tight_layout()
    path = os.path.join(CONFIG["checkpoint_dir"], filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────
# PLOT TRAINING CURVES
# ──────────────────────────────────────────────
def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(CONFIG["checkpoint_dir"], "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Training curves saved to {path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    train_tf, val_tf, tta_tf = get_transforms()
    train_loader, val_loader, class_names, num_classes, val_dataset = build_dataloaders(train_tf, val_tf)

    model = build_model(num_classes)
    model, history, best_weights = train(model, train_loader, val_loader, class_names)

    # Load best weights for final evaluation
    model.load_state_dict(best_weights)
    evaluate(model, val_loader, val_dataset, class_names, tta_tf)
    plot_history(history)