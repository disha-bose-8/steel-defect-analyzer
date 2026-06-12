import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from pathlib import Path
from PIL import Image

# =====================================================
# UPDATE THESE PATHS
# =====================================================

CSV_PATH = r"C:\Users\disha\Downloads\severstal-steel-defect-detection\train.csv"

IMG_DIR = Path(
    r"C:\Users\disha\Downloads\severstal-steel-defect-detection\train_images"
)

# =====================================================

random.seed(42)

df = pd.read_csv(CSV_PATH)


# -----------------------------------------------------
# Decode Severstal RLE mask
# -----------------------------------------------------
def rle_decode(mask_rle, shape=(256, 1600)):
    """
    mask_rle: string
    shape: (height, width)
    """

    if pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()

    starts = np.asarray(s[0::2], dtype=int)
    lengths = np.asarray(s[1::2], dtype=int)

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


# -----------------------------------------------------
# Generate overlay galleries
# -----------------------------------------------------
for cls in [1, 2, 3, 4]:

    print(f"\nProcessing Class {cls}")

    class_df = df[
        (df["ClassId"] == cls)
        & (df["EncodedPixels"].notna())
    ]

    image_ids = class_df["ImageId"].unique().tolist()

    sample_ids = random.sample(
        image_ids,
        min(20, len(image_ids))
    )

    fig, axes = plt.subplots(
        4,
        5,
        figsize=(18, 10)
    )

    for ax, img_id in zip(axes.flatten(), sample_ids):

        row = class_df[class_df["ImageId"] == img_id].iloc[0]

        mask = rle_decode(row["EncodedPixels"])

        img = np.array(
            Image.open(IMG_DIR / img_id).convert("RGB")
        )

        overlay = img.copy()

        overlay[mask == 1] = [255, 0, 0]

        ax.imshow(overlay)

        ax.set_title(
            img_id[:10],
            fontsize=8
        )

        ax.axis("off")

    plt.suptitle(
        f"Severstal Class {cls} (Mask Overlay)",
        fontsize=18
    )

    plt.tight_layout()

    output_file = f"class_{cls}_overlay.png"

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_file}")

print("\nDone.")