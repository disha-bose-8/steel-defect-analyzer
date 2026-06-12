import random
import requests
import pandas as pd
from collections import Counter
from pathlib import Path

CSV_PATH = r"C:\Users\disha\steel-defect-analyzer\dataset\severstal-steel-defect-detection\train.csv"
IMG_DIR = r"C:\Users\disha\steel-defect-analyzer\dataset\severstal-steel-defect-detection\train_images"

API_URL = "http://127.0.0.1:8000/predict"

random.seed(42)

df = pd.read_csv(CSV_PATH)

for cls in [1, 2, 3, 4]:

    print("\n" + "=" * 60)
    print(f"SEVERSTAL CLASS {cls}")
    print("=" * 60)

    class_df = df[
        (df["ClassId"] == cls)
        & (df["EncodedPixels"].notna())
    ]

    image_ids = class_df["ImageId"].unique().tolist()

    sample_ids = random.sample(
        image_ids,
        min(10, len(image_ids))
    )

    predictions = []

    for img_id in sample_ids:



        img_path = Path(IMG_DIR) / img_id

        with open(img_path , "rb") as f:
            response = requests.post(
                "http://127.0.0.1:8000/predict" ,
                files={
                    "file": (
                        img_path.name ,
                        f ,
                        "image/jpeg"
                    )
                }
            )

        if response.status_code != 200:
            print(f"Failed: {img_id}")
            continue

        result = response.json()

        pred = result["predicted_defect"]

        predictions.append(pred)

        print(
            f"{img_id[:12]:15s} -> "
            f"{pred:15s} "
            f"({result['confidence']:.2f}%)"
        )

    print("\nSummary:")

    counts = Counter(predictions)

    for defect, count in counts.most_common():
        print(
            f"{defect:15s} : "
            f"{count}/10"
        )