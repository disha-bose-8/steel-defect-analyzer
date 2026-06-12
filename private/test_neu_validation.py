import random
import requests
from pathlib import Path
from collections import Counter

VAL_DIR = Path(
    r"C:\Users\disha\steel-defect-analyzer\dataset\NEU-DET\validation\images"
)

API_URL = "http://127.0.0.1:8000/predict"

random.seed(42)

correct = 0
total = 0

class_stats = Counter()

classes = [d.name for d in VAL_DIR.iterdir() if d.is_dir()]

print("\nTesting random validation images...\n")

for cls in classes:

    images = list((VAL_DIR / cls).glob("*"))

    sample_count = min(5, len(images))

    samples = random.sample(images, sample_count)

    for img_path in samples:

        with open(img_path, "rb") as f:

            response = requests.post(
                API_URL,
                files={
                    "file": (
                        img_path.name,
                        f,
                        "image/jpeg"
                    )
                }
            )

        if response.status_code != 200:
            print(f"FAILED: {img_path.name}")
            continue

        result = response.json()

        pred = result["predicted_defect"]

        is_correct = pred == cls

        if is_correct:
            correct += 1

        total += 1

        class_stats[(cls, pred)] += 1

        mark = "✓" if is_correct else "✗"

        print(
            f"{mark} "
            f"TRUE={cls:16s} "
            f"PRED={pred:16s} "
            f"{result['confidence']:.2f}%"
        )

print("\n" + "=" * 60)

accuracy = 100 * correct / total

print(f"Correct: {correct}/{total}")
print(f"Accuracy: {accuracy:.2f}%")

print("\nMisclassifications:")

for (true_cls, pred_cls), count in class_stats.items():

    if true_cls != pred_cls:

        print(
            f"{true_cls} -> {pred_cls} : {count}"
        )