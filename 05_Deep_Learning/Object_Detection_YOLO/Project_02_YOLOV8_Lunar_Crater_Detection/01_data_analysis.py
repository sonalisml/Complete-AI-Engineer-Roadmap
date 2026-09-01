# ============================================
# Project 02: Lunar Crater Detection
# Step 1: Dataset Analysis
# ============================================

import os
from collections import Counter

# ============================================
# 1. Dataset Path
# ============================================

dataset_path = "dataset"

class_names = {
    0: "crater"
}

# ============================================
# 2. Count Images
# ============================================

def count_images(split):

    image_path = os.path.join(
        dataset_path,
        split,
        "images"
    )

    if not os.path.exists(image_path):
        print(f"{split}: Image folder not found")
        return 0

    images = [
        file for file in os.listdir(image_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    return len(images)


# ============================================
# 3. Count Object Instances
# ============================================

def count_instances(split):

    label_path = os.path.join(
        dataset_path,
        split,
        "labels"
    )

    class_count = Counter()
    total_instances = 0

    if not os.path.exists(label_path):
        print(f"{split}: Label folder not found")
        return class_count, total_instances

    for file in os.listdir(label_path):

        if file.endswith(".txt"):

            file_path = os.path.join(
                label_path,
                file
            )

            with open(file_path, "r") as f:

                for line in f:

                    values = line.strip().split()

                    if len(values) >= 5:

                        class_id = int(values[0])

                        class_count[class_id] += 1
                        total_instances += 1

    return class_count, total_instances


# ============================================
# 4. Analyze Dataset
# ============================================

splits = ["train", "valid", "test"]

print("\n" + "=" * 50)
print("LUNAR CRATER DATASET ANALYSIS")
print("=" * 50)

for split in splits:

    image_count = count_images(split)

    class_count, total_instances = count_instances(split)

    print("\n" + "=" * 50)
    print(split.upper())
    print("=" * 50)

    print("Images     :", image_count)
    print("Instances  :", total_instances)

    for class_id, count in class_count.items():

        class_name = class_names.get(
            class_id,
            f"Unknown ({class_id})"
        )

        print(
            f"{class_name:<12}: {count} instances"
        )