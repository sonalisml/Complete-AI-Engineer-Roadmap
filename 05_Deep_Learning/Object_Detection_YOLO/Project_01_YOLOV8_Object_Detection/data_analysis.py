#==============================
#Import libararies
#==============================
import os
from collections import Counter
#==============================
#dataset path and clas snames
#==============================
dataset_path = "dataset_yolo"
class_names = {
    0: "bolt",
    1: "bottle",
    2: "washer"
}
#========
#Create the function to count objects
#============
def count_instances(split):
    label_path = os.path.join(
        dataset_path,
        split,
        "labels"
    )
    class_count = Counter()
    total_annotataions = 0
    for file in os.listdir(label_path):
        if file.endswith(".txt"):
            file_path = os.path.join(
                label_path,
                file
            )
            with open(file_path, "r") as f:
                for line in f:
                    values = line.strip().split()
                    if len(values)>5:
                        class_id = int(values[0])
                        class_count[class_id] +=1
                        total_annotataions +=1
            return class_count, total_annotataions
# ==========================================================
# Analyze Train / Validation / Test
# ==========================================================
for split in ["train", "valid", "test"]:

    counts, total = count_instances(split)

    print("\n" + "=" * 50)
    print(split.upper())
    print("=" * 50)

    for class_id, class_name in class_names.items():

        print(
            f"{class_name:10} : "
            f"{counts[class_id]} instances"
        )

    print(
        f"Total      : {total} instances"
    )