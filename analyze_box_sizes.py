import os

DATASET_DIR = os.path.join("data", "merged_dataset")
SPLIT = "train"

CLASS_NAMES = ["head", "helmet", "License_Plate"]
SUSPICIOUS_HEIGHT_THRESHOLD = 0.35


def analyze():
    labels_dir = os.path.join(DATASET_DIR, SPLIT, "labels")
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

    stats = {i: {"count": 0, "heights": [], "suspicious": 0} for i in range(len(CLASS_NAMES))}

    for fname in label_files:
        path = os.path.join(labels_dir, fname)
        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            box_h = float(parts[4])

            if class_id not in stats:
                continue

            stats[class_id]["count"] += 1
            stats[class_id]["heights"].append(box_h)

            if class_id in (0, 1) and box_h > SUSPICIOUS_HEIGHT_THRESHOLD:
                stats[class_id]["suspicious"] += 1

    print(f"\n--- Box size analysis ({SPLIT} split) ---\n")
    for class_id, name in enumerate(CLASS_NAMES):
        s = stats[class_id]
        count = s["count"]
        if count == 0:
            print(f"{name}: no instances found")
            continue

        avg_height = sum(s["heights"]) / count
        max_height = max(s["heights"])
        suspicious_pct = (s["suspicious"] / count * 100) if class_id in (0, 1) else None

        print(f"{name}:")
        print(f"  total boxes: {count}")
        print(f"  avg height (normalized): {avg_height:.3f}")
        print(f"  max height (normalized): {max_height:.3f}")
        if suspicious_pct is not None:
            print(f"  suspicious (height > {SUSPICIOUS_HEIGHT_THRESHOLD}): "
                  f"{s['suspicious']} boxes ({suspicious_pct:.1f}%)")
        print()


if __name__ == "__main__":
    analyze()