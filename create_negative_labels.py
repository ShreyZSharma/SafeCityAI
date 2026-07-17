import os

IMAGES_DIR = os.path.join("data", "negative_examples", "images")
LABELS_DIR = os.path.join("data", "negative_examples", "labels")

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def create_empty_labels():
    os.makedirs(LABELS_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(VALID_EXTENSIONS)]

    for fname in image_files:
        label_fname = os.path.splitext(fname)[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_fname)

        # Create an empty file (no bounding boxes = "nothing to detect here")
        open(label_path, "w").close()

    print(f"[done] Created {len(image_files)} empty label files in '{LABELS_DIR}'")


if __name__ == "__main__":
    create_empty_labels()