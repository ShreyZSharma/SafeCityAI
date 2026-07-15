import os
import random
import cv2
DATASET_DIR= os.path.join("data", "merged_dataset")
SPLIT= "train"  
NUM_SAMPLES=12  
CLASS_NAMES = ["head", "helmet", "License_Plate"]
CLASS_COLORS = [(0, 255, 0),(0, 0, 255),(255, 165, 0),]
OUTPUT_DIR = "annotation_check"
def draw_boxes(image_path, label_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[warn] Could not read image: {image_path}")
        return

    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"[warn] No label file for: {image_path}")
        cv2.imwrite(save_path, img)
        return

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        x_center, y_center, box_w, box_h = map(float, parts[1:5])

        x_center *= w
        y_center *= h
        box_w *= w
        box_h *= h

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        color = CLASS_COLORS[class_id] if class_id < len(CLASS_COLORS) else (255, 255, 255)
        label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imwrite(save_path, img)


def main():
    images_dir = os.path.join(DATASET_DIR, SPLIT, "images")
    labels_dir = os.path.join(DATASET_DIR, SPLIT, "labels")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_images = os.listdir(images_dir)
    sample = random.sample(all_images, min(NUM_SAMPLES, len(all_images)))

    for fname in sample:
        image_path = os.path.join(images_dir, fname)
        label_fname = os.path.splitext(fname)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_fname)
        save_path = os.path.join(OUTPUT_DIR, fname)

        draw_boxes(image_path, label_path, save_path)

    print(f"\nDone. Saved {len(sample)} annotated sample images to '{OUTPUT_DIR}/'")
    print("Open that folder and visually check the bounding boxes look correct.")


if __name__ == "__main__":
    main()