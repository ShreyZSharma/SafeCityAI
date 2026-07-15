import os
import shutil

# ---- CONFIG ----
BASE = "data"
HELMET_DIR = os.path.join(BASE, "helmet_dataset")
PLATE_DIR = os.path.join(BASE, "plate_dataset")
MERGED_DIR = os.path.join(BASE, "merged_dataset")

# Final class order
# 0 = head, 1 = helmet, 2 = License_Plate
PLATE_OLD_CLASS_ID = 0   # plate dataset's only class is currently id 0
PLATE_NEW_CLASS_ID = 2   # we remap it to 2 in the merged dataset

SPLITS = ["train", "valid"]  # helmet_dataset has no test folder, so we skip test


def ensure_dirs():
    for split in SPLITS:
        os.makedirs(os.path.join(MERGED_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(MERGED_DIR, split, "labels"), exist_ok=True)


def copy_helmet_split(split):
    src_img_dir = os.path.join(HELMET_DIR, split, "images")
    src_lbl_dir = os.path.join(HELMET_DIR, split, "labels")
    dst_img_dir = os.path.join(MERGED_DIR, split, "images")
    dst_lbl_dir = os.path.join(MERGED_DIR, split, "labels")

    if not os.path.isdir(src_img_dir):
        print(f"[skip] No helmet {split} folder found at {src_img_dir}")
        return

    for fname in os.listdir(src_img_dir):
        src_path = os.path.join(src_img_dir, fname)
        dst_fname = f"helmet_{fname}"
        shutil.copy2(src_path, os.path.join(dst_img_dir, dst_fname))

    for fname in os.listdir(src_lbl_dir):
        src_path = os.path.join(src_lbl_dir, fname)
        dst_fname = f"helmet_{fname}"
        shutil.copy2(src_path, os.path.join(dst_lbl_dir, dst_fname))

    print(f"[done] Copied helmet {split}: {len(os.listdir(src_img_dir))} images")


def copy_plate_split(split):
    src_img_dir = os.path.join(PLATE_DIR, split, "images")
    src_lbl_dir = os.path.join(PLATE_DIR, split, "labels")
    dst_img_dir = os.path.join(MERGED_DIR, split, "images")
    dst_lbl_dir = os.path.join(MERGED_DIR, split, "labels")

    if not os.path.isdir(src_img_dir):
        print(f"[skip] No plate {split} folder found at {src_img_dir}")
        return

    for fname in os.listdir(src_img_dir):
        src_path = os.path.join(src_img_dir, fname)
        dst_fname = f"plate_{fname}"
        shutil.copy2(src_path, os.path.join(dst_img_dir, dst_fname))

    count = 0
    for fname in os.listdir(src_lbl_dir):
        src_path = os.path.join(src_lbl_dir, fname)
        dst_fname = f"plate_{fname}"
        dst_path = os.path.join(dst_lbl_dir, dst_fname)

        with open(src_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id == PLATE_OLD_CLASS_ID:
                parts[0] = str(PLATE_NEW_CLASS_ID)
            new_lines.append(" ".join(parts))

        with open(dst_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")
        count += 1

    print(f"[done] Copied plate {split}: {count} label files remapped to class {PLATE_NEW_CLASS_ID}")


def write_merged_yaml():
    yaml_content = """train: train/images
val: valid/images

nc: 3
names: ['head', 'helmet', 'License_Plate']
"""
    yaml_path = os.path.join(MERGED_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[done] Wrote merged data.yaml at {yaml_path}")


if __name__ == "__main__":
    ensure_dirs()
    for split in SPLITS:
        copy_helmet_split(split)
        copy_plate_split(split)
    write_merged_yaml()
    print("\nMerge complete. Check data/merged_dataset/ for the unified dataset.")