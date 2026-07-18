# SafeCityAI: Traffic Violation Detection System
## Project Report

---

## 1. Problem Statement

Traffic police in smart cities spend thousands of hours manually reviewing CCTV footage to
identify traffic violations, primarily missing helmet usage and illegible license plates.
This project builds an automated object detection system that identifies:
1. Riders without helmets (`head` class)
2. Riders with helmets (`helmet` class)
3. Vehicle license plates (`License_Plate` class)

The goal was to build a working end-to-end pipeline — from raw data to a deployable API —
capable of assisting automated traffic violation review.

---

## 2. Approach

### 2.1 Data Sourcing
Rather than manually annotating a dataset from scratch (a multi-week task), publicly available
pre-annotated datasets were sourced from Roboflow Universe:
- A head/helmet detection dataset (2,041 images)
- A license plate detection dataset (483 images)

### 2.2 Data Quality Verification (Key Decision Point)
An initial helmet dataset, despite having a large image count (1.9k images), was found to have
poor annotation quality upon inspection — bounding boxes covered entire upper bodies rather than
just heads (average box height was ~51% of image height, versus an expected ~10-15% for
head-level detection). This was caught using a custom script that statistically analyzed
bounding box dimensions across the dataset before any training time was invested.

This dataset was discarded and replaced with a properly head-level-annotated dataset
(average box height dropped to ~9-14%), verified both statistically and visually before
proceeding to training.

**This decision — pausing to verify data quality rather than immediately training — was the
single most consequential choice in the project.** Training on the original dataset would have
produced a model with poor localization precision despite potentially misleading loss curves.

### 2.3 Dataset Merging
The two source datasets (head/helmet + license plate) used different class indexing schemes.
A custom Python script (`merge_datasets.py`) was written to:
- Merge both datasets into a unified structure
- Remap class indices to a consistent 3-class scheme (`head`, `helmet`, `License_Plate`)
- Generate a unified `data.yaml` configuration for YOLOv5

### 2.4 Model Training
YOLOv5s (small variant) was fine-tuned from COCO-pretrained weights for 100 epochs, using:
- Image size: 640px
- Batch size: 16
- Local training on an NVIDIA RTX 5060 Laptop GPU (CUDA 12.8)

### 2.5 Hard-Negative Mining
After initial training, video inference testing revealed the model occasionally misclassified
caps, turbans, and unrelated round/colorful objects (e.g., decorations near head height) as
helmets. To address this, ~25 real-world images of caps, turbans, and similar objects were
manually gathered and added to training as hard-negative examples (images with no bounding
boxes, teaching the model these regions contain nothing worth detecting). The model was
retrained with this expanded dataset.

### 2.6 Deployment
A FastAPI server (`server.py`) was built to wrap the trained model as a REST API, accepting
image uploads and returning structured JSON detections (class, confidence, bounding box).

---

## 3. Results

| Metric | Initial Model | After Hard-Negative Mining |
|---|---|---|
| mAP@0.5 | 97.3% | 97.4% |
| mAP@0.5:0.95 | 67.5% | 67.3% |
| Precision | 93.2% | 95.2% |
| Recall | 94.0% | 93.7% |

Adding hard-negative examples improved precision (fewer false positives) with no meaningful
regression in overall detection quality (mAP@0.5 held steady).

Video inference testing confirmed:
- Strong performance on close-to-medium range, clearly visible subjects
- Accurate multi-person detection in crowded scenes
- Reduced (though not fully eliminated) false positives on cap/decoy objects after the
  hard-negative mining step

---

## 4. Challenges Faced

1. **Environment setup friction** — initial Git, PowerShell execution policy, and path-quoting
   issues (spaces in the Windows username path) consumed early setup time; resolved through
   systematic troubleshooting.
2. **Silent data quality issues** — the first helmet dataset "looked fine" at a glance but had
   fundamentally wrong annotation granularity, caught only through deliberate statistical
   verification rather than visual spot-checks alone.
3. **YOLOv5 relative path resolution** — the merged dataset's `data.yaml` initially failed to
   resolve correctly since YOLOv5 interprets relative paths relative to the working directory,
   not the yaml's location; fixed by adding an explicit `path:` key.
4. **Mid-training crash** — a training run crashed at epoch 93/99 due to a memory allocation
   error, traced to the laptop entering sleep mode during a long training run. Resolved using
   YOLOv5's `--resume` functionality to continue from the last checkpoint, and mitigated going
   forward by disabling sleep during training.

---

## 5. Known Limitations

- **Distant/small-scale objects:** detection accuracy drops for subjects far from the camera.
- **Motion blur:** fast-moving vehicles can produce blurred, undetectable license plates.
- **Residual false positives:** cap/decoy object misclassification is reduced but not fully
  eliminated; would benefit from a larger, more diverse hard-negative example set.

---

## 6. Future Work

- Expand hard-negative examples further for improved precision
- Add training examples with distant/small-scale and motion-blurred subjects
- Integrate object tracking (e.g., DeepSORT) for consistent violator identification across
  video frames
- Containerize the API (Docker) for production deployment
- Build a simple frontend for interactive, non-technical demonstration

---

## 7. Conclusion

This project delivered a working end-to-end traffic violation detection system — from data
sourcing through deployment — achieving strong detection performance (97.4% mAP@0.5) on a
custom 3-class fine-tuned YOLOv5 model. The most valuable lesson was the importance of
verifying data quality *before* investing training time, a decision that meaningfully shaped
the project's final quality.