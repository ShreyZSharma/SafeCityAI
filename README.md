# SafeCityAI — Traffic Violation Detection System

An AI-powered object detection system that identifies helmet violations and license plates
in traffic footage, built using YOLOv5 and deployed as a FastAPI inference service.

## Project Overview
This system detects three classes in traffic images/video:
- **head** — a bare head (no helmet)
- **helmet** — a helmeted head
- **License_Plate** — vehicle license plates

## Results
- **mAP@0.5:** 97.4%
- **Precision:** 95.2%
- **Recall:** 93.7%
- Trained for 100 epochs on a merged, quality-verified dataset (~2,000+ head/helmet images,
  ~450 license plate images, ~25 hard-negative examples) using YOLOv5s fine-tuned from
  COCO-pretrained weights.

## Project Structure

Pro1/
├── data/
│   ├── plate_dataset/          # source license plate dataset
│   ├── helmet_dataset/         # source head/helmet dataset
│   ├── negative_examples/      # hard-negative examples (caps, turbans, decoys)
│   └── merged_dataset/         # unified 3-class dataset used for training
├── yolov5/                     # YOLOv5 repo (training + inference code)
│   └── runs/train/safecity_v2/weights/best.pt   # trained model weights (latest)
├── merge_datasets.py           # merges source datasets into unified format
├── analyze_box_sizes.py        # dataset annotation quality checker
├── check_annotations.py        # visual bounding box verification tool
├── create_negative_labels.py   # auto-generates empty labels for negative examples
├── server.py                   # FastAPI inference server
├── training_notebook.ipynb     # documented training process and results
└── README.md

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:
```bash
pip install -r yolov5/requirements.txt
pip install fastapi uvicorn python-multipart
```

3. Verify GPU/CUDA is available (optional but recommended for training):
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Running the API

Start the server:
```bash
uvicorn server:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

- `GET /` — health check
- `POST /detect` — upload an image, returns detected violations as JSON
- Interactive docs: `http://127.0.0.1:8000/docs`

### Example response
```json
{
  "detections": [
    {
      "class": "helmet",
      "confidence": 0.93,
      "box": [78.4, 156.6, 28, 45.2]
    }
  ]
}
```
Box format: `[x, y, width, height]` in pixel coordinates.

## Running Video Inference
```bash
cd yolov5
python detect.py --weights runs/train/safecity_v2/weights/best.pt --img 640 --conf 0.5 --source path/to/video.mp4
```

## Known Limitations
- **Distant/small-scale objects:** detection accuracy drops for subjects far from the camera,
  since training data primarily featured medium-to-close range subjects.
- **Motion blur:** fast-moving vehicles can produce blurred license plates that the model
  struggles to detect reliably, as training data consisted of static images.
- **Helmet vs. cap / decoy objects:** an earlier version of the model occasionally
  misclassified caps, turbans, and round/colorful decoy objects (e.g., decorations near
  head height) as helmets. This was mitigated by adding ~25 hard-negative training examples
  (real-world images of caps, turbans, and similar objects with empty labels), which improved
  precision from 93.2% to 95.2% while maintaining mAP@0.5 at ~97.4%. Some edge cases may
  still occur; further improvement would benefit from a larger and more diverse set of
  hard-negative examples.

## Future Improvements
- Expand hard-negative examples further (more caps/hats, decorative objects, varied contexts)
  to continue reducing false positives.
- Expand dataset with more distant/small-scale and motion-blurred examples.
- Add tracking (e.g., DeepSORT) for consistent multi-frame identification of violators
  in video streams.
- Containerize the API (Docker) for easier deployment.
- Build a simple frontend for interactive demonstration of detections.