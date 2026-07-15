import io
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI(title="SafeCityAI Detection API")

# ---- Load the trained YOLOv5 model once, at startup ----
MODEL_PATH = "yolov5/runs/train/safecity_full/weights/best.pt"

model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=MODEL_PATH,
    force_reload=False,
)
model.conf = 0.5  # default confidence threshold, adjustable


@app.get("/")
def root():
    return {"message": "SafeCityAI Detection API is running. POST an image to /detect"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(image)

    detections = results.xyxy[0].tolist()
    class_names = results.names  # {0: 'head', 1: 'helmet', 2: 'License_Plate'}

    output = []
    for x1, y1, x2, y2, conf, class_id in detections:
        output.append({
            "class": class_names[int(class_id)],
            "confidence": round(conf, 2),
            "box": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]
        })

    return {"detections": output}