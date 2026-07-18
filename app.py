import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch

st.set_page_config(page_title="SafeCityAI Detection Demo", layout="centered")

st.title("🚦 SafeCityAI — Traffic Violation Detection")
st.write("Upload a traffic image to detect helmets, bare heads, and license plates.")

MODEL_PATH = "yolov5/runs/train/safecity_v2/weights/best.pt"

CLASS_COLORS = {
    "head": (255, 0, 0),        # red - no helmet violation
    "helmet": (0, 200, 0),      # green - compliant
    "License_Plate": (255, 165, 0),  # orange
}


@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=False)
    model.conf = 0.5
    return model


model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Running detection..."):
        results = model(image)

    detections = results.xyxy[0].tolist()
    class_names = results.names

    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    for x1, y1, x2, y2, conf, class_id in detections:
        label = class_names[int(class_id)]
        color = CLASS_COLORS.get(label, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, max(y1 - 15, 0)), f"{label} {conf:.2f}", fill=color)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detected")
        st.image(draw_image, use_container_width=True)

    st.subheader("Detection Details")
    if detections:
        for x1, y1, x2, y2, conf, class_id in detections:
            label = class_names[int(class_id)]
            st.write(f"**{label}** — confidence: {conf:.2f}, box: [{x1:.0f}, {y1:.0f}, {x2-x1:.0f}, {y2-y1:.0f}]")
    else:
        st.write("No detections found above the confidence threshold.")