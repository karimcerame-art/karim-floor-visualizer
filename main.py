from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import io

app = FastAPI()

# ===============================
# LOAD SEGFORMER (CPU ONLY)
# ===============================
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.eval()

FLOOR_LABEL = 3  # ADE20K floor

# ===============================
# HELPERS
# ===============================
def keep_largest_component(mask):
    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels <= 1:
        return mask
    largest = max(range(1, num_labels), key=lambda i: np.sum(labels == i))
    return np.where(labels == largest, 255, 0).astype(np.uint8)

def feather_mask(mask, ksize=25):
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)

def tile_texture(texture, target_shape):
    th, tw = texture.shape[:2]
    h, w = target_shape
    tiled = np.tile(texture, (h // th + 1, w // tw + 1, 1))
    return tiled[:h, :w]

# ===============================
# REQUEST MODEL
# ===============================
class ApplyRequest(BaseModel):
    room: str      # base64 image
    laminate: str  # base64 image

# ===============================
# API ENDPOINT
# ===============================
@app.post("/apply-floor")
def apply_floor(data: ApplyRequest):

    # Decode images
    room_bytes = base64.b64decode(data.room)
    floor_bytes = base64.b64decode(data.laminate)

    room = Image.open(io.BytesIO(room_bytes)).convert("RGB")
    floor = Image.open(io.BytesIO(floor_bytes)).convert("RGB")

    # Segmentation
    inputs = processor(images=room, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    seg = torch.argmax(outputs.logits, dim=1)[0].cpu().numpy()
    mask = (seg == FLOOR_LABEL).astype(np.uint8) * 255
    mask = cv2.resize(mask, room.size, interpolation=cv2.INTER_NEAREST)

    # Clean mask
    mask = keep_largest_component(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = feather_mask(mask, 25)

    room_np = np.array(room)
    floor_np = np.array(floor)

    tiled_floor = tile_texture(floor_np, room_np.shape[:2])

    alpha = mask.astype(np.float32) / 255.0
    alpha = np.stack([alpha] * 3, axis=-1)

    blended = (room_np * (1 - alpha) + tiled_floor * alpha).astype(np.uint8)

    # Encode result
    result_img = Image.fromarray(blended)
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG", quality=95)
    result_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "image": f"data:image/jpeg;base64,{result_b64}"
    }
