from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import io

app = FastAPI()

# ===============================
# LOAD SEGFORMER
# ===============================
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.eval()

FLOOR_LABEL = 3

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
# CORE PROCESS
# ===============================
def apply_floor(room_img, floor_img):
    room = Image.open(io.BytesIO(room_img)).convert("RGB")
    floor = Image.open(io.BytesIO(floor_img)).convert("RGB")

    inputs = processor(images=room, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    seg = torch.argmax(outputs.logits, dim=1)[0].cpu().numpy()

    mask = (seg == FLOOR_LABEL).astype(np.uint8) * 255
    mask = cv2.resize(mask, room.size, interpolation=cv2.INTER_NEAREST)

    mask = keep_largest_component(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = feather_mask(mask, 25)

    room_np = np.array(room)
    floor_np = np.array(floor)
    tiled_floor = tile_texture(floor_np, room_np.shape[:2])

    alpha = mask.astype(np.float32) / 255.0
    alpha = np.stack([alpha] * 3, axis=-1)

    result = (room_np * (1 - alpha) + tiled_floor * alpha).astype(np.uint8)
    return result

# ===============================
# API ENDPOINT (THIS WAS MISSING ❌)
# ===============================
@app.post("/apply-floor")
async def apply_floor_api(
    room: UploadFile = File(...),
    laminate: UploadFile = File(...)
):
    room_bytes = await room.read()
    laminate_bytes = await laminate.read()

    result = apply_floor(room_bytes, laminate_bytes)

    _, buffer = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return Response(content=buffer.tobytes(), media_type="image/png")
