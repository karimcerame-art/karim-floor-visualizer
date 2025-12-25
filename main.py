from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Allow Hugging Face to call Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Render backend running"}

@app.post("/upload")
async def upload_image(
    room: UploadFile = File(...),
    laminate: UploadFile = File(...)
):
    room_bytes = await room.read()
    laminate_bytes = await laminate.read()

    room_img = Image.open(io.BytesIO(room_bytes)).convert("RGB")
    laminate_img = Image.open(io.BytesIO(laminate_bytes)).convert("RGB")

    # TEMP: no AI yet, just confirmation
    return {
        "room_size": room_img.size,
        "laminate_size": laminate_img.size,
        "message": "Images received successfully"
    }
