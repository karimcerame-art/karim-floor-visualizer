from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

app = FastAPI()

# Allow Hugging Face + browser calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/apply-floor")
async def apply_floor(
    room: UploadFile = File(...),
    laminate: UploadFile = File(...)
):
    # Load images
    room_img = Image.open(io.BytesIO(await room.read())).convert("RGB")
    laminate_img = Image.open(io.BytesIO(await laminate.read())).convert("RGB")

    w, h = room_img.size

    # Resize laminate to tile size
    tile = laminate_img.resize((200, 200))

    # Create tiled laminate
    pattern = Image.new("RGB", (w, h))
    for x in range(0, w, tile.width):
        for y in range(int(h * 0.55), h, tile.height):
            pattern.paste(tile, (x, y))

    # Blend laminate onto bottom of room
    result = room_img.copy()
    mask = Image.new("L", (w, h), 0)
    for y in range(int(h * 0.55), h):
        for x in range(w):
            mask.putpixel((x, y), 200)

    result = Image.composite(pattern, result, mask)

    # Return image
    buf = io.BytesIO()
    result.save(buf, format="JPEG")
    buf.seek(0)

    return Response(content=buf.read(), media_type="image/jpeg")
