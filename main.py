from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ApplyFloorRequest(BaseModel):
    room: str
    laminate: str

@app.get("/")
def home():
    return {"status": "backend running"}

@app.post("/apply-floor")
def apply_floor(data: ApplyFloorRequest):
    return {
        "result": data.room
    }
