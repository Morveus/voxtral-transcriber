import os
import time
import tempfile
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.model import TranscriptionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voxtral Transcriber", version="1.0.0")

model = TranscriptionModel()


@app.on_event("startup")
async def startup():
    logger.info("Loading Voxtral model...")
    model.load()
    logger.info("Model loaded and ready.")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model.is_loaded}


ALLOWED_EXTENSIONS = {".mp3", ".wav", ".ogg", ".oga", ".flac", ".m4a", ".webm"}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    ext = os.path.splitext(audio.filename or "audio.ogg")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp.flush()

        start = time.time()
        text, duration = model.transcribe(tmp.name)
        elapsed = time.time() - start

    return JSONResponse(
        content={
            "text": text,
            "duration_seconds": round(duration, 2),
            "processing_time_seconds": round(elapsed, 2),
        }
    )
