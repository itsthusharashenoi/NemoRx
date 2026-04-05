"""
FastAPI server: upload audio → multilingual transcription + per-segment language guesses.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from transcribe import load_whisper_model, run_transcription

app = FastAPI(title="Multilingual STT pipeline", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_whisper_model()
    return _model


@app.on_event("startup")
def startup_load_model():
    if os.getenv("WHISPER_LAZY_LOAD", "").lower() in ("1", "true", "yes"):
        return
    get_model()


@app.get("/health")
def health():
    ready = _model is not None
    return {"status": "ok", "service": "whisper-pipeline", "model_loaded": ready}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Missing filename")

    suffix = Path(file.filename).suffix.lower() or ".bin"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        data = await file.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB} MB)")
        Path(tmp).write_bytes(data)
        model = get_model()
        result = run_transcription(tmp, model)
        return result
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}") from e
    finally:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
