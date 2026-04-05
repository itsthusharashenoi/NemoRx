"""
Offline multilingual ASR + per-segment language ID using faster-whisper.
Suitable for English, Indian languages, and code-mixed speech (heuristic per segment).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf

# Whisper ISO 639-1 (and a few Whisper-specific) → display names
LANG_LABELS: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "ne": "Nepali",
    "si": "Sinhala",
    "or": "Odia",
    "as": "Assamese",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "mai": "Maithili",
    "doi": "Dogri",
    "kok": "Konkani",
    "mni": "Manipuri",
    "sat": "Santali",
    "brx": "Bodo",
    "ks": "Kashmiri",
}

SAMPLE_RATE = 16000
MIN_LID_SAMPLES = int(0.35 * SAMPLE_RATE)


def load_whisper_model() -> Any:
    from faster_whisper import WhisperModel

    model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _which_ffmpeg() -> Any:
    import shutil

    return shutil.which("ffmpeg")


def ensure_wav_16k_mono(src_path: str) -> Tuple[str, bool]:
    """
    Return path to 16 kHz mono WAV. If conversion happened, second value is True
    (caller should delete temp file).
    """
    p = Path(src_path)
    if p.suffix.lower() == ".wav":
        data, sr = sf.read(str(p), always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr == SAMPLE_RATE:
            return str(p), False

    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg (e.g. brew install ffmpeg) to convert webm/mp4/m4a."
        )

    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-y",
            "-i",
            str(src_path),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            out,
        ],
        check=True,
        capture_output=True,
    )
    return out, True


def load_wav_float32(path: str) -> np.ndarray:
    data, sr = sf.read(path, always_2d=False, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        ratio = SAMPLE_RATE / float(sr)
        n = max(1, int(len(data) * ratio))
        x_old = np.linspace(0, len(data) - 1, num=len(data))
        x_new = np.linspace(0, len(data) - 1, num=n)
        data = np.interp(x_new, x_old, data).astype(np.float32)
    return np.clip(data, -1.0, 1.0)


def _segment_language(
    model: Any, audio_mono: np.ndarray, start: float, end: float, fallback_lang: str, fallback_p: float
) -> Tuple[str, float]:
    i0 = max(0, int(start * SAMPLE_RATE))
    i1 = min(len(audio_mono), int(end * SAMPLE_RATE))
    chunk = audio_mono[i0:i1]
    if len(chunk) < MIN_LID_SAMPLES:
        return fallback_lang, float(fallback_p)
    try:
        code, prob, _ = model.detect_language(audio=chunk)
        return code, float(prob)
    except Exception:
        return fallback_lang, float(fallback_p)


def run_transcription(input_path: str, model: Any) -> Dict[str, Any]:
    model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
    wav_path, wav_temp = ensure_wav_16k_mono(input_path)
    try:
        audio_mono = load_wav_float32(wav_path)
        segments_iter, info = model.transcribe(
            wav_path,
            language=None,
            vad_filter=True,
            beam_size=5,
        )

        segments_out: List[Dict[str, Any]] = []
        full_parts: List[str] = []

        primary = info.language or "unknown"
        primary_p = float(getattr(info, "language_probability", 0.0) or 0.0)

        for seg in segments_iter:
            text = (seg.text or "").strip()
            if not text:
                continue
            full_parts.append(text)
            sl, sp = _segment_language(
                model, audio_mono, seg.start, seg.end, primary, primary_p
            )
            segments_out.append(
                {
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "language": sl,
                    "language_label": LANG_LABELS.get(sl, sl),
                    "language_probability": round(sp, 4),
                    "text": text,
                }
            )

        full_text = " ".join(full_parts).strip()
        return {
            "engine": "faster-whisper",
            "model_size": model_size,
            "primary_language": primary,
            "primary_language_label": LANG_LABELS.get(primary, primary),
            "primary_language_probability": round(primary_p, 4),
            "segments": segments_out,
            "full_text": full_text,
        }
    finally:
        if wav_temp and os.path.isfile(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass
