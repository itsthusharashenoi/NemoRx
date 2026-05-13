"""
vexyl_stt_server.py
VEXYL-STT Server
------------------------------------------------------
Wraps ai4bharat/indic-conformer-600m-multilingual in a WebSocket server.
Accepts 16kHz 16-bit mono PCM audio chunks, returns transcripts as JSON.
Also exposes a Sarvam-style batch transcription API (POST /batch/transcribe).
POST /conversation/transcribe accepts a full recording (WebM/WAV/…) and returns a Doc/Patient/Voice-segmented document.

Usage:
    pip install transformers torchaudio websockets numpy torch soundfile
    python vexyl_stt_server.py

Optional env vars:
    PORT                      (default: 8080, Cloud Run injects this)
    VEXYL_STT_HOST            (default: 0.0.0.0)
    VEXYL_STT_PORT            (fallback if PORT unset)
    VEXYL_STT_DECODE          (default: ctc)   options: ctc, rnnt
    VEXYL_STT_DEVICE          (default: auto)  options: auto, cpu, cuda
    VEXYL_STT_API_KEY         (default: empty) shared secret; if set, clients must send X-API-Key header
    GEMINI_API_KEY / GOOGLE_API_KEY   optional; enables POST /online/gemini/transcribe (Gemini 2.5 Flash STT)
    GEMINI_MODEL              (default: gemini-2.5-flash)
"""

import asyncio
import websockets
from websockets.asyncio.server import ServerConnection
import json
import numpy as np
import torch
import torchaudio
import os
import sys
import logging
import time
import signal
import threading
import io
import hmac
import uuid
import re
import base64
import urllib.request
import urllib.error
import soundfile as sf
from dataclasses import dataclass
from enum import Enum
from http import HTTPStatus
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VexylSTT] %(levelname)s %(message)s"
)
log = logging.getLogger("vexyl_stt")

# ─── Config ────────────────────────────────────────────────────────────────────
HOST        = os.getenv("VEXYL_STT_HOST",   "0.0.0.0")
PORT        = int(os.getenv("PORT", os.getenv("VEXYL_STT_PORT", "8080")))
DECODE_MODE = os.getenv("VEXYL_STT_DECODE", "ctc")   # ctc = faster, rnnt = more accurate
DEVICE_PREF = os.getenv("VEXYL_STT_DEVICE", "auto")
API_KEY     = os.getenv("VEXYL_STT_API_KEY", "")

# Google Gemini (online STT via generateContent — key stays on server)
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")).strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_MAX_UPLOAD_BYTES = int(os.getenv("GEMINI_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))

# Audio input: 16kHz 16-bit mono PCM
TARGET_SAMPLE_RATE = 16000

# VAD parameters - detect silence to trigger transcription
SILENCE_THRESHOLD    = 0.015   # RMS energy threshold
MIN_SPEECH_DURATION  = 0.3     # seconds of speech before attempting transcription
SILENCE_DURATION     = 0.6     # seconds of silence to consider utterance complete
MAX_BUFFER_DURATION  = 12.0    # force transcription after this many seconds

# Batch transcription config
BATCH_MAX_FILE_SIZE     = 25 * 1024 * 1024  # 25MB
BATCH_MAX_AUDIO_DURATION = 300.0             # 5 minutes
BATCH_MAX_JOBS          = 1000
BATCH_JOB_TTL           = 3600               # 1 hour

# Supported upload extensions (browser recordings often use WebM)
SUPPORTED_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm")

# Conversation document: merge same-utterance gaps shorter than this (hesitations)
CONV_MERGE_MAX_GAP_SEC = 0.22
# VAD frame hop for offline segmentation (seconds)
CONV_VAD_HOP_SEC = 0.02
# Smoothing window for offline VAD (odd number of hops)
CONV_VAD_SMOOTH_WIN = 7

# Speaker labels assigned in order of first appearance (cluster → role)
SPEAKER_LABEL_ORDER = ("Doc", "Patient", "Voice 1", "Voice 2")

# Language code map — VEXYL language codes → model codes
LANG_MAP = {
    "ml-IN": "ml",  # Malayalam
    "hi-IN": "hi",  # Hindi
    "ta-IN": "ta",  # Tamil
    "te-IN": "te",  # Telugu
    "kn-IN": "kn",  # Kannada
    "bn-IN": "bn",  # Bengali
    "gu-IN": "gu",  # Gujarati
    "mr-IN": "mr",  # Marathi
    "pa-IN": "pa",  # Punjabi
    "or-IN": "or",  # Odia
    "as-IN": "as",  # Assamese
    "ur-IN": "ur",  # Urdu
    "sa-IN": "sa",  # Sanskrit
    "ne-IN": "ne",  # Nepali
    # Pass-through if already short code
    "ml": "ml", "hi": "hi", "ta": "ta", "te": "te",
    "kn": "kn", "bn": "bn", "gu": "gu", "mr": "mr",
    "pa": "pa", "or": "or", "as": "as", "ur": "ur",
    "sa": "sa", "ne": "ne",
}

# ─── Connection Limits ────────────────────────────────────────────────────────
MAX_CONNECTIONS = int(os.getenv("VEXYL_STT_MAX_CONN", "50"))
_conn_semaphore: asyncio.Semaphore  # initialized in main() (needs running loop)
active_sessions: dict[str, "STTSession"] = {}
_server_start_time: float = 0.0

# ─── Batch Job Types ─────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BatchJob:
    job_id: str
    status: JobStatus
    language: str
    created_at: float
    audio_pcm: Optional[np.ndarray] = None
    audio_duration: float = 0.0
    transcript: Optional[str] = None
    latency_ms: Optional[int] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None

_batch_jobs: dict[str, BatchJob] = {}
_batch_queue: asyncio.Queue = None    # initialized in main()
_batch_worker_task: asyncio.Task = None
_batch_cleanup_task: asyncio.Task = None

# ─── Model Loader ──────────────────────────────────────────────────────────────
model = None
device = None
_infer_lock = threading.Lock()

def load_model():
    global model, device

    if DEVICE_PREF == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = DEVICE_PREF

    log.info(f"Loading ai4bharat/indic-conformer-600m-multilingual on {device}...")
    start = time.time()

    # HF_HUB_OFFLINE=1 (or VEXYL_STT_LOCAL_MODEL_ONLY=1) = no network; needs full cache under ~/.cache/huggingface
    local_only = os.getenv("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes") or os.getenv(
        "VEXYL_STT_LOCAL_MODEL_ONLY", ""
    ).lower() in ("1", "true", "yes")
    if local_only:
        log.info("Offline / local-only model load (no Hugging Face Hub requests)")

    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True,
        local_files_only=local_only,
    )
    if device == "cuda":
        model = model.cuda()
    model.eval()

    elapsed = time.time() - start
    log.info(f"Model loaded in {elapsed:.1f}s on {device} | decode_mode={DECODE_MODE}")


# ─── VAD Helper ────────────────────────────────────────────────────────────────
def compute_rms(pcm_float32: np.ndarray) -> float:
    """Compute root-mean-square energy of audio chunk."""
    if len(pcm_float32) == 0:
        return 0.0
    return float(np.sqrt(np.mean(pcm_float32 ** 2)))


# ─── Audio Conversion ─────────────────────────────────────────────────────────

def _convert_audio_to_pcm_sync(audio_bytes: bytes) -> tuple[np.ndarray, float]:
    """Decode audio bytes (WAV/MP3/FLAC/OGG) to 16kHz mono float32 PCM.
    Uses soundfile (libsndfile) which supports WAV, FLAC, OGG, AIFF natively.
    For MP3/M4A, ffmpeg must be available on PATH as a subprocess fallback."""
    buf = io.BytesIO(audio_bytes)
    try:
        data, sample_rate = sf.read(buf, dtype="float32")
    except Exception:
        # soundfile can't handle MP3/M4A — fall back to ffmpeg subprocess
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            result = subprocess.run(
                ["ffmpeg", "-i", tmp.name, "-f", "wav", "-acodec", "pcm_s16le",
                 "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1", "-"],
                capture_output=True, timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')[:200]}")
            wav_buf = io.BytesIO(result.stdout)
            data, sample_rate = sf.read(wav_buf, dtype="float32")

    # Mono mixdown if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to 16kHz if needed
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torch.from_numpy(data).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(int(sample_rate), TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
        data = waveform.squeeze(0).numpy()

    pcm = data.astype(np.float32)
    duration = len(pcm) / TARGET_SAMPLE_RATE
    return pcm, duration


async def _convert_audio_to_pcm(audio_bytes: bytes) -> tuple[np.ndarray, float]:
    """Async wrapper — runs audio conversion in thread pool."""
    return await asyncio.to_thread(_convert_audio_to_pcm_sync, audio_bytes)


# ─── Multipart Parser ─────────────────────────────────────────────────────────

def _parse_multipart(content_type: str, body: bytes) -> dict:
    """Parse multipart/form-data. Returns dict of field_name → value or {filename, data}."""
    match = re.search(r'boundary=([^\s;]+)', content_type)
    if not match:
        return {}
    boundary = match.group(1).strip('"').encode()

    parts = body.split(b"--" + boundary)
    fields = {}

    for part in parts:
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue

        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        headers_raw = part[:header_end].decode("utf-8", errors="replace")
        part_body = part[header_end + 4:]

        # Strip trailing \r\n left by boundary split
        if part_body.endswith(b"\r\n"):
            part_body = part_body[:-2]

        name_match = re.search(r'name="([^"]*)"', headers_raw)
        if not name_match:
            continue
        name = name_match.group(1)

        filename_match = re.search(r'filename="([^"]*)"', headers_raw)
        if filename_match:
            fields[name] = {"filename": filename_match.group(1), "data": part_body}
        else:
            fields[name] = part_body.decode("utf-8", errors="replace").strip()

    return fields


# ─── Batch Worker ──────────────────────────────────────────────────────────────

async def _batch_worker():
    """Background coroutine — pulls jobs from queue and runs inference.
    Wraps the loop in an outer try/except so unexpected errors don't kill the worker."""
    log.info("Batch worker started")
    while True:
        try:
            job_id = await _batch_queue.get()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.error("[batch] Error getting from queue", exc_info=True)
            await asyncio.sleep(1)
            continue

        try:
            job = _batch_jobs.get(job_id)
            if not job or job.status != JobStatus.QUEUED:
                continue

            job.status = JobStatus.PROCESSING
            log.info(f"[batch] Processing job {job_id} ({job.language}, {job.audio_duration:.1f}s)")

            start = time.time()
            text = await transcribe(job.audio_pcm, job.language)
            latency = int((time.time() - start) * 1000)

            job.transcript = text
            job.latency_ms = latency
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.audio_pcm = None  # free memory

            log.info(f"[batch] Job {job_id} completed: '{text}' ({latency}ms)")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(f"[batch] Job {job_id} failed: {e}", exc_info=True)
            if job_id in _batch_jobs:
                _batch_jobs[job_id].status = JobStatus.FAILED
                _batch_jobs[job_id].error_message = "Transcription failed"
                _batch_jobs[job_id].audio_pcm = None
                _batch_jobs[job_id].completed_at = time.time()
        finally:
            try:
                _batch_queue.task_done()
            except ValueError:
                pass  # task_done called too many times


async def _batch_cleanup_loop():
    """Remove completed/failed jobs older than BATCH_JOB_TTL every 5 minutes."""
    while True:
        await asyncio.sleep(300)
        now = time.time()
        expired = [
            jid for jid, job in _batch_jobs.items()
            if job.completed_at and (now - job.completed_at) > BATCH_JOB_TTL
        ]
        for jid in expired:
            del _batch_jobs[jid]
        if expired:
            log.info(f"[batch] Cleaned up {len(expired)} expired jobs")


# ─── Transcription ─────────────────────────────────────────────────────────────
def _run_inference(pcm_float32: np.ndarray, lang_code: str) -> str:
    """Synchronous inference — runs in thread pool so it doesn't block the event loop."""
    indic_lang = LANG_MAP.get(lang_code, "ml")
    wav = torch.from_numpy(pcm_float32).unsqueeze(0)
    if device == "cuda":
        wav = wav.cuda()
    with _infer_lock:
        with torch.no_grad():
            result = model(wav, indic_lang, DECODE_MODE)
    return result.strip() if isinstance(result, str) else str(result).strip()


async def transcribe(pcm_float32: np.ndarray, lang_code: str) -> str:
    """Run model inference off the event loop via asyncio.to_thread()."""
    if len(pcm_float32) == 0:
        return ""
    return await asyncio.to_thread(_run_inference, pcm_float32, lang_code)


# ─── Conversation document (full recording → segmented + speaker labels) ─────

def _smooth_bool_flags(flags: list[bool], win: int) -> list[bool]:
    if not flags or win < 1:
        return flags
    half = win // 2
    n = len(flags)
    out: list[bool] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = flags[lo:hi]
        out.append(sum(1 for x in chunk if x) > len(chunk) / 2)
    return out


def _segment_utterance_ranges(pcm: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> list[tuple[int, int]]:
    """Energy-based segmentation on full PCM; returns (start, end_excl) sample ranges."""
    n = len(pcm)
    if n < int(0.05 * sr):
        if compute_rms(pcm) > SILENCE_THRESHOLD * 0.4:
            return [(0, n)]
        return []

    hop = max(int(CONV_VAD_HOP_SEC * sr), 160)
    flags: list[bool] = []
    positions: list[int] = []
    i = 0
    while i < n:
        end = min(i + hop * 2, n)
        chunk = pcm[i:end]
        if len(chunk) >= hop // 2:
            flags.append(compute_rms(chunk) > SILENCE_THRESHOLD)
            positions.append(i)
        i += hop

    if not flags:
        return [(0, n)] if compute_rms(pcm) > SILENCE_THRESHOLD * 0.4 else []

    flags = _smooth_bool_flags(flags, CONV_VAD_SMOOTH_WIN)

    min_speech = max(1, int(MIN_SPEECH_DURATION / (hop / sr)))
    raw_ranges: list[tuple[int, int]] = []
    j = 0
    while j < len(flags):
        if not flags[j]:
            j += 1
            continue
        k = j
        while k < len(flags) and flags[k]:
            k += 1
        if k - j >= min_speech:
            start_s = max(0, positions[j] - int(0.08 * sr))
            end_s = min(n, positions[k - 1] + hop * 2 + int(0.08 * sr))
            raw_ranges.append((start_s, end_s))
        j = k

    if not raw_ranges:
        return [(0, n)] if compute_rms(pcm) > SILENCE_THRESHOLD * 0.35 else []

    merged: list[tuple[int, int]] = []
    for start_s, end_s in raw_ranges:
        if not merged:
            merged.append((start_s, end_s))
            continue
        ps, pe = merged[-1]
        gap_sec = (start_s - pe) / sr
        if gap_sec >= 0 and gap_sec < CONV_MERGE_MAX_GAP_SEC:
            merged[-1] = (ps, max(pe, end_s))
        else:
            merged.append((start_s, end_s))

    return merged


def _segment_audio_features(seg: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Lightweight embedding for same-speaker clustering (no extra deps)."""
    if len(seg) == 0:
        return np.zeros(16, dtype=np.float64)
    x = seg.astype(np.float64)
    e = float(np.sqrt(np.mean(x * x)) + 1e-8)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x))))) if len(x) > 1 else 0.0
    n_fft = min(4096, max(256, len(x)))
    spec = np.abs(np.fft.rfft(x[:n_fft]))
    bands = 8
    m = max(len(spec) // bands, 1)
    band_e = [float(np.mean(spec[i * m : (i + 1) * m])) for i in range(bands)]
    feat = np.array([e, zcr, float(np.std(x))] + band_e, dtype=np.float64)
    nrm = np.linalg.norm(feat) + 1e-8
    return feat / nrm


def _kmeans_labels(features: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Return cluster id 0..k-1 per row. Pure numpy; k clamped to n."""
    rng = np.random.default_rng(seed)
    n, d = features.shape
    k = int(max(1, min(k, n)))
    if k == 1:
        return np.zeros(n, dtype=np.int32)
    idx = rng.choice(n, k, replace=False)
    centroids = features[idx].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(35):
        dists = np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1).astype(np.int32)
        new_c = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_c[j] = features[mask].mean(axis=0)
            else:
                new_c[j] = centroids[j] + rng.normal(0, 1e-3, d)
        if np.allclose(new_c, centroids, rtol=1e-5, atol=1e-5):
            break
        centroids = new_c
    return labels


def _pick_speaker_cluster_count(num_segments: int) -> int:
    """Heuristic cluster count for Doc / Patient / extra voices (max 4)."""
    n = num_segments
    if n <= 1:
        return 1
    if n <= 4:
        return min(2, n)
    if n <= 9:
        return min(3, n)
    return min(4, n)


def _map_clusters_to_speaker_names(labels: np.ndarray) -> list[str]:
    """Order clusters by time of first segment; assign Doc, Patient, Voice 1, Voice 2."""
    first_pos: dict[int, int] = {}
    for i, lab in enumerate(labels.tolist()):
        if lab not in first_pos:
            first_pos[lab] = i
    unique_ids = sorted(int(u) for u in np.unique(labels))
    cluster_ids = sorted(unique_ids, key=lambda c: first_pos.get(c, 10**9))
    rename: dict[int, str] = {}
    for rank, cid in enumerate(cluster_ids):
        if rank < len(SPEAKER_LABEL_ORDER):
            rename[cid] = SPEAKER_LABEL_ORDER[rank]
        else:
            rename[cid] = f"Voice {rank - 1}"
    return [rename[int(l)] for l in labels.tolist()]


def _format_conversation_document(
    segments: list[dict],
) -> tuple[str, str]:
    """Plain text + simple Markdown with Doc/Patient/Voice headers in time order."""
    lines_txt: list[str] = []
    lines_md: list[str] = []
    for seg in segments:
        sp = seg["speaker"]
        tx = (seg.get("text") or "").strip()
        if not tx:
            continue
        lines_txt.append(f"{sp}:")
        lines_txt.append(tx)
        lines_txt.append("")
        lines_md.append(f"### {sp}\n")
        lines_md.append(f"{tx}\n\n")
    return "\n".join(lines_txt).strip(), "".join(lines_md).strip()


def _process_conversation_sync(pcm: np.ndarray, lang_code: str) -> dict:
    """Segment full recording, cluster coarse speakers, transcribe each chunk, build document."""
    sr = TARGET_SAMPLE_RATE
    ranges = _segment_utterance_ranges(pcm, sr)
    if not ranges:
        return {
            "language": lang_code,
            "segments": [],
            "document": "",
            "document_md": "",
            "note": "No speech detected in recording.",
        }

    chunks: list[np.ndarray] = []
    starts: list[float] = []
    ends: list[float] = []
    for a, b in ranges:
        a = max(0, min(a, len(pcm)))
        b = max(a, min(b, len(pcm)))
        seg = pcm[a:b]
        if len(seg) < int(0.12 * sr):
            continue
        chunks.append(seg)
        starts.append(a / sr)
        ends.append(b / sr)

    if not chunks:
        return {
            "language": lang_code,
            "segments": [],
            "document": "",
            "document_md": "",
            "note": "No usable utterances after segmentation.",
        }

    feats = np.stack([_segment_audio_features(c, sr) for c in chunks], axis=0)
    k = _pick_speaker_cluster_count(len(chunks))
    labels = _kmeans_labels(feats, k)
    speaker_names = _map_clusters_to_speaker_names(labels)

    segments_out: list[dict] = []
    for idx, chunk in enumerate(chunks):
        text = _run_inference(chunk, lang_code).strip()
        segments_out.append({
            "speaker": speaker_names[idx],
            "text": text,
            "start_sec": round(starts[idx], 2),
            "end_sec": round(ends[idx], 2),
        })

    doc_txt, doc_md = _format_conversation_document(segments_out)
    return {
        "language": lang_code,
        "segments": segments_out,
        "document": doc_txt,
        "document_md": doc_md,
    }


async def process_conversation_document(pcm: np.ndarray, lang_code: str) -> dict:
    return await asyncio.to_thread(_process_conversation_sync, pcm, lang_code)


# ─── Google Gemini 2.5 Flash (online STT) ────────────────────────────────────

def gemini_online_enabled() -> bool:
    return bool(GEMINI_API_KEY)


def _mime_from_upload_filename(filename: str) -> str:
    fn = filename.lower()
    if fn.endswith(".webm"):
        return "audio/webm"
    if fn.endswith(".wav"):
        return "audio/wav"
    if fn.endswith(".mp3") or fn.endswith(".mpga") or fn.endswith(".mpeg"):
        return "audio/mpeg"
    if fn.endswith(".m4a") or fn.endswith(".mp4"):
        return "audio/mp4"
    if fn.endswith(".flac"):
        return "audio/flac"
    if fn.endswith(".ogg"):
        return "audio/ogg"
    return "application/octet-stream"


def _gemini_multilingual_core_rules() -> str:
    """Shared rules so Gemini transcribes any language (not Indic-only)."""
    return (
        "Multilingual speech rules: identify and transcribe every language actually spoken in the "
        "audio. Support code-switching and mixed-language sentences. Use the normal writing system "
        "(script) for each language as it would appear in standard text. Do not translate the speech "
        "into another language unless the speaker is explicitly translating; otherwise preserve what "
        "was said in the original language(s)."
    )


def _gemini_language_instruction(language_code: str) -> str:
    """How to apply optional UI locale vs full auto-detect."""
    code = (language_code or "").strip().lower()
    if code in ("", "auto", "any", "multi", "*"):
        return _gemini_multilingual_core_rules()

    return (
        _gemini_multilingual_core_rules()
        + f" Optional user hint for ambiguous stretches: locale / tag `{language_code}` "
        "(use as soft bias only; still transcribe any other language if it appears)."
    )


def _gemini_verbatim_prompt(language_code: str) -> str:
    li = _gemini_language_instruction(language_code)
    return (
        f"{li}\n\n"
        "Task: transcribe all speech in the attached audio.\n"
        "Output rules: return ONLY the spoken words as plain text. "
        "No timestamps, no speaker labels, no preamble or markdown."
    )


def _gemini_conversation_prompt(language_code: str) -> str:
    li = _gemini_language_instruction(language_code)
    return (
        f"{li}\n\n"
        "You are a medical documentation assistant. The audio may be a consultation with a doctor "
        "(Doc), a patient (Patient), and possibly other people — label extras as Voice 1, Voice 2, "
        "in order of first appearance after Doc and Patient.\n"
        "Each segment's text must stay in whatever language(s) that speaker used (multilingual allowed).\n\n"
        "Return ONLY valid JSON (no markdown fences) with this exact shape:\n"
        '{"segments":[{"speaker":"Doc|Patient|Voice 1|Voice 2","text":"..."}],'
        '"document":"plain text: each turn starts with Speaker: then newline then text, blank line between turns"}\n'
        "Rules: segments must be in chronological order. "
        "If you cannot distinguish roles, still split by speaker change and use Voice 1, Voice 2. "
        "The document field must match the same content as segments formatted for a human reader."
    )


def _gemini_generate_text(user_prompt: str, mime_type: str, audio_bytes: bytes) -> str:
    """Call Gemini generateContent; returns assistant text or raises RuntimeError."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    body_obj = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.standard_b64encode(audio_bytes).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8192,
        },
    }
    data = json.dumps(body_obj).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"Gemini API HTTP {e.code}: {err}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Gemini API network error: {e}") from e

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Gemini invalid JSON response: {e}") from e

    if "error" in payload:
        msg = payload.get("error", {}).get("message", str(payload["error"]))
        raise RuntimeError(f"Gemini API error: {msg}")

    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates (empty or blocked response)")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts: list[str] = []
    for p in parts:
        if isinstance(p, dict) and "text" in p:
            texts.append(p["text"])
    if not texts:
        raise RuntimeError("Gemini returned no text parts")

    return "\n".join(texts).strip()


def _strip_json_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _parse_gemini_conversation_json(text: str) -> Optional[dict]:
    s = _strip_json_fence(text)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    segs = obj.get("segments")
    if not isinstance(segs, list):
        return None
    cleaned: list[dict] = []
    for item in segs:
        if not isinstance(item, dict):
            continue
        sp = str(item.get("speaker", "Speaker")).strip() or "Speaker"
        tx = str(item.get("text", "")).strip()
        if tx:
            cleaned.append({"speaker": sp, "text": tx})
    doc = str(obj.get("document", "")).strip()
    if not doc and cleaned:
        doc, _ = _format_conversation_document(
            [{"speaker": s["speaker"], "text": s["text"]} for s in cleaned]
        )
    doc_md = str(obj.get("document_md", "")).strip()
    if not doc_md and cleaned:
        _, doc_md = _format_conversation_document(
            [{"speaker": s["speaker"], "text": s["text"]} for s in cleaned]
        )
    return {"segments": cleaned, "document": doc, "document_md": doc_md}


def _gemini_transcribe_upload_sync(
    audio_bytes: bytes,
    filename: str,
    language_code: str,
    prompt_type: str,
) -> dict:
    """Sync: call Gemini; returns JSON-serializable dict for HTTP response."""
    mime = _mime_from_upload_filename(filename)
    if mime == "application/octet-stream":
        raise RuntimeError("Unsupported or unknown audio MIME type for Gemini")

    if prompt_type == "conversation_doc":
        prompt = _gemini_conversation_prompt(language_code)
        raw_text = _gemini_generate_text(prompt, mime, audio_bytes)
        parsed = _parse_gemini_conversation_json(raw_text)
        if parsed is None:
            plain = raw_text.strip()
            parsed = {
                "segments": [{"speaker": "Transcript", "text": plain}] if plain else [],
                "document": plain,
                "document_md": f"### Transcript\n\n{plain}\n" if plain else "",
            }
        return {
            "provider": "google-gemini",
            "model": GEMINI_MODEL,
            "language": language_code,
            "prompt_type": prompt_type,
            "segments": parsed["segments"],
            "document": parsed["document"],
            "document_md": parsed["document_md"],
        }

    prompt = _gemini_verbatim_prompt(language_code)
    raw_text = _gemini_generate_text(prompt, mime, audio_bytes)
    return {
        "provider": "google-gemini",
        "model": GEMINI_MODEL,
        "language": language_code,
        "prompt_type": "verbatim",
        "transcript": raw_text.strip(),
    }


# ─── Session Handler ───────────────────────────────────────────────────────────
class STTSession:
    """Manages audio buffering + VAD + transcription for one WebSocket connection."""

    def __init__(self, session_id: str, lang_code: str, websocket):
        self.session_id   = session_id
        self.lang_code    = lang_code
        self.websocket    = websocket
        self.audio_buffer = np.array([], dtype=np.float32)
        self.speech_active = False
        self.silence_frames = 0
        self.speech_frames  = 0
        self.total_buffered = 0.0  # seconds

        log.info(f"[{session_id}] Session started | lang={lang_code} | decode={DECODE_MODE}")

    def add_audio(self, pcm_bytes: bytes) -> None:
        """Ingest raw 16-bit PCM bytes (already at 16kHz)."""
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        pcm_float = pcm_int16.astype(np.float32) / 32768.0
        self.audio_buffer = np.concatenate([self.audio_buffer, pcm_float])
        self.total_buffered = len(self.audio_buffer) / TARGET_SAMPLE_RATE

    def check_vad(self) -> Optional[str]:
        """
        Check energy-based VAD on current buffer.
        Returns 'transcribe' if we should run STT, else None.
        """
        if len(self.audio_buffer) == 0:
            return None

        # Use last 100ms window for VAD decision
        window_size = int(0.1 * TARGET_SAMPLE_RATE)
        recent = self.audio_buffer[-window_size:] if len(self.audio_buffer) >= window_size else self.audio_buffer
        rms = compute_rms(recent)

        if rms > SILENCE_THRESHOLD:
            self.speech_active  = True
            self.silence_frames = 0
            self.speech_frames += 1
        else:
            if self.speech_active:
                self.silence_frames += 1

        speech_secs  = self.speech_frames * 0.1
        silence_secs = self.silence_frames * 0.1

        # Trigger: enough speech followed by silence
        if (self.speech_active and
                speech_secs >= MIN_SPEECH_DURATION and
                silence_secs >= SILENCE_DURATION):
            return "transcribe"

        # Force trigger: buffer too long
        if self.total_buffered >= MAX_BUFFER_DURATION:
            return "transcribe"

        return None

    async def process_if_ready(self) -> None:
        """Run transcription if VAD says so, then send result over WebSocket."""
        action = self.check_vad()
        if action != "transcribe":
            return
        if len(self.audio_buffer) < TARGET_SAMPLE_RATE * MIN_SPEECH_DURATION:
            return

        audio_to_transcribe  = self.audio_buffer.copy()
        duration             = len(audio_to_transcribe) / TARGET_SAMPLE_RATE

        # Reset buffer and VAD state
        self.audio_buffer   = np.array([], dtype=np.float32)
        self.speech_active  = False
        self.silence_frames = 0
        self.speech_frames  = 0
        self.total_buffered = 0.0

        start = time.time()
        text  = await transcribe(audio_to_transcribe, self.lang_code)
        latency = int((time.time() - start) * 1000)

        log.info(f"[{self.session_id}] Transcribed {duration:.1f}s → '{text}' ({latency}ms)")

        if text:
            await self.websocket.send(json.dumps({
                "type":      "final",
                "text":      text,
                "lang":      self.lang_code,
                "duration":  round(duration, 2),
                "latency_ms": latency
            }))

    async def flush(self) -> None:
        """Force transcribe any remaining audio on session stop."""
        if len(self.audio_buffer) < TARGET_SAMPLE_RATE * 0.2:
            return

        audio_to_transcribe = self.audio_buffer.copy()
        duration            = len(audio_to_transcribe) / TARGET_SAMPLE_RATE
        self.audio_buffer   = np.array([], dtype=np.float32)

        start = time.time()
        text  = await transcribe(audio_to_transcribe, self.lang_code)
        latency = int((time.time() - start) * 1000)

        log.info(f"[{self.session_id}] Flush transcribed {duration:.1f}s → '{text}' ({latency}ms)")

        if text:
            await self.websocket.send(json.dumps({
                "type":      "final",
                "text":      text,
                "lang":      self.lang_code,
                "duration":  round(duration, 2),
                "latency_ms": latency,
                "flushed":   True
            }))


# ─── WebSocket Handler ─────────────────────────────────────────────────────────
async def handle_connection(websocket):
    """
    Protocol:
      → Client sends JSON init:  {"type":"start","lang":"ml-IN","session_id":"abc"}
      → Client sends binary:     raw 16kHz 16-bit mono PCM bytes
      → Client sends JSON stop:  {"type":"stop"}
      ← Server sends JSON:       {"type":"final","text":"...","latency_ms":120}
      ← Server sends JSON:       {"type":"ready"} after model confirms loaded
      ← Server sends JSON:       {"type":"error","message":"..."}
    """
    session = None
    conn_id = f"conn_{id(websocket)}"
    remote  = websocket.remote_address

    try:
        await websocket.send(json.dumps({"type": "ready", "model": "indic-conformer-600m-multilingual"}))
        log.info(f"New connection from {remote}")

        async for message in websocket:

            # ── Binary audio chunk ──
            if isinstance(message, bytes):
                if session is None:
                    log.warning(f"{remote}: received audio before init message, ignoring")
                    continue
                session.add_audio(message)
                await session.process_if_ready()

            # ── JSON control message ──
            elif isinstance(message, str):
                try:
                    msg = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
                    continue

                msg_type = msg.get("type")

                if msg_type == "start":
                    lang        = msg.get("lang", "ml-IN")
                    session_id  = msg.get("session_id", f"sess_{int(time.time())}")
                    # Sanitize session_id (max 64 chars, strip non-printable)
                    session_id = re.sub(r'[^\w\-.]', '_', str(session_id))[:64]
                    # Validate language code
                    if lang not in LANG_MAP:
                        lang = "ml-IN"
                    session     = STTSession(session_id, lang, websocket)
                    active_sessions[conn_id] = session
                    await websocket.send(json.dumps({"type": "started", "session_id": session_id, "lang": lang}))

                elif msg_type == "stop":
                    if session:
                        await session.flush()
                        log.info(f"[{session.session_id}] Session stopped")
                    await websocket.send(json.dumps({"type": "stopped"}))
                    active_sessions.pop(conn_id, None)
                    session = None

                elif msg_type == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))

    except websockets.exceptions.ConnectionClosed:
        log.info(f"Connection closed: {remote}")
    except Exception as e:
        log.error(f"Handler error for {remote}: {e}", exc_info=True)
        try:
            await websocket.send(json.dumps({"type": "error", "message": "Internal server error"}))
        except Exception:
            pass
    finally:
        active_sessions.pop(conn_id, None)
        if session:
            try:
                await session.flush()
            except Exception:
                pass


async def _limited_handler(websocket):
    """Wrap handle_connection with a semaphore to cap concurrent connections."""
    if _conn_semaphore.locked() and _conn_semaphore._value == 0:
        await websocket.close(1013, "Server at capacity")
        log.warning(f"Rejected connection from {websocket.remote_address} — at capacity ({MAX_CONNECTIONS})")
        return
    async with _conn_semaphore:
        await handle_connection(websocket)


# ─── Batch-Capable Connection ────────────────────────────────────────────────
# websockets 16.x rejects POST requests at the HTTP/1.1 parsing level before
# _process_request() is ever called.  We subclass ServerConnection and override
# data_received() to intercept POST requests at the transport level.

class BatchCapableConnection(ServerConnection):
    """ServerConnection subclass that intercepts HTTP POST for batch endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._post_buffer = b""
        self._is_post: Optional[bool] = None  # None = undetermined
        self._handled_as_http = False

    async def handshake(self, *args, **kwargs):
        """Override to suppress the EOF error when we already handled as HTTP.
        The race: handshake() starts awaiting protocol data, then data_received()
        intercepts POST/OPTIONS and closes the transport, causing an EOF here."""
        try:
            return await super().handshake(*args, **kwargs)
        except Exception:
            if self._handled_as_http:
                return  # suppress — we already sent an HTTP response
            raise

    def data_received(self, data: bytes) -> None:
        # First chunk: determine request type
        if self._is_post is None:
            self._post_buffer = data
            if data[:7] == b"OPTIONS":
                self._handled_as_http = True
                self._send_cors_preflight()
                return
            elif data[:4] == b"POST":
                self._is_post = True
                self._handled_as_http = True
                self._try_handle_post()
                return
            else:
                self._is_post = False
                super().data_received(data)
                return

        if self._is_post:
            # Cap buffer to prevent unbounded memory growth
            # Allow headers (~8KB) + body (BATCH_MAX_FILE_SIZE)
            max_buffer = BATCH_MAX_FILE_SIZE + 64 * 1024
            if len(self._post_buffer) + len(data) > max_buffer:
                self._send_json_response(413, "Payload Too Large",
                                         {"error": "Request too large"})
                return
            self._post_buffer += data
            self._try_handle_post()
        else:
            super().data_received(data)

    def _try_handle_post(self):
        """Check if we have the full POST request, then handle it."""
        header_end = self._post_buffer.find(b"\r\n\r\n")
        if header_end == -1:
            return  # need more header data

        headers_section = self._post_buffer[:header_end]
        body_start = header_end + 4

        # Parse Content-Length (with validation)
        content_length = 0
        for line in headers_section.decode("utf-8", errors="replace").split("\r\n"):
            if line.lower().startswith("content-length:"):
                try:
                    content_length = int(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    self._send_json_response(400, "Bad Request",
                                             {"error": "Invalid Content-Length"})
                    return
                if content_length < 0 or content_length > BATCH_MAX_FILE_SIZE + 4096:
                    self._send_json_response(413, "Payload Too Large",
                                             {"error": f"Content-Length exceeds limit"})
                    return
                break

        body_so_far = self._post_buffer[body_start:]
        if len(body_so_far) < content_length:
            return  # need more body data

        # We have the full request
        body = body_so_far[:content_length]
        headers_raw = headers_section.decode("utf-8", errors="replace")
        task = asyncio.ensure_future(self._handle_post(headers_raw, body))
        task.add_done_callback(self._post_task_done)

    def _post_task_done(self, task: asyncio.Task):
        """Callback for POST handler task — log unhandled exceptions."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            log.error(f"[batch] Unhandled POST handler error: {exc}", exc_info=exc)

    async def _handle_post(self, headers_raw: str, body: bytes):
        """Route and handle the POST request."""
        try:
            lines = headers_raw.split("\r\n")
            request_line = lines[0]  # e.g. "POST /batch/transcribe HTTP/1.1"
            parts = request_line.split(" ", 2)
            path = parts[1] if len(parts) > 1 else "/"

            # Parse headers into dict
            headers = {}
            for line in lines[1:]:
                if ":" in line:
                    key, val = line.split(":", 1)
                    headers[key.strip().lower()] = val.strip()

            # API key check (timing-safe)
            if API_KEY:
                client_key = headers.get("x-api-key", "")
                if not hmac.compare_digest(client_key, API_KEY):
                    self._send_json_response(403, "Forbidden",
                                             {"error": "Invalid or missing API key"})
                    return

            if path == "/batch/transcribe":
                await self._handle_batch_transcribe(headers, body)
            elif path == "/conversation/transcribe":
                await self._handle_conversation_transcribe(headers, body)
            elif path == "/online/gemini/transcribe":
                await self._handle_gemini_transcribe(headers, body)
            else:
                self._send_json_response(404, "Not Found",
                                         {"error": f"Unknown endpoint: {path}"})
        except Exception as e:
            log.error(f"[batch] POST handler error: {e}", exc_info=True)
            self._send_json_response(500, "Internal Server Error",
                                     {"error": "Internal server error"})

    async def _handle_batch_transcribe(self, headers: dict, body: bytes):
        """Handle POST /batch/transcribe — accept audio file for async transcription."""
        content_type = headers.get("content-type", "")

        if "multipart/form-data" not in content_type:
            self._send_json_response(400, "Bad Request",
                                     {"error": "Content-Type must be multipart/form-data"})
            return

        # Check file size before parsing
        if len(body) > BATCH_MAX_FILE_SIZE:
            self._send_json_response(413, "Payload Too Large",
                                     {"error": f"File exceeds {BATCH_MAX_FILE_SIZE // (1024*1024)}MB limit"})
            return

        fields = _parse_multipart(content_type, body)

        if "file" not in fields or not isinstance(fields["file"], dict):
            self._send_json_response(400, "Bad Request",
                                     {"error": "Missing 'file' field in multipart form"})
            return

        file_info = fields["file"]
        filename = file_info["filename"].lower()
        audio_data = file_info["data"]

        # Validate file extension
        if not any(filename.endswith(ext) for ext in SUPPORTED_AUDIO_EXTS):
            self._send_json_response(400, "Bad Request",
                                     {"error": f"Unsupported format. Supported: {', '.join(SUPPORTED_AUDIO_EXTS)}"})
            return

        # Check job limit
        pending_count = sum(1 for j in _batch_jobs.values()
                           if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING))
        if pending_count >= BATCH_MAX_JOBS:
            self._send_json_response(429, "Too Many Requests",
                                     {"error": f"Too many pending jobs (max {BATCH_MAX_JOBS})"})
            return

        language_code = fields.get("language_code", "hi-IN")

        # Convert audio to PCM
        try:
            pcm, duration = await _convert_audio_to_pcm(audio_data)
        except Exception as e:
            log.error(f"[batch] Audio conversion failed: {e}", exc_info=True)
            self._send_json_response(400, "Bad Request",
                                     {"error": "Failed to decode audio file. Ensure the file is a valid audio format."})
            return

        if duration > BATCH_MAX_AUDIO_DURATION:
            self._send_json_response(400, "Bad Request",
                                     {"error": f"Audio too long ({duration:.1f}s). Max {BATCH_MAX_AUDIO_DURATION:.0f}s"})
            return

        # Create job
        job_id = f"batch_{uuid.uuid4().hex[:16]}"
        job = BatchJob(
            job_id=job_id,
            status=JobStatus.QUEUED,
            language=language_code,
            created_at=time.time(),
            audio_pcm=pcm,
            audio_duration=round(duration, 2),
        )
        _batch_jobs[job_id] = job
        await _batch_queue.put(job_id)

        log.info(f"[batch] Job {job_id} queued: {filename} ({duration:.1f}s, {language_code})")

        self._send_json_response(201, "Created", {
            "job_id": job_id,
            "status": "queued",
            "language": language_code,
            "audio_duration": round(duration, 2),
        })

    async def _handle_conversation_transcribe(self, headers: dict, body: bytes):
        """POST /conversation/transcribe — full recording → segmented transcript + Doc/Patient/Voice doc."""
        content_type = headers.get("content-type", "")

        if "multipart/form-data" not in content_type:
            self._send_json_response(400, "Bad Request",
                                     {"error": "Content-Type must be multipart/form-data"})
            return

        if len(body) > BATCH_MAX_FILE_SIZE:
            self._send_json_response(413, "Payload Too Large",
                                     {"error": f"File exceeds {BATCH_MAX_FILE_SIZE // (1024*1024)}MB limit"})
            return

        fields = _parse_multipart(content_type, body)

        if "file" not in fields or not isinstance(fields["file"], dict):
            self._send_json_response(400, "Bad Request",
                                     {"error": "Missing 'file' field in multipart form"})
            return

        file_info = fields["file"]
        filename = file_info["filename"].lower()
        audio_data = file_info["data"]

        if not any(filename.endswith(ext) for ext in SUPPORTED_AUDIO_EXTS):
            self._send_json_response(400, "Bad Request",
                                     {"error": f"Unsupported format. Supported: {', '.join(SUPPORTED_AUDIO_EXTS)}"})
            return

        language_code = fields.get("language_code", "hi-IN")
        if language_code not in LANG_MAP:
            language_code = "hi-IN"

        try:
            pcm, duration = await _convert_audio_to_pcm(audio_data)
        except Exception as e:
            log.error(f"[conversation] Audio conversion failed: {e}", exc_info=True)
            self._send_json_response(400, "Bad Request",
                                     {"error": "Failed to decode audio file. Ensure valid format or ffmpeg for WebM/MP3."})
            return

        if duration > BATCH_MAX_AUDIO_DURATION:
            self._send_json_response(400, "Bad Request",
                                     {"error": f"Audio too long ({duration:.1f}s). Max {BATCH_MAX_AUDIO_DURATION:.0f}s"})
            return

        try:
            result = await process_conversation_document(pcm, language_code)
        except Exception as e:
            log.error(f"[conversation] Processing failed: {e}", exc_info=True)
            self._send_json_response(500, "Internal Server Error",
                                     {"error": "Conversation processing failed"})
            return

        result["audio_duration_sec"] = round(duration, 2)
        self._send_json_response(200, "OK", result)

    async def _handle_gemini_transcribe(self, headers: dict, body: bytes):
        """POST /online/gemini/transcribe — proxy to Google Gemini 2.5 Flash (speech → text)."""
        content_type = headers.get("content-type", "")

        if not gemini_online_enabled():
            self._send_json_response(
                503,
                "Service Unavailable",
                {"error": "Gemini is not configured. Set GEMINI_API_KEY (or GOOGLE_API_KEY) on the server."},
            )
            return

        if "multipart/form-data" not in content_type:
            self._send_json_response(400, "Bad Request",
                                     {"error": "Content-Type must be multipart/form-data"})
            return

        if len(body) > BATCH_MAX_FILE_SIZE:
            self._send_json_response(413, "Payload Too Large",
                                     {"error": f"File exceeds {BATCH_MAX_FILE_SIZE // (1024*1024)}MB limit"})
            return

        fields = _parse_multipart(content_type, body)

        if "file" not in fields or not isinstance(fields["file"], dict):
            self._send_json_response(400, "Bad Request",
                                     {"error": "Missing 'file' field in multipart form"})
            return

        file_info = fields["file"]
        filename = file_info["filename"].lower()
        audio_data = file_info["data"]

        if not any(filename.endswith(ext) for ext in SUPPORTED_AUDIO_EXTS):
            self._send_json_response(400, "Bad Request",
                                     {"error": f"Unsupported format. Supported: {', '.join(SUPPORTED_AUDIO_EXTS)}"})
            return

        if len(audio_data) > GEMINI_MAX_UPLOAD_BYTES:
            self._send_json_response(
                413,
                "Payload Too Large",
                {"error": f"Audio exceeds Gemini upload limit ({GEMINI_MAX_UPLOAD_BYTES // (1024*1024)}MB)"},
            )
            return

        language_code = (fields.get("language_code") or "auto").strip()
        prompt_type = (fields.get("prompt_type") or "verbatim").strip().lower()
        if prompt_type not in ("verbatim", "conversation_doc"):
            prompt_type = "verbatim"

        log.info(f"[gemini] Transcribe request | model={GEMINI_MODEL} | prompt={prompt_type} | bytes={len(audio_data)}")

        try:
            result = await asyncio.to_thread(
                _gemini_transcribe_upload_sync,
                audio_data,
                filename,
                language_code,
                prompt_type,
            )
        except RuntimeError as e:
            log.warning(f"[gemini] Failed: {e}")
            self._send_json_response(502, "Bad Gateway", {"error": str(e)})
            return
        except Exception as e:
            log.error(f"[gemini] Unexpected error: {e}", exc_info=True)
            self._send_json_response(500, "Internal Server Error",
                                     {"error": "Gemini transcription failed"})
            return

        self._send_json_response(200, "OK", result)

    def _send_cors_preflight(self):
        """Respond to an OPTIONS preflight request."""
        response = (
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, X-API-Key\r\n"
            "Access-Control-Max-Age: 86400\r\n"
            "Content-Length: 0\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("utf-8")
        try:
            self.transport.write(response)
            self.transport.close()
        except Exception:
            pass

    def _send_json_response(self, status_code: int, status_text: str, body_dict: dict):
        """Write a raw HTTP JSON response to the transport and close."""
        body = json.dumps(body_dict).encode("utf-8")
        response = (
            f"HTTP/1.1 {status_code} {status_text}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            f"Access-Control-Allow-Headers: Content-Type, X-API-Key\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode("utf-8") + body
        try:
            self.transport.write(response)
            self.transport.close()
        except Exception:
            pass


# ─── CORS & HTTP Helpers ──────────────────────────────────────────────────────

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-API-Key",
}

def _json_response(connection, status: HTTPStatus, body_dict: dict):
    """Helper to build a JSON HTTP response via websockets' connection.respond()."""
    body = json.dumps(body_dict)
    response = connection.respond(status, body)
    response.headers["Content-Type"] = "application/json"
    for k, v in _CORS_HEADERS.items():
        response.headers[k] = v
    return response


def _process_request(connection, request):
    """Intercept HTTP requests before WebSocket upgrade.
    Serves /health, /batch/status/{id}, /batch/result/{id}.
    websockets 16.x API: (ServerConnection, Request) -> Response | None."""

    # ── Health check (no auth required) ──
    if request.path == "/health":
        queued = sum(1 for j in _batch_jobs.values() if j.status == JobStatus.QUEUED)
        return _json_response(connection, HTTPStatus.OK, {
            "status":           "ok",
            "model":            "indic-conformer-600m-multilingual",
            "device":           device,
            "decode_mode":      DECODE_MODE,
            "active_sessions":  len(active_sessions),
            "max_connections":  MAX_CONNECTIONS,
            "uptime_seconds":   round(time.time() - _server_start_time, 1),
            "batch_jobs_queued": queued,
            "batch_jobs_total":  len(_batch_jobs),
            "gemini_online":    gemini_online_enabled(),
            "gemini_model":     GEMINI_MODEL if gemini_online_enabled() else None,
        })

    # API key check — skip if no key configured (backwards compatible for local dev)
    if API_KEY:
        client_key = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(client_key, API_KEY):
            log.warning(f"Rejected connection — invalid or missing API key from {request.headers.get('Host', 'unknown')}")
            return connection.respond(HTTPStatus.FORBIDDEN, "Invalid or missing API key")

    # ── Batch status endpoint ──
    if request.path.startswith("/batch/status/"):
        job_id = request.path[len("/batch/status/"):]
        job = _batch_jobs.get(job_id)
        if not job:
            return _json_response(connection, HTTPStatus.NOT_FOUND,
                                  {"error": "Job not found", "job_id": job_id})

        result = {
            "job_id": job.job_id,
            "status": job.status.value,
            "language": job.language,
            "duration": job.audio_duration,
            "created_at": job.created_at,
        }
        if job.status == JobStatus.COMPLETED:
            result["transcript"] = job.transcript
            result["latency_ms"] = job.latency_ms
            result["completed_at"] = job.completed_at
        elif job.status == JobStatus.FAILED:
            result["error_message"] = job.error_message
            result["completed_at"] = job.completed_at

        return _json_response(connection, HTTPStatus.OK, result)

    # ── Batch result endpoint ──
    if request.path.startswith("/batch/result/"):
        job_id = request.path[len("/batch/result/"):]
        job = _batch_jobs.get(job_id)
        if not job:
            return _json_response(connection, HTTPStatus.NOT_FOUND,
                                  {"error": "Job not found", "job_id": job_id})

        if job.status == JobStatus.COMPLETED:
            return _json_response(connection, HTTPStatus.OK, {
                "job_id": job.job_id,
                "status": "completed",
                "transcript": job.transcript,
                "language": job.language,
                "duration": job.audio_duration,
                "latency_ms": job.latency_ms,
            })
        elif job.status == JobStatus.FAILED:
            return _json_response(connection, HTTPStatus.OK, {
                "job_id": job.job_id,
                "status": "failed",
                "error_message": job.error_message,
            })
        else:
            # Still processing — 202 Accepted
            return _json_response(connection, HTTPStatus.ACCEPTED, {
                "job_id": job.job_id,
                "status": job.status.value,
                "language": job.language,
                "duration": job.audio_duration,
            })

    # ── Fix headers mangled by reverse proxies (e.g. Cloudflare Tunnel) ──
    # Cloudflare rewrites "Connection: Upgrade" → "Connection: keep-alive".
    # If Sec-WebSocket-Key is present, this is a genuine WebSocket client,
    # so restore the expected headers before websockets validates them.
    if request.headers.get("Sec-WebSocket-Key"):
        conn_values = [v.lower() for v in request.headers.get_all("Connection")]
        if not any("upgrade" in v for v in conn_values):
            log.info(f"Fixing Connection header mangled by reverse proxy (was: {request.headers.get('Connection')})")
            del request.headers["Connection"]
            request.headers["Connection"] = "Upgrade"

        upgrade_values = [v.lower() for v in request.headers.get_all("Upgrade")]
        if not any("websocket" in v for v in upgrade_values):
            log.info(f"Fixing Upgrade header mangled by reverse proxy (was: {request.headers.get('Upgrade')})")
            if "Upgrade" in request.headers:
                del request.headers["Upgrade"]
            request.headers["Upgrade"] = "websocket"

    return None


# ─── Main ──────────────────────────────────────────────────────────────────────
async def main():
    global _conn_semaphore, _server_start_time, _batch_queue
    global _batch_worker_task, _batch_cleanup_task

    load_model()

    log.info("Running warm-up inference...")
    dummy = np.zeros(16000, dtype=np.float32)
    _run_inference(dummy, "hi")
    log.info("Warm-up complete")

    _conn_semaphore = asyncio.Semaphore(MAX_CONNECTIONS)
    _server_start_time = time.time()

    # Initialize batch processing
    _batch_queue = asyncio.Queue()
    _batch_worker_task = asyncio.create_task(_batch_worker())
    _batch_cleanup_task = asyncio.create_task(_batch_cleanup_loop())

    log.info(f"Starting VEXYL-STT WebSocket server on ws://{HOST}:{PORT}")

    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: _handle_signal(s, stop_event))

    async with websockets.serve(
        _limited_handler,
        HOST,
        PORT,
        max_size=10 * 1024 * 1024,   # 10MB max message
        ping_interval=30,
        ping_timeout=10,
        close_timeout=5,
        process_request=_process_request,
        create_connection=BatchCapableConnection,
    ) as server:
        log.info(f"VEXYL-STT server ready | ws://{HOST}:{PORT} | max_conn={MAX_CONNECTIONS} | batch=enabled")
        await stop_event.wait()

        log.info("Shutting down... cancelling batch tasks")
        _batch_worker_task.cancel()
        _batch_cleanup_task.cancel()
        try:
            await _batch_worker_task
        except asyncio.CancelledError:
            pass
        try:
            await _batch_cleanup_task
        except asyncio.CancelledError:
            pass

        log.info("Closing active connections")
        server.close()
        await server.wait_closed()
        log.info("Server stopped cleanly")


def _handle_signal(sig, stop_event: asyncio.Event):
    log.info(f"Received {signal.Signals(sig).name}, initiating shutdown...")
    stop_event.set()


if __name__ == "__main__":
    asyncio.run(main())
