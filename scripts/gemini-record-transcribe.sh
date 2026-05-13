#!/usr/bin/env bash
# =============================================================================
# Terminal-only: record microphone → Ctrl+C stops → save WAV under recordings/,
# then Gemini 2.5 Flash transcribes to Transcriptis/ with the same basename (.txt).
#
# Requires: ffmpeg, python3 (stdlib only for transcription helper)
# API key: GEMINI_API_KEY env, or scripts/.env.secrets, or vexyl-stt/.env.secrets
#
# Usage:
#   ./scripts/gemini-record-transcribe.sh
#   LANGUAGE=hi-IN ./scripts/gemini-record-transcribe.sh   # soft bias (optional)
#   SKIP_NETWORK_CHECK=1 ./scripts/gemini-record-transcribe.sh   # skip curl connectivity check
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RECORDINGS="${RECORDINGS_DIR:-$REPO_ROOT/recordings}"
TRANSCRIPTS="${TRANSCRIPTS_DIR:-$REPO_ROOT/Transcriptis}"
LANGUAGE="${LANGUAGE:-auto}"

mkdir -p "$RECORDINGS" "$TRANSCRIPTS"

if [[ "${SKIP_NETWORK_CHECK:-0}" != "1" ]]; then
  if command -v curl >/dev/null 2>&1; then
    if ! curl -sf --max-time 8 "https://www.google.com" >/dev/null 2>&1; then
      echo "Network check failed (no response from https://www.google.com). Gemini needs internet." >&2
      echo "Fix Wi‑Fi/Ethernet or set SKIP_NETWORK_CHECK=1 to bypass." >&2
      exit 1
    fi
  else
    echo "Warning: curl not found; skipping network check." >&2
  fi
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install: brew install ffmpeg" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
BASE="gemini-${STAMP}"
OUT_WAV="${RECORDINGS}/${BASE}.wav"
OUT_TXT="${TRANSCRIPTS}/${BASE}.txt"

OS="$(uname -s)"
FFPID=""

record_macos() {
  local dev="${FFMPEG_AUDIO_DEVICE:-:0}"
  echo "Recording from avfoundation audio device ${dev} (set FFMPEG_AUDIO_DEVICE if wrong)."
  echo "List devices: ffmpeg -f avfoundation -list_devices true -i \"\""
  # :0 = default capture device pair on many Macs; override if needed.
  ffmpeg -hide_banner -loglevel info -y -nostdin \
    -f avfoundation -i "$dev" \
    -ar 16000 -ac 1 -c:a pcm_s16le \
    "$OUT_WAV" &
  FFPID=$!
}

record_linux() {
  local dev="${ALSA_DEVICE:-default}"
  echo "Recording from ALSA device: $dev"
  ffmpeg -hide_banner -loglevel info -y -nostdin \
    -f alsa -thread_queue_size 4096 -i "$dev" \
    -ar 16000 -ac 1 -c:a pcm_s16le \
    "$OUT_WAV" &
  FFPID=$!
}

on_int() {
  echo ""
  echo "Interrupt received — stopping ffmpeg and finalizing ${OUT_WAV} …"
  if [[ -n "$FFPID" ]] && kill -0 "$FFPID" 2>/dev/null; then
    kill -INT "$FFPID" 2>/dev/null || true
    wait "$FFPID" 2>/dev/null || true
  fi
}
trap on_int INT TERM

echo "============================================================================="
echo " Gemini 2.5 Flash — terminal capture"
echo " Repo:       $REPO_ROOT"
echo " Recording:  $OUT_WAV"
echo " Transcript: $OUT_TXT"
echo " Language:   $LANGUAGE  (use auto for Indian languages + English mixed)"
if [[ "${GEMINI_VERBATIM:-0}" == "1" ]]; then
  echo " Mode:       verbatim (plain text)"
else
  echo " Mode:       Doc/Patient conversation (set GEMINI_VERBATIM=1 for plain only)"
fi
echo "============================================================================="
echo "Press Ctrl+C when you are done speaking."
echo ""

if [[ "$OS" == "Darwin" ]]; then
  record_macos
elif [[ "$OS" == "Linux" ]]; then
  record_linux
else
  echo "Unsupported OS: $OS (add ffmpeg flags for your platform in this script)" >&2
  exit 1
fi

wait "$FFPID" || true
trap - INT TERM
FFPID=""

if [[ ! -f "$OUT_WAV" ]]; then
  echo "No WAV file produced." >&2
  exit 1
fi
SZ="$(wc -c < "$OUT_WAV" | tr -d ' ')"
if [[ "$SZ" -lt 500 ]]; then
  echo "Recording too small (${SZ} bytes); check microphone / ffmpeg device." >&2
  exit 1
fi

echo "Saved recording (${SZ} bytes). Calling Gemini…"
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash}"
PY_ARGS=( "$OUT_WAV" "$OUT_TXT" --language "$LANGUAGE" --repo-root "$REPO_ROOT" --stamp "$STAMP" )
if [[ "${GEMINI_VERBATIM:-0}" != "1" ]]; then
  PY_ARGS+=( --conversation )
fi
python3 "$SCRIPT_DIR/_gemini_transcribe.py" "${PY_ARGS[@]}"
echo "Done. (If a patient name was found: patient .txt is written and gemini .txt removed; WAV moved to <name>-${STAMP}.wav.)"

if [[ "${PRESCRIBE_AFTER_TRANSCRIBE:-0}" == "1" ]]; then
  PRE_DIR="$REPO_ROOT/prescription_gen"
  if [[ -f "$PRE_DIR/run.py" ]]; then
    LATEST="$(ls -t "$TRANSCRIPTS"/*.txt 2>/dev/null | head -1 || true)"
    if [[ -n "${LATEST:-}" && -f "$LATEST" ]]; then
      echo "PRESCRIBE_AFTER_TRANSCRIBE=1 — running prescription pipeline on: $LATEST"
      python3 "$PRE_DIR/run.py" --input "$LATEST" || echo "Warning: prescription step exited non-zero." >&2
    else
      echo "Warning: PRESCRIBE_AFTER_TRANSCRIBE set but no .txt found under $TRANSCRIPTS" >&2
    fi
  else
    echo "Warning: prescription_gen missing at $PRE_DIR (skipping prescription)." >&2
  fi
fi
