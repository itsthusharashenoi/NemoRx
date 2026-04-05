#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  python3 -m venv venv
fi
# shellcheck source=/dev/null
source venv/bin/activate
pip install -q -r requirements.txt

HOST="${WHISPER_PIPELINE_HOST:-127.0.0.1}"
PORT="${WHISPER_PIPELINE_PORT:-8092}"

echo "Whisper pipeline on http://${HOST}:${PORT} (POST /transcribe, GET /health)"
exec uvicorn server:app --host "$HOST" --port "$PORT"
