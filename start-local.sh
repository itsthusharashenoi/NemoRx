#!/bin/bash
# Start VEXYL-STT server + React UI (both on localhost).
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
STT="${ROOT}/vexyl-stt"
UI="${ROOT}/vexyl-stt-ui"

cd "$STT"
if [ -f .env ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
    esac
    export "$line"
  done < .env
fi

if [ ! -d venv ]; then
  echo "Missing vexyl-stt/venv. Run: cd vexyl-stt && ./setup.sh"
  echo "Or: install deps manually, then export HF_TOKEN=... && ./download-model.sh"
  exit 1
fi
# shellcheck source=/dev/null
source venv/bin/activate

if ! python3 -c "
from transformers import AutoModel
AutoModel.from_pretrained(
    'ai4bharat/indic-conformer-600m-multilingual',
    trust_remote_code=True,
    local_files_only=True,
)
" 2>/dev/null; then
  echo "Model is not downloaded yet."
  echo "  cd \"${STT}\""
  echo "  export HF_TOKEN=hf_your_token"
  echo "  ./download-model.sh"
  exit 1
fi

cleanup() {
  if [ -n "${STT_PID:-}" ] && kill -0 "$STT_PID" 2>/dev/null; then
    kill "$STT_PID" 2>/dev/null || true
    wait "$STT_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

PORT="${VEXYL_STT_PORT:-8091}"
echo "Starting VEXYL-STT on ws://127.0.0.1:${PORT}/ ..."
(
  cd "$STT"
  if [ -f .env ]; then
    while IFS= read -r line || [ -n "$line" ]; do
      case "$line" in
        ''|\#*) continue ;;
      esac
      export "$line"
    done < .env
  fi
  source venv/bin/activate
  exec python3 vexyl_stt_server.py
) &
STT_PID=$!

for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if ! kill -0 "$STT_PID" 2>/dev/null; then
    echo "VEXYL-STT process exited early. Check logs above."
    exit 1
  fi
done

if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "Timed out waiting for /health on port ${PORT}."
  exit 1
fi

echo "STT server is up. Starting UI..."
cd "$UI"
exec npm run dev
