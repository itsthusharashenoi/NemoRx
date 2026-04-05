#!/bin/bash
# Download the gated IndicConformer model (requires Hugging Face access + token).
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate

MODEL_ID="ai4bharat/indic-conformer-600m-multilingual"

if [ -f .env.secrets ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
    esac
    export "$line"
  done < .env.secrets
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set."
  echo ""
  echo "1. Create a token: https://huggingface.co/settings/tokens (read access)"
  echo "2. Accept the model license: https://huggingface.co/${MODEL_ID}"
  echo "3. Copy .env.secrets.example to .env.secrets and set HF_TOKEN=..."
  echo "   Or: export HF_TOKEN=hf_... && ./download-model.sh"
  exit 1
fi

echo "Logging in to Hugging Face..."
export HF_TOKEN
python3 -c "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'])"

echo "Downloading ${MODEL_ID} (~2.4 GB)..."
set +e
python3 -c "
from transformers import AutoModel
AutoModel.from_pretrained('${MODEL_ID}', trust_remote_code=True)
print('Model download complete.')
"
DL=$?
set -e
if [ "$DL" -ne 0 ]; then
  echo ""
  echo "Download failed. Common causes:"
  echo "  • Gated model: open https://huggingface.co/${MODEL_ID}"
  echo "    while logged in as your HF user, accept the license, wait for approval, then re-run this script."
  echo "  • Invalid or expired token: update HF_TOKEN in .env.secrets"
  exit "$DL"
fi
