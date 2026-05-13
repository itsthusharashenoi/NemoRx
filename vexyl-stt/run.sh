#!/bin/bash
# Start the VEXYL-STT server
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment config (.env then .env.secrets; latter is gitignored)
for _envf in .env .env.secrets; do
    if [ -f "$_envf" ]; then
        while IFS= read -r line || [ -n "$line" ]; do
            case "$line" in ''|\#*) continue ;; esac
            export "$line"
        done < "$_envf"
    fi
done
unset _envf

source venv/bin/activate
python3 vexyl_stt_server.py
