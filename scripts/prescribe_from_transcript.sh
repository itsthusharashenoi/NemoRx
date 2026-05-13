#!/usr/bin/env bash
# =============================================================================
# Generate hospital-style prescription PDF from a DocScribe transcript .txt
#
# Input: path to a transcript (Doc/Patient lines), or omit to use the newest
#        *.txt under Transcriptis/ (same folder as gemini-record-transcribe.sh).
#
# Requires: Python 3 with prescription_gen/requirements.txt; sibling
#           docscribe_akhil/ at monorepo root; Node + puppeteer at repo root
#           for PDF (optional — fpdf2 fallback if Puppeteer fails).
#
# Usage:
#   ./scripts/prescribe_from_transcript.sh
#   ./scripts/prescribe_from_transcript.sh Transcriptis/some-patient-20260101-120000.txt
#   ./scripts/prescribe_from_transcript.sh path/to.txt --use-gemini off --skip-pdf
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRANSCRIPTS="${TRANSCRIPTS_DIR:-$REPO_ROOT/Transcriptis}"
PRE_DIR="$REPO_ROOT/prescription_gen"

if [[ ! -f "$PRE_DIR/run.py" ]]; then
  echo "prescription_gen not found at $PRE_DIR" >&2
  exit 1
fi

INPUT="${1:-}"
if [[ -n "$INPUT" ]]; then
  shift || true
fi
if [[ -z "$INPUT" ]]; then
  INPUT="$(ls -t "$TRANSCRIPTS"/*.txt 2>/dev/null | head -1 || true)"
fi
if [[ -z "$INPUT" || ! -f "$INPUT" ]]; then
  echo "No transcript .txt found. Record with ./scripts/gemini-record-transcribe.sh or pass a path." >&2
  exit 1
fi

echo "Prescription input: $INPUT"
exec python3 "$PRE_DIR/run.py" --input "$INPUT" "$@"
