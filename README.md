# Docscribe — terminal capture + Gemini transcription

**Primary workflow:** record from the microphone in the terminal, stop with **Ctrl+C**, then get:

- **`recordings/gemini-*.wav`** (initially) — raw audio  
- **`Transcriptis/gemini-*.txt`** — transcript (Doc/Patient style by default, or plain if `GEMINI_VERBATIM=1`)

After transcription, a short follow-up call asks Gemini for the **patient’s name** when it is clear in the text. If found, the **same** transcript is written to **`<patient-name>-<timestamp>.txt`**, then the **`gemini-<timestamp>.txt`** file is **deleted**. The WAV is still **moved** to **`<patient-name>-<timestamp>.wav`** (same stem). If the name is unclear, only **`gemini-<timestamp>.txt`** remains.

Uses **Google Gemini 2.5 Flash** over the network. Transcription is tuned for **any language** (including Indian languages, English, and code-switching). **Internet is required** for this path.

**Remote:** [github.com/itsthusharashenoi/DocScribe](https://github.com/itsthusharashenoi/DocScribe)

## Quick start (terminal only)

1. **Install** [ffmpeg](https://ffmpeg.org/) (e.g. `brew install ffmpeg`).
2. **API key:** put your Google AI key in **`scripts/.env.secrets`**:
   ```bash
   cp scripts/.env.secrets.example scripts/.env.secrets
   # edit: GEMINI_API_KEY=...
   ```
3. **Run** from repo root:
   ```bash
   ./scripts/gemini-record-transcribe.sh
   ```
   Speak, then **Ctrl+C** when finished. Outputs appear under `recordings/` and `Transcriptis/`.

### Environment variables

| Variable | Meaning |
|----------|---------|
| `GEMINI_VERBATIM=1` | Plain continuous text only (no Doc/Patient). |
| `LANGUAGE=hi-IN` | Optional soft locale hint (default `auto` = any language). |
| `GEMINI_MODEL` | Override model id (default `gemini-2.5-flash`). |
| `SKIP_NETWORK_CHECK=1` | Skip connectivity probe before calling Gemini. |
| `GEMINI_JSON_RESPONSE=0` | Doc/Patient mode: disable JSON MIME hint if your API returns 400. |
| `FFMPEG_AUDIO_DEVICE` | macOS only, e.g. `:1` if default `:0` is wrong. |
| `GEMINI_SKIP_PATIENT_RENAME=1` | Keep `gemini-<timestamp>.*` names (skip name extraction + rename). |
| `GEMINI_EXTRACT_MODEL` | Model for the patient-name line (default: same as `GEMINI_MODEL`). |

## Optional: local offline STT (`vexyl-stt/`)

The **`vexyl-stt/`** directory is a patched [VEXYL-STT](https://github.com/vexyl-ai/vexyl-stt) server (Indic Conformer) for **offline / no-Google** use later. It is **not** required for the terminal Gemini flow above.

## Optional: `vexyl-stt-ui/` (browser UI)

Legacy Vite + React client. **Not part of the recommended workflow** for this project.

## License

`vexyl-stt/` retains upstream licensing (Apache 2.0). Other project files unless noted are MIT.
