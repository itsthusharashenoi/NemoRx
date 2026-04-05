# Witch Hunt — local Indic STT + React UI

**Remote:** [github.com/itsthusharashenoi/witch-hunt](https://github.com/itsthusharashenoi/witch-hunt)

Monorepo layout:

| Path | Purpose |
|------|---------|
| `vexyl-stt/` | [VEXYL-STT](https://github.com/vexyl-ai/vexyl-stt) server (patched for Python 3.9 + optional offline load). |
| `vexyl-stt-ui/` | Vite + React client: **VEXYL** (Indic streaming) or **Multilingual** (Whisper after stop). |
| `whisper-pipeline/` | FastAPI + **faster-whisper**: auto language + per-segment language guesses (English, Indian langs, code-mix). |
| `start-local.sh` | Starts VEXYL STT then the UI (waits for `/health`). |

## Quick start

1. **Hugging Face:** request access to `ai4bharat/indic-conformer-600m-multilingual`, create a read token.
2. **Secrets:** `cp vexyl-stt/.env.secrets.example vexyl-stt/.env.secrets` and set `HF_TOKEN=...` (file is gitignored).
3. **Model:** `cd vexyl-stt && ./download-model.sh`
4. **Run (VEXYL mode):** from repo root, `./start-local.sh` (or `vexyl-stt/./run.sh` + `vexyl-stt-ui/npm run dev`).

5. **Multilingual mode (English + Indian + mix):** install [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg`). Then in a second terminal: `cd whisper-pipeline && ./run.sh` (first start downloads the Whisper model). In the UI choose **Multilingual (Whisper)**. On stop, the app POSTs the recording to `http://127.0.0.1:8092/transcribe` and saves `transcript-*.txt`, `transcript-*.json` (segments with `language` / `language_label`), and the audio file.

Server defaults: VEXYL `ws://127.0.0.1:8091`, pipeline `http://127.0.0.1:8092`. Override with `VITE_VEXYL_WS_URL` and `VITE_WHISPER_PIPELINE_URL` in `vexyl-stt-ui/.env` (see `.env.example`).

**Env (Whisper):** `WHISPER_MODEL_SIZE` (default `small`), `WHISPER_DEVICE=cpu`, `WHISPER_COMPUTE_TYPE=int8`. Larger models are more accurate but slower on CPU.

## Offline use

After the model is cached, uncomment `HF_HUB_OFFLINE=1` in `vexyl-stt/.env` (see comments there) so startup does not call the Hugging Face Hub.

## License

`vexyl-stt/` retains upstream licensing (Apache 2.0). UI code in `vexyl-stt-ui/` is MIT unless you specify otherwise.
