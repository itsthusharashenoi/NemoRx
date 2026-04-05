# Witch Hunt — local Indic STT + React UI

Monorepo layout:

| Path | Purpose |
|------|---------|
| `vexyl-stt/` | [VEXYL-STT](https://github.com/vexyl-ai/vexyl-stt) server (patched for Python 3.9 + optional offline load). |
| `vexyl-stt-ui/` | Vite + React client: mic toggle, WebSocket PCM streaming, transcript + recording downloads. |
| `start-local.sh` | Starts STT then the UI (waits for `/health`). |

## Quick start

1. **Hugging Face:** request access to `ai4bharat/indic-conformer-600m-multilingual`, create a read token.
2. **Secrets:** `cp vexyl-stt/.env.secrets.example vexyl-stt/.env.secrets` and set `HF_TOKEN=...` (file is gitignored).
3. **Model:** `cd vexyl-stt && ./download-model.sh`
4. **Run:** from repo root, `./start-local.sh` (or `vexyl-stt/./run.sh` + `vexyl-stt-ui/npm run dev`).

Server defaults: `ws://127.0.0.1:8091`. Override with `VITE_VEXYL_WS_URL` in `vexyl-stt-ui/.env` (copy from `.env.example`).

## Offline use

After the model is cached, uncomment `HF_HUB_OFFLINE=1` in `vexyl-stt/.env` (see comments there) so startup does not call the Hugging Face Hub.

## License

`vexyl-stt/` retains upstream licensing (Apache 2.0). UI code in `vexyl-stt-ui/` is MIT unless you specify otherwise.
