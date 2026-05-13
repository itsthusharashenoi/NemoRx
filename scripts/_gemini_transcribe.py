#!/usr/bin/env python3
"""
Transcribe one audio file with Google Gemini (generateContent). Used by gemini-record-transcribe.sh.
API key: environment, then scripts/.env.secrets, then vexyl-stt/.env.secrets.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


def _parse_key_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("GEMINI_API_KEY=") or line.startswith("GOOGLE_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _load_api_key(repo_root: Path, script_dir: Path) -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    for secrets in (script_dir / ".env.secrets", repo_root / "vexyl-stt" / ".env.secrets"):
        got = _parse_key_file(secrets)
        if got:
            return got
    print(
        "No API key: export GEMINI_API_KEY, or add GEMINI_API_KEY=... to "
        "scripts/.env.secrets or vexyl-stt/.env.secrets",
        file=sys.stderr,
    )
    sys.exit(2)


def _mime(path: Path) -> str:
    s = path.suffix.lower()
    return {
        ".wav": "audio/wav",
        ".webm": "audio/webm",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }.get(s, "application/octet-stream")


def _verbatim_prompt(language_code: str) -> str:
    code = (language_code or "").strip().lower()
    multilingual = (
        "You are an expert multilingual transcriptionist. Transcribe every language in the audio "
        "exactly as spoken—including any mix of Indian languages, English, Arabic, Chinese, "
        "Japanese, Korean, European languages, or others. Support code-switching within a sentence. "
        "Use the normal native script for each language. Never rewrite everything into English unless "
        "the speech is only English. Do not translate; do not add commentary."
    )
    if code in ("", "auto", "any", "multi", "*"):
        hint = multilingual
    else:
        hint = (
            multilingual
            + f" Optional hint only when ambiguous: locale `{language_code}` "
            "(still transcribe any other language spoken)."
        )
    return (
        f"{hint}\n\n"
        "Task: transcribe all speech in the attached audio.\n"
        "Output rules: return ONLY the spoken words as plain text. "
        "No timestamps, no speaker labels, no preamble or markdown."
    )


def _conversation_prompt(language_code: str) -> str:
    """Ask Gemini for Doc / Patient / Voice N turns (multilingual)."""
    code = (language_code or "").strip().lower()
    multilingual = (
        "You are an expert multilingual transcriptionist. Each speaker may use any language or mix "
        "(Indian languages, English, or others) with correct script. Do not translate."
    )
    if code in ("", "auto", "any", "multi", "*"):
        hint = multilingual
    else:
        hint = multilingual + f" Optional hint when ambiguous: `{language_code}`."
    return (
        f"{hint}\n\n"
        "You are a medical documentation assistant. The recording may contain two or more people. "
        "Identify distinct speakers from voice, turn-taking, and dialogue. Label the clinical "
        "professional **Doc** and the person receiving care **Patient**. Further speakers in order of "
        "first appearance: **Voice 1**, **Voice 2**.\n"
        "Return a JSON object with keys segments (array of {speaker, text}) and document (string). "
        "The document is plain text: each turn starts with Speaker: then the words, blank line "
        "between turns, chronological order, matching segments."
    )


def _strip_json_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _format_doc_from_segments(segments: list[dict]) -> tuple[str, str]:
    """Plain document + simple Markdown."""
    lines_txt: list[str] = []
    lines_md: list[str] = []
    for seg in segments:
        sp = str(seg.get("speaker", "Speaker")).strip() or "Speaker"
        tx = str(seg.get("text", "")).strip()
        if not tx:
            continue
        lines_txt.append(f"{sp}:")
        lines_txt.append(tx)
        lines_txt.append("")
        lines_md.append(f"### {sp}\n\n{tx}\n\n")
    return "\n".join(lines_txt).strip(), "".join(lines_md).strip()


def _parse_conversation_response(raw: str) -> tuple[str, str]:
    """Returns (document_plain, markdown)."""
    s = _strip_json_fence(raw)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return raw.strip(), f"### Transcript\n\n{raw.strip()}\n"
    if not isinstance(obj, dict):
        return raw.strip(), f"### Transcript\n\n{raw.strip()}\n"
    segs = obj.get("segments")
    doc = str(obj.get("document", "")).strip()
    doc_md = str(obj.get("document_md", "")).strip()
    if isinstance(segs, list) and segs:
        cleaned: list[dict] = []
        for item in segs:
            if not isinstance(item, dict):
                continue
            sp = str(item.get("speaker", "Speaker")).strip() or "Speaker"
            tx = str(item.get("text", "")).strip()
            if tx:
                cleaned.append({"speaker": sp, "text": tx})
        if cleaned:
            fd, fm = _format_doc_from_segments(cleaned)
            if not doc:
                doc = fd
            if not doc_md:
                doc_md = fm
    if not doc:
        doc = raw.strip()
    if not doc_md:
        doc_md = f"### Transcript\n\n{doc}\n"
    return doc, doc_md


def _call_gemini(
    api_key: str,
    model: str,
    prompt: str,
    mime: str,
    audio: bytes,
    generation_config: Optional[dict] = None,
) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    gen: dict = {
        "temperature": float(os.environ.get("GEMINI_TEMPERATURE", "0.12")),
        "maxOutputTokens": int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "65536")),
    }
    if generation_config:
        gen.update(generation_config)
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime,
                            "data": base64.standard_b64encode(audio).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": gen,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"Gemini HTTP {e.code}: {err}") from e
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    cands = payload.get("candidates") or []
    if not cands:
        raise RuntimeError("No candidates in Gemini response")
    parts = (cands[0].get("content") or {}).get("parts") or []
    texts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
    if not texts:
        raise RuntimeError("No text in Gemini response")
    return "\n".join(texts).strip()


def _call_gemini_text_only(
    api_key: str,
    model: str,
    prompt: str,
    generation_config: Optional[dict] = None,
) -> str:
    """Single-turn text-only generateContent (no audio)."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    gen: dict = {
        "temperature": 0.0,
        "maxOutputTokens": int(os.environ.get("GEMINI_EXTRACT_MAX_TOKENS", "256")),
    }
    if generation_config:
        gen.update(generation_config)
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": gen,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"Gemini HTTP {e.code}: {err}") from e
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    cands = payload.get("candidates") or []
    if not cands:
        raise RuntimeError("No candidates in Gemini response")
    parts = (cands[0].get("content") or {}).get("parts") or []
    texts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
    if not texts:
        raise RuntimeError("No text in Gemini response")
    return "\n".join(texts).strip()


def _slug_for_filename(name: str) -> str:
    """Safe basename fragment (no path separators or OS-forbidden chars)."""
    s = (name or "").strip()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    bad = set('<>:"/\\|?*')
    s = "".join(c for c in s if ord(c) >= 32 and c not in bad)
    s = s.strip(". ")
    return s[:120] if s else ""


def _extract_patient_slug(transcript: str, api_key: str, model: str) -> str:
    """Ask Gemini for a one-line patient label; returns slug or 'unknown'."""
    t = (transcript or "").strip()
    if len(t) < 10:
        return "unknown"
    extract_model = os.environ.get("GEMINI_EXTRACT_MODEL", model).strip() or model
    prompt = (
        "You read a clinical transcript (it may include Doc: / Patient: / Voice labels).\n"
        "Task: extract the patient's full name ONLY if it is clearly stated or clearly "
        "identifiable as the patient (e.g. self-introduction, staff addressing them by name).\n"
        "Rules:\n"
        "- Reply with exactly ONE line: the name using letters/spaces/hyphens only. "
        "If names are in another script, use a sensible ASCII transliteration.\n"
        "- Prefer given name and family name when both appear.\n"
        "- If the patient name cannot be determined from the transcript, reply exactly: unknown\n"
        "- Do not invent names. Do not include titles like Dr. for the patient.\n\n"
        "Transcript:\n---\n"
        f"{t[:120000]}\n---"
    )
    try:
        line = _call_gemini_text_only(api_key, extract_model, prompt).splitlines()[0].strip()
    except Exception:
        return "unknown"
    low = line.lower()
    if not line or low == "unknown" or low.startswith("unknown "):
        return "unknown"
    slug = _slug_for_filename(line)
    if not slug or slug == "unknown":
        return "unknown"
    return slug


def _stamp_from_gemini_basename(path: Path) -> Optional[str]:
    m = re.match(r"^gemini-(.+)\.wav$", path.name, flags=re.IGNORECASE)
    return m.group(1) if m else None


def _unique_wav_txt_pair(
    wav_dir: Path,
    txt_dir: Path,
    stem: str,
    wav_suffix: str,
) -> tuple[Path, Path]:
    """Paths for wav and txt with the same stem; add -2, -3 if either exists."""
    for n in range(0, 1000):
        s = stem if n == 0 else f"{stem}-{n}"
        w = wav_dir / f"{s}{wav_suffix}"
        t = txt_dir / f"{s}.txt"
        if not w.exists() and not t.exists():
            return w, t
    return wav_dir / f"{stem}{wav_suffix}", txt_dir / f"{stem}.txt"


def _maybe_rename_with_patient(
    audio_path: Path,
    out_path: Path,
    stamp: str,
    doc_plain_for_extract: str,
    txt_body: str,
    api_key: str,
    model: str,
) -> tuple[Path, Path]:
    """If patient slug is known: move wav to {slug}-{stamp}.wav; write patient .txt then remove gemini .txt."""
    if os.environ.get("GEMINI_SKIP_PATIENT_RENAME", "").lower() in ("1", "true", "yes"):
        return audio_path, out_path
    if not stamp.strip():
        return audio_path, out_path
    slug = _extract_patient_slug(doc_plain_for_extract, api_key, model)
    if slug == "unknown":
        return audio_path, out_path
    new_base = f"{slug}-{stamp}"
    new_wav, new_txt = _unique_wav_txt_pair(
        audio_path.parent,
        out_path.parent,
        new_base,
        audio_path.suffix,
    )
    try:
        audio_path.rename(new_wav)
    except OSError:
        return audio_path, out_path
    try:
        new_txt.write_text(txt_body, encoding="utf-8")
    except OSError:
        try:
            new_wav.rename(audio_path)
        except OSError:
            pass
        return audio_path, out_path
    try:
        out_path.unlink()
    except OSError as e:
        print(f"Warning: could not remove gemini transcript {out_path}: {e}", file=sys.stderr)
    return new_wav, new_txt


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe audio with Gemini 2.5 Flash")
    ap.add_argument("audio", type=Path, help="Input audio file")
    ap.add_argument("output", type=Path, help="Output transcript (.txt)")
    ap.add_argument(
        "--language",
        default="auto",
        help="auto (default) or e.g. hi-IN as soft hint",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root (parent of vexyl-stt); default: parent of this script's directory",
    )
    ap.add_argument(
        "--conversation",
        action="store_true",
        help="Doc/Patient/Voice N to-and-fro document (plain .txt)",
    )
    ap.add_argument(
        "--stamp",
        default="",
        help="Timestamp suffix used when renaming to <patient>-<stamp>.* (from bash date stamp)",
    )
    args = ap.parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = args.repo_root or script_dir.parent

    audio_path: Path = args.audio
    out_path: Path = args.output
    if not audio_path.is_file():
        print(f"Not a file: {audio_path}", file=sys.stderr)
        sys.exit(1)
    data = audio_path.read_bytes()
    if len(data) < 200:
        print("Audio file too small", file=sys.stderr)
        sys.exit(1)
    max_b = int(os.environ.get("GEMINI_MAX_BYTES", str(20 * 1024 * 1024)))
    if len(data) > max_b:
        print(f"Audio exceeds {max_b} bytes; shorten recording or raise GEMINI_MAX_BYTES", file=sys.stderr)
        sys.exit(1)

    mime = _mime(audio_path)
    if mime == "application/octet-stream":
        print(f"Unsupported extension {audio_path.suffix}", file=sys.stderr)
        sys.exit(1)

    api_key = _load_api_key(repo_root, script_dir)
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
    prompt = (
        _conversation_prompt(args.language)
        if args.conversation
        else _verbatim_prompt(args.language)
    )

    gen: dict = {
        "temperature": float(os.environ.get("GEMINI_TEMPERATURE", "0.12")),
        "maxOutputTokens": int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "65536")),
    }

    try:
        if args.conversation and os.environ.get("GEMINI_JSON_RESPONSE", "1").lower() not in (
            "0",
            "false",
            "no",
        ):
            try:
                raw = _call_gemini(
                    api_key,
                    model,
                    prompt,
                    mime,
                    data,
                    {**gen, "responseMimeType": "application/json"},
                )
            except RuntimeError as e:
                if "400" in str(e) or "INVALID_ARGUMENT" in str(e):
                    print(
                        "Note: Gemini JSON response mode not accepted; retrying with default output.",
                        file=sys.stderr,
                    )
                    raw = _call_gemini(api_key, model, prompt, mime, data, gen)
                else:
                    raise
        else:
            raw = _call_gemini(api_key, model, prompt, mime, data, gen)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = (args.stamp or "").strip() or (_stamp_from_gemini_basename(audio_path) or "")

    if args.conversation:
        doc_plain, _ = _parse_conversation_response(raw)
        txt_body = doc_plain + ("\n" if doc_plain and not doc_plain.endswith("\n") else "")
        out_path.write_text(txt_body, encoding="utf-8")
        print(f"Wrote gemini transcript: {out_path}")
        final_wav, final_txt = _maybe_rename_with_patient(
            audio_path, out_path, stamp, doc_plain, txt_body, api_key, model
        )
        if final_txt != out_path:
            print(f"Wrote patient transcript (gemini .txt removed): {final_txt}")
        if final_wav != audio_path:
            print(f"Renamed audio to: {final_wav}")
    else:
        txt_body = raw + ("\n" if raw and not raw.endswith("\n") else "")
        out_path.write_text(txt_body, encoding="utf-8")
        print(f"Wrote gemini transcript: {out_path}")
        doc_for_extract = raw.strip()
        final_wav, final_txt = _maybe_rename_with_patient(
            audio_path, out_path, stamp, doc_for_extract, txt_body, api_key, model
        )
        if final_txt != out_path:
            print(f"Wrote patient transcript (gemini .txt removed): {final_txt}")
        if final_wav != audio_path:
            print(f"Renamed audio to: {final_wav}")


if __name__ == "__main__":
    main()
