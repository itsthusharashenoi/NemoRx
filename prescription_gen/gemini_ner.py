"""
Optional Gemini NER: extract prescription fields as JSON, merged with rule-based parse.
Loads API key from prescription_gen/.env (see .env.example).
Default model: gemini-3-flash-preview (override GEMINI_MODEL).
Uses SDK with fallbacks (no timeout param, non-JSON mime, then REST+certifi like curl).

Uses ``google.generativeai`` (Google may migrate to ``google.genai``; swap SDK when needed).
"""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_PKG = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(_PKG / ".env")
except ImportError:
    pass

# Default: strong cost/performance balance for structured extraction (Google AI Studio).
_DEFAULT_MODEL = "gemini-3-flash-preview"

_JSON_INSTRUCTION = """You extract a concise OPD prescription from doctor–patient dialogue.
Typical pattern: doctor in English/Hinglish, patient in Hindi/Hinglish. The written prescription must be
easy for BOTH doctor and patient: use English first, add regional wording only in parentheses when the
patient (or doctor) used a non-English term for that same idea.

Return ONE JSON object only (no markdown). Shape:
{
  "chief_complaint": ["string"],
  "diagnosis": ["string"],
  "medications": [
    {
      "name": "string",
      "generic_name": "string",
      "dose": "string",
      "frequency": "OD|BD|TDS|QID|HS|SOS|STAT or empty",
      "frequency_expanded": "string",
      "duration": "string",
      "route": "Oral|Topical|Inhalation|Injection|Drops or empty",
      "instructions": "string",
      "category": "string"
    }
  ],
  "investigations": ["string"],
  "follow_up": "string",
  "general_advice": ["string"],
  "vitals": {
    "weight_kg": "numeric string or empty",
    "blood_pressure_mmhg": "e.g. 120/80 or empty",
    "spo2_percent": "e.g. 96 or empty",
    "temperature_c": "string or empty",
    "pulse_bpm": "string or empty",
    "respiratory_rate": "string or empty"
  },
  "clinical_notes": ["string"]
}

LANGUAGE & FORMAT (strict)
- Default line shape: **English (Hindi in Devanagari inside parentheses)** whenever the dialogue used Hindi
  (including Devanagari script in the transcript). **Never** put Hindi as Roman transliteration inside parentheses.
  Good: **"Fever for 3 days (तीन दिन से बुखार)"**, **"Cough (खाँसी)"**, **"Headache (सर में दर्द)"**,
  **"After food (खाना खाने के बाद)"** in medication instructions when Hindi was used.
  Wrong: **(Teen din se bukhar)**, **(Khansi)**, **(Khana khane ke baad)** — do not Romanize Hindi glosses.
- If the whole turn is already English with no regional synonym spoken, output plain English (no empty parentheses).
- diagnosis: same rule — e.g. "Viral URTI" or with Hindi gloss only in Devanagari if Hindi was spoken.
- investigations: short English labels; optional Hindi gloss in Devanagari in parentheses only if spoken that way.
- follow_up: one tight line; if you add a Hindi gloss, use Devanagari in parentheses (e.g. review / blood test advice).
- general_advice: include all distinct advice; **no paraphrase duplicates** (e.g. do not repeat both "rest" and "take rest" — keep one clear line).

CHIEF COMPLAINT (strict)
- One list entry per clinical concept. Never duplicate the same concept as two list items.
- Merge patient Hindi and doctor English into a single string per concept using **English (Devanagari Hindi gloss)** when Hindi applies.
- When the patient gives a **duration** (e.g. "तीन दिन से", "teen din se", "3 days", "for 5 days"), include it in that same line, e.g. **"Fever for 3 days (तीन दिन से बुखार)"**, **"Headache (सर में दर्द)"** — not a duration-free fever line if duration was stated.
- **Negated / screening negatives:** If the doctor only **asks** about a symptom and the patient **denies** it (e.g. "Any vomiting?" → "No" / "Nahi" / "Ulti nahi"), **do not** put that symptom in chief_complaint. Chief complaint = **positive** symptoms the patient is actually complaining about.
- **No redundant gloss:** Do not add a separate bullet that is only a generic word (e.g. bare "Dard" / "Pain") when another line already states the specific complaint (e.g. "Headache (Sar dard)"). Parentheses hold the patient's phrase for **that** symptom, not a duplicate concept.

MEDICATIONS (strict)
- **One row per distinct drug product** — e.g. do not output both "Dolo 650" and a separate bare "Dolo" for the same paracetamol course; merge to the strength the doctor stated (e.g. Dolo 650).
- Extract only drugs the doctor actually prescribed or confirmed.
- **frequency** is mandatory whenever the regimen is clear from speech, including colloquial phrasing:
  - "empty stomach in the morning", "morning before food", "once in the morning", "subah khali pet" → frequency **OD**,
    frequency_expanded "Once daily — morning, before food (AC)", instructions should reflect AC/morning empty stomach.
  - "at bedtime", "night", "sone se pehle" → **HS**.
  - "after food twice daily", "BD after meals", "subah sham" → **BD**; similar for TDS/OD/QID/SOS/STAT.
- If dose milligram is explicit in speech, put it in dose (e.g. 650MG). Do not invent strengths.
- instructions: short English; if a Hindi timing/route phrase was spoken, append **(Devanagari Hindi)** only, not Roman Hinglish.

GENERAL (strict)
- Omit empty strings; use [] for unknown lists. Do not add sections not grounded in the dialogue.
- Be precise: no filler, no duplicate advice, no duplicate symptoms.

VITALS & CLINICAL NOTES (when spoken)
- vitals: fill object keys only when the dialogue states a measurement (BP, weight, SpO₂, temp, pulse, RR).
  Use blood_pressure_mmhg like "120/80" (no unit suffix needed). spo2_percent as digits only unless "%" was explicit.
- clinical_notes: short objective lines only if stated (e.g. oximeter placed, vitals stable, exam setup). Do not invent vitals.
"""


def _gemini_configured() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def _normalize_token(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def _list_union_existing_first(existing: list[str], additions: list[str]) -> list[str]:
    out = list(existing)
    seen = {_normalize_token(x) for x in out if x}
    for a in additions or []:
        if not isinstance(a, str):
            continue
        t = a.strip()
        if not t:
            continue
        k = _normalize_token(t)
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


_BRAND_STEM_TRAIL = re.compile(r"^(.+?)\s+(\d+(?:\.\d+)?)\s*$", re.I)


def _brand_stem_for_med_key(name: str) -> str:
    """e.g. 'Dolo 650' -> 'dolo' so Gemini does not add duplicate bare 'Dolo' for same generic."""
    s = (name or "").strip().lower()
    if not s:
        return ""
    m = _BRAND_STEM_TRAIL.match(s)
    if m:
        return _normalize_token(m.group(1))
    return _normalize_token(s)


def _med_key(name: str, generic: str) -> str:
    stem = _brand_stem_for_med_key(name)
    return _normalize_token(f"{stem}|{generic or name}")


def _debug(msg: str) -> None:
    if os.environ.get("GEMINI_DEBUG", "").strip() in ("1", "true", "yes"):
        print(f"[gemini_ner] {msg}", flush=True)


def _extract_response_text(resp: Any) -> str:
    """
    ``response.text`` raises ValueError when there is no aggregated text
    (blocked output, missing candidates, or SDK quirks with JSON mode).
    Always walk candidates.parts as a fallback — matches what REST returns.
    """
    try:
        t = getattr(resp, "text", None)
        if t:
            return str(t).strip()
    except ValueError as e:
        _debug(f"resp.text unavailable: {e}")
    chunks: list[str] = []
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            continue
        for part in parts:
            txt = getattr(part, "text", None)
            if txt:
                chunks.append(str(txt))
    return "".join(chunks).strip()


def _strip_json_fences(raw: str) -> str:
    s = raw.strip()
    if not s.startswith("```"):
        return s
    s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    text = _strip_json_fences(text)
    if not text:
        return None
    try:
        out = json.loads(text)
        if isinstance(out, dict):
            return out
        if isinstance(out, list) and out and isinstance(out[0], dict):
            return out[0]
        return None
    except json.JSONDecodeError as e:
        _debug(f"JSON decode error: {e}; snippet: {text[:240]!r}")
        return None


def _ssl_context():
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


def _rest_gemini_generate_json(api_key: str, model_name: str, prompt: str, timeout_s: float) -> tuple[str, str]:
    """
    Same wire format as curl to generativelanguage.googleapis.com.
    Returns (response_text, error_message). error_message empty on HTTP success with body.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={urllib.parse.quote(api_key, safe='')}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.15,
            "responseMimeType": "application/json",
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ctx = _ssl_context()
    opener = (
        urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        if ctx
        else urllib.request.build_opener()
    )
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")[:800]
        except Exception:
            detail = str(e)
        return "", f"HTTP {e.code}: {detail}"
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        return "", f"Network/SSL error: {reason!r}"
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"

    try:
        outer = json.loads(raw)
    except json.JSONDecodeError as e:
        return "", f"Invalid JSON from API: {e}"

    cands = outer.get("candidates") or []
    if not cands:
        fb = outer.get("promptFeedback") or outer.get("error")
        return "", f"No candidates in response: {fb!r}"[:500]

    parts = (cands[0].get("content") or {}).get("parts") or []
    text = "".join((p.get("text") or "") for p in parts if isinstance(p, dict))
    return text.strip(), ""


def _sdk_generate(
    genai: Any,
    model_name: str,
    prompt: str,
    *,
    use_json_mime: bool,
    use_request_timeout: bool,
) -> tuple[Any | None, str]:
    """Returns (response, err). err empty on success."""
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "").strip())
    gcfg: dict[str, Any] = {"temperature": 0.15}
    if use_json_mime:
        gcfg["response_mime_type"] = "application/json"
    model = genai.GenerativeModel(model_name=model_name, generation_config=gcfg)
    try:
        timeout_ms = int(os.environ.get("GEMINI_TIMEOUT_MS", "120000"))
        timeout_s = max(30.0, timeout_ms / 1000.0)
        if use_request_timeout:
            resp = model.generate_content(prompt, request_options={"timeout": timeout_s})
        else:
            resp = model.generate_content(prompt)
        return resp, ""
    except TypeError as e:
        # Older clients may reject request_options
        if use_request_timeout:
            return _sdk_generate(genai, model_name, prompt, use_json_mime=use_json_mime, use_request_timeout=False)
        return None, f"TypeError: {e}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _gemini_extract_json_with_reason(clinical_text: str) -> tuple[dict[str, Any] | None, str]:
    """
    Returns (payload, error_reason).
    On success: (dict, "") — dict may be empty {}.
    On failure: (None, short human-readable reason for UI / logs).
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None, "GEMINI_API_KEY not set"
    model_name = os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

    try:
        import google.generativeai as genai
    except ImportError as e:
        return None, f"google-generativeai not installed: {e}"

    prompt = _JSON_INSTRUCTION + "\n\n--- INPUT ---\n" + clinical_text.strip()
    timeout_ms = int(os.environ.get("GEMINI_TIMEOUT_MS", "120000"))
    timeout_s = max(45.0, timeout_ms / 1000.0)

    # --- Pass 1: SDK + JSON mime + timeout
    resp, err = _sdk_generate(genai, model_name, prompt, use_json_mime=True, use_request_timeout=True)
    if resp is not None:
        text = _extract_response_text(resp)
        parsed = _parse_json_payload(text) if text else None
        if parsed is not None:
            return parsed, ""
        if text:
            return None, f"SDK returned text but JSON parse failed (first 120 chars): {text[:120]!r}"
        # empty text — try fallbacks below
        _debug(f"SDK pass1 empty text; err={err!r}")

    # --- Pass 2: SDK + JSON mime, no request_options timeout
    resp2, err2 = _sdk_generate(genai, model_name, prompt, use_json_mime=True, use_request_timeout=False)
    if resp2 is not None:
        text = _extract_response_text(resp2)
        parsed = _parse_json_payload(text) if text else None
        if parsed is not None:
            return parsed, ""
        if text:
            return None, f"SDK (no timeout) JSON parse failed: {text[:120]!r}"

    # --- Pass 3: SDK without JSON mime — model returns prose; parse JSON substring
    resp3, err3 = _sdk_generate(genai, model_name, prompt, use_json_mime=False, use_request_timeout=False)
    if resp3 is not None:
        text = _extract_response_text(resp3)
        parsed = _parse_json_payload(text) if text else None
        if parsed is not None:
            return parsed, ""
        if text:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                parsed = _parse_json_payload(m.group(0))
                if parsed is not None:
                    return parsed, ""

    # --- Pass 4: REST (curl-equivalent), fixes macOS Python SSL issues vs urllib defaults
    rest_text, rest_err = _rest_gemini_generate_json(key, model_name, prompt, timeout_s=timeout_s)
    if rest_text:
        parsed = _parse_json_payload(rest_text)
        if parsed is not None:
            return parsed, ""
        return None, f"REST body JSON parse failed: {rest_text[:160]!r}"
    if rest_err:
        return None, f"REST fallback failed: {rest_err}"

    tail = "; ".join(x for x in (err, err2, err3) if x)
    return None, f"All Gemini attempts failed. SDK hints: {tail or 'no response text'}"


def _gemini_extract_json(clinical_text: str) -> dict[str, Any] | None:
    """Backward-compatible: payload only (no error detail)."""
    payload, _reason = _gemini_extract_json_with_reason(clinical_text)
    return payload


def _gemini_vitals_to_lines(v: Any) -> list[str]:
    """Turn Gemini vitals object or list of strings into display lines."""
    if v is None:
        return []
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            t = str(x).strip()
            if t and t.lower() not in ("null", "none", "n/a"):
                out.append(t)
        return out
    if not isinstance(v, dict):
        return []

    def sk(val: Any) -> str:
        s = str(val or "").strip()
        if not s or s.lower() in ("null", "none", "n/a"):
            return ""
        return s

    lines: list[str] = []
    w = sk(v.get("weight_kg") or v.get("weight"))
    if w:
        w_clean = re.sub(r"\s*kg\s*$", "", w, flags=re.I)
        lines.append(f"Weight — {w_clean} kg")

    bp = sk(v.get("blood_pressure_mmhg") or v.get("bp"))
    if bp:
        if "/" in bp and "mmhg" not in bp.lower():
            lines.append(f"BP — {bp} mmHg")
        else:
            lines.append(f"BP — {bp}")

    sp = sk(v.get("spo2_percent") or v.get("spo2"))
    if sp:
        sp = re.sub(r"\s*%\s*$", "", sp)
        lines.append(f"SpO₂ — {sp}%")

    tmp = sk(v.get("temperature_c") or v.get("temperature"))
    if tmp:
        if not re.search(r"[°CFcf]|celsius|fahrenheit", tmp, re.I):
            lines.append(f"Temperature — {tmp} °C")
        else:
            lines.append(f"Temperature — {tmp}")

    pulse = sk(v.get("pulse_bpm") or v.get("pulse"))
    if pulse:
        if "bpm" not in pulse.lower():
            lines.append(f"Pulse — {pulse} bpm")
        else:
            lines.append(f"Pulse — {pulse}")

    rr = sk(v.get("respiratory_rate") or v.get("rr"))
    if rr:
        if "/min" not in rr.lower() and "rpm" not in rr.lower():
            lines.append(f"Resp. rate — {rr}/min")
        else:
            lines.append(f"Resp. rate — {rr}")

    return lines


def merge_gemini_payload(rx: Any, payload: dict[str, Any] | None) -> bool:
    """
    Merges Gemini NER into ParsedPrescription in place. Returns True if any field changed.
    """
    if payload is None or not isinstance(payload, dict):
        return False

    from nlp.medical_nlp import AMBIGUOUS_STRENGTH_DRUGS, Medication  # noqa: WPS433

    changed = False

    add_cc = payload.get("chief_complaint") or []
    if isinstance(add_cc, list):
        new_cc = _list_union_existing_first(rx.chief_complaint, [str(x) for x in add_cc])
        if new_cc != rx.chief_complaint:
            rx.chief_complaint = new_cc
            changed = True

    add_dx = payload.get("diagnosis") or []
    if isinstance(add_dx, list):
        new_dx = _list_union_existing_first(rx.diagnosis, [str(x) for x in add_dx])
        if new_dx != rx.diagnosis:
            rx.diagnosis = new_dx
            changed = True

    add_inv = payload.get("investigations") or []
    if isinstance(add_inv, list):
        new_inv = _list_union_existing_first(rx.investigations, [str(x) for x in add_inv])
        if new_inv != rx.investigations:
            rx.investigations = new_inv
            changed = True

    add_adv = payload.get("general_advice") or []
    if isinstance(add_adv, list):
        new_adv = _list_union_existing_first(rx.general_advice, [str(x) for x in add_adv])
        if new_adv != rx.general_advice:
            rx.general_advice = new_adv
            changed = True

    gv = payload.get("vitals")
    v_lines = _gemini_vitals_to_lines(gv)
    if v_lines:
        new_v = _list_union_existing_first(rx.vitals, v_lines)
        if new_v != rx.vitals:
            rx.vitals = new_v
            changed = True

    cn = payload.get("clinical_notes")
    if isinstance(cn, list):
        new_cn = _list_union_existing_first(rx.clinical_notes, [str(x) for x in cn if str(x).strip()])
        if new_cn != rx.clinical_notes:
            rx.clinical_notes = new_cn
            changed = True

    fu = payload.get("follow_up")
    if isinstance(fu, str) and fu.strip() and not (rx.follow_up or "").strip():
        rx.follow_up = fu.strip()
        changed = True

    existing_keys = {_med_key(m.name, m.generic_name) for m in rx.medications}
    for raw in payload.get("medications") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        gen = str(raw.get("generic_name") or "").strip()
        k = _med_key(name, gen)
        if k in existing_keys:
            continue
        dose = str(raw.get("dose") or "").strip().upper().replace(" ", "")
        dose_src = "explicit" if dose else "missing"
        gnorm = (gen or name).lower().replace(" ", "").replace("-", "")
        ambiguous = False
        for amb in AMBIGUOUS_STRENGTH_DRUGS:
            amb_n = amb.lower().replace("+", "")
            if amb_n in gnorm or amb_n in name.lower().replace(" ", ""):
                ambiguous = True
                break
        if ambiguous and dose_src != "explicit":
            rx.block_signoff = True
            rx.confidence_flags.append(
                f"BLOCK_SIGNOFF: Gemini suggested {name} ({gen or 'unknown'}) "
                f"without explicit strength — confirm before signing."
            )
        med = Medication(
            name=name,
            generic_name=gen,
            dose=dose,
            dose_source=dose_src,
            frequency=str(raw.get("frequency") or "").strip(),
            frequency_expanded=str(raw.get("frequency_expanded") or "").strip(),
            duration=str(raw.get("duration") or "").strip(),
            route=str(raw.get("route") or "Oral").strip() or "Oral",
            instructions=str(raw.get("instructions") or "").strip(),
            category=str(raw.get("category") or "").strip(),
        )
        rx.medications.append(med)
        existing_keys.add(k)
        changed = True
        rx.confidence_flags.append(f"INFO: Medication from Gemini NER — verify: {name}")

    return changed


def maybe_apply_gemini_ner(rx: Any, conversation_text: str, use_gemini: str) -> bool:
    """
    use_gemini: 'auto' | 'on' | 'off'
    Returns True if Gemini merge changed the prescription.
    """
    mode = use_gemini.lower().strip()
    if mode == "off":
        return False
    if mode == "auto" and not _gemini_configured():
        return False
    if mode == "on" and not _gemini_configured():
        rx.confidence_flags.append(
            "WARNING: --use-gemini on but GEMINI_API_KEY is missing in prescription_gen/.env"
        )
        return False

    payload, reason = _gemini_extract_json_with_reason(conversation_text)
    model_label = os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

    if payload is None:
        if mode in ("on", "auto"):
            detail = reason if len(reason) <= 400 else reason[:397] + "..."
            rx.confidence_flags.append(f"WARNING: Gemini NER failed: {detail}")
        return False

    merged = merge_gemini_payload(rx, payload)
    if merged:
        msg = f"INFO: Gemini NER enrichment applied (model {model_label})."
        if msg not in rx.confidence_flags:
            rx.confidence_flags.append(msg)
    else:
        info = (
            f"INFO: Gemini NER ran (model {model_label}) but nothing new to merge "
            "(rules already captured the same entities)."
        )
        if info not in rx.confidence_flags:
            rx.confidence_flags.append(info)
    return merged
