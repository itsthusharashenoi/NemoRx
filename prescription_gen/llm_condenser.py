"""
Optional LLM step: rewrite noisy dialogue into concise clinical prose.
Disabled when OPENAI_API_KEY is unset (fully offline).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def _openai_condense(clinical_text: str) -> str | None:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You convert doctor-patient dialogue into a short clinical note "
                    "in English or Hinglish (Roman script only). Preserve all drug names, "
                    "doses (mg/ml), frequencies (BD/TDS/OD), durations, investigations, "
                    "and follow-up. No bullet labels — plain prose only."
                ),
            },
            {"role": "user", "content": clinical_text},
        ],
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError):
        return None


def maybe_condense(text: str, use_llm: str = "auto") -> tuple[str, bool]:
    """
    use_llm: 'auto' | 'on' | 'off'
    Returns (text, used_llm).
    """
    mode = use_llm.lower().strip()
    if mode == "off":
        return text, False
    if mode == "on" and not os.environ.get("OPENAI_API_KEY", "").strip():
        return text, False
    if mode == "auto" and not os.environ.get("OPENAI_API_KEY", "").strip():
        return text, False
    out = _openai_condense(text)
    if out:
        return out, True
    return text, False
