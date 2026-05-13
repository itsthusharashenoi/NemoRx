"""
Normalize raw doctor–patient conversation text for the rule-based NLP pipeline.
"""

from __future__ import annotations

import re
from typing import Tuple

# Basic Devanagari block detection
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def contains_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text))


def transliterate_devanagari_to_roman(text: str) -> Tuple[str, bool]:
    """
    Returns (text, applied). If indic-transliteration is missing, returns original.
    """
    if not contains_devanagari(text):
        return text, False
    try:
        from indic_transliteration.sanscript import DEVANAGARI, ITRANS, transliterate
    except ImportError:
        return text, False
    try:
        out = transliterate(text, DEVANAGARI, ITRANS)
        return out, True
    except Exception:
        return text, False


_SPEAKER_LINE = re.compile(
    r"^\s*(?:doctor|dr\.?|patient|nurse|pt\.?|p\.?)\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)


def strip_speaker_prefixes(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lines.append(_SPEAKER_LINE.sub("", line).strip())
    return "\n".join(lines)


def merge_conversation(text: str) -> str:
    """Collapse excessive blank lines; preserve paragraph breaks."""
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def clinical_slice_boost(text: str) -> str:
    """
    If the block is very long, prepend sentences that look like prescribing cues
    so drug context windows still match (light heuristic, fully offline).
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 12:
        return text
    cue = re.compile(
        r"(?i)\b(tab|cap|syrup|syp|mg|ml|bd|tds|od|hs|ac|pc|"
        r"after food|before food|din|dino|hafte|len|lena|dena|tablet|capsule)\b"
    )
    priority: list[str] = []
    rest: list[str] = []
    for ln in lines:
        if cue.search(ln):
            priority.append(ln)
        else:
            rest.append(ln)
    if not priority or len(priority) == len(lines):
        return text
    merged = "\n".join(priority + [""] + rest)
    return merged


def prepare_conversation_text(raw: str) -> tuple[str, list[str]]:
    """
    Full offline prep. Returns (text_for_nlp, notes) where notes are non-fatal flags
    for the template (e.g. transliteration skipped).
    """
    notes: list[str] = []
    t = merge_conversation(raw)
    t = strip_speaker_prefixes(t)
    t, did_tr = transliterate_devanagari_to_roman(t)
    if contains_devanagari(raw) and not did_tr:
        notes.append(
            "INFO: Devanagari detected but transliteration unavailable "
            "(install indic-transliteration or check text)."
        )
    t = clinical_slice_boost(t)
    return t, notes
