"""
Post-parse cleanup: dedupe advice & chief complaints, infer missing medication frequencies
from instructions and from the raw transcript (e.g. Pan 40 + 'empty stomach in the morning' → OD).
"""

from __future__ import annotations

import re
from typing import Any


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _word_tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]{2,}", s.lower()))


def dedupe_general_advice(lines: list[str]) -> list[str]:
    """Drop redundant advice (substring or high word-overlap); preserve first-seen order."""
    raw = [str(x).strip() for x in lines if x and str(x).strip()]
    kept: list[str] = []
    for s in raw:
        sl = _norm(s)
        if len(sl) < 4:
            continue
        replaced = False
        for j, k in enumerate(kept):
            kl = _norm(k)
            if sl in kl:
                replaced = True  # s is redundant vs longer existing
                break
            if kl in sl and len(sl) > len(kl):
                kept[j] = s
                replaced = True
                break
            a, b = _word_tokens(s), _word_tokens(k)
            if a and b:
                jacc = len(a & b) / len(a | b)
                if jacc >= 0.55:
                    if len(sl) <= len(kl):
                        replaced = True
                    else:
                        kept[j] = s
                        replaced = True
                    break
        if not replaced:
            kept.append(s)
    return kept


def dedupe_chief_complaints(lines: list[str]) -> list[str]:
    """
    Merge overlapping symptom lines; prefer 'English (Regional)' over English-only duplicate.
    """
    out: list[str] = []
    for s in lines:
        s = (s or "").strip()
        if not s:
            continue
        ws = _word_tokens(s)
        merged = False
        for j, k in enumerate(out):
            wk = _word_tokens(k)
            if ws and wk and len(ws & wk) / len(ws | wk) >= 0.35:
                if "(" in s and ")" in s and "(" not in k:
                    out[j] = s
                elif "(" in k and ")" in k and "(" not in s:
                    pass
                elif len(s) > len(k):
                    out[j] = s
                merged = True
                break
            if ws and not wk and _norm(s) == _norm(k):
                out[j] = s
                merged = True
                break
        if not merged:
            out.append(s)
    return out


_MORNING_AC = re.compile(
    r"(?i)(?:empty\s+stomach|before\s+food|ac\b|khali\s+pet|khane\s+se\s+pehle).{0,50}"
    r"(?:morning|subah|mornings)|(?:morning|subah|mornings).{0,50}"
    r"(?:empty\s+stomach|before\s+food|khali\s+pet|khane\s+se\s+pehle)"
)

_HAS_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Roman Hindi often emitted by LLMs; replace when source transcript contained Devanagari Hindi.
_ROMAN_TO_DEV_PARENS: list[tuple[str, str]] = sorted(
    [
        (
            "Teen din ke baad agar bukhar ya kamzori bani rahe to dobara aayein",
            "तीन दिन के बाद अगर बुखार या कमजोरी बनी रहे तो दोबारा आएं",
        ),
        (
            "Teen din ke baad yadi bukhar ya kamzori bani rahe to dobara aayein",
            "तीन दिन के बाद यदि बुखार या कमजोरी बनी रहे तो दोबारा आएं",
        ),
        ("Teen din baad agar bukhar ya kamzori rahe", "तीन दिन के बाद अगर बुखार या कमजोरी बनी रहे"),
        ("Teen din se bukhar", "तीन दिन से बुखार"),
        ("Teen din se", "तीन दिन से"),
        ("Sar mein dard", "सर में दर्द"),
        ("Sar dard", "सर दर्द"),
        ("Khansi", "खाँसी"),
        ("Khana khane ke baad", "खाना खाने के बाद"),
        ("Khane ke baad", "खाने के बाद"),
        ("Khane se pehle", "खाने से पहले"),
        ("Khali pet", "खाली पेट"),
        ("Subah khali pet", "सुबह खाली पेट"),
        ("Bukhar", "बुखार"),
        ("Ulti", "उल्टी"),
    ],
    key=lambda t: len(t[0]),
    reverse=True,
)

_ROMAN_PAREN_FIX_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\(\s*{re.escape(rim)}\s*\)", re.I), f"({dev})")
    for rim, dev in _ROMAN_TO_DEV_PARENS
]


def _raw_has_devanagari(raw: str) -> bool:
    return bool(_HAS_DEVANAGARI.search(raw or ""))


def _fix_roman_hindi_parentheses(text: str) -> str:
    if not text:
        return text
    s = text
    for pat, repl in _ROMAN_PAREN_FIX_PATTERNS:
        s = pat.sub(repl, s)
    return s


def localize_medication_instructions_devanagari(rx: Any) -> None:
    """Append Devanagari timing gloss when the source conversation used Devanagari Hindi."""
    raw = getattr(rx, "raw_text", "") or ""
    if not _raw_has_devanagari(raw):
        return
    for med in getattr(rx, "medications", None) or []:
        ins = (getattr(med, "instructions", None) or "").strip()
        if re.fullmatch(r"(?i)after\s+food", ins) or ins == "After food":
            med.instructions = "After food (खाना खाने के बाद)"
        elif re.fullmatch(r"(?i)before\s+food", ins) or ins == "Before food":
            med.instructions = "Before food (खाली पेट / भोजन से पहले)"


def devanagari_gloss_pass(rx: Any) -> None:
    """When the conversation included Devanagari Hindi, swap common Roman-Hindi glosses in parentheses."""
    raw = getattr(rx, "raw_text", "") or ""
    if not _raw_has_devanagari(raw):
        return

    def fix_list(lines: list[str]) -> list[str]:
        return [_fix_roman_hindi_parentheses(x) for x in lines]

    rx.follow_up = _fix_roman_hindi_parentheses(rx.follow_up or "")
    rx.chief_complaint = fix_list(list(rx.chief_complaint))
    rx.diagnosis = fix_list(list(rx.diagnosis))
    rx.investigations = fix_list(list(rx.investigations))
    rx.general_advice = fix_list(list(rx.general_advice))
    rx.vitals = fix_list(list(getattr(rx, "vitals", None) or []))
    rx.clinical_notes = fix_list(list(getattr(rx, "clinical_notes", None) or []))
    for med in getattr(rx, "medications", None) or []:
        if getattr(med, "instructions", None):
            med.instructions = _fix_roman_hindi_parentheses(med.instructions)
        if getattr(med, "frequency_expanded", None):
            med.frequency_expanded = _fix_roman_hindi_parentheses(med.frequency_expanded)


_FU_COMPLEX = re.compile(
    r"(?is)(?:teen|3|three|३|तीन)\s*(?:din|दिन|dino|day|days).{0,320}"
    r"(?:bukhar|बुखार|fever).{0,320}"
    r"(?:weak|kamzori|कमजोरी|weekness|weakness|वीक)",
)
_FU_COMPLEX_DEV = re.compile(
    r"तीन\s*दिन.{0,320}बुखार.{0,320}(?:कमजोरी|कम\s*जोरी|weakness|weekness|weak\b|वीक)",
    re.UNICODE | re.I,
)


def enrich_follow_up_bilingual(rx: Any) -> None:
    """Set a clear review line when the transcript mentions 3-day fever/weakness follow-up (Hinglish / Devanagari)."""
    raw = getattr(rx, "raw_text", "") or ""
    if not raw.strip():
        return
    cur = (rx.follow_up or "").strip()
    if cur and _HAS_DEVANAGARI.search(cur) and re.search(r"\([^)]*[\u0900-\u097F][^)]*\)", cur):
        return
    low = raw.lower()
    if not (_FU_COMPLEX.search(raw) or _FU_COMPLEX.search(low) or _FU_COMPLEX_DEV.search(raw)):
        return
    rx.follow_up = (
        "Review after 3 days if fever or weakness persists "
        "(तीन दिन के बाद यदि बुखार या कमजोरी बनी रहे तो दोबारा आएं; रक्त परीक्षण की आवश्यकता हो सकती है)"
    )


_BRAND_STEM = re.compile(r"^(.+?)\s+(\d+(?:\.\d+)?)\s*$", re.I)


def _med_brand_stem(name: str) -> str:
    n = (name or "").strip().lower()
    if not n:
        return ""
    m = _BRAND_STEM.match(n)
    if m:
        return re.sub(r"\s+", " ", m.group(1).strip())
    return re.sub(r"\s+", " ", n)


def _generic_bucket(med: Any) -> str:
    g = (getattr(med, "generic_name", None) or "").strip().lower()
    if g:
        return g
    return (getattr(med, "name", None) or "").strip().lower()


def _dose_source_rank(med: Any) -> int:
    return {
        "explicit": 4,
        "inferred_from_brand": 3,
        "standard_fallback": 1,
        "missing": 0,
    }.get((getattr(med, "dose_source", None) or "missing").lower(), 0)


def _name_dose_consistency_bonus(med: Any) -> int:
    name = (med.name or "").lower()
    dose = (getattr(med, "dose", None) or "").upper()
    m = re.search(r"\s+(\d+)\s*$", name.strip())
    if not m:
        return 0
    num = m.group(1)
    digits = re.sub(r"\D", "", dose)
    return 2 if num and num in digits else 0


def dedupe_medications_same_brand(rx: Any) -> None:
    """
    Collapse duplicate rows that refer to the same branded product (e.g. 'Dolo 650' + bare 'Dolo'
    from 'Dolo tabhi…' PRN phrasing in the transcript). Prefer explicit / brand-inferred dose over standard fallback.
    """
    meds = list(getattr(rx, "medications", None) or [])
    if len(meds) <= 1:
        rx.medications = meds
        return
    seen_keys: set[tuple[str, str]] = set()
    out: list[Any] = []
    for m in meds:
        key = (_generic_bucket(m), _med_brand_stem(m.name or ""))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        grp = [x for x in meds if (_generic_bucket(x), _med_brand_stem(x.name or "")) == key]
        if len(grp) == 1:
            out.append(grp[0])
            continue

        def score(med: Any) -> tuple[int, int]:
            return (
                _dose_source_rank(med) + _name_dose_consistency_bonus(med),
                len((med.name or "").strip()),
            )

        out.append(max(grp, key=score))
    rx.medications = out


def _infer_od_hs_from_blob(blob: str) -> tuple[str, str] | None:
    if not blob.strip():
        return None
    if _MORNING_AC.search(blob) or re.search(
        r"(?i)(?:once\s+(?:daily\s+)?in\s+the\s+morning|morning\s+only|ek\s+baar\s+subah)", blob
    ):
        return ("OD", "Once daily — morning, before food (AC)")
    if re.search(r"(?i)\b(?:bedtime|at\s+night|\bhs\b|raat|sone\s+se\s+pehle)\b", blob):
        return ("HS", "At bedtime")
    return None


def infer_missing_frequencies(rx: Any) -> None:
    """Fill frequency from instructions / expanded text on each medication."""
    for med in rx.medications:
        if med.frequency:
            continue
        blob = f"{med.instructions} {med.frequency_expanded}".strip()
        hit = _infer_od_hs_from_blob(blob)
        if hit:
            med.frequency, med.frequency_expanded = hit[0], hit[1] or med.frequency_expanded


def infer_missing_frequencies_from_raw(rx: Any) -> None:
    """Second pass: doctor often lists several drugs in one sentence — scan near each drug name in raw_text."""
    raw = (getattr(rx, "raw_text", None) or "").lower()
    if not raw:
        return
    for med in rx.medications:
        if med.frequency:
            continue
        needles = [med.name, getattr(med, "generic_name", "") or ""]
        idx = -1
        for n in needles:
            n = (n or "").strip().lower()
            if len(n) < 2:
                continue
            idx = raw.find(n)
            if idx != -1:
                break
        if idx == -1:
            continue
        chunk = raw[max(0, idx - 40) : idx + 160]
        hit = _infer_od_hs_from_blob(chunk)
        if hit:
            med.frequency, med.frequency_expanded = hit[0], hit[1] or med.frequency_expanded


_BARE_PAIN_LINE = re.compile(r"(?i)^(pain|dard)(\s*\([^)]*\))?\s*$")


def refine_chief_drop_redundant_pain(lines: list[str]) -> list[str]:
    """Remove bare 'Pain' / 'Dard' lines when a specific complaint (e.g. headache) already covers it."""
    blob = " ".join(_norm(x) for x in lines)
    has_specific = bool(
        re.search(r"(?i)\b(headache|migraine|fever|cough|bodyache|body\s*ache|chest|abdominal)\b", blob)
        or ("sar" in blob and "dard" in blob)
        or ("sirdard" in blob.replace(" ", ""))
    )
    if not has_specific:
        return lines
    return [x for x in lines if x and not _BARE_PAIN_LINE.match(x.strip())]


def _vomiting_screening_denied(raw: str) -> bool:
    """Doctor asked about vomiting; patient denied — common false positive for chief_complaint."""
    t = (raw or "").lower()
    if not re.search(r"(?i)vomit|vomiting|\bulti\b", t):
        return False
    return bool(
        re.search(
            r"(?i)(?:nahi|nahin|no\b|not|never|deny|denies).{0,80}(?:vomit|vomiting|\bulti\b)|"
            r"(?:vomit|vomiting|\bulti\b).{0,80}(?:nahi|nahin|no\b|not|never)",
            t,
        )
    )


def filter_denied_symptoms_chief(rx: Any) -> None:
    """Drop chief-complaint lines for symptoms the patient denied in the transcript."""
    raw = getattr(rx, "raw_text", None) or ""
    if not raw.strip():
        return
    out: list[str] = []
    for line in rx.chief_complaint:
        s = (line or "").strip()
        if not s:
            continue
        sl = s.lower()
        if _vomiting_screening_denied(raw) and re.search(r"(?i)vomit|vomiting|\bulti\b|nausea", sl):
            continue
        out.append(s)
    rx.chief_complaint = out


def strip_stale_medication_warnings(rx: Any) -> None:
    """Remove frequency/dose warnings that no longer apply after inference merge."""
    meds = {m.name.strip().lower(): m for m in rx.medications}
    kept: list[str] = []
    for flag in rx.confidence_flags:
        m = re.match(r"^WARNING: Frequency missing for\s+(.+)$", flag, re.I)
        if m:
            key = m.group(1).strip().lower()
            med = meds.get(key)
            if med is None:
                for mk, mv in meds.items():
                    if mk in key or key in mk:
                        med = mv
                        break
            if med is not None and (med.frequency or "").strip():
                continue
        m2 = re.match(r"^WARNING: Dose missing for\s+(.+)$", flag, re.I)
        if m2:
            key = m2.group(1).strip().lower()
            med = meds.get(key)
            if med is None:
                for mk, mv in meds.items():
                    if mk in key or key in mk:
                        med = mv
                        break
            if med is not None and (med.dose or "").strip():
                continue
        kept.append(flag)
    rx.confidence_flags = kept


_MED_NAME_FLAG_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^INFO: Standard fallback dose used for\s+(.+?)\s*\(", re.I),
    re.compile(r"^INFO: Dose inferred from brand name for\s+(.+?)\s*\(", re.I),
)


def prune_validation_flags_for_absent_medication_names(rx: Any) -> None:
    """
    Drop INFO lines that name a medication no longer on the Rx (e.g. duplicate bare 'Dolo'
    removed after dedupe, leaving 'INFO: Standard fallback dose used for Dolo (500MG)').
    WARNING lines are left to strip_stale_medication_warnings (fuzzy name match).
    """
    meds = list(getattr(rx, "medications", None) or [])
    names = {(m.name or "").strip().lower() for m in meds if (m.name or "").strip()}
    if not names:
        return
    kept: list[str] = []
    for flag in list(rx.confidence_flags):
        drop = False
        for pat in _MED_NAME_FLAG_PATTERNS:
            mm = pat.match((flag or "").strip())
            if not mm:
                continue
            drug = (mm.group(1) or "").strip().lower()
            if drug and drug not in names:
                drop = True
            break
        if not drop:
            kept.append(flag)
    rx.confidence_flags = kept


def unique_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in lines:
        t = (x or "").strip()
        if not t:
            continue
        k = _norm(t)
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def polish_prescription(rx: Any) -> None:
    """In-place precision pass after rules + optional Gemini merge."""
    infer_missing_frequencies(rx)
    infer_missing_frequencies_from_raw(rx)
    dedupe_medications_same_brand(rx)
    strip_stale_medication_warnings(rx)
    localize_medication_instructions_devanagari(rx)
    filter_denied_symptoms_chief(rx)
    rx.chief_complaint = refine_chief_drop_redundant_pain(rx.chief_complaint)
    rx.chief_complaint = dedupe_chief_complaints(rx.chief_complaint)
    rx.diagnosis = dedupe_chief_complaints(rx.diagnosis)
    rx.investigations = unique_lines(rx.investigations)
    rx.general_advice = dedupe_general_advice(rx.general_advice)
    enrich_follow_up_bilingual(rx)
    devanagari_gloss_pass(rx)
    rx.vitals = unique_lines(getattr(rx, "vitals", None) or [])
    rx.clinical_notes = unique_lines(getattr(rx, "clinical_notes", None) or [])
    strip_stale_medication_warnings(rx)
    prune_validation_flags_for_absent_medication_names(rx)
