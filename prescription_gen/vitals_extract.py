"""
Heuristic vitals extraction from Hindi/English/Hinglish clinic transcripts (offline).
"""

from __future__ import annotations

import re
from typing import List


def extract_vitals_lines(text: str) -> List[str]:
    """Return display lines like 'BP — 120/80 mmHg' when patterns match."""
    if not (text or "").strip():
        return []
    out: list[str] = []
    seen: set[str] = set()

    def add(line: str) -> None:
        k = line.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(line)

    # Blood pressure — label optional (Hindi / English)
    m = re.search(
        r"(?i)(?:bp|बीपी|blood\s*pressure)[^\d\n]{0,20}(\d{2,3})\s*/\s*(\d{2,3})",
        text,
    )
    if m:
        add(f"BP — {m.group(1)}/{m.group(2)} mmHg")
    else:
        m2 = re.search(
            r"(?<![\d/])(\d{2,3})\s*/\s*(\d{2,3})(?![\d/])",
            text,
        )
        if m2 and re.search(
            r"(?i)(?:normal|नॉर्म|सीम|seem|bp|बीपी|pressure|relax)",
            text[max(0, m2.start() - 30) : m2.end() + 40],
        ):
            add(f"BP — {m2.group(1)}/{m2.group(2)} mmHg")

    # Weight — वेट 68 किलो / weight 68 kg
    m = re.search(
        r"(?i)(?:weight|वेट|वजन)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(?:kg|किलो|kilos?)?",
        text,
    )
    if m:
        add(f"Weight — {m.group(1)} kg")
    else:
        m = re.search(r"(?i)(\d+(?:\.\d+)?)\s*(?:kg|किलो)\b", text)
        if m and re.search(r"(?i)(?:weight|वेट|वजन|weigh)", text[max(0, m.start() - 12) : m.end() + 4]):
            add(f"Weight — {m.group(1)} kg")

    # SpO2 / oxygen saturation
    m = re.search(
        r"(?i)(?:spo2|sp\s*o2|oxygen|ऑक्सी|saturation)[^\d\n]{0,25}(\d{2,3})\s*%",
        text,
    )
    if m:
        add(f"SpO₂ — {m.group(1)}%")
    else:
        # "ऑक्सीजन लेवल 96%" — allow Devanagari between label and digits
        m = re.search(r"(?i)(?:oxygen|ऑक्सीजन|ऑक्सी).{0,35}(\d{2,3})\s*%", text)
        if m:
            add(f"SpO₂ — {m.group(1)}%")

    # Temperature (optional)
    m = re.search(
        r"(?i)(?:temp|temperature|बुखार)[^\d\n]{0,12}(\d{2}(?:\.\d+)?)\s*(?:°?\s*[cf]|degree|deg\b)",
        text,
    )
    if m:
        add(f"Temperature — {m.group(1)}")

    return out


def extract_clinical_note_lines(text: str) -> List[str]:
    """Short non-vital exam context (oximeter placed, vitals stable) when clearly stated."""
    if not (text or "").strip():
        return []
    notes: list[str] = []
    if re.search(r"(?i)oximeter|ऑक्सीमीटर", text) and re.search(
        r"(?i)(?:normal|नॉर्म|96|97|98|99)\s*%?", text
    ):
        notes.append("Pulse oximetry performed (reading documented in vitals)")
    return notes[:3]
