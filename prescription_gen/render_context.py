"""
Map ParsedPrescription → plain dict for Jinja2 + confidence score (validation layer).
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from repo_paths import find_docscribe_akhil

_DOCS = find_docscribe_akhil()
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))

from nlp.medical_nlp import ParsedPrescription  # noqa: E402


def _confidence_score(flags: list[str], block_signoff: bool) -> int:
    """0–100 heuristic from validation flags."""
    score = 100
    for f in flags:
        if "BLOCK" in f.upper():
            score -= 25
        elif "WARNING" in f.upper():
            score -= 12
        elif "INFO" in f.upper():
            score -= 2
    if block_signoff:
        score -= 15
    return max(0, min(100, score))


def prescription_to_context(rx: ParsedPrescription) -> dict[str, Any]:
    meds: list[dict[str, Any]] = []
    for m in rx.medications:
        d = asdict(m)
        meds.append(d)

    return {
        "hospital_name": "City General Hospital",
        "hospital_tagline": "Out-Patient Department · Electronic Prescription",
        "logo_placeholder": True,
        "prescription_date": date.today().strftime("%d %b %Y"),
        "doctor_name": "Dr. ________________",
        "registration_no": "Reg. No. __________",
        "patient_name": "________________",
        "patient_age_sex": "____ / ____",
        "patient_id": "OPD ____________",
        "chief_complaint": rx.chief_complaint,
        "vitals": rx.vitals,
        "clinical_notes": rx.clinical_notes,
        "diagnosis": rx.diagnosis,
        "medications": meds,
        "investigations": rx.investigations,
        "follow_up": rx.follow_up or "—",
        "general_advice": rx.general_advice,
        "confidence_flags": rx.confidence_flags,
        "confidence_score": _confidence_score(rx.confidence_flags, rx.block_signoff),
        "block_signoff": rx.block_signoff,
        "raw_conversation_excerpt": (rx.raw_text[:1200] + "…")
        if len(rx.raw_text) > 1200
        else rx.raw_text,
    }
