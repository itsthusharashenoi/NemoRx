"""
Tests for prescription_gen pipeline (offline, no LLM).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_GEN = Path(__file__).resolve().parent.parent
if str(_GEN) not in sys.path:
    sys.path.insert(0, str(_GEN))
from repo_paths import find_docscribe_akhil  # noqa: E402

_DOCS = find_docscribe_akhil()
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))

from pipeline import parse_from_conversation  # noqa: E402
from prescription_polish import (  # noqa: E402
    filter_denied_symptoms_chief,
    infer_missing_frequencies_from_raw,
    strip_stale_medication_warnings,
)
from render_context import prescription_to_context  # noqa: E402
from vitals_extract import extract_vitals_lines  # noqa: E402


def test_parse_from_conversation_extracts_meds():
    raw = (_GEN / "tests" / "fixtures" / "sample_conversation.txt").read_text(encoding="utf-8")
    rx = parse_from_conversation(raw, use_llm="off", use_gemini="off")
    names = {m.name.lower() for m in rx.medications}
    assert "dolo 650" in names or "dolo" in names
    assert any("cbc" in inv.lower() for inv in rx.investigations)


def test_prescription_context_has_score():
    raw = "Tab Paracetamol 500mg OD for 3 days.\n"
    rx = parse_from_conversation(raw, use_llm="off", use_gemini="off")
    ctx = prescription_to_context(rx)
    assert "confidence_score" in ctx
    assert 0 <= ctx["confidence_score"] <= 100
    assert isinstance(ctx["medications"], list)


def test_devanagari_transliteration_path():
    raw = "डॉक्टर: बुखार है।\nDolo 650 BD khane ke baad."
    rx = parse_from_conversation(raw, use_llm="off", use_gemini="off")
    names = {m.name.lower() for m in rx.medications}
    assert "dolo 650" in names or "dolo" in names


def test_polish_infers_od_from_raw_transcript():
    from nlp.medical_nlp import Medication, ParsedPrescription  # noqa: WPS433

    rx = ParsedPrescription(
        raw_text="Also Pan 40 empty stomach in the morning for 5 days.",
        medications=[
            Medication(name="Pan 40", generic_name="Pantoprazole", dose="40MG", instructions="Before food")
        ],
    )
    infer_missing_frequencies_from_raw(rx)
    m = rx.medications[0]
    assert m.frequency == "OD"


def test_strip_stale_frequency_warning_after_infer():
    from nlp.medical_nlp import Medication, ParsedPrescription  # noqa: WPS433

    rx = ParsedPrescription(
        raw_text="Pan 40 empty stomach in the morning.",
        medications=[Medication(name="Pan 40", dose="40MG")],
        confidence_flags=["WARNING: Frequency missing for Pan 40"],
    )
    infer_missing_frequencies_from_raw(rx)
    strip_stale_medication_warnings(rx)
    assert not any("Frequency missing for Pan 40" in f for f in rx.confidence_flags)


def test_vitals_from_input_2_transcript_offline():
    raw = (_GEN / "input_2.txt").read_text(encoding="utf-8")
    rx = parse_from_conversation(raw, use_llm="off", use_gemini="off")
    assert rx.vitals
    joined = " ".join(rx.vitals).lower()
    assert "120/80" in joined
    assert "68" in joined
    assert "spo" in joined or "96" in joined
    ctx = prescription_to_context(rx)
    assert ctx["vitals"] == rx.vitals
    assert ctx.get("clinical_notes")


def test_input_2_dedupes_dolo_and_devanagari_follow_up():
    raw = (_GEN / "input_2.txt").read_text(encoding="utf-8")
    rx = parse_from_conversation(raw, use_llm="off", use_gemini="off")
    dolo_rows = [m for m in rx.medications if "dolo" in m.name.lower()]
    assert len(dolo_rows) == 1
    assert dolo_rows[0].dose == "650MG"
    assert rx.follow_up and "तीन दिन" in rx.follow_up and "रक्त" in rx.follow_up


def test_localize_after_food_devanagari_when_raw_hindi():
    from nlp.medical_nlp import Medication, ParsedPrescription  # noqa: WPS433

    from prescription_polish import localize_medication_instructions_devanagari  # noqa: E402

    rx = ParsedPrescription(
        raw_text="डॉक्टर: खाना खाने के बाद लीजिए।",
        medications=[Medication(name="Azithromycin 500", generic_name="Azithromycin", instructions="After food")],
    )
    localize_medication_instructions_devanagari(rx)
    assert "खाना" in (rx.medications[0].instructions or "")


def test_gemini_med_key_collapses_dolo_brand_strength():
    from gemini_ner import _med_key

    assert _med_key("Dolo 650", "Paracetamol") == _med_key("Dolo", "Paracetamol")


def test_prune_standard_fallback_info_when_deduped_dolo_removed():
    from nlp.medical_nlp import Medication, ParsedPrescription  # noqa: WPS433

    from prescription_polish import polish_prescription  # noqa: E402

    rx = ParsedPrescription(
        raw_text="नमस्ते",
        medications=[
            Medication(
                name="Dolo 650",
                generic_name="Paracetamol",
                dose="650MG",
                dose_source="inferred_from_brand",
            ),
            Medication(
                name="Dolo",
                generic_name="Paracetamol",
                dose="500MG",
                dose_source="standard_fallback",
            ),
        ],
        confidence_flags=[
            "INFO: Standard fallback dose used for Dolo (500MG)",
            "INFO: Dose inferred from brand name for Dolo 650 (650MG)",
        ],
    )
    polish_prescription(rx)
    assert len(rx.medications) == 1
    assert not any("Standard fallback" in f for f in rx.confidence_flags)
    assert any("Dolo 650" in f for f in rx.confidence_flags)


def test_extract_vitals_lines_oxygen_level_devanagari():
    text = "Doc: ऑक्सीजन लेवल 96%।"
    lines = extract_vitals_lines(text)
    assert any("96" in ln for ln in lines)


def test_filter_vomiting_when_patient_denied():
    from nlp.medical_nlp import ParsedPrescription  # noqa: WPS433

    rx = ParsedPrescription(
        raw_text="Doctor: Any vomiting?\nPatient: Nahi, ulti nahi hui.\nPatient: bukhar hai.",
        chief_complaint=["Fever (Bukhar)", "Vomiting"],
    )
    filter_denied_symptoms_chief(rx)
    assert all("vomit" not in c.lower() and "ulti" not in c.lower() for c in rx.chief_complaint)
