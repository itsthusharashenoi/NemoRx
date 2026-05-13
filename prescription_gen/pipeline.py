"""
Orchestration: conversation prep → optional LLM → MedicalNLPPipeline.parse
"""

from __future__ import annotations

import sys
from pathlib import Path

from repo_paths import find_docscribe_akhil

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

_DOCS = find_docscribe_akhil()
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))

from nlp.medical_nlp import MedicalNLPPipeline, ParsedPrescription  # noqa: E402

from conversation_prep import prepare_conversation_text
from gemini_ner import maybe_apply_gemini_ner
from llm_condenser import maybe_condense
from prescription_polish import polish_prescription, unique_lines
from vitals_extract import extract_clinical_note_lines, extract_vitals_lines

_pipeline = MedicalNLPPipeline()


def parse_from_conversation(
    raw: str,
    use_llm: str = "auto",
    use_gemini: str = "auto",
) -> ParsedPrescription:
    """
    use_llm: 'auto' (use OpenAI if key set), 'on', 'off' — condenses dialogue before rules.
    use_gemini: 'auto' (use Gemini if GEMINI_API_KEY set), 'on', 'off' — NER merge after rules.
    """
    prepared, prep_notes = prepare_conversation_text(raw)
    condensed, used_llm = maybe_condense(prepared, use_llm=use_llm)
    rx = _pipeline.parse(condensed)
    # Preserve original conversation in output for audit
    rx.raw_text = raw.strip()
    for n in prep_notes:
        if n not in rx.confidence_flags:
            rx.confidence_flags.append(n)
    if used_llm:
        rx.confidence_flags.append("INFO: OpenAI condenser applied before rule extraction.")
    src = raw.strip()
    rx.vitals = unique_lines(list(rx.vitals) + extract_vitals_lines(src))
    rx.clinical_notes = unique_lines(list(rx.clinical_notes) + extract_clinical_note_lines(src))
    # Full conversation for NER (multilingual / speaker context); rules already used Roman-normalized text.
    maybe_apply_gemini_ner(rx, src, use_gemini)
    polish_prescription(rx)
    return rx
