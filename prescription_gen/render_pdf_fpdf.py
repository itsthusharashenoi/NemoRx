"""
Hospital-style prescription PDF using fpdf2 (no Chrome required).
Used when Puppeteer is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fpdf import FPDF


class _RxPDF(FPDF):
    def header(self) -> None:
        pass


def write_prescription_pdf(ctx: dict[str, Any], out_path: Path) -> None:
    pdf = _RxPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(16, 16, 16)

    # Fonts — core fonts only (ASCII-friendly); avoid Unicode issues on default PDF fonts
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(13, 79, 108)
    pdf.cell(0, 10, _ascii(ctx["hospital_name"]), ln=1)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, _ascii(ctx.get("hospital_tagline", "")), ln=1)
    pdf.set_draw_color(13, 79, 108)
    pdf.set_line_width(0.6)
    pdf.line(16, pdf.get_y() + 2, 194, pdf.get_y() + 2)
    pdf.ln(8)

    if ctx.get("block_signoff"):
        pdf.set_fill_color(253, 234, 234)
        pdf.set_text_color(139, 26, 26)
        pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(0, 6, _ascii("Sign-off blocked: resolve critical issues before signature."))
        pdf.ln(4)
        pdf.set_text_color(0, 0, 0)

    pdf.set_font("Helvetica", "", 10)
    _meta_row(pdf, "Date", ctx.get("prescription_date", ""))
    _meta_row(pdf, "Patient ID", ctx.get("patient_id", ""))
    _meta_row(pdf, "Patient name", ctx.get("patient_name", ""))
    _meta_row(pdf, "Age / Sex", ctx.get("patient_age_sex", ""))
    _meta_row(pdf, "Physician", ctx.get("doctor_name", ""))
    pdf.ln(4)

    if ctx.get("vitals"):
        _section(pdf, "Vitals", ctx["vitals"])
    if ctx.get("clinical_notes"):
        _section(pdf, "Clinical notes", ctx["clinical_notes"])

    _section(pdf, "Chief complaint", ctx.get("chief_complaint") or [])
    _section(pdf, "Diagnosis", ctx.get("diagnosis") or [])

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(13, 79, 108)
    pdf.cell(0, 6, _ascii("PRESCRIBED MEDICATIONS"), ln=1)
    pdf.set_text_color(0, 0, 0)
    meds = ctx.get("medications") or []
    if meds:
        pdf.set_font("Helvetica", "", 8)
        for m in meds:
            name = str(m.get("name", ""))
            if m.get("generic_name") and str(m.get("generic_name")).lower() != name.lower():
                name = f"{name} ({m['generic_name']})"
            dose = str(m.get("dose") or "").strip()
            freq = str(m.get("frequency") or "").strip()
            dur = str(m.get("duration") or "").strip()
            instr = (str(m.get("instructions") or "").strip() or str(m.get("route") or "").strip())
            line = f"{name} | {dose} | {freq} | {dur} | {instr}"
            pdf.multi_cell(0, 5, _ascii(line))
            pdf.ln(1)
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, _ascii("No medications detected."), ln=1)

    pdf.ln(2)
    _section(pdf, "Investigations", ctx.get("investigations") or [])
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(13, 79, 108)
    pdf.cell(0, 6, _ascii("FOLLOW-UP"), ln=1)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 5, _ascii(str(ctx.get("follow_up") or "")))
    _section(pdf, "Advice", ctx.get("general_advice") or [])

    pdf.ln(2)
    pdf.set_fill_color(248, 244, 232)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, _ascii(f"Validation (confidence {ctx.get('confidence_score', 0)}/100)"), ln=1)
    pdf.set_font("Helvetica", "", 8)
    for f in ctx.get("confidence_flags") or ["None"]:
        pdf.multi_cell(0, 4, _ascii(f"- {f}"))

    pdf.ln(6)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4, _ascii("Source conversation (excerpt):"))
    ex = ctx.get("raw_conversation_excerpt", "")[:2000]
    pdf.multi_cell(0, 4, _ascii(ex))

    pdf.output(str(out_path))


def _ascii(s: str) -> str:
    return s.encode("latin-1", "replace").decode("latin-1")


def _meta_row(pdf: FPDF, label: str, value: str) -> None:
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(42, 4, _ascii(label.upper()), ln=0)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _ascii(value or ""), ln=1)


def _section(pdf: FPDF, title: str, items: list[str]) -> None:
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(13, 79, 108)
    pdf.cell(0, 6, _ascii(title.upper()), ln=1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 9)
    if items:
        for it in items:
            pdf.multi_cell(0, 5, _ascii(f"- {it}"))
    else:
        pdf.cell(0, 5, _ascii("-"), ln=1)
    pdf.ln(1)
