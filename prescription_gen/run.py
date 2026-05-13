#!/usr/bin/env python3
"""
Read input.txt → parse conversation → render hospital prescription PDF.
Run from anywhere: python prescription_gen/run.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(_DIR / ".env")
except ImportError:
    pass

if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from repo_paths import find_node_workspace  # noqa: E402

_NODE_CWD = find_node_workspace()

from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from pipeline import parse_from_conversation  # noqa: E402
from render_context import prescription_to_context  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate prescription PDF from input.txt")
    ap.add_argument("--input", type=Path, default=_DIR / "input.txt", help="Raw conversation text file")
    ap.add_argument("--out", type=Path, default=_DIR / "output.pdf", help="Output PDF path")
    ap.add_argument("--html", type=Path, default=_DIR / "filled_prescription.html", help="Intermediate HTML path")
    ap.add_argument("--use-llm", choices=["auto", "on", "off"], default="auto", help="OpenAI condenser (OPENAI_API_KEY)")
    ap.add_argument(
        "--use-gemini",
        choices=["auto", "on", "off"],
        default="auto",
        help="Gemini JSON NER merge after rules (GEMINI_API_KEY in prescription_gen/.env)",
    )
    ap.add_argument("--skip-pdf", action="store_true", help="Only write HTML, skip Puppeteer")
    args = ap.parse_args()

    raw = args.input.read_text(encoding="utf-8")
    if not raw.strip():
        print("Input file is empty.", file=sys.stderr)
        return 1

    rx = parse_from_conversation(raw, use_llm=args.use_llm, use_gemini=args.use_gemini)
    ctx = prescription_to_context(rx)

    env = Environment(
        loader=FileSystemLoader(_DIR / "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    tpl = env.get_template("prescription.html")
    html_out = tpl.render(**ctx)
    args.html.write_text(html_out, encoding="utf-8")
    print(f"Wrote {args.html}")

    if args.skip_pdf:
        return 0

    html_abs = args.html.resolve()
    pdf_abs = args.out.resolve()
    cmd = ["node", str(_DIR / "render_pdf.mjs"), str(html_abs), str(pdf_abs)]
    try:
        subprocess.run(cmd, cwd=str(_NODE_CWD), check=True)
        print(f"Wrote {pdf_abs}")
        return 0
    except FileNotFoundError:
        print("Node.js not found; falling back to fpdf2.", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Puppeteer PDF failed ({e}); falling back to fpdf2.", file=sys.stderr)

    from render_pdf_fpdf import write_prescription_pdf

    write_prescription_pdf(ctx, pdf_abs)
    print(f"Wrote {pdf_abs} (fpdf2 fallback)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
