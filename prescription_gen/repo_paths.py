"""
Resolve shared dependencies when ``prescription_gen`` lives under the hackathon repo
(``…/prescription_gen``) or under ``DocScribe/prescription_gen``.
"""

from __future__ import annotations

from pathlib import Path


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def find_docscribe_akhil() -> Path:
    """
    Locate ``docscribe_akhil`` (contains ``nlp/medical_nlp.py``) by walking parents
    from ``prescription_gen`` — supports:

    - ``<root>/prescription_gen`` with ``<root>/docscribe_akhil``
    - ``<root>/DocScribe/prescription_gen`` with ``<root>/docscribe_akhil``
    """
    p = package_dir().parent
    for _ in range(8):
        cand = p / "docscribe_akhil"
        if (cand / "nlp" / "medical_nlp.py").is_file():
            return cand
        if p.parent == p:
            break
        p = p.parent
    raise ImportError(
        "Could not find docscribe_akhil next to this repo. "
        "Expected a sibling folder docscribe_akhil/ with nlp/medical_nlp.py "
        "(Witch Hunt monorepo layout)."
    )


def find_node_workspace() -> Path:
    """
    Directory that contains ``node_modules/puppeteer`` (used as subprocess cwd for
    ``render_pdf.mjs``). Walks upward from ``prescription_gen``; falls back to
    ``prescription_gen``'s parent if Puppeteer is not installed yet.
    """
    p = package_dir()
    for _ in range(10):
        if (p / "node_modules" / "puppeteer" / "package.json").is_file():
            return p
        if p.parent == p:
            break
        p = p.parent
    return package_dir().parent
