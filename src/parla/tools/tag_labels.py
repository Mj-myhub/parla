"""Human-readable names for ERRANT-style error tags.
Operation prefixes: M = missing, R = replace, U = unnecessary."""
from __future__ import annotations

_BASE = {
    "DET": "article/determiner", "PREP": "preposition",
    "VERB:SVA": "subject-verb agreement", "VERB:TENSE": "verb tense",
    "VERB:FORM": "verb form", "VERB": "verb",
    "NOUN:NUM": "noun number (singular/plural)", "NOUN": "noun",
    "PRON": "pronoun", "WO": "word order", "SPELL": "spelling",
    "WORD": "word form", "PUNCT": "punctuation", "MORPH": "word form", "OTHER": "other",
}
_OP = {"M": "missing", "R": "wrong", "U": "unnecessary"}


def label(tag: str) -> str:
    """'R:VERB:TENSE' -> 'wrong verb tense'; 'STYLE' -> 'style / word choice'."""
    if not tag or tag == "STYLE":
        return "style / word choice"
    parts = tag.split(":")
    op = _OP.get(parts[0], "")
    base = _BASE.get(":".join(parts[1:]), ":".join(parts[1:]).lower() or tag.lower())
    return f"{op} {base}".strip()
