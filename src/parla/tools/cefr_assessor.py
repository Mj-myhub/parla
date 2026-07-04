"""Tool 3: CEFR level assessment (A1-C2). LLM-based with a heuristic fallback so the
graph runs even without a model/key. Evaluated in docs/evaluation.md section 4."""
from __future__ import annotations
import re

_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
_SYSTEM = ("Classify the CEFR level of the learner's English writing. "
           "Reply with ONLY one code from: A1, A2, B1, B2, C1, C2.")


def _parse_level(text: str) -> str | None:
    m = re.search(r"\b([ABC][12])\b", text.upper())
    return m.group(1) if m and m.group(1) in _LEVELS else None


def _heuristic(text: str) -> str:
    words = text.split()
    sents = max(1, text.count(".") + text.count("!") + text.count("?"))
    wps = len(words) / sents
    if wps < 6:
        return "A2"
    if wps < 12:
        return "B1"
    return "B2"


def assess(text: str, prior: str | None = None, model=None) -> str:
    try:
        from parla.tools.error_detection import _default_llm
        llm = model or _default_llm()
        resp = llm.invoke([("system", _SYSTEM), ("human", text)])
        level = _parse_level(resp.content if hasattr(resp, "content") else str(resp))
        if level:
            return level
    except Exception:
        pass
    return prior or _heuristic(text)
