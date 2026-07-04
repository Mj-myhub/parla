"""Tool 4: generate targeted practice for the learner's specific recurring errors.
LLM-based with a templated fallback so the graph always returns something useful."""
from __future__ import annotations
from parla.graph.state import DetectedError

_SYSTEM = ("You are an English teacher. Given a learner's CEFR level and the grammar "
           "error types they just made, write 2 short practice exercises (fill-in-the-blank "
           "or correct-the-sentence) targeting those specific error types. Keep it concise "
           "and level-appropriate. Do not include answer keys.")


def _fallback(errors: list[DetectedError]) -> str:
    kinds = sorted({e.error_type for e in errors})
    return "Practice focus: " + ", ".join(kinds) + ".\nTry writing three new sentences that use these correctly."


def generate(errors: list[DetectedError], level: str, model=None) -> str:
    if not errors:
        return ""
    try:
        from parla.tools.error_detection import _default_llm
        llm = model or _default_llm()
        kinds = ", ".join(sorted({e.error_type for e in errors}))
        resp = llm.invoke([("system", _SYSTEM),
                           ("human", f"Level: {level}. Error types: {kinds}.")])
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception:
        return _fallback(errors)
