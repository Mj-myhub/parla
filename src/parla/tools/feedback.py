"""Grounded feedback generation — the heart of the tutor.

Two responsibilities, deliberately separated:
1. ground_corrections(): DETERMINISTICALLY attach a rule_id to each detected error by
   matching its ERRANT-style type to a retrieved rule. This is what makes the verifier
   meaningful — corrections are tied to real corpus rules, not to the LLM's imagination.
2. generate_feedback(): the LLM writes warm, level-appropriate prose EXPLAINING those
   grounded corrections. It may not introduce rules beyond the ones supplied.
"""
from __future__ import annotations
from parla.graph.state import DetectedError, Feedback


def _family(tag: str) -> str:
    """'R:VERB:SVA' -> 'VERB:SVA', 'M:DET' -> 'DET' — operation-agnostic subtype."""
    parts = tag.split(":")
    return ":".join(parts[1:]) if len(parts) > 1 else tag


def ground_corrections(errors: list[DetectedError], retrieved_rules: list[dict]) -> list[DetectedError]:
    """Attach rule_id to each error via exact type match, then family fallback."""
    grounded: list[DetectedError] = []
    for e in errors:
        rid = None
        for rule in retrieved_rules:
            types = rule.get("error_types", [])
            if e.error_type in types or _family(e.error_type) in {_family(t) for t in types}:
                rid = rule["rule_id"]
                break
        grounded.append(DetectedError(span=e.span, error_type=e.error_type,
                                      suggestion=e.suggestion, rule_id=rid))
    return grounded


_FB_SYSTEM = (
    "You are a warm, encouraging English tutor. The learner's CEFR level is {level}, so "
    "pitch your explanations at that level (simpler for A1-A2, more detail for B2-C1). "
    "You are given the learner's text and a list of corrections, each paired with the "
    "grammar rule it illustrates. Write short feedback that: (1) opens with one genuine "
    "positive, (2) explains each correction in plain language, naming the rule, (3) ends "
    "with one encouraging sentence. Do NOT introduce any rule that is not in the list."
)


def _corrections_block(corrections: list[DetectedError], rules_by_id: dict[str, dict]) -> str:
    lines = []
    for c in corrections:
        rule = rules_by_id.get(c.rule_id or "", {})
        rule_txt = f"{rule.get('title', 'general usage')}: {rule.get('rule', '')}".strip()
        lines.append(f'- "{c.span}" -> "{c.suggestion}"  [{c.error_type}]  rule = {rule_txt}')
    return "\n".join(lines) if lines else "(no corrections)"


def generate_feedback(submission: str, level: str, corrections: list[DetectedError],
                      retrieved_rules: list[dict], model=None) -> Feedback:
    grounded = ground_corrections(corrections, retrieved_rules)
    if not grounded:
        return Feedback(text="This looks good — I didn't spot any grammar issues. Nice work!",
                        corrections=[])
    rules_by_id = {r["rule_id"]: r for r in retrieved_rules}
    human = (f"Learner text:\n{submission}\n\nCorrections and their rules:\n"
             f"{_corrections_block(grounded, rules_by_id)}")
    try:
        from parla.tools.error_detection import _default_llm
        llm = model or _default_llm()
        resp = llm.invoke([("system", _FB_SYSTEM.format(level=level)), ("human", human)])
        text = resp.content if hasattr(resp, "content") else str(resp)
    except Exception:
        text = "Here are some corrections:\n" + "\n".join(
            f'• "{c.span}" -> "{c.suggestion}"' for c in grounded)
    return Feedback(text=text, corrections=grounded)
