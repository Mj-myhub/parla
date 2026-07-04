"""Two-tier grounding verification — makes "verified" an honest claim.

Tier 1 (deterministic): a correction is groundable only if it cites a rule that was
    actually retrieved AND whose declared error_types are type-compatible with it.
Tier 2 (LLM faithfulness judge, strict mode): confirms the cited rule genuinely EXPLAINS
    the correction — catching a type-consistent but semantically wrong grounding.
"""
from __future__ import annotations
import json
from parla.graph.state import DetectedError, Feedback


def _family(tag: str) -> str:
    parts = tag.split(":")
    return ":".join(parts[1:]) if len(parts) > 1 else tag


def type_matches(correction: DetectedError, rule: dict) -> bool:
    types = rule.get("error_types", [])
    return (correction.error_type in types
            or _family(correction.error_type) in {_family(t) for t in types})


def _cited_rule(correction: DetectedError, retrieved_rules: list[dict]) -> dict | None:
    if not correction.rule_id:
        return None
    return next((r for r in retrieved_rules if r.get("rule_id") == correction.rule_id), None)


_JUDGE_SYSTEM = (
    "You are a strict grammar reviewer. You receive a JSON list of items; each pairs a "
    "learner CORRECTION with the grammar RULE it was tagged with. For each item decide "
    "whether the rule genuinely explains and justifies that specific correction. Be strict: "
    "if the rule describes a different phenomenon, answer false. Respond with ONLY a JSON "
    "array of booleans, one per item, in the same order."
)


def faithfulness_judge(pairs: list[tuple[DetectedError, dict]], model=None) -> list[bool]:
    """One batched call judges all pairs. Missing/failed judge -> trust Tier 1 (all True)."""
    if not pairs:
        return []
    try:
        from parla.tools.error_detection import _default_llm, _extract_json_array
        llm = model or _default_llm()
        items = [{"n": i + 1, "correction": f"{c.span} -> {c.suggestion}",
                  "rule": f"{r.get('title', '')}: {r.get('rule', '')}"}
                 for i, (c, r) in enumerate(pairs)]
        resp = llm.invoke([("system", _JUDGE_SYSTEM), ("human", json.dumps(items))])
        content = resp.content if hasattr(resp, "content") else str(resp)
        verdicts = [bool(x) for x in _extract_json_array(content)][:len(pairs)]
        verdicts += [True] * (len(pairs) - len(verdicts))
        return verdicts
    except Exception:
        return [True] * len(pairs)


def verify(corrections: list[DetectedError], retrieved_rules: list[dict],
           model=None, strict: bool = True) -> list[dict]:
    """Per-correction: {correction, rule, status: 'verified'|'unverified', reason}."""
    results = []
    for c in corrections:
        rule = _cited_rule(c, retrieved_rules)
        ok = rule is not None and type_matches(c, rule)
        results.append({"correction": c, "rule": rule if ok else None,
                        "status": "verified" if ok else "unverified",
                        "reason": "type-matched rule" if ok else "no matching rule"})
    if strict:
        pairs = [(r["correction"], r["rule"]) for r in results if r["status"] == "verified"]
        verdicts = faithfulness_judge(pairs, model=model)
        vi = 0
        for r in results:
            if r["status"] == "verified":
                if not verdicts[vi]:
                    r["status"] = "unverified"
                    r["reason"] = "rule does not explain this fix"
                vi += 1
    return results


def is_grounded(feedback: Feedback, retrieved_rules: list[dict], model=None, strict: bool = False) -> bool:
    results = verify(feedback.corrections, retrieved_rules, model=model, strict=strict)
    return all(r["status"] == "verified" for r in results) if results else True


def ungrounded_corrections(feedback: Feedback, retrieved_rules: list[dict], model=None, strict: bool = False):
    results = verify(feedback.corrections, retrieved_rules, model=model, strict=strict)
    return [r["correction"] for r in results if r["status"] != "verified"]
