"""Tool 5: the guardrail. Rejects any correction not grounded in a retrieved rule.

LLM06 'excessive agency' mitigation: the tutor may not invent a grammar rule. A
correction is grounded only if it cites a rule_id that was actually retrieved.
"""
from __future__ import annotations
from parla.graph.state import Feedback


def is_grounded(feedback: Feedback, retrieved_rules: list[dict]) -> bool:
    """True only if EVERY correction cites a rule_id present in the retrieved set."""
    if not feedback.corrections:
        return True
    allowed = {r.get("rule_id") for r in retrieved_rules if r.get("rule_id")}
    return all(c.rule_id in allowed for c in feedback.corrections)


def ungrounded_corrections(feedback: Feedback, retrieved_rules: list[dict]) -> list:
    """The offenders — useful for the eval report's before/after ungrounded rate."""
    allowed = {r.get("rule_id") for r in retrieved_rules if r.get("rule_id")}
    return [c for c in feedback.corrections if c.rule_id not in allowed]
