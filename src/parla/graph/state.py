"""Agent state — the single source of truth that flows through the LangGraph."""
from __future__ import annotations
from typing import TypedDict, Literal
from dataclasses import dataclass, field

CEFRLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]


@dataclass
class DetectedError:
    span: str
    error_type: str          # ERRANT-style tag, e.g. "M:DET", "R:VERB:TENSE"
    suggestion: str
    rule_id: str | None = None   # filled once grounded against the corpus


@dataclass
class Feedback:
    text: str
    corrections: list[DetectedError] = field(default_factory=list)
    grounded: bool = False       # verifier sets this


class TutorState(TypedDict, total=False):
    # inputs
    learner_id: str
    submission: str
    # working memory (populated by nodes)
    level: CEFRLevel
    errors: list[DetectedError]
    retrieved_rules: list[dict]
    feedback: Feedback
    exercise: str
    verification: list
    # control
    verify_attempts: int
    max_verify_attempts: int
