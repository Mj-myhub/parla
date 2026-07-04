"""Per-learner memory: recurring error types, level trajectory, mastered vs. persistent.

This is what makes the tutor personal ("you've made this article error 6 times") and is
why memory is justified here rather than bolted on.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from parla.graph.state import DetectedError


@dataclass
class Profile:
    learner_id: str
    level: str | None = None
    error_counts: dict[str, int] = field(default_factory=dict)


def load(learner_id: str) -> Profile:
    # TODO: read from a small store (SQLite / json / vector store for semantic recall).
    raise NotImplementedError


def update(learner_id: str, errors: list[DetectedError], level: str) -> None:
    # TODO: increment error_counts, append level to trajectory, persist.
    raise NotImplementedError
