"""Per-learner memory: recurring error types + level trajectory, persisted as JSON.

Makes the tutor personal ("you've made this article error 6 times") and is why memory
is justified here rather than bolted on. Store dir is overridable for tests.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from parla.graph.state import DetectedError

STORE_DIR = Path(".profiles")


@dataclass
class Profile:
    learner_id: str
    level: str | None = None
    error_counts: dict[str, int] = field(default_factory=dict)


def _path(learner_id: str) -> Path:
    safe = "".join(c for c in learner_id if c.isalnum() or c in "-_") or "anon"
    return STORE_DIR / f"{safe}.json"


def load(learner_id: str) -> Profile:
    p = _path(learner_id)
    if p.exists():
        d = json.loads(p.read_text())
        return Profile(d["learner_id"], d.get("level"), d.get("error_counts", {}))
    return Profile(learner_id=learner_id)


def update(learner_id: str, errors: list[DetectedError], level: str) -> Profile:
    prof = load(learner_id)
    prof.level = level
    for e in errors:
        prof.error_counts[e.error_type] = prof.error_counts.get(e.error_type, 0) + 1
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    _path(learner_id).write_text(json.dumps(
        {"learner_id": prof.learner_id, "level": prof.level, "error_counts": prof.error_counts},
        indent=2))
    return prof
