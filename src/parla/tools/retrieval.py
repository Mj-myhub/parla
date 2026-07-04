"""Tool 2: RAG retrieval over the pedagogical corpus (CEFR descriptors + grammar refs).

Hybrid-ready (BM25 now, dense+rerank seam in retrieval/index.py). Returns the rules the
agent must ground its feedback in.
"""
from __future__ import annotations
from functools import lru_cache
from parla.graph.state import DetectedError
from parla.retrieval.index import RuleIndex


@lru_cache(maxsize=1)
def _index() -> RuleIndex:
    return RuleIndex.load()


def rules_for(errors: list[DetectedError], level: str | None = None) -> list[dict]:
    """For each detected error, fetch the best-matching rule(s), deduped by rule_id."""
    idx = _index()
    seen: dict[str, dict] = {}
    for err in errors:
        query = f"{err.error_type} {err.span} {err.suggestion}"
        for hit in idx.search(query, level=level, k=3):
            seen.setdefault(hit["rule_id"], hit)
    return sorted(seen.values(), key=lambda r: r["score"], reverse=True)
