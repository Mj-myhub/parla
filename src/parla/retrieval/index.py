"""Rule index over the pedagogical corpus.

Ships with a BM25 backend (pure-python, zero downloads) so the retrieval spine works
immediately. The `dense` + `rerank` seams are marked for when you're on your machine with
HF access — turning this from lexical into the hybrid retriever the README advertises.
"""
from __future__ import annotations
import json, pickle, argparse
from pathlib import Path
from rank_bm25 import BM25Okapi

CORPUS_PATH = Path("data/corpus/grammar_rules.json")
INDEX_PATH = Path(".index/bm25.pkl")

_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


def _doc_text(rule: dict) -> str:
    """The searchable surface for a rule: tags + title + rule + examples."""
    return " ".join([
        " ".join(rule.get("error_types", [])),
        rule.get("title", ""),
        rule.get("rule", ""),
        " ".join(rule.get("examples", [])),
    ]).lower()


def _tok(text: str) -> list[str]:
    return "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower()).split()


class RuleIndex:
    def __init__(self, rules: list[dict]):
        self.rules = rules
        self._bm25 = BM25Okapi([_tok(_doc_text(r)) for r in rules])

    @classmethod
    def load(cls, corpus_path: Path = CORPUS_PATH) -> "RuleIndex":
        return cls(json.loads(Path(corpus_path).read_text()))

    def search(self, query: str, level: str | None = None, k: int = 5) -> list[dict]:
        scores = self._bm25.get_scores(_tok(query))
        ranked = sorted(zip(self.rules, scores), key=lambda x: x[1], reverse=True)
        results = []
        for rule, score in ranked:
            if score <= 0:
                continue
            adj = score - 0.15 * _level_distance(level, rule.get("cefr"))  # gentle level bias
            results.append({**rule, "score": round(float(adj), 3)})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:k]
        # DENSE SEAM: embed query + docs (sentence-transformers), fuse with BM25 via RRF.
        # RERANK SEAM: cross-encoder over the top-k before returning.


def _level_distance(a: str | None, b: str | None) -> int:
    if not a or not b or a not in _LEVELS or b not in _LEVELS:
        return 0
    return abs(_LEVELS.index(a) - _LEVELS.index(b))


def _build_cli() -> None:
    idx = RuleIndex.load()
    INDEX_PATH.parent.mkdir(exist_ok=True)
    INDEX_PATH.write_bytes(pickle.dumps(idx))
    print(f"built index over {len(idx.rules)} rules -> {INDEX_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    if p.parse_args().build:
        _build_cli()
