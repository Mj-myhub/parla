"""Evaluation harness for grammatical error detection: edit-level P / R / F0.5.

Self-contained exact-match scorer — fully offline and deterministic, so every number in
the README is reproducible with one command. ERRANT over BEA-2019 is the standard-
benchmark upgrade (docs/evaluation.md); this gives honest numbers today and, crucially,
the baseline-vs-system discipline the whole project rests on. F0.5 (beta=0.5) weights
precision over recall — the right choice for a tutor, where a false correction misleads.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


@dataclass
class Example:
    id: str
    original: str
    gold: list[dict]


def load_dataset(path: str | Path) -> list[Example]:
    rows = json.loads(Path(path).read_text())
    return [Example(r["id"], r["original"], r.get("gold", [])) for r in rows]


def _key(span: str, correction: str) -> tuple[str, str]:
    return (_norm(span), _norm(correction))


def score_example(pred_edits: list[dict], gold_edits: list[dict]) -> tuple[int, int, int]:
    """(tp, fp, fn) via exact (span, correction) match. pred uses 'suggestion'."""
    gold = {_key(g["span"], g["correction"]) for g in gold_edits}
    pred = {_key(p["span"], p["suggestion"]) for p in pred_edits}
    return len(gold & pred), len(pred - gold), len(gold - pred)


def prf(tp: int, fp: int, fn: int, beta: float = 0.5) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    if p == 0.0 and r == 0.0:
        f = 0.0
    else:
        b2 = beta * beta
        denom = b2 * p + r
        f = (1 + b2) * p * r / denom if denom else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f_beta": round(f, 3),
            "beta": beta, "tp": tp, "fp": fp, "fn": fn}
