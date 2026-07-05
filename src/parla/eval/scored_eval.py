"""Scored evaluation on real annotated learner essays (exact + overlap scorers).

    python -m parla.eval.scored_eval
"""
from __future__ import annotations
from pathlib import Path

from parla.eval.harness import load_dataset, score_example, prf
from parla.tools.error_detection import detect as _default_detect


def _predict(detect_fn, text: str, use_llm: bool) -> list[dict]:
    return [{"span": e.span, "suggestion": e.suggestion, "error_type": e.error_type}
            for e in detect_fn(text, use_llm=use_llm)]


def _locate(span: str, original: str) -> tuple[int, int] | None:
    i = original.lower().find(span.strip().lower())
    return (i, i + len(span.strip())) if i >= 0 else None


def score_overlap(preds: list[dict], gold: list[dict], original: str) -> tuple[int, int, int]:
    """(tp, fp, fn) by span OVERLAP in the original. Detection-level, wording-agnostic."""
    gold_spans = [loc for g in gold if (loc := _locate(g["span"], original))]
    pred_spans = [loc for p in preds if (loc := _locate(p["span"], original))]
    unlocatable = len(preds) - len(pred_spans)
    used, tp = set(), 0
    for gs, ge in gold_spans:
        for i, (ps, pe) in enumerate(pred_spans):
            if i not in used and gs < pe and ps < ge:
                used.add(i); tp += 1; break
    fn = len(gold_spans) - tp
    fp = (len(pred_spans) - len(used)) + unlocatable
    return tp, fp, fn


def evaluate(gold_path: str = "data/eval/gold_essays.json", detect_fn=_default_detect) -> dict:
    examples = load_dataset(gold_path)
    out = {}
    for name, use_llm in [("deterministic", False), ("det+LLM", True)]:
        ex_tp = ex_fp = ex_fn = 0
        ov_tp = ov_fp = ov_fn = 0
        for ex in examples:
            preds = _predict(detect_fn, ex.original, use_llm)
            a, b, c = score_example(preds, ex.gold)
            ex_tp += a; ex_fp += b; ex_fn += c
            a, b, c = score_overlap(preds, ex.gold, ex.original)
            ov_tp += a; ov_fp += b; ov_fn += c
        out[name] = {"exact": prf(ex_tp, ex_fp, ex_fn), "overlap": prf(ov_tp, ov_fp, ov_fn)}
    return out


def render(report: dict) -> str:
    lines = ["# Detection evaluation (real annotated essays)", "",
             "| config | scorer | precision | recall | F0.5 | tp | fp | fn |",
             "|---|---|---|---|---|---|---|---|"]
    for name, sc in report.items():
        for scorer in ("exact", "overlap"):
            m = sc[scorer]
            lines.append(f"| {name} | {scorer} | {m['precision']} | {m['recall']} | "
                         f"{m['f_beta']} | {m['tp']} | {m['fp']} | {m['fn']} |")
    lines += ["", "_EXACT = strict string match (a floor). OVERLAP = detection by span "
              "overlap, wording-agnostic (fair to the LLM). ERRANT is the standard upgrade._"]
    return "\n".join(lines) + "\n"


def main() -> None:
    report = evaluate()
    md = render(report)
    out = Path("eval/reports/scored_eval.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"(written to {out})")


if __name__ == "__main__":
    main()
