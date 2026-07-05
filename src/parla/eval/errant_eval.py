"""Standard GEC evaluation with ERRANT — detection + correction F0.5.

ERRANT scores parallel sentences (original vs corrected). Our gold and detector produce
edit lists, so we reconstruct corrected sentences by applying the edits, then let ERRANT
extract/classify and compare. Reports, for deterministic and det+LLM:
  - DETECTION F0.5 : did the system find the error span?
  - CORRECTION F0.5: did it also produce the right fix? (the strict, standard number)

    python -m parla.eval.errant_eval
"""
from __future__ import annotations
from pathlib import Path

from parla.eval.harness import load_dataset, prf
from parla.tools.error_detection import detect as _default_detect


def _apply_edits(original: str, edits: list[dict]) -> str:
    """Apply span->correction replacements, left-to-right, non-overlapping."""
    spans = []
    for e in edits:
        span = e["span"].strip()
        corr = (e.get("correction") or e.get("suggestion") or "").strip()
        i = original.lower().find(span.lower())
        if i >= 0:
            spans.append((i, i + len(span), corr))
    spans.sort()
    out, last = [], 0
    for s, end, corr in spans:
        if s < last:
            continue
        out.append(original[last:s]); out.append(corr); last = end
    out.append(original[last:])
    return "".join(out)


def _errant_edits(annotator, original: str, corrected: str) -> set[tuple]:
    orig = annotator.parse(original)
    cor = annotator.parse(corrected)
    return {(e.o_start, e.o_end, e.c_str) for e in annotator.annotate(orig, cor)}


def _score(gold: set[tuple], sys: set[tuple]) -> dict:
    det_gold = {(s, e) for (s, e, _) in gold}
    det_sys = {(s, e) for (s, e, _) in sys}
    det = prf(len(det_gold & det_sys), len(det_sys - det_gold), len(det_gold - det_sys))
    cor = prf(len(gold & sys), len(sys - gold), len(gold - sys))
    return {"detection": det, "correction": cor}


def evaluate(gold_path: str = "data/eval/gold_essays.json", detect_fn=_default_detect,
             annotator=None) -> dict:
    if annotator is None:
        import errant
        annotator = errant.load("en")
    examples = load_dataset(gold_path)
    out = {}
    for name, use_llm in [("deterministic", False), ("det+LLM", True)]:
        det_tp = det_fp = det_fn = cor_tp = cor_fp = cor_fn = 0
        for ex in examples:
            gold_c = _apply_edits(ex.original, ex.gold)
            preds = [{"span": e.span, "correction": e.suggestion}
                     for e in detect_fn(ex.original, use_llm=use_llm)]
            sys_c = _apply_edits(ex.original, preds)
            gold_e = _errant_edits(annotator, ex.original, gold_c)
            sys_e = _errant_edits(annotator, ex.original, sys_c)
            s = _score(gold_e, sys_e)
            det_tp += s["detection"]["tp"]; det_fp += s["detection"]["fp"]; det_fn += s["detection"]["fn"]
            cor_tp += s["correction"]["tp"]; cor_fp += s["correction"]["fp"]; cor_fn += s["correction"]["fn"]
        out[name] = {"detection": prf(det_tp, det_fp, det_fn),
                     "correction": prf(cor_tp, cor_fp, cor_fn)}
    return out


def render(report: dict) -> str:
    lines = ["# ERRANT evaluation (standard GEC scorer)", "",
             "| config | metric | precision | recall | F0.5 |",
             "|---|---|---|---|---|"]
    for name, sc in report.items():
        for metric in ("detection", "correction"):
            m = sc[metric]
            lines.append(f"| {name} | {metric} | {m['precision']} | {m['recall']} | {m['f_beta']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    report = evaluate()
    md = render(report)
    out = Path("eval/reports/errant_eval.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"(written to {out})")


if __name__ == "__main__":
    main()
