"""Run the detector over the eval set and report edit-level P/R/F0.5 + per-type recall.

    python -m parla.eval.run_eval                    # deterministic detector
    python -m parla.eval.run_eval --use-llm          # once Layer 3 is wired
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from parla.tools.error_detection import detect
from parla.eval.harness import load_dataset, score_example, prf


def run(data_path: str, use_llm: bool = False) -> dict:
    examples = load_dataset(data_path)
    TP = FP = FN = 0
    gold_by_type: dict[str, int] = defaultdict(int)
    hit_by_type: dict[str, int] = defaultdict(int)
    for ex in examples:
        preds = [{"span": e.span, "suggestion": e.suggestion, "error_type": e.error_type}
                 for e in detect(ex.original, use_llm=use_llm)]
        tp, fp, fn = score_example(preds, ex.gold)
        TP += tp; FP += fp; FN += fn
        pred_keys = {(p["span"].strip().lower(), p["suggestion"].strip().lower()) for p in preds}
        for g in ex.gold:
            gold_by_type[g["type"]] += 1
            if (g["span"].strip().lower(), g["correction"].strip().lower()) in pred_keys:
                hit_by_type[g["type"]] += 1
    return {"overall": prf(TP, FP, FN),
            "per_type": {t: {"recall": round(hit_by_type[t] / n, 3), "gold": n}
                         for t, n in sorted(gold_by_type.items())}}


def _render(report: dict, use_llm: bool) -> str:
    o = report["overall"]
    lines = [f"# Detection eval ({'det+LLM' if use_llm else 'deterministic'})", "",
             f"- Precision: **{o['precision']}**  ·  Recall: **{o['recall']}**  ·  "
             f"F0.5: **{o['f_beta']}**   (tp={o['tp']} fp={o['fp']} fn={o['fn']})", "",
             "| error type | recall | gold n |", "|---|---|---|"]
    for t, v in report["per_type"].items():
        lines.append(f"| {t} | {v['recall']} | {v['gold']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval/dev.json")
    ap.add_argument("--use-llm", action="store_true")
    args = ap.parse_args()
    report = run(args.data, use_llm=args.use_llm)
    md = _render(report, args.use_llm)
    out = Path("eval/reports/detection_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"(written to {out})")


if __name__ == "__main__":
    main()
