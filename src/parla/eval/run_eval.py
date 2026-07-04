"""One command regenerates every number and plot in the README. See docs/evaluation.md.

    python -m parla.eval.run_eval --dataset bea2019-dev --compare-baseline
"""
from __future__ import annotations
import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bea2019-dev")
    p.add_argument("--compare-baseline", action="store_true")
    args = p.parse_args()

    # TODO:
    #   1) run baseline (single-shot) + Parla over the dataset
    #   2) score GEC with ERRANT/M2 -> precision, recall, F0.5, error-type breakdown
    #   3) faithfulness (RAGAS) before/after verifier
    #   4) pedagogical LLM-as-judge (validated rubric)
    #   5) cost + latency
    #   6) write eval/reports/{table.md, plots/*.png}
    raise NotImplementedError


if __name__ == "__main__":
    main()
