# Evaluation methodology

This is the most important document in the repository. It is what separates a
computational linguist who *owns evaluation* from someone who wired up an LLM and
called it done. Invest here.

## Principle

Every claim about quality is measured against (a) a standardized benchmark and (b) a
baseline. We never report an absolute number without a comparison, because a number
without a baseline is meaningless.

## 1. Grammatical Error Correction (the headline metric)

**Task.** Given a learner sentence, produce corrections. This is the classic GEC task,
so we can borrow its established evaluation machinery instead of inventing our own.

**Datasets** (check licensing/access before use; all are standard in the GEC literature):
- **BEA-2019 Shared Task** (W&I+LOCNESS) — the modern standard; evaluated with **ERRANT**.
- **CoNLL-2014 Shared Task** — the classic benchmark; evaluated with the **M2 scorer**.
- **FCE** — Cambridge First Certificate learner essays with error annotations.
- **JFLEG** — fluency-oriented references, good as a secondary lens.

**Metric.** **F0.5** (ERRANT / M2). F0.5 weights precision twice as heavily as recall,
which is the *correct* choice for a tutor: a false correction actively misleads a
learner, so precision matters more than catching every error. Stating this rationale in
an interview signals you understand the task, not just the toolkit.

**Baseline.** Single-shot LLM correction (one prompt, no retrieval, no verifier).

**What to report.** Precision, recall, F0.5 for baseline vs. Parla, plus a short error-
type breakdown from ERRANT (e.g., does the agent help most on determiner/preposition
errors and least on word-order?). The breakdown is what a reviewer remembers.

## 2. Feedback faithfulness (is the advice grounded?)

The agent must ground each correction in a retrieved rule. We measure the **residual
rate of ungrounded corrections** — corrections the verifier let through that cannot be
traced to a rule in the corpus — using RAGAS faithfulness plus a manual spot-check on a
sample. Report the rate *before* and *after* the verifier to show the guardrail's effect.

## 3. Pedagogical appropriateness (LLM-as-judge)

A rubric-based judge scores feedback on: correctness, level-appropriateness (is the
explanation pitched at the learner's CEFR level?), tone, and actionability. Publish the
rubric in this doc. Validate the judge against ~30 human-rated examples and report
agreement — a judge you haven't validated is just another ungrounded model.

## 4. CEFR level assessment

Classify learner text into A1–C2. Evaluate accuracy / macro-F1 against a CEFR-labeled
corpus (e.g., EFCAMDAT or a CEFR-labeled set you can access). Report the confusion
matrix — adjacent-level confusion (B1 vs B2) is expected and worth discussing honestly.

## 5. Systems metrics

Cost per interaction and latency per interaction, for baseline vs. agent. If you add a
model router (cheap model for planning, stronger for synthesis), show the cost delta.
Cost-awareness is repeatedly flagged as an underrated interview signal.

## Reproducibility

- Eval set versions are pinned; the exact question/sentence sets live under `eval/`.
- `python -m parla.eval.run_eval` regenerates every number and plot in one command.
- Every reported figure in the README maps to a committed artifact in `eval/reports/`.
