"""Scored-eval orchestration + overlap scorer — validated with a mock detector."""
from parla.graph.state import DetectedError
from parla.eval import scored_eval


def _mock(text, use_llm=False):
    out = []
    if "hard worker" in text:
        out.append(DetectedError(span="hard worker", error_type="R:NOUN:NUM", suggestion="hard workers"))
    if use_llm and "discrimination" in text:
        out.append(DetectedError(span="zzz", error_type="R:VERB:SVA", suggestion="y"))
    return out


def test_evaluate_reports_both_configs_and_scorers():
    rep = scored_eval.evaluate(detect_fn=_mock)
    assert set(rep) == {"deterministic", "det+LLM"}
    assert set(rep["deterministic"]) == {"exact", "overlap"}


def test_overlap_beats_exact_for_llm_phrasing():
    orig = "there are still some discrimination against women"
    gold = [{"span": "are still some discrimination", "correction": "is still some discrimination"}]
    pred = [{"span": "there are still some discrimination", "suggestion": "there is still some"}]
    assert scored_eval.score_example(pred, gold) == (0, 1, 1)
    assert scored_eval.score_overlap(pred, gold, orig) == (1, 0, 0)


def test_render_produces_table():
    md = scored_eval.render(scored_eval.evaluate(detect_fn=_mock))
    assert "overlap" in md and "precision" in md
