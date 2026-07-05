"""ERRANT eval: pure edit-application + scoring; evaluate() run with a mock annotator."""
from parla.graph.state import DetectedError
from parla.eval import errant_eval as ee


class _Edit:
    def __init__(self, s, e, c): self.o_start, self.o_end, self.c_str = s, e, c

class _Ann:
    def parse(self, t): return t.split()
    def annotate(self, o, c):
        return [_Edit(i, i + 1, c[i]) for i in range(min(len(o), len(c))) if o[i] != c[i]]


def _mock_detect(text, use_llm=False):
    if "hard worker" in text:
        return [DetectedError(span="hard worker", error_type="R:NOUN:NUM", suggestion="hard workers")]
    return []


def test_apply_edits_non_overlapping_and_order_independent():
    o = "He go and she have two cat."
    e = [{"span": "two cat", "correction": "two cats"}, {"span": "He go", "correction": "He goes"}]
    assert ee._apply_edits(o, e) == "He goes and she have two cats."


def test_score_detection_vs_correction():
    gold = {(0, 1, "x"), (2, 3, "y")}
    sys = {(0, 1, "x"), (2, 3, "z")}
    s = ee._score(gold, sys)
    assert s["detection"]["tp"] == 2 and s["correction"]["tp"] == 1


def test_evaluate_with_mock_annotator():
    rep = ee.evaluate(detect_fn=_mock_detect, annotator=_Ann())
    assert set(rep) == {"deterministic", "det+LLM"}
    assert set(rep["deterministic"]) == {"detection", "correction"}
