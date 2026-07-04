"""Two-tier verifier: deterministic type match + LLM faithfulness judge (mocked)."""
import json
from parla.graph.state import DetectedError, Feedback
from parla.tools import verifier as v

DET_RULE = {"rule_id": "DET-001", "error_types": ["M:DET", "R:DET"],
            "title": "Articles", "rule": "Use a/an correctly."}
WO_RULE = {"rule_id": "WO-002", "error_types": ["R:WO"],
           "title": "Question word order", "rule": "Direct questions invert aux and subject."}


def _c(etype, rid):
    return DetectedError(span="x", error_type=etype, suggestion="y", rule_id=rid)


class _Resp:
    def __init__(self, c): self.content = c

class _Judge:
    def __init__(self, arr): self._c = json.dumps(arr)
    def invoke(self, msgs): return _Resp(self._c)


def test_type_match_is_verified():
    assert v.verify([_c("R:DET", "DET-001")], [DET_RULE], strict=False)[0]["status"] == "verified"

def test_missing_rule_id_is_unverified():
    assert v.verify([_c("R:DET", None)], [DET_RULE], strict=False)[0]["status"] == "unverified"

def test_rule_not_retrieved_is_unverified():
    assert v.verify([_c("R:DET", "DET-999")], [DET_RULE], strict=False)[0]["status"] == "unverified"

def test_type_mismatch_is_unverified():
    assert v.verify([_c("R:VERB:TENSE", "DET-001")], [DET_RULE], strict=False)[0]["status"] == "unverified"

def test_judge_flips_wrong_grounding_to_unverified():
    res = v.verify([_c("R:WO", "WO-002")], [WO_RULE], model=_Judge([False]), strict=True)
    assert res[0]["status"] == "unverified" and "explain" in res[0]["reason"]

def test_judge_keeps_faithful_grounding():
    res = v.verify([_c("R:DET", "DET-001")], [DET_RULE], model=_Judge([True]), strict=True)
    assert res[0]["status"] == "verified"

def test_is_grounded_true_and_false():
    assert v.is_grounded(Feedback(text="", corrections=[_c("R:DET", "DET-001")]), [DET_RULE]) is True
    assert v.is_grounded(Feedback(text="", corrections=[_c("R:DET", None)]), [DET_RULE]) is False

def test_no_corrections_is_grounded():
    assert v.is_grounded(Feedback(text="ok"), []) is True
