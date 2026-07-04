"""Grounded feedback generation — grounding is pure; generation uses a mock model."""
from parla.graph.state import DetectedError, Feedback
from parla.tools import feedback as fb, retrieval, verifier


class _Resp:
    content = "Nice work! 'He goes' takes -s in the third person. Keep it up!"

class _MockLLM:
    def invoke(self, msgs): return _Resp()


def _errs():
    return [DetectedError(span="He go", error_type="R:VERB:SVA", suggestion="He goes"),
            DetectedError(span="a apple", error_type="R:DET", suggestion="an apple")]


def test_grounding_attaches_retrieved_rule_ids():
    rules = retrieval.rules_for(_errs(), level="A2")
    grounded = fb.ground_corrections(_errs(), rules)
    assert all(c.rule_id for c in grounded)


def test_grounded_output_passes_verifier():
    rules = retrieval.rules_for(_errs(), level="A2")
    out = fb.generate_feedback("He go to a apple.", "A2", _errs(), rules, model=_MockLLM())
    assert verifier.is_grounded(Feedback(text="", corrections=out.corrections), rules)


def test_no_errors_returns_encouragement():
    out = fb.generate_feedback("I have a cat.", "A2", [], [], model=_MockLLM())
    assert out.corrections == [] and "good" in out.text.lower()
