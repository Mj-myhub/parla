"""Unit tests for each tool. Start here on day 1 — write the test, then the tool.

A green CI badge on a junior repo is a real differentiator; these make it green.
"""
import pytest
from parla.graph.state import DetectedError, Feedback
from parla.tools import verifier


def test_verifier_rejects_ungrounded_correction():
    fb = Feedback(text="use 'a' before 'apple'",
                  corrections=[DetectedError(span="a apple", error_type="M:DET",
                                             suggestion="an apple", rule_id=None)])
    # a correction with no rule_id must NOT be considered grounded
    assert verifier.is_grounded(fb, retrieved_rules=[]) is False


@pytest.mark.skip(reason="implement detect() first")
def test_detect_finds_subject_verb_agreement():
    from parla.tools import error_detection
    errs = error_detection.detect("He go to school.")
    assert any("VERB" in e.error_type for e in errs)
