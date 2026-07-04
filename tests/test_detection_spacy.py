"""Layer 2 (spaCy) agreement checks — needs en_core_web_sm."""
import pytest
from parla.tools import error_detection as ed

spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_sm")
except OSError:
    pytest.skip("en_core_web_sm not installed", allow_module_level=True)


def _sug(text):
    return [(e.error_type, e.suggestion) for e in ed.detect(text)]


def test_sva_third_singular_base_verb():
    assert ("R:VERB:SVA", "He goes") in _sug("He go to school every day.")

def test_sva_plural_subject_s_verb():
    assert ("R:VERB:SVA", "They go") in _sug("They goes home after class.")

def test_sva_irregular_have():
    assert ("R:VERB:SVA", "She has") in _sug("She have two cat.")

def test_number_agreement_numeral():
    assert ("R:NOUN:NUM", "two cats") in _sug("She have two cat.")

def test_sva_noun_subject():
    assert ("R:VERB:SVA", "father works") in _sug("My father work in a bank.")

def test_no_false_positive_on_correct_sentences():
    for ok in ["He goes to school.", "They go home.", "I have a cat.", "She has two cats."]:
        assert ed.detect(ok) == [], f"false positive on: {ok}"

def test_present_perfect_left_to_llm():
    assert ed.detect("I have visited Rome last year.") == []
