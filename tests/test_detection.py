"""Deterministic a/an detection — validated without any model or API key."""
import pytest
from parla.tools import error_detection as ed


@pytest.mark.parametrize("text,should_flag,expected", [
    ("I saw a apple on the table.", True, "an apple"),
    ("She is an university student.", True, "a university"),
    ("We waited an hour.", False, None),
    ("It was a European city.", False, None),
    ("He has an honest face.", False, None),
    ("I bought a book.", False, None),
])
def test_article_detection(text, should_flag, expected):
    errs = ed.detect(text)
    if should_flag:
        assert errs, f"expected a flag for: {text}"
        assert errs[0].error_type == "R:DET"
        assert errs[0].suggestion == expected
    else:
        assert not errs, f"false positive on: {text} -> {[e.span for e in errs]}"


def test_parse_llm_errors_is_defensive():
    payload = [{"span": "he go", "error_type": "R:VERB:SVA", "suggestion": "he goes"},
               {"junk": "no span"}]
    out = ed.parse_llm_errors(payload)
    assert len(out) == 1 and out[0].suggestion == "he goes"
