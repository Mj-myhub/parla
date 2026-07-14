"""Spellcheck layer + sentence-chunked LLM pass — validated with mocks."""
from parla.tools import error_detection as ed


def test_spellcheck_catches_nonwords_only():
    errs = ed.detect("The aventages of the protein cointend in it, but flash is a real word.",
                     use_spacy=False)
    spans = {e.span for e in errs}
    assert "aventages" in spans and "cointend" in spans
    assert "flash" not in spans
    assert all(e.error_type == "R:SPELL" for e in errs if e.span in {"aventages", "cointend"})


def test_spellcheck_skips_proper_nouns_and_short_words():
    errs = ed.detect("Rome is in Lazio.", use_spacy=False)
    assert errs == []


class _Resp:
    content = "[]"

class _Counting:
    def __init__(self): self.calls = 0
    def invoke(self, msgs):
        self.calls += 1
        return _Resp()


def test_llm_pass_is_chunked_per_sentence():
    m = _Counting()
    ed.detect("First sentence. Second one! Third here?", use_spacy=False, use_llm=True, model=m)
    assert m.calls == 3


def test_single_sentence_makes_one_call():
    m = _Counting()
    ed.detect("Just one sentence here.", use_spacy=False, use_llm=True, model=m)
    assert m.calls == 1
