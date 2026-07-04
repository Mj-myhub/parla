"""Layer 3 structure — validated with a mock model, no API key needed."""
from parla.tools import error_detection as ed


class _Resp:
    def __init__(self, content): self.content = content

class _FakeLLM:
    def __init__(self, content): self._c = content
    def invoke(self, messages): return _Resp(self._c)

class _BrokenLLM:
    def invoke(self, messages): raise RuntimeError("no key")


def test_extract_json_handles_fences_and_prose():
    assert ed._extract_json_array('```json\n[{"span":"x","suggestion":"y"}]\n```')
    assert ed._extract_json_array('sure: [{"span":"x","suggestion":"y"}] done')
    assert ed._extract_json_array('no json here') == []


def test_llm_pass_parses_mocked_response():
    llm = _FakeLLM('[{"span":"have visited","error_type":"R:VERB:TENSE","suggestion":"visited"}]')
    errs = ed._llm_pass("I have visited Rome last year.", model=llm)
    assert len(errs) == 1 and errs[0].error_type == "R:VERB:TENSE"


def test_detect_reconciles_layers():
    llm = _FakeLLM('[{"span":"have visited","error_type":"R:VERB:TENSE","suggestion":"visited"}]')
    out = ed.detect("I saw a apple.", use_spacy=False, use_llm=True, model=llm)
    kinds = {e.error_type for e in out}
    assert "R:DET" in kinds and "R:VERB:TENSE" in kinds


def test_llm_failure_is_graceful():
    assert ed._llm_pass("x", model=_BrokenLLM()) == []
