"""Retrieval spine works end-to-end over the real seed corpus."""
from parla.graph.state import DetectedError
from parla.retrieval.index import RuleIndex
from parla.tools import retrieval


def test_index_finds_determiner_rule():
    idx = RuleIndex.load()
    hits = idx.search("M:DET a apple article", k=3)
    assert hits, "expected at least one hit"
    assert any(h["rule_id"].startswith("DET") for h in hits[:2])


def test_rules_for_maps_errors_to_grounding():
    errs = [DetectedError(span="he go", error_type="R:VERB:SVA", suggestion="he goes")]
    rules = retrieval.rules_for(errs, level="A2")
    assert any(r["rule_id"].startswith("SVA") for r in rules)


def test_level_bias_prefers_closer_level():
    idx = RuleIndex.load()
    hits = idx.search("prepositions of time in on at", level="A2", k=5)
    assert hits[0]["cefr"] in {"A2", "B1"}
