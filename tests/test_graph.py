"""Graph: compiles, retry gate works, and runs end-to-end (LLM falls back; no key needed)."""
import tempfile
from pathlib import Path
from parla.graph.graph import build_graph, should_retry
from parla.graph.state import Feedback
from parla.memory import learner_profile


def test_graph_compiles():
    assert build_graph() is not None


def test_retry_logic_stops_at_max_attempts():
    state = {"feedback": Feedback(text="x", grounded=False),
             "verify_attempts": 2, "max_verify_attempts": 2}
    assert should_retry(state) == "commit"


def test_retry_when_ungrounded_under_budget():
    state = {"feedback": Feedback(text="x", grounded=False),
             "verify_attempts": 1, "max_verify_attempts": 2}
    assert should_retry(state) == "retry"


def test_end_to_end_article_error(tmp_path, monkeypatch):
    monkeypatch.setattr(learner_profile, "STORE_DIR", Path(tmp_path))
    graph = build_graph()
    result = graph.invoke({"learner_id": "t1", "submission": "I saw a apple.",
                           "max_verify_attempts": 2})
    corr = result["feedback"].corrections
    assert any(c.error_type == "R:DET" and c.rule_id for c in corr)
    assert result["feedback"].grounded is True
    assert learner_profile.load("t1").error_counts.get("R:DET") == 1
