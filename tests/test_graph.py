"""Smoke test: the graph compiles and has the retry edge wired correctly."""
from parla.graph.graph import build_graph, should_retry
from parla.graph.state import Feedback


def test_graph_compiles():
    assert build_graph() is not None


def test_retry_logic_stops_at_max_attempts():
    fb = Feedback(text="x", grounded=False)
    state = {"feedback": fb, "verify_attempts": 2, "max_verify_attempts": 2}
    assert should_retry(state) == "commit"
