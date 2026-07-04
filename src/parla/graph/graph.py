"""LangGraph orchestration.

Design note (see docs/decisions.md): LangGraph, not bare LangChain, because we need a
conditional retry edge (verifier -> plan) and a persisted learner profile. As of the
2026 stack, LangChain is the toolkit and LangGraph is the orchestrator.
"""
from __future__ import annotations
from langgraph.graph import StateGraph, START, END

from parla.graph.state import TutorState
from parla.tools import cefr_assessor, error_detection, retrieval, exercise_gen, verifier
from parla.memory import learner_profile


def assess_node(state: TutorState) -> TutorState:
    profile = learner_profile.load(state["learner_id"])
    state["level"] = cefr_assessor.assess(state["submission"], prior=profile.level)
    return state


def detect_node(state: TutorState) -> TutorState:
    state["errors"] = error_detection.detect(state["submission"])
    return state


def retrieve_node(state: TutorState) -> TutorState:
    state["retrieved_rules"] = retrieval.rules_for(state["errors"], level=state["level"])
    return state


def plan_and_generate_node(state: TutorState) -> TutorState:
    # TODO: build the feedback with an LLM, constrained to cite retrieved_rules,
    #       pitched at state["level"], and aware of the learner's recurring errors.
    state["feedback"] = ...  # -> Feedback(...)
    return state


def verify_node(state: TutorState) -> TutorState:
    state["feedback"].grounded = verifier.is_grounded(
        state["feedback"], state["retrieved_rules"]
    )
    state["verify_attempts"] = state.get("verify_attempts", 0) + 1
    return state


def should_retry(state: TutorState) -> str:
    grounded = state["feedback"].grounded
    attempts = state.get("verify_attempts", 0)
    if grounded or attempts >= state.get("max_verify_attempts", 2):
        return "commit"
    return "retry"


def commit_node(state: TutorState) -> TutorState:
    state["exercise"] = exercise_gen.generate(state["errors"], level=state["level"])
    learner_profile.update(state["learner_id"], state["errors"], state["level"])
    return state


def build_graph():
    g = StateGraph(TutorState)
    g.add_node("assess", assess_node)
    g.add_node("detect", detect_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("plan_generate", plan_and_generate_node)
    g.add_node("verify", verify_node)
    g.add_node("commit", commit_node)

    g.add_edge(START, "assess")
    g.add_edge("assess", "detect")
    g.add_edge("detect", "retrieve")
    g.add_edge("retrieve", "plan_generate")
    g.add_edge("plan_generate", "verify")
    g.add_conditional_edges("verify", should_retry,
                            {"retry": "plan_generate", "commit": "commit"})
    g.add_edge("commit", END)
    return g.compile()
