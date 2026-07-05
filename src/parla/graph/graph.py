"""LangGraph orchestration — the agent loop.

assess -> detect -> retrieve -> plan_generate -> verify -(retry?)- commit

Design (docs/decisions.md): LangGraph, not bare LangChain, for the conditional verifier
retry edge and persisted learner memory. On retry we drop corrections that could not be
grounded, so the loop converges within max_verify_attempts.
"""
from __future__ import annotations
from langgraph.graph import StateGraph, START, END

from parla.graph.state import TutorState
from parla.tools import cefr_assessor, error_detection, retrieval, exercise_gen, verifier
from parla.tools.feedback import generate_feedback, ground_corrections
from parla.memory import learner_profile


def assess_node(state: TutorState) -> TutorState:
    profile = learner_profile.load(state["learner_id"])
    state["level"] = cefr_assessor.assess(state["submission"], prior=profile.level)
    return state


def detect_node(state: TutorState) -> TutorState:
    state["errors"] = error_detection.detect(state["submission"], use_llm=True)
    return state


def retrieve_node(state: TutorState) -> TutorState:
    state["retrieved_rules"] = retrieval.rules_for(state["errors"], level=state.get("level"))
    return state


def plan_and_generate_node(state: TutorState) -> TutorState:
    errors = state["errors"]
    rules = state["retrieved_rules"]
    if state.get("verify_attempts", 0) > 0:
        grounded = ground_corrections(errors, rules)
        errors = [e for e, g in zip(errors, grounded) if g.rule_id]
    state["feedback"] = generate_feedback(state["submission"], state.get("level", "B1"),
                                          errors, rules)
    return state


def verify_node(state: TutorState) -> TutorState:
    results = verifier.verify(state["feedback"].corrections, state["retrieved_rules"], strict=True)
    state["verification"] = results
    grammar = [r for r in results if r["status"] != "suggestion"]
    state["feedback"].grounded = all(r["status"] == "verified" for r in grammar) if grammar else True
    state["verify_attempts"] = state.get("verify_attempts", 0) + 1
    return state


def should_retry(state: TutorState) -> str:
    if state["feedback"].grounded or state.get("verify_attempts", 0) >= state.get("max_verify_attempts", 2):
        return "commit"
    return "retry"


def commit_node(state: TutorState) -> TutorState:
    state["exercise"] = exercise_gen.generate(state["errors"], state.get("level", "B1"))
    learner_profile.update(state["learner_id"], state["feedback"].corrections, state.get("level", "B1"))
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
    g.add_conditional_edges("verify", should_retry, {"retry": "plan_generate", "commit": "commit"})
    g.add_edge("commit", END)
    return g.compile()
