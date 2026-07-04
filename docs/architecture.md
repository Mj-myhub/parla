# Architecture

See the diagram in the README. Key decisions:

- **LangGraph orchestrator, LangChain toolkit.** The conditional verifier→plan retry edge
  and the persisted learner profile are exactly what graph orchestration is for.
- **Deterministic + LLM error detection.** The rule checker gives precision and a signal
  to reason over; the LLM adds recall. Reconciled before feedback.
- **Grounding gate.** Nothing reaches the learner unless the verifier can trace each
  correction to a retrieved rule. This is the safety story.
- **Memory earns its place.** The learner profile drives personalization and is the
  reason feedback improves across sessions.
