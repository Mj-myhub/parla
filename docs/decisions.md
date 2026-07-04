# Decision log (ADR-lite)

Cheap way to signal engineering maturity: record *why*, not just *what*.

1. **LangGraph over bare LangChain** — need cycles (verifier retry), state, checkpointing.
2. **Hybrid retrieval (dense + BM25) + reranker** — grammar rule lookup benefits from
   lexical match on error tags; report retrieval quality with numbers.
3. **F0.5 as the headline GEC metric** — precision matters more than recall for a tutor;
   a wrong correction misleads a learner.
4. **Provider-agnostic LLM layer + optional router** — swap models in one line; enables
   the cost story (cheap model to plan, stronger to synthesize).
5. **Verifier as a hard gate** — mitigates LLM06 excessive agency; measured effect.
