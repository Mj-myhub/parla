"""Parla — Streamlit demo. Shows an HONEST split: grammar corrections verified against a
rule vs. suggestions (unverified grammar + style/word-choice)."""
from dotenv import load_dotenv
load_dotenv(".env")

import streamlit as st
from parla.graph.graph import build_graph
from parla.memory import learner_profile

st.set_page_config(page_title="Parla — English writing tutor", page_icon="✍️", layout="centered")


@st.cache_resource
def _graph():
    return build_graph()


st.title("Parla ✍️")
st.caption("An evaluation-driven agentic English writing tutor. Grammar corrections are "
           "verified against a real rule; everything else is shown honestly as a suggestion.")

with st.sidebar:
    st.header("Learner")
    learner_id = st.text_input("Learner ID", value="demo")
    st.markdown("---")
    prof = learner_profile.load(learner_id)
    if prof.error_counts:
        st.subheader("Recurring errors")
        st.bar_chart(prof.error_counts)
    else:
        st.caption("No history yet — submit some writing.")

text = st.text_area("Paste your writing:", height=180,
                    placeholder="He go to school and she have two cat. Yesterday I have visited Rome.")

if st.button("Get feedback", type="primary") and text.strip():
    with st.spinner("Analysing…"):
        result = _graph().invoke({"learner_id": learner_id, "submission": text,
                                  "max_verify_attempts": 2})
    results = result.get("verification", [])
    rules_by_id = {r["rule_id"]: r for r in result.get("retrieved_rules", [])}
    verified = [r for r in results if r["status"] == "verified"]
    suggestions = [r for r in results if r["status"] != "verified"]

    c1, c2 = st.columns(2)
    c1.metric("Assessed level", result.get("level", "—"))
    c2.metric("Verified grammar fixes", f"{len(verified)} of {len(results)}")

    if verified:
        st.subheader("✅ Corrections (verified against a rule)")
        for r in verified:
            c = r["correction"]
            rule = rules_by_id.get(c.rule_id or "", {})
            with st.container(border=True):
                st.markdown(f"**{c.span}** → **{c.suggestion}**  &nbsp; `{c.error_type}`")
                if rule:
                    st.caption(f"📖 {rule.get('title','')} ({rule.get('cefr','')}) — {rule.get('rule','')}")

    if suggestions:
        st.subheader("💡 Suggestions (not verified as grammar)")
        st.caption("These come from the language model and are not grounded in a rule — "
                   "treat them as optional: possible improvements or style/word choice.")
        for r in suggestions:
            c = r["correction"]
            label = "style / word choice" if c.error_type == "STYLE" else c.error_type
            with st.container(border=True):
                st.markdown(f"**{c.span}** → **{c.suggestion}**  &nbsp; `{label}`")

    st.subheader("Feedback")
    st.write(result["feedback"].text)

    if result.get("exercise"):
        with st.expander("Practice exercises"):
            st.write(result["exercise"])
