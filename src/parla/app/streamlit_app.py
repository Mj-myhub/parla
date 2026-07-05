"""Parla — Streamlit demo. Honest split: verified grammar corrections vs suggestions."""
from dotenv import load_dotenv
load_dotenv(".env")

import streamlit as st
from parla.graph.graph import build_graph
from parla.memory import learner_profile
from parla.tools.tag_labels import label as tag_label

st.set_page_config(page_title="Parla — English writing tutor", page_icon="✍️", layout="centered")


@st.cache_resource
def _graph():
    return build_graph()


learner_id = st.sidebar.text_input("Learner ID", value="demo")
if st.sidebar.button("Reset history"):
    f = learner_profile.STORE_DIR / f"{learner_id}.json"
    if f.exists():
        f.unlink()
st.sidebar.markdown("---")

st.title("Parla ✍️")
st.caption("An evaluation-driven agentic English writing tutor. Grammar corrections are "
           "verified against a real rule; everything else is shown honestly as a suggestion.")

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
                st.markdown(f"**{c.span}** → **{c.suggestion}**")
                st.caption(f"🏷️ {tag_label(c.error_type)}  ·  `{c.error_type}`")
                if rule:
                    st.caption(f"📖 {rule.get('title','')} ({rule.get('cefr','')}) — {rule.get('rule','')}")

    if suggestions:
        st.subheader("💡 Suggestions (not verified as grammar)")
        st.caption("These come from the language model and are not grounded in a rule — "
                   "treat them as optional: possible improvements or style/word choice.")
        for r in suggestions:
            c = r["correction"]
            with st.container(border=True):
                st.markdown(f"**{c.span}** → **{c.suggestion}**")
                st.caption(f"🏷️ {tag_label(c.error_type)}  ·  `{c.error_type}`")

    st.subheader("Feedback")
    st.write(result["feedback"].text)

    if result.get("exercise"):
        with st.expander("Practice exercises"):
            st.write(result["exercise"])

st.sidebar.subheader("Recurring errors")
prof = learner_profile.load(learner_id)
if prof.error_counts:
    st.sidebar.bar_chart(prof.error_counts)
    st.sidebar.caption(f"Total logged: {sum(prof.error_counts.values())}  ·  level {prof.level}")
    with st.sidebar.expander("What do these tags mean?"):
        for t in sorted(prof.error_counts):
            st.caption(f"**{t}** — {tag_label(t)}")
else:
    st.sidebar.caption("No history yet — submit some writing.")
