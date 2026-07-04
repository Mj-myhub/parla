"""Streamlit demo — deploy to Hugging Face Spaces for the public live link."""
import streamlit as st
from parla.graph.graph import build_graph

st.set_page_config(page_title="Parla — English writing tutor", page_icon="✍️")
st.title("Parla ✍️  — grounded English writing feedback")

graph = build_graph()
learner_id = st.text_input("Learner ID", value="demo-user")
text = st.text_area("Paste your writing:", height=200)

if st.button("Get feedback") and text.strip():
    with st.spinner("Analyzing…"):
        result = graph.invoke(
            {"learner_id": learner_id, "submission": text, "max_verify_attempts": 2}
        )
    st.subheader("Feedback")
    st.write(result["feedback"].text)
    st.caption(f"Assessed level: {result.get('level')}  ·  grounded: "
               f"{result['feedback'].grounded}")
    if result.get("exercise"):
        st.subheader("Practice")
        st.write(result["exercise"])
