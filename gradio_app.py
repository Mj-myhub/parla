"""Parla — Gradio UI for Hugging Face Spaces. Same agent, different front-end."""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from parla.graph.graph import build_graph
from parla.tools.tag_labels import label as tag_label

_graph = build_graph()
_PLACEHOLDER = "He go to school and she have two cat. Yesterday I have visited Rome."


def _fmt(items, rules_by_id, with_rule):
    if not items:
        return "_None._"
    lines = []
    for r in items:
        c = r["correction"]
        lines.append(f"- **{c.span}** → **{c.suggestion}**  ·  _{tag_label(c.error_type)}_")
        if with_rule:
            rule = rules_by_id.get(c.rule_id or "", {})
            if rule:
                lines.append(f"    - 📖 {rule.get('title','')} ({rule.get('cefr','')})")
    return "\n".join(lines)


def run(text, learner_id):
    if not text.strip():
        return "Please paste some writing above.", "", "", ""
    result = _graph.invoke({"learner_id": learner_id or "demo",
                            "submission": text, "max_verify_attempts": 2})
    results = result.get("verification", [])
    rules_by_id = {r["rule_id"]: r for r in result.get("retrieved_rules", [])}
    verified = [r for r in results if r["status"] == "verified"]
    suggestions = [r for r in results if r["status"] != "verified"]

    header = (f"**Assessed level:** {result.get('level', '—')}  ·  "
              f"**Verified grammar fixes:** {len(verified)} of {len(results)}")
    verified_md = "### ✅ Corrections (verified against a rule)\n" + _fmt(verified, rules_by_id, True)
    suggestions_md = ("### 💡 Suggestions (not verified as grammar)\n"
                      "_From the language model; optional improvements or style._\n\n"
                      + _fmt(suggestions, rules_by_id, False))
    feedback_md = "### Feedback\n" + result["feedback"].text
    if result.get("exercise"):
        feedback_md += "\n\n### Practice\n" + result["exercise"]
    return header, verified_md, suggestions_md, feedback_md


with gr.Blocks(title="Parla — English writing tutor") as demo:
    gr.Markdown("# Parla ✍️\nAn evaluation-driven agentic English writing tutor. "
                "Grammar corrections are **verified against a real rule**; everything else "
                "is shown honestly as a **suggestion**.")
    with gr.Row():
        inp = gr.Textbox(label="Paste your writing", lines=8, placeholder=_PLACEHOLDER, scale=3)
        lid = gr.Textbox(label="Learner ID", value="demo", scale=1)
    btn = gr.Button("Get feedback", variant="primary")
    header = gr.Markdown()
    with gr.Row():
        verified_out = gr.Markdown()
        suggestions_out = gr.Markdown()
    feedback_out = gr.Markdown()
    btn.click(run, inputs=[inp, lid], outputs=[header, verified_out, suggestions_out, feedback_out])

if __name__ == "__main__":
    demo.launch()
