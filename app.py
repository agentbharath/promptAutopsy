import streamlit as st
from agent import run_pipeline
import re

def clean_prompt(text):
    # Remove citation comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Remove extra blank lines left behind
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

st.set_page_config(
    page_title="PromptAutopsy",
    layout="centered"
)

st.title("Prompt Autopsy")
st.caption("Pre-flight checklist for your LLM prompts")

user_prompt = st.text_area(
    label="Enter your query here",
    height=200
)

analyse_clicked = st.button(
    label="🔍 Analyse My Prompt"
)

if analyse_clicked and user_prompt:
    with st.spinner("Analysing your prompt..."):
        st.session_state["result"] = run_pipeline(user_prompt)

if "result" in st.session_state:
    result = st.session_state["result"]
    if result["stopped"]:
        st.warning(result["stopped"])
        if result.get("hint"):
            st.info(result["hint"])
    else:
        st.success("✅ Analysis complete")
        st.subheader("🩺 Diagnosis")
        st.metric(label="Overall health", value=result["diagnosis"]["overall_health"])
        st.metric(label="Issues found", value=result["diagnosis"]["issues_found"])

        for mode in ["vague_instruction", "missing_context", "wrong_format", "conflicting_instructions", "missing_examples"]:
            details = result["diagnosis"][mode]
            if details["detected"]:
                st.error(f"**{mode.replace('_', ' ').title()}** ({details['severity']}): {details['reason']}")
            else:
                st.success(f"**{mode.replace('_', ' ').title()}**: ✅ Not detected")

        st.subheader("📊 Improvement Score")
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        dims = ["clarity", "specificity", "completeness"]

        for i, dim in enumerate(dims):
            with cols[i]:
                st.metric(
                    label=dim.capitalize(),
                    value=f"{result['scores'][dim]['after']}/5",
                    delta=result['scores'][dim]['delta']
                )

        st.metric(label="Overall Delta", value=result["scores"]["overall_delta"])

        st.subheader("✏️ Rewritten Prompt")
        if result["scores"]["show_rewrite"]:

            tab1, tab2 = st.tabs(["✅ Ready to Use", "🧩 Editable Template"])
            with tab1:
                st.code(clean_prompt(result["rewritten_filled"]))
                st.download_button("⬇️ Download", result["rewritten_filled"], 
                                "prompt_ready.txt")
            with tab2:
                st.code(clean_prompt(result["rewritten_template"]))
                st.download_button("⬇️ Download", result["rewritten_template"], 
                                "prompt_template.txt")
        else:
            st.info("✅ Your prompt is already well-structured — no rewrite needed.")
