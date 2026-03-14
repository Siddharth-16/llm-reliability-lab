import streamlit as st

st.set_page_config(page_title="LLM Reliability Lab", layout="wide")

st.title("LLM Reliability Lab")
st.subheader("RAG reliability evaluation dashboard")

st.markdown(
    """
This dashboard will be used to:

- inspect evaluation runs
- analyze retrieval quality
- track hallucination and faithfulness
- detect prompt injection patterns
- compare RAG configurations
"""
)

st.info("dashboard is running.")