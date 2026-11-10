"""Streamlit UI for the original Day 7 ABC Technologies HR assistant."""
from pathlib import Path
import os

import streamlit as st

from ingest import DEFAULT_DB_DIR, ingest_documents
from rag import PROFILES, answer_question


DB_DIR = Path(os.getenv("DAY7_CHROMA_DIR", str(DEFAULT_DB_DIR)))

st.set_page_config(page_title="ABC Technologies HR Knowledge Assistant", page_icon="📚")
st.title("ABC Technologies HR Knowledge Assistant")
st.caption("Fictional course project • three representative documents indexed")

with st.sidebar:
    st.header("Simulated user profile")
    profile_name = st.selectbox("Profile", list(PROFILES))
    generation_label = st.selectbox(
        "Generation mode",
        ["Deterministic no-key rehearsal", "Live OpenAI"],
    )
    st.warning("The profile selector simulates identity. Production roles must come from trusted authentication.")

question = st.text_input(
    "Ask an HR question",
    value="How many days per week can employees work remotely?",
)

if st.button("Ask", type="primary"):
    if not DB_DIR.exists():
        with st.spinner("Indexing the three sample documents..."):
            ingest_documents(DB_DIR)
    generation = "openai" if generation_label == "Live OpenAI" else "deterministic"
    try:
        result = answer_question(
            question,
            PROFILES[profile_name],
            generation=generation,
            db_dir=DB_DIR,
        )
    except Exception as error:
        st.error(f"Unable to answer: {error}")
    else:
        if generation == "deterministic":
            st.info("Deterministic no-key rehearsal — this is not live LLM generation.")
        st.subheader("Answer")
        st.write(result["answer"])
        st.subheader("Authorized sources")
        if not result["sources"]:
            st.write("No authorized sources retrieved.")
        for source in result["sources"]:
            with st.expander(source["metadata"]["title"]):
                st.write(source["text"])
                st.caption(
                    f"Source: {source['metadata']['source']} • Access: {source['metadata']['access']}"
                )
