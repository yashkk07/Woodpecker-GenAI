import streamlit as st

from rag.ingest import ingest_pdf
from rag.embeddings import get_embedder
from rag.vector_store import VectorStore
from agent.agent import build_agent

st.set_page_config(page_title="Agentic Research Analyst", layout="wide")
st.title("📄 Agentic AI Research & Insight Agent")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    chunks = ingest_pdf(uploaded_file)

    st.success(f"Document ingested. Total chunks: {len(chunks)}")

    embedder = get_embedder()
    vector_store = VectorStore(embedder)
    vector_store.build(chunks)

    agent = build_agent(vector_store)

    user_goal = st.text_input(
        "Enter your goal",
        placeholder="e.g. Summarize the document and give risks and recommendations"
    )

    if st.button("Run Agent"):
        with st.spinner("Agent reasoning..."):
            response = agent.invoke({"input": user_goal})

        st.subheader("📌 Agent Output")
        st.write(response["output"])
