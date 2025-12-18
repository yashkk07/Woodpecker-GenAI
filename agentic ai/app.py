import streamlit as st
from pypdf import PdfReader
from agent import Agent
from tools import build_vector_store

if "result" not in st.session_state:
    st.session_state.result = None

st.set_page_config(page_title="Agentic RAG", layout="wide")

st.title("📄 Agentic RAG System")
st.markdown("Upload a PDF and ask questions. Answers come ONLY from the document.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

if uploaded_file:
    raw_text = extract_text(uploaded_file)
    chunks = chunk_text(raw_text)

    st.subheader("📄 Extracted Text Preview")
    st.write(raw_text[:1000] + "...")

    st.info(f"Total chunks created: {len(chunks)}")

    build_vector_store(chunks)

    query = st.text_input("Ask a question about the document")

    if st.button("Run Agent"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Agent is thinking..."):
                agent = Agent()
                st.session_state.result = agent.run(query)

            if st.session_state.result:
                summary = st.session_state.result["summary"]
                actions = st.session_state.result["actions"]

                final_text = f"""
                    SUMMARY
                    -------
                    {summary}

                    ACTIONABLE INSIGHTS
                    ------------------
                    {actions}
                    """.strip()

                st.subheader("📌 Summary")
                st.write(summary)

                st.subheader("✅ Actionable Insights")
                st.write(actions)

                st.download_button(
                    label="📥 Download results as text file",
                    data=final_text,
                    file_name="agentic_rag_output.txt",
                    mime="text/plain"
                )