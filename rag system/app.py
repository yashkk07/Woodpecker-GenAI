import streamlit as st

from pdf_loader import extract_text_from_pdf
from chunker import chunk_text
from embeddings import build_vector_store
from rag import retrieve_chunks
from llm import generate_answer
from prompt import build_rag_prompt

def context_coverage_score(answer: str, context: str) -> float:
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())

    if not answer_tokens:
        return 0.0

    overlap = answer_tokens.intersection(context_tokens)
    return len(overlap) / len(answer_tokens)

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="PDF RAG System",
    layout="wide"
)

st.title("📄 PDF-based Question Answering (RAG)")
st.markdown(
    "Upload a PDF and ask questions. "
    "**Answers are generated strictly from the document content.**"
)

st.divider()

# -------------------------------
# File upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"]
)

# -------------------------------
# Main RAG pipeline
# -------------------------------
if uploaded_file is not None:

    # Step 1: Extract + clean text
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    if not text.strip():
        st.error("No readable text found in the PDF.")
        st.stop()

    # Preview extracted text
    st.subheader("📄 Extracted Text Preview")
    st.text(text[:1000] + ("..." if len(text) > 1000 else ""))

    st.divider()

    # Step 2: Chunk text
    with st.spinner("Chunking document..."):
        chunks = chunk_text(text)

    st.write(f"🔹 Total chunks created: **{len(chunks)}**")

    st.divider()

    # Step 3: Build vector store
    with st.spinner("Building vector index..."):
        index, _ = build_vector_store(chunks)

    # -------------------------------
    # Question input
    # -------------------------------
    question = st.text_input(
        "Ask a question about the document:",
        placeholder="e.g. Summarize in less than 100 words"
    )

    if question:

        # Step 4: Retrieve relevant chunks
        with st.spinner("Retrieving relevant context..."):
            retrieved_chunks, distances = retrieve_chunks(
                query=question,
                index=index,
                chunks=chunks,
                k=5
            )

        context = "\n\n".join(retrieved_chunks)
        
        avg_distance = sum(distances) / len(distances)
        retrieval_confidence = 1 / (1 + avg_distance)

        # Step 5: Build grounded prompt (FROM prompt.py)
        prompt = build_rag_prompt(
            context=context,
            question=question
        )

        # Step 6: Generate answer
        with st.spinner("Generating answer..."):
            answer = generate_answer(prompt)
        coverage_score = context_coverage_score(answer, context)
        trust_score = (
            0.7 * retrieval_confidence +
            0.3 * coverage_score
        )



        # -------------------------------
        # Display answer
        # -------------------------------
        st.subheader("📌 Answer")
        st.write(answer)

        st.subheader("📊 Answer Confidence")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Retrieval Confidence",
            f"{retrieval_confidence:.2f}"
        )

        col2.metric(
            "Context Coverage",
            f"{coverage_score:.2f}"
        )

        label = (
            "High Trust" if trust_score > 0.65
            else "Medium Trust" if trust_score > 0.45
            else "Low Trust"
        )

        col3.metric(
            "Trust Score",
            f"{trust_score:.2f}",
            label
        )


        # Optional: show retrieved chunks
        with st.expander("🔍 View retrieved context"):
            for i, chunk in enumerate(retrieved_chunks, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.write(chunk)
                st.divider()

else:
    st.info("Please upload a PDF file to begin.")