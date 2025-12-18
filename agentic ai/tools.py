import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llm import generate_text
from prompt import build_rag_answer_prompt

# Global objects (cached once)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = None
chunks = []

def summarize(context: str) -> str:
    return generate_text(f"Summarize the following:\n{context}")

def extract_action_items(context: str) -> str:
    return generate_text(
        f"""
    From the text below, extract 5–7 concrete, actionable insights.
    Each action should:
    - Start with a verb
    - Be implementable by an organization
    - Be grounded in the document only

    Return as bullet points.

    TEXT:
    {context}
    """
    )


def build_vector_store(text_chunks):
    """Builds a FAISS index from `text_chunks`.

    This caches `chunks` and `index` globally for later retrieval.
    """
    global index, chunks
    if not text_chunks:
        raise ValueError("text_chunks must be a non-empty list of strings")

    chunks = list(text_chunks)

    # Ensure numpy output and float32 dtype for FAISS
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    embeddings = embeddings.astype('float32')

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)


def retrieve_context(query, k=5):
    """Retrieves top-k chunks for `query` from the FAISS index.

    Raises a RuntimeError if the index hasn't been built yet.
    """
    global index, chunks
    if index is None:
        raise RuntimeError("Vector store is not built. Call build_vector_store() first.")

    query_vec = embedder.encode([query], convert_to_numpy=True).astype('float32')
    distances, indices = index.search(query_vec, k)

    results = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(chunks):
            continue
        results.append(chunks[idx])

    return "\n".join(results)


def answer_question(context, question):
    prompt = build_rag_answer_prompt(context, question)
    return generate_text(prompt)
