from langchain.tools import tool
from typing import List
import os

# These will be injected at runtime
_vector_store = None
_llm = None
_last_context = None

# configurable retrieve k (can be set via env or at runtime)
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "4"))


def set_retrieval_k(k: int):
    """Set the number of chunks to retrieve for semantic search."""
    global RETRIEVE_K
    try:
        k = int(k)
    except Exception:
        return
    RETRIEVE_K = max(1, k)


def initialize_tools(vector_store, llm):
    """
    Initialize shared objects for tools.
    This MUST be called once after vector store creation.
    """
    global _vector_store, _llm
    _vector_store = vector_store
    _llm = llm
    global _last_context
    _last_context = None


@tool
def retrieve_context(query: str) -> str:
    """
    Retrieve the most relevant document chunks for a given query.
    Uses semantic similarity search over the vector store.
    """
    if _vector_store is None:
        raise RuntimeError("Vector store not initialized.")

    # Retrieve top-k chunks (configurable) and truncate to keep payload reasonable
    docs = _vector_store.similarity_search(query, k=RETRIEVE_K)

    def _truncate(s: str, n: int = 1500):
        if not s:
            return s
        if len(s) <= n:
            return s
        # try to cut at a newline for readability
        part = s[:n]
        if "\n" in part:
            part = part.rsplit("\n", 1)[0]
        return part + ("..." if len(s) > n else "")

    result = "\n\n".join(
        f"[Chunk {i+1}]\n{_truncate(doc.page_content, 1500)}"
        for i, doc in enumerate(docs)
    )

    # cache last retrieved context so subsequent tool calls can use it
    global _last_context
    _last_context = result

    return result


@tool
def summarize_context(context: str) -> str:
    """
    Summarize the provided document context.
    """
    if _llm is None:
        raise RuntimeError("LLM not initialized.")

    # If the model passed a placeholder string indicating it expects the
    # previously retrieved context, substitute the cached context.
    global _last_context
    if not context or "document context retrieved" in context.lower() or "you haven't provided" in context.lower():
        if _last_context:
            # truncate cached context to keep prompt size reasonable
            context = _last_context[:4000]
        else:
            return "No document context available. Please run `retrieve_context` first or provide the document text."

    prompt = f"""
Summarize the following document context clearly and concisely:

{context}
"""
    response = _llm.invoke(prompt)
    return getattr(response, "content", str(response))


@tool
def extract_action_items(context: str) -> str:
    """
    Extract clear, actionable insights from the document context.
    """
    if _llm is None:
        raise RuntimeError("LLM not initialized.")

    global _last_context
    if not context or "document context retrieved" in context.lower() or "you haven't provided" in context.lower():
        if _last_context:
            # truncate cached context to keep prompt size reasonable
            context = _last_context[:4000]
        else:
            return "No document context available. Please run `retrieve_context` first or provide the document text."

    prompt = f"""
From the following document context, extract 5–7 clear, actionable insights.
Present them as bullet points.
Only use the provided content.

{context}
"""
    response = _llm.invoke(prompt)
    return getattr(response, "content", str(response))


@tool
def brave_search(query: str) -> str:
    """Stub for web search calls.

    This environment is configured to answer STRICTLY from uploaded
    documents. External web search is disabled to prevent tool misuse.
    The model may still attempt to call `brave_search`; return a clear
    message instructing it to use `retrieve_context` instead.
    """
    return (
        "External web search disabled. "
        "Use the `retrieve_context` tool to fetch document-based evidence."
    )
