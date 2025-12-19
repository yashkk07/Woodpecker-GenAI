from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import numpy as np

# Try to import Embeddings base class from installed langchain packages
try:
    from langchain_core.embeddings.base import Embeddings
except Exception:
    try:
        from langchain.embeddings.base import Embeddings
    except Exception:
        Embeddings = object


class SentenceTransformerEmbeddingWrapper(Embeddings):
    """Adapter that wraps a SentenceTransformer-style encoder and
    implements the LangChain `Embeddings` interface.

    This avoids passing a raw function/callable to vectorstores and
    prevents the deprecation warning about `embedding_function`.
    """
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        # return List[List[float]]
        arr = self.model.encode(texts, convert_to_numpy=True)
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        return [list(x) for x in arr]

    def embed_query(self, text):
        arr = self.model.encode([text], convert_to_numpy=True)
        if isinstance(arr, np.ndarray):
            return arr[0].tolist()
        return list(arr[0])


def build_vector_store(chunks, embedder):
    """
    Builds and returns a LangChain FAISS vector store.

    Args:
        chunks (List[str]): Text chunks extracted from the PDF
        embedder: LangChain-compatible embedding model

    Returns:
        FAISS: LangChain FAISS vector store with similarity_search support
    """

    # Convert text chunks into LangChain Documents
    documents = [
        Document(page_content=chunk) for chunk in chunks
    ]

    # If user passed a raw SentenceTransformer (or any object with
    # `encode`), wrap it so LangChain can call `embed_documents`.
    if not hasattr(embedder, "embed_documents") and hasattr(embedder, "encode"):
        embedding = SentenceTransformerEmbeddingWrapper(embedder)
    else:
        embedding = embedder

    # Build FAISS vector store
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding
    )

    return vector_store
