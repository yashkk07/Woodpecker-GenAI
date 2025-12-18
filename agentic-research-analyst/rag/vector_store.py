import faiss
import numpy as np

class VectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.index = None
        self.chunks = []

    def build(self, chunks):
        self.chunks = chunks
        vectors = self.embeddings.encode(chunks, convert_to_numpy=True).astype("float32")

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

    def search(self, query, k=5):
        query_vec = self.embeddings.encode([query], convert_to_numpy=True).astype("float32")
        _, indices = self.index.search(query_vec, k)

        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
