import numpy as np
from embeddings import model

def retrieve_chunks(query, index, chunks, k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    retrieved_distances = distances[0]

    return retrieved_chunks, retrieved_distances
