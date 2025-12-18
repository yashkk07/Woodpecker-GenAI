def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap

    return chunks
