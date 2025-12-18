def build_rag_prompt(context, question):
    return f"""
You are a document question-answering assistant.

CONTEXT:
{context}

RULES:
- Answer ONLY from the context above.
- If the answer is not present, say:
  "The document does not contain this information."

QUESTION:
{question}
"""
