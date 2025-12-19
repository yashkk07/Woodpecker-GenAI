import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def get_llm():
    # Cap output tokens to reduce cost. Model and token cap can be overridden via env.
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=model,
        temperature=float(os.getenv("GROQ_TEMPERATURE", "0.2")),
    )
