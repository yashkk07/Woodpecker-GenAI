from langchain_community.chat_models import ChatOllama

def get_llm():
    return ChatOllama(
        model="llama3:latest",   # or mistral, phi-3, etc.
        temperature=0.1
    )
