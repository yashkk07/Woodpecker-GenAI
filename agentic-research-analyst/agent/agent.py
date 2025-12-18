from langchain.agents import create_agent
from llm.ollama import get_llm
from agent.tools import create_tools


def build_agent(vector_store):
    model = get_llm()
    tools = create_tools(vector_store, model)

    agent = create_agent(
        model=model,
        tools=tools
    )

    return agent
