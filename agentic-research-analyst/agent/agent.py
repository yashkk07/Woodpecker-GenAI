from langchain.agents import create_agent
from llm.groq_llm import get_llm
from agent.tools import (
    plan_steps,
    retrieve_context,
    summarize_context,
    extract_action_items,
)

def build_agent():
    llm = get_llm()

    agent = create_agent(
        model=llm,
        tools=[
            plan_steps,
            retrieve_context,
            summarize_context,
            extract_action_items,
        ],
        system_prompt="""
You are an autonomous AI Research Analyst.

You MUST follow this workflow:
1. First, create a plan using the planning tool.
2. Then retrieve relevant document context.
3. Then summarize findings.
4. Then extract actionable insights.

Rules:
- Never answer without retrieving context.
- Base all outputs strictly on retrieved document chunks.
- If information is missing, explicitly say so.
- Act like a Chief Data Officer when proposing actions.
"""
    )

    return agent
