def build_planner_prompt(goal: str, memory: str) -> str:
    return f"""
You are an autonomous AI agent.

GOAL:
{goal}

PAST STEPS:
{memory}

Decide the NEXT action.

AVAILABLE ACTIONS:
- RETRIEVE_CONTEXT
- ANSWER
- FINISH

RULES:
- Choose exactly ONE action
- Respond ONLY in JSON
- No explanations

FORMAT:
{{
  "action": "<ACTION>",
  "input": "<string>"
}}
"""

def build_rag_answer_prompt(context: str, question: str) -> str:
    return f"""
Answer the question STRICTLY using the context below.
If the answer is not present, say "Not found in document".

CONTEXT:
{context}

QUESTION:
{question}
"""
