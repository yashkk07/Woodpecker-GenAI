def format_memory(memory):
    if not memory:
        return "None"

    lines = []
    for step in memory:
        lines.append(
            f"Action: {step['action']} | Observation: {step['observation'][:200]}"
        )
    return "\n".join(lines)
