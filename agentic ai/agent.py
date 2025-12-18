import tools

class Agent:
    def __init__(self):
        self.context = ""

    def run(self, goal: str) -> dict:
        # Step 1: Retrieve context
        self.context = tools.retrieve_context(goal)

        # Step 2: Summarize
        summary = tools.summarize(self.context)

        # Step 3: Extract actionable insights
        actions = tools.extract_action_items(summary)

        return {
            "summary": summary,
            "actions": actions
        }
