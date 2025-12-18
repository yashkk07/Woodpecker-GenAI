# Agent memory placeholder
# Implement short-term / long-term memory mechanisms

class Memory:
    def __init__(self):
        self.items = []

    def add(self, entry: dict):
        self.items.append(entry)

    def format(self) -> str:
        return "\n".join(str(i) for i in self.items)
