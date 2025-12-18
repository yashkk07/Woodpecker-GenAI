from langchain_core.tools import Tool

def create_tools(vector_store, llm):

    def search_document(query: str) -> str:
        results = vector_store.search(query)
        return "\n".join(results)

    def summarize_context(text: str) -> str:
        return llm.invoke(f"Summarize the following:\n{text}")

    def extract_risks(text: str) -> str:
        return llm.invoke(f"Extract business risks as bullet points:\n{text}")

    def extract_opportunities(text: str) -> str:
        return llm.invoke(f"Extract business opportunities as bullet points:\n{text}")

    def generate_recommendations(text: str) -> str:
        return llm.invoke(f"Generate actionable recommendations:\n{text}")

    return [
        Tool(
            name="SearchDocument",
            func=search_document,
            description="Retrieve relevant document sections"
        ),
        Tool(
            name="Summarize",
            func=summarize_context,
            description="Summarize retrieved context"
        ),
        Tool(
            name="ExtractRisks",
            func=extract_risks,
            description="Extract business risks"
        ),
        Tool(
            name="ExtractOpportunities",
            func=extract_opportunities,
            description="Extract business opportunities"
        ),
        Tool(
            name="GenerateRecommendations",
            func=generate_recommendations,
            description="Generate actionable recommendations"
        ),
    ]
