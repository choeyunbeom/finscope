"""Query Rewriter node — rewrites the search query based on Critic feedback."""

from groq import Groq

from src.api.core.config import settings
from src.agents.graph import AgentState

try:
    from langfuse import observe
except ImportError:
    def observe(fn=None, **kwargs):
        return fn if fn is not None else lambda f: f

REWRITE_PROMPT = """You are a search query optimiser for financial document retrieval.

The previous search query did not retrieve sufficient evidence. The critic provided this feedback:
{feedback}

Original query: {query}

Rewrite the query to retrieve MORE RELEVANT financial document excerpts. Strategies:
- Use different financial terminology (e.g. "revenue" → "net sales", "risk" → "uncertainties")
- Be more specific about the aspect that lacked citations
- Add relevant financial keywords (EBITDA, operating income, segment revenue, etc.)

Return ONLY the rewritten query, nothing else."""


@observe(name="query-rewriter-node")
async def query_rewriter_node(state: AgentState) -> dict:
    feedback = state.get("critique_feedback", "")
    original_query = state["query"]

    if not feedback:
        return {"query": original_query}

    client = Groq(api_key=settings.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[{
            "role": "user",
            "content": REWRITE_PROMPT.format(feedback=feedback, query=original_query),
        }],
        temperature=0.3,
    )

    rewritten = response.choices[0].message.content.strip().strip('"')
    return {"query": rewritten}
