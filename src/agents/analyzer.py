"""Analyzer agent node — parallel Risk / Growth / Competitor analysis."""

import asyncio

from groq import Groq

from src.api.core.config import settings
from src.agents.graph import AgentState

try:
    from langfuse import observe
except ImportError:
    def observe(fn=None, **kwargs):
        return fn if fn is not None else lambda f: f

RISK_PROMPT = """You are a financial risk analyst. Based on the following excerpts from a financial filing,
identify and summarise the key risk factors. Be specific and cite the source text.

Excerpts:
{context}

Provide a concise risk analysis with citations."""

GROWTH_PROMPT = """You are a financial growth analyst. Based on the following excerpts from a financial filing,
identify and summarise the key growth drivers, revenue trends, and future outlook. Cite the source text.

Excerpts:
{context}

Provide a concise growth analysis with citations."""

COMPETITOR_PROMPT = """You are a competitive intelligence analyst. Based on the following excerpts from a financial filing,
identify mentions of competitors, market position, and competitive advantages or threats. Cite the source text.

Excerpts:
{context}

Provide a concise competitive analysis with citations."""


def _build_context(documents: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{d['metadata'].get('filing_type', 'filing')} {d['metadata'].get('filing_date', '')}]\n{d['text']}"
        for d in documents
    )


def _call_groq(prompt: str) -> str:
    client = Groq(api_key=settings.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content


@observe(name="analyze-risk")
async def analyze_risk(documents: list[dict]) -> str:
    context = _build_context(documents)
    return await asyncio.to_thread(_call_groq, RISK_PROMPT.format(context=context))


@observe(name="analyze-growth")
async def analyze_growth(documents: list[dict]) -> str:
    context = _build_context(documents)
    return await asyncio.to_thread(_call_groq, GROWTH_PROMPT.format(context=context))


@observe(name="analyze-competitors")
async def analyze_competitors(documents: list[dict]) -> str:
    context = _build_context(documents)
    return await asyncio.to_thread(_call_groq, COMPETITOR_PROMPT.format(context=context))


@observe(name="analyzer-node")
async def analyzer_node(state: AgentState) -> dict:
    risk, growth, competitors = await asyncio.gather(
        analyze_risk(state["documents"]),
        analyze_growth(state["documents"]),
        analyze_competitors(state["documents"]),
    )

    analysis = f"## Risk Analysis\n{risk}\n\n## Growth Analysis\n{growth}\n\n## Competitive Analysis\n{competitors}"
    return {"analysis": analysis}
