"""Analyzer agent node — parallel Risk / Growth / Competitor analysis."""
# TODO Week 2
import asyncio
from .graph import AgentState


async def analyze_risk(documents: list[dict]) -> str:
    raise NotImplementedError


async def analyze_growth(documents: list[dict]) -> str:
    raise NotImplementedError


async def analyze_competitors(documents: list[dict]) -> str:
    raise NotImplementedError


async def analyzer_node(state: AgentState) -> AgentState:
    results = await asyncio.gather(
        analyze_risk(state["documents"]),
        analyze_growth(state["documents"]),
        analyze_competitors(state["documents"]),
    )
    return {**state, "analysis": "\n\n".join(results)}
