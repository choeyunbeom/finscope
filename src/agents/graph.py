"""LangGraph StateGraph — Orchestrator entry point."""
# TODO Week 2: implement full graph
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
import operator


class AgentState(TypedDict):
    query: str
    documents: list[dict]
    analysis: str
    critique: str
    final_report: str
    retry_count: int


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Nodes (stubs — implement in Week 2)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("critic", critic_node)

    # Edges
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "analyzer")
    graph.add_edge("analyzer", "critic")
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {"retry": "retriever", "done": END},
    )
    return graph.compile()


def should_retry(state: AgentState) -> str:
    if state["critique"] == "insufficient" and state["retry_count"] < 2:
        return "retry"
    return "done"


async def retriever_node(state: AgentState) -> AgentState:
    raise NotImplementedError


async def analyzer_node(state: AgentState) -> AgentState:
    raise NotImplementedError


async def critic_node(state: AgentState) -> AgentState:
    raise NotImplementedError
