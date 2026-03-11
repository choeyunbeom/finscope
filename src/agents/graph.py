"""LangGraph StateGraph — Orchestrator entry point."""

from typing import TypedDict

from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    query: str
    company: str
    documents: list[dict]
    analysis: str
    critique: str
    final_report: str
    retry_count: int


def should_retry(state: AgentState) -> str:
    if state.get("critique") == "insufficient" and state.get("retry_count", 0) < 2:
        return "retry"
    return "done"


def build_graph():
    from src.agents.retriever import retriever_node
    from src.agents.analyzer import analyzer_node
    from src.agents.critic import critic_node

    graph = StateGraph(AgentState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "analyzer")
    graph.add_edge("analyzer", "critic")
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {"retry": "retriever", "done": END},
    )

    return graph.compile()
