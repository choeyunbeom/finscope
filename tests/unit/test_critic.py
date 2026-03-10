"""Unit tests for critic agent node."""

import pytest
from unittest.mock import MagicMock, patch

from src.agents.graph import AgentState


@pytest.fixture
def state_with_analysis(sample_retrieved_docs) -> AgentState:
    return AgentState(
        query="What are Apple's main risk factors?",
        documents=sample_retrieved_docs,
        analysis="Apple faces supply chain risks [10-K 2022-10-28]. Revenue was $394 billion [10-K 2022-10-28].",
        critique="",
        final_report="",
        retry_count=1,
    )


@patch("src.agents.critic.Groq")
async def test_critic_sufficient_sets_final_report(mock_groq_class, state_with_analysis):
    """When verdict is sufficient, final_report should be set."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "CITED_COUNT: 2\nUNCITED_COUNT: 0\nVERDICT: sufficient\nFEEDBACK: All claims are cited."
    )
    mock_groq_class.return_value = mock_client

    from src.agents.critic import critic_node
    result = await critic_node(state_with_analysis)

    assert result["critique"] == "sufficient"
    assert result["final_report"] == state_with_analysis["analysis"]


@patch("src.agents.critic.Groq")
async def test_critic_insufficient_clears_final_report(mock_groq_class, state_with_analysis):
    """When verdict is insufficient, final_report should remain empty."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "CITED_COUNT: 1\nUNCITED_COUNT: 3\nVERDICT: insufficient\nFEEDBACK: Too many uncited claims."
    )
    mock_groq_class.return_value = mock_client

    from src.agents.critic import critic_node
    result = await critic_node(state_with_analysis)

    assert result["critique"] == "insufficient"
    assert result["final_report"] == ""


@patch("src.agents.critic.Groq")
async def test_critic_forces_done_at_max_retries(mock_groq_class, sample_retrieved_docs):
    """At retry_count=2, final_report should be set even if insufficient."""
    state = AgentState(
        query="test",
        documents=sample_retrieved_docs,
        analysis="Some analysis.",
        critique="",
        final_report="",
        retry_count=2,
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "CITED_COUNT: 0\nUNCITED_COUNT: 5\nVERDICT: insufficient\nFEEDBACK: No citations found."
    )
    mock_groq_class.return_value = mock_client

    from src.agents.critic import critic_node
    result = await critic_node(state)

    assert result["final_report"] == "Some analysis."


def test_parse_verdict_sufficient():
    from src.agents.critic import _parse_verdict
    verdict, feedback = _parse_verdict(
        "CITED_COUNT: 3\nUNCITED_COUNT: 0\nVERDICT: sufficient\nFEEDBACK: All good."
    )
    assert verdict == "sufficient"
    assert feedback == "All good."


def test_parse_verdict_insufficient():
    from src.agents.critic import _parse_verdict
    verdict, feedback = _parse_verdict(
        "CITED_COUNT: 1\nUNCITED_COUNT: 4\nVERDICT: insufficient\nFEEDBACK: Too many uncited."
    )
    assert verdict == "insufficient"
    assert feedback == "Too many uncited."


def test_parse_verdict_fallback():
    """Malformed response should default to sufficient."""
    from src.agents.critic import _parse_verdict
    verdict, _ = _parse_verdict("Some random response with no format.")
    assert verdict == "sufficient"
