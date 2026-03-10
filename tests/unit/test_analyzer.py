"""Unit tests for analyzer agent node."""

import pytest
from unittest.mock import MagicMock, patch


@patch("src.agents.analyzer.asyncio.to_thread")
async def test_analyzer_runs_three_analyses(mock_to_thread, base_state):
    """Analyzer should call all three analysis functions in parallel."""
    mock_to_thread.side_effect = [
        "Risk: supply chain disruptions.",
        "Growth: revenue increased 8% YoY.",
        "Competitors: Samsung, Google.",
    ]

    from src.agents.analyzer import analyzer_node
    result = await analyzer_node(base_state)

    assert "analysis" in result
    assert "## Risk Analysis" in result["analysis"]
    assert "## Growth Analysis" in result["analysis"]
    assert "## Competitive Analysis" in result["analysis"]


@patch("src.agents.analyzer.asyncio.to_thread")
async def test_analyzer_combines_results(mock_to_thread, base_state):
    """Analysis output should contain content from all three subtasks."""
    mock_to_thread.side_effect = [
        "supply chain risk",
        "8% revenue growth",
        "competition from Samsung",
    ]

    from src.agents.analyzer import analyzer_node
    result = await analyzer_node(base_state)

    assert "supply chain risk" in result["analysis"]
    assert "8% revenue growth" in result["analysis"]
    assert "competition from Samsung" in result["analysis"]


@patch("src.agents.analyzer._call_groq")
async def test_analyze_risk_calls_groq(mock_groq, sample_retrieved_docs):
    """analyze_risk should call Groq with risk prompt."""
    mock_groq.return_value = "Risk: supply chain issues."

    from src.agents.analyzer import analyze_risk
    result = await analyze_risk(sample_retrieved_docs)

    assert mock_groq.called
    assert "supply chain" in result
