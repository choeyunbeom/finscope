"""Unit tests for retriever agent node."""

import pytest
from unittest.mock import MagicMock, patch

from src.agents.graph import AgentState


@pytest.fixture
def empty_state() -> AgentState:
    return AgentState(
        query="What are Apple's main risk factors?",
        documents=[],
        analysis="",
        critique="",
        final_report="",
        retry_count=0,
    )


@patch("src.agents.retriever._embed")
@patch("src.agents.retriever.chromadb.PersistentClient")
async def test_retriever_returns_documents(mock_chroma, mock_embed, empty_state, mock_embedding):
    """Retriever node should return non-empty documents list."""
    mock_embed.return_value = mock_embedding

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Revenue was $394 billion.", "Risk factors include supply chain issues."]],
        "metadatas": [[
            {"filing_type": "10-K", "filing_date": "2022-10-28"},
            {"filing_type": "10-K", "filing_date": "2022-10-28"},
        ]],
    }
    mock_chroma.return_value.get_collection.return_value = mock_collection

    from src.agents.retriever import retriever_node
    result = await retriever_node(empty_state)

    assert "documents" in result
    assert len(result["documents"]) == 2
    assert result["documents"][0]["text"] == "Revenue was $394 billion."


@patch("src.agents.retriever._embed")
@patch("src.agents.retriever.chromadb.PersistentClient")
async def test_retriever_increments_retry_count(mock_chroma, mock_embed, empty_state, mock_embedding):
    """retry_count should increment on each retriever call."""
    mock_embed.return_value = mock_embedding
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
    mock_chroma.return_value.get_collection.return_value = mock_collection

    from src.agents.retriever import retriever_node
    result = await retriever_node(empty_state)

    assert result["retry_count"] == 1


@patch("src.agents.retriever._embed")
@patch("src.agents.retriever.chromadb.PersistentClient")
async def test_retriever_increments_from_existing_count(mock_chroma, mock_embed, mock_embedding):
    """retry_count should increment from existing value on retry."""
    state = AgentState(
        query="test",
        documents=[],
        analysis="",
        critique="insufficient",
        final_report="",
        retry_count=1,
    )
    mock_embed.return_value = mock_embedding
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
    mock_chroma.return_value.get_collection.return_value = mock_collection

    from src.agents.retriever import retriever_node
    result = await retriever_node(state)

    assert result["retry_count"] == 2
