"""Unit tests for retriever agent node."""

import pytest
from unittest.mock import MagicMock, patch

from src.agents.graph import AgentState
from src.ingestion.base import Document


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


@pytest.fixture
def mock_docs() -> list[Document]:
    return [
        Document(
            content="Revenue was $394 billion.",
            metadata={"chunk_id": "abc123", "filing_type": "10-K", "filing_date": "2022-10-28"},
        ),
        Document(
            content="Risk factors include supply chain issues.",
            metadata={"chunk_id": "def456", "filing_type": "10-K", "filing_date": "2022-10-28"},
        ),
    ]


@patch("src.agents.retriever.HybridRetriever")
@patch("src.agents.retriever._load_all_documents")
async def test_retriever_returns_documents(mock_load, mock_retriever_cls, empty_state, mock_docs):
    """Retriever node should return non-empty documents list."""
    mock_load.return_value = mock_docs

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = mock_docs
    mock_retriever_cls.return_value = mock_retriever

    from src.agents.retriever import retriever_node
    result = await retriever_node(empty_state)

    assert "documents" in result
    assert len(result["documents"]) == 2
    assert result["documents"][0]["text"] == "Revenue was $394 billion."


@patch("src.agents.retriever.HybridRetriever")
@patch("src.agents.retriever._load_all_documents")
async def test_retriever_increments_retry_count(mock_load, mock_retriever_cls, empty_state, mock_docs):
    """retry_count should increment on each retriever call."""
    mock_load.return_value = mock_docs
    mock_retriever_cls.return_value.retrieve.return_value = []

    from src.agents.retriever import retriever_node
    result = await retriever_node(empty_state)

    assert result["retry_count"] == 1


@patch("src.agents.retriever.HybridRetriever")
@patch("src.agents.retriever._load_all_documents")
async def test_retriever_increments_from_existing_count(mock_load, mock_retriever_cls, mock_docs):
    """retry_count should increment from existing value on retry."""
    state = AgentState(
        query="test",
        documents=[],
        analysis="",
        critique="insufficient",
        final_report="",
        retry_count=1,
    )
    mock_load.return_value = mock_docs
    mock_retriever_cls.return_value.retrieve.return_value = []

    from src.agents.retriever import retriever_node
    result = await retriever_node(state)

    assert result["retry_count"] == 2
