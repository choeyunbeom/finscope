"""Shared pytest fixtures."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.ingestion.base import Document
from src.agents.graph import AgentState


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            content="Apple Inc. reported revenue of $394 billion in fiscal 2022.",
            metadata={"source": "sec", "company": "Apple", "filing_type": "10-K", "period": "2022", "page_number": 1},
        ),
        Document(
            content="Key risk factors include supply chain disruptions and geopolitical tensions.",
            metadata={"source": "sec", "company": "Apple", "filing_type": "10-K", "period": "2022", "page_number": 12},
        ),
    ]


@pytest.fixture
def sample_retrieved_docs() -> list[dict]:
    return [
        {
            "text": "Apple Inc. reported revenue of $394 billion in fiscal 2022.",
            "metadata": {"filing_type": "10-K", "filing_date": "2022-10-28", "source": "sec_edgar"},
        },
        {
            "text": "Key risk factors include supply chain disruptions and geopolitical tensions.",
            "metadata": {"filing_type": "10-K", "filing_date": "2022-10-28", "source": "sec_edgar"},
        },
        {
            "text": "The company faces competition from Samsung, Google, and other technology companies.",
            "metadata": {"filing_type": "10-K", "filing_date": "2022-10-28", "source": "sec_edgar"},
        },
    ]


@pytest.fixture
def base_state(sample_retrieved_docs) -> AgentState:
    return AgentState(
        query="What are Apple's main risk factors?",
        documents=sample_retrieved_docs,
        analysis="",
        critique="",
        final_report="",
        retry_count=0,
    )


@pytest.fixture
def mock_embedding() -> list[float]:
    return [0.1] * 768
