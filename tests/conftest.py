"""Shared pytest fixtures."""
import pytest
from src.ingestion.base import Document


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
