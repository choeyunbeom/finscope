"""Unit tests for Companies House loader."""

import pytest
from unittest.mock import MagicMock, patch

from src.ingestion.companies_house import CompaniesHouseLoader


@pytest.fixture
def loader():
    return CompaniesHouseLoader(api_key="test_api_key")


def test_resolve_company_number_found(loader):
    """Should return company number from search results."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "items": [
            {"company_number": "00102498", "title": "HSBC HOLDINGS PLC"},
        ]
    }
    with patch("src.ingestion.companies_house.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
        result = loader.resolve_company_number("HSBC")
    assert result == "00102498"


def test_resolve_company_number_not_found(loader):
    """Should raise ValueError when search returns no items."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"items": []}
    with patch("src.ingestion.companies_house.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
        with pytest.raises(ValueError, match="Company not found"):
            loader.resolve_company_number("UnknownCompanyXYZ")


def test_fetch_returns_filing_metadata(loader):
    """fetch() should return filing metadata with document_id extracted."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "items": [
            {
                "description": "annual-report",
                "date": "2023-12-31",
                "type": "AA",
                "links": {"document_metadata": "/document/abc123def456"},
            }
        ]
    }
    with patch("src.ingestion.companies_house.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
        results = loader.fetch("00102498", category="accounts", limit=1)

    assert len(results) == 1
    assert results[0]["document_id"] == "abc123def456"
    assert results[0]["type"] == "AA"
    assert results[0]["date"] == "2023-12-31"


def test_fetch_raises_when_no_filings(loader):
    """fetch() should raise ValueError when no filings found."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"items": []}
    with patch("src.ingestion.companies_house.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
        with pytest.raises(ValueError, match="No accounts filings found"):
            loader.fetch("00102498")
