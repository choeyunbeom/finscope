"""Unit tests for SEC EDGAR loader."""

import pytest
from unittest.mock import MagicMock, patch

from src.ingestion.sec_edgar import SecEdgarLoader


@pytest.fixture
def loader():
    return SecEdgarLoader(user_agent="test test@test.com")


def test_resolve_to_cik_routes_ticker_uppercase(loader):
    """All-uppercase 1-5 char input should route to ticker lookup."""
    with patch.object(loader, "_ticker_to_cik", return_value="320193") as mock_ticker:
        with patch.object(loader, "_search_company") as mock_search:
            result = loader.resolve_to_cik("AAPL")
            mock_ticker.assert_called_once_with("AAPL")
            mock_search.assert_not_called()
            assert result == "320193"


def test_resolve_to_cik_routes_company_name(loader):
    """Mixed case / long input should route to company name search."""
    with patch.object(loader, "_search_company", return_value="320193") as mock_search:
        with patch.object(loader, "_ticker_to_cik") as mock_ticker:
            result = loader.resolve_to_cik("Apple Inc")
            mock_search.assert_called_once_with("Apple Inc")
            mock_ticker.assert_not_called()
            assert result == "320193"


def test_ticker_to_cik_found(loader):
    """Should return CIK string when ticker is found."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc"},
        "1": {"ticker": "MSFT", "cik_str": 789019, "title": "Microsoft"},
    }
    with patch("src.ingestion.sec_edgar.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
        result = loader._ticker_to_cik("AAPL")
    assert result == "320193"


def test_ticker_to_cik_not_found(loader):
    """Should raise ValueError when ticker is not in the list."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "0": {"ticker": "MSFT", "cik_str": 789019, "title": "Microsoft"},
    }
    with patch("src.ingestion.sec_edgar.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
        with pytest.raises(ValueError, match="Ticker not found"):
            loader._ticker_to_cik("AAPL")


def test_search_company_found(loader):
    """Should return CIK from EDGAR search results."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "hits": {
            "hits": [{"_source": {"ciks": ["320193"], "display_names": ["Apple Inc"]}}]
        }
    }
    with patch("src.ingestion.sec_edgar.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
        result = loader._search_company("Apple")
    assert result == "320193"


def test_search_company_not_found(loader):
    """Should raise ValueError when search returns no hits."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"hits": {"hits": []}}
    with patch("src.ingestion.sec_edgar.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
        with pytest.raises(ValueError, match="Company not found"):
            loader._search_company("UnknownCompanyXYZ")


def test_strip_html_removes_tags(loader):
    """Should strip HTML tags and decode entities."""
    raw = "<p>Revenue was &#36;394&#160;billion</p>"
    result = loader._strip_html(raw)
    assert "<p>" not in result
    assert "$394" in result


def test_fetch_filters_by_filing_type(loader):
    """fetch() should only return filings matching the requested type."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "10-K"],
                "accessionNumber": ["0001-2022", "0002-2022", "0003-2021"],
                "primaryDocument": ["doc1.htm", "doc2.htm", "doc3.htm"],
                "filingDate": ["2022-10-28", "2022-07-29", "2021-10-29"],
            }
        }
    }
    with patch("src.ingestion.sec_edgar.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp
        results = loader.fetch("320193", filing_type="10-K", limit=2)

    assert len(results) == 2
    assert all(r["form"] == "10-K" for r in results)
