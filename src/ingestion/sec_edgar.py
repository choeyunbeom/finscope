"""SEC EDGAR document loader — fetches 10-K / 10-Q filings."""
# TODO Week 1: implement SecEdgarLoader
from .base import BaseDocumentLoader, Document


class SecEdgarLoader(BaseDocumentLoader):
    """Fetch filings from SEC EDGAR EFTS API.

    Endpoints used:
      - /submissions/{CIK}.json  → filing metadata
      - /Archives/edgar/data/... → actual PDF/HTML filing
    """

    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self.base_url = "https://data.sec.gov"

    def fetch(self, cik: str, filing_type: str = "10-K", limit: int = 5) -> list[dict]:
        raise NotImplementedError

    def parse(self, raw: list[dict]) -> list[Document]:
        raise NotImplementedError
