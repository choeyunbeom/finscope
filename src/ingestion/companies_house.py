"""Companies House document loader — fetches UK annual reports."""
# TODO Week 1: implement CompaniesHouseLoader
from .base import BaseDocumentLoader, Document


class CompaniesHouseLoader(BaseDocumentLoader):
    """Fetch filings from Companies House API.

    Endpoints used:
      - /search/companies                          → company search
      - /company/{company_number}/filing-history   → filing list
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.company-information.service.gov.uk"

    def fetch(self, company_number: str, category: str = "accounts", limit: int = 5) -> list[dict]:
        raise NotImplementedError

    def parse(self, raw: list[dict]) -> list[Document]:
        raise NotImplementedError
