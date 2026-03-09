"""Companies House document loader — fetches UK annual reports."""

import io

import httpx
import pdfplumber

from src.api.core.config import settings
from src.ingestion.base import BaseDocumentLoader, Document

SEARCH_URL = "https://api.company-information.service.gov.uk/search/companies"
FILING_HISTORY_URL = "https://api.company-information.service.gov.uk/company/{company_number}/filing-history"
DOCUMENT_URL = "https://document-api.company-information.service.gov.uk/document/{document_id}/content"


class CompaniesHouseLoader(BaseDocumentLoader):
    """Fetch annual reports from Companies House API.

    Endpoints:
      - /search/companies                        → company name → company_number
      - /company/{number}/filing-history         → filing list (category=accounts)
      - document-api.../document/{id}/content    → PDF download
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.COMPANIES_HOUSE_API_KEY
        # Companies House uses HTTP Basic Auth: API key as username, empty password
        self.auth = (self.api_key, "")

    # ------------------------------------------------------------------
    # Input resolution
    # ------------------------------------------------------------------

    def resolve_company_number(self, company_name: str) -> str:
        """Search for a company and return its company number."""
        with httpx.Client(auth=self.auth, timeout=30) as client:
            resp = client.get(SEARCH_URL, params={"q": company_name, "items_per_page": 5})
            resp.raise_for_status()

        items = resp.json().get("items", [])
        if not items:
            raise ValueError(f"Company not found: {company_name}")

        return items[0]["company_number"]

    # ------------------------------------------------------------------
    # Fetch filing metadata
    # ------------------------------------------------------------------

    def fetch(self, company_number: str, category: str = "accounts", limit: int = 1) -> list[dict]:
        """Fetch filing metadata for a given company number."""
        url = FILING_HISTORY_URL.format(company_number=company_number)

        with httpx.Client(auth=self.auth, timeout=30) as client:
            resp = client.get(url, params={"category": category, "items_per_page": limit})
            resp.raise_for_status()

        items = resp.json().get("items", [])
        if not items:
            raise ValueError(f"No {category} filings found for company {company_number}")

        results = []
        for item in items[:limit]:
            # Extract document ID from links
            doc_links = item.get("links", {}).get("document_metadata", "")
            document_id = doc_links.split("/")[-1] if doc_links else None

            results.append({
                "company_number": company_number,
                "description": item.get("description", ""),
                "date": item.get("date", ""),
                "type": item.get("type", ""),
                "document_id": document_id,
            })

        return results

    # ------------------------------------------------------------------
    # Parse: download PDF and extract text
    # ------------------------------------------------------------------

    def parse(self, raw: list[dict]) -> list[Document]:
        """Download PDFs and extract text into Document objects."""
        docs = []
        for filing in raw:
            doc_id = filing.get("document_id")
            if not doc_id:
                continue

            text = self._extract_pdf_text(doc_id)
            if not text:
                continue

            docs.append(Document(
                content=text,
                metadata={
                    "source": "companies_house",
                    "company_number": filing["company_number"],
                    "filing_type": filing["type"],
                    "filing_date": filing["date"],
                    "description": filing["description"],
                    "document_id": doc_id,
                },
            ))

        return docs

    def _extract_pdf_text(self, document_id: str) -> str | None:
        """Download a filing PDF from Companies House and extract text."""
        url = DOCUMENT_URL.format(document_id=document_id)
        try:
            with httpx.Client(
                auth=self.auth,
                timeout=120,
                follow_redirects=True,
                headers={"Accept": "application/pdf"},
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()

            pages = []
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)

            return "\n\n".join(pages) if pages else None

        except Exception as e:
            print(f"[CompaniesHouseLoader] Failed to extract {document_id}: {e}")
            return None
