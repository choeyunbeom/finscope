"""SEC EDGAR document loader — fetches 10-K / 10-Q filings."""

import io
import re

import httpx
import pdfplumber

from src.api.core.config import settings
from src.ingestion.base import BaseDocumentLoader, Document


TICKER_JSON_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FILING_BASE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{company}%22&dateRange=custom&startdt=2020-01-01&forms=10-K"


class SecEdgarLoader(BaseDocumentLoader):
    """Fetch filings from SEC EDGAR.

    Endpoints:
      - /files/company_tickers.json      → ticker → CIK mapping
      - /submissions/CIK{cik}.json       → filing history
      - /Archives/edgar/data/...         → PDF/HTML filing
    """

    def __init__(self, user_agent: str | None = None):
        self.user_agent = user_agent or settings.SEC_EDGAR_USER_AGENT
        self.headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}

    # ------------------------------------------------------------------
    # Input resolution
    # ------------------------------------------------------------------

    def resolve_to_cik(self, user_input: str) -> str:
        """Resolve ticker symbol or company name to CIK."""
        stripped = user_input.strip()
        if re.fullmatch(r"[A-Za-z]{1,5}", stripped):
            return self._ticker_to_cik(stripped.upper())
        return self._search_company(stripped)

    def _ticker_to_cik(self, ticker: str) -> str:
        with httpx.Client(headers=self.headers, timeout=30) as client:
            resp = client.get(TICKER_JSON_URL)
            resp.raise_for_status()

        data = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry["ticker"].upper() == ticker_upper:
                return str(entry["cik_str"])

        raise ValueError(f"Ticker not found: {ticker}")

    def _search_company(self, company_name: str) -> str:
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{company_name}%22&forms=10-K"
        with httpx.Client(headers=self.headers, timeout=30) as client:
            resp = client.get(url)
            resp.raise_for_status()

        hits = resp.json().get("hits", {}).get("hits", [])
        if not hits:
            raise ValueError(f"Company not found: {company_name}")

        return str(hits[0]["_source"]["entity_id"])

    # ------------------------------------------------------------------
    # Fetch filing metadata
    # ------------------------------------------------------------------

    def fetch(self, cik: str, filing_type: str = "10-K", limit: int = 1) -> list[dict]:
        """Fetch filing metadata for a given CIK."""
        cik_int = int(cik)
        url = SUBMISSIONS_URL.format(cik=cik_int)

        with httpx.Client(headers=self.headers, timeout=30) as client:
            resp = client.get(url)
            resp.raise_for_status()

        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        filing_dates = recent.get("filingDate", [])

        results = []
        for form, accession, doc, date in zip(forms, accessions, primary_docs, filing_dates):
            if form == filing_type:
                accession_clean = accession.replace("-", "")
                results.append({
                    "cik": cik_int,
                    "form": form,
                    "accession": accession_clean,
                    "primary_document": doc,
                    "filing_date": date,
                    "filing_url": FILING_BASE_URL.format(
                        cik=cik_int,
                        accession=accession_clean,
                        filename=doc,
                    ),
                })
                if len(results) >= limit:
                    break

        if not results:
            raise ValueError(f"No {filing_type} filings found for CIK {cik}")

        return results

    # ------------------------------------------------------------------
    # Parse: download PDF and extract text
    # ------------------------------------------------------------------

    def parse(self, raw: list[dict]) -> list[Document]:
        """Download PDFs and extract text into Document objects."""
        docs = []
        for filing in raw:
            url = filing["filing_url"]
            text = self._extract_pdf_text(url)
            if not text:
                continue

            docs.append(Document(
                content=text,
                metadata={
                    "source": "sec_edgar",
                    "company_cik": str(filing["cik"]),
                    "filing_type": filing["form"],
                    "filing_date": filing["filing_date"],
                    "accession": filing["accession"],
                    "url": url,
                },
            ))

        return docs

    def _extract_pdf_text(self, url: str) -> str | None:
        """Download a PDF and extract text using pdfplumber."""
        try:
            with httpx.Client(headers=self.headers, timeout=120, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")

                # Some filings are HTML, not PDF
                if "html" in content_type:
                    return self._strip_html(resp.text)

                pdf_bytes = resp.content

            pages = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)

            return "\n\n".join(pages) if pages else None

        except Exception as e:
            print(f"[SecEdgarLoader] Failed to extract {url}: {e}")
            return None

    @staticmethod
    def _strip_html(html: str) -> str:
        """Minimal HTML tag stripping for non-PDF filings."""
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()
