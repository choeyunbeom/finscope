"""
Ingest CLI — fetch, parse, chunk, and index filings into ChromaDB.

Usage:
  python -m src.ingestion.ingest --company "Apple" --source sec --filing 10-K
  python -m src.ingestion.ingest --company "AAPL" --source sec --filing 10-K
  python -m src.ingestion.ingest --company "Barclays" --source ch
"""

import argparse

from src.api.core.config import settings
from src.ingestion.companies_house import CompaniesHouseLoader
from src.ingestion.indexer import index_documents
from src.ingestion.sec_edgar import SecEdgarLoader
from src.retrieval.chunker import chunk_documents


def ingest_sec(company: str, filing_type: str) -> int:
    print(f"\n[SEC EDGAR] Resolving '{company}' → CIK...")
    loader = SecEdgarLoader()
    cik = loader.resolve_to_cik(company)
    print(f"  CIK: {cik}")

    print(f"  Fetching latest {filing_type}...")
    raw = loader.fetch(cik=cik, filing_type=filing_type, limit=1)
    print(f"  Found filing: {raw[0]['filing_date']} — {raw[0]['filing_url']}")

    print("  Downloading and parsing PDF...")
    docs = loader.parse(raw)
    if not docs:
        print("  [ERROR] No text extracted from filing.")
        return 0
    print(f"  Extracted {len(docs)} document(s), {sum(len(d.content) for d in docs):,} chars")

    print("  Chunking...")
    chunks = chunk_documents(docs)
    print(f"  {len(chunks)} chunks created")

    print("  Indexing into ChromaDB...")
    return index_documents(chunks)


def ingest_companies_house(company: str) -> int:
    print(f"\n[Companies House] Searching '{company}'...")
    loader = CompaniesHouseLoader()
    company_number = loader.resolve_company_number(company)
    print(f"  Company number: {company_number}")

    print("  Fetching latest accounts filing...")
    raw = loader.fetch(company_number=company_number, category="accounts", limit=1)
    print(f"  Found filing: {raw[0]['date']} — {raw[0]['description']}")

    print("  Downloading and parsing PDF...")
    docs = loader.parse(raw)
    if not docs:
        print("  [ERROR] No text extracted from filing.")
        return 0
    print(f"  Extracted {len(docs)} document(s), {sum(len(d.content) for d in docs):,} chars")

    print("  Chunking...")
    chunks = chunk_documents(docs)
    print(f"  {len(chunks)} chunks created")

    print("  Indexing into ChromaDB...")
    return index_documents(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest financial filings into ChromaDB")
    parser.add_argument("--company", required=True, help="Company name or ticker (e.g. 'Apple' or 'AAPL')")
    parser.add_argument("--source", choices=["sec", "ch"], default="sec", help="Data source: sec (SEC EDGAR) or ch (Companies House)")
    parser.add_argument("--filing", default="10-K", help="Filing type for SEC (default: 10-K)")
    args = parser.parse_args()

    if args.source == "sec":
        n = ingest_sec(args.company, args.filing)
    else:
        n = ingest_companies_house(args.company)

    print(f"\nDone. {n} chunks indexed.")


if __name__ == "__main__":
    main()
