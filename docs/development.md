# Development Log

## Day 0 (Setup)
- `pyproject.toml` — added `langchain-groq` dependency
- `.env.example` — added `LLM_PROVIDER`, `GROQ_API_KEY`, `GROQ_MODEL`
- `uv sync` — confirmed all packages installed
- `README.md` — rewritten with user-first framing (value prop → architecture)

---

## Day 1
**Goal:** Full ingestion + retrieval pipeline + Q&A smoke test

**Completed:**
- `src/api/core/config.py` — `Settings` with Groq/Ollama switching via `LLM_PROVIDER` env var
- `src/ingestion/sec_edgar.py` — `SecEdgarLoader` fully implemented
  - `resolve_to_cik(user_input)` — ticker (1–5 chars) → CIK, else company name search via EDGAR EFTS
  - `fetch(cik, filing_type, limit)` — pulls filing metadata from `/submissions/CIK{cik}.json`
  - `parse(raw)` — downloads PDF/HTML, extracts text via `pdfplumber`
- `src/ingestion/companies_house.py` — `CompaniesHouseLoader` fully implemented
  - `resolve_company_number(company_name)` — name → company number via `/search/companies`
  - `fetch(company_number, category, limit)` — filing history via `/filing-history`
  - `parse(raw)` — downloads PDF via Companies House document API, extracts text via `pdfplumber`
  - HTTP Basic Auth: API key as username, empty password
- `src/retrieval/chunker.py` — fixed deprecated import (`langchain_text_splitters`)
- `src/retrieval/hybrid_retriever.py` — full pipeline implemented
  - Stage 1: ChromaDB dense search + BM25 sparse search
  - Stage 2: RRF fusion
  - Stage 3: cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `src/ingestion/indexer.py` — `index_documents(chunks)` ChromaDB indexing pipeline
  - Batch embedding via Ollama (`nomic-embed-text`), individual fallback if batch fails
  - SHA-256 chunk ID generation for deduplication
- `src/ingestion/ingest.py` — ingest CLI
  - `--source sec` or `--source ch`
  - Full pipeline: resolve → fetch → parse → chunk → index
- `scripts/qa_test.py` — single-agent Q&A smoke test (ChromaDB retrieval + Groq)

**Key decisions:**
- `chromadb.PersistentClient` (local) instead of `HttpClient` (remote) — no infra needed
- `pdfplumber` instead of `pymupdf4llm` — better financial table extraction
- HTML fallback in `_extract_pdf_text` — some EDGAR filings are HTML, not PDF
- Companies House document API requires `Accept: application/pdf` header
- ChromaDB metadata values must be `str` — applied `str(v)` coercion in indexer
- `html.unescape()` in `_strip_html` — EDGAR HTML filings contain encoded entities (`&#160;` etc.)

**Bugs fixed:**
- `DuplicateIDError` in ChromaDB — chunk ID was based on `content[:100]` only, causing collisions
  - Fix: added global `index` to hash: `source:filing_date:index:content[:80]`
- HTML entities in retrieved chunks (`&#160;`, `&#8217;` etc.) — replaced regex-only stripping with `html.unescape()`

**Smoke test result:**
- `uv run python -m src.ingestion.ingest --company "AAPL" --source sec --filing 10-K`
- AAPL 10-K (2025-10-31, HTML) → 575 chunks → 575/575 indexed into `financial_filings` ✓
- `uv run python scripts/qa_test.py "What are Apple's main risk factors?"`
- Retrieved 5 relevant chunks, Groq answered with citations ✓
