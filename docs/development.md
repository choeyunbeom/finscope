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

---

## Day 2
**Goal:** Multi-Agent Graph + Unit Tests

**Completed:**
- `src/agents/graph.py` — LangGraph `StateGraph` (Retriever → Analyzer → Critic → conditional retry)
- `src/agents/retriever.py` — ChromaDB vector search node, increments `retry_count`
- `src/agents/analyzer.py` — parallel Risk / Growth / Competitor via `asyncio.gather` + Groq
- `src/agents/critic.py` — citation check, parses `VERDICT: sufficient|insufficient`, sets `final_report`
- `tests/conftest.py` — shared fixtures (`sample_retrieved_docs`, `base_state`, `mock_embedding`)
- `tests/unit/test_retriever.py` — 3 tests
- `tests/unit/test_analyzer.py` — 3 tests
- `tests/unit/test_critic.py` — 6 tests (including `_parse_verdict` edge cases)

**Test result:** 12/12 passed ✓

**Key decisions:**
- `asyncio.to_thread` wraps sync Groq calls inside async analyzer — avoids blocking event loop
- Critic forces `final_report` when `retry_count >= 2` regardless of verdict
- `_parse_verdict` defaults to `"sufficient"` on malformed LLM response

---

## Day 3
**Goal:** Polish + Ship — UI, Langfuse, README

**Completed:**
- `src/api/main.py` — `/analyze` endpoint wired up (ingest → graph → response)
- `ui/app.py` — Streamlit UI with sidebar settings, 2-column layout, sources expander
- `monitoring/langfuse_config.py` — `trace_graph` context manager (no-op if keys not set)
- Langfuse trace attached to `/analyze` endpoint
- `README.md` — LangGraph Mermaid diagram + status updated

**Key decisions:**
- Langfuse is optional (no-op if `LANGFUSE_PUBLIC_KEY` not set) — avoids hard failure in local dev
- UI connects to FastAPI via `httpx` — keeps agents server-side, UI stays thin

---

## Post Day 3 — Code Review Fixes

**Issues addressed after external review:**

- `retriever_node` now calls `HybridRetriever.retrieve()` — previously used dense-only ChromaDB search, README claimed hybrid
- `_ticker_to_cik` bug — `resp.json()` moved inside `with httpx.Client(...)` block
- `tests/unit/test_sec_edgar.py` — 8 tests added (resolve, ticker, search, fetch, strip_html)
- `tests/unit/test_companies_house.py` — 4 tests added (resolve, fetch, error handling)
- README: CI badge, Results table, "What's Different from arXiv RAG" comparison table
- README: unit test count corrected (12 → 20)
- `_load_all_documents()`: added docstring noting prototype-scale tradeoff and production improvement path (cache HybridRetriever per collection)
- `.github/workflows/ci.yml`: GitHub Actions on push/PR to main

**Test result:** 24/24 passed ✓

---

## Post Day 3 — Bug Fix: Cross-Company Contamination & CIK Misresolution

**Root cause identified:**

`_search_company()` was using EDGAR full-text search (`/LATEST/search-index`), which returns filings ranked by text match — not company relevance. Querying `"apple"` returned **Apple Hospitality REIT** (CIK 1418121) instead of **Apple Inc.** (CIK 320193), because the REIT had more recent filings matching the keyword.

Additionally, `resolve_to_cik` only checked uppercase ticker regex on the raw input, so lowercase inputs like `"apple"` bypassed ticker lookup entirely and fell through to the broken company search.

**Fixes:**

- `sec_edgar.py` — `resolve_to_cik`: uppercase input before ticker regex check; ticker lookup wrapped in try/except to fall back to name search on miss
- `sec_edgar.py` — `_search_company`: replaced EFTS full-text search with EDGAR `/browse-edgar` Atom API, which ranks results by company relevance and size — `"apple"` now correctly returns Apple Inc. first
- `agents/retriever.py` — removed fallback-to-all-docs when company filter returns empty; returns empty documents instead of leaking other companies' data

**Verification:**

```
apple  → CIK 320193  (Apple Inc.)        ✓
AAPL   → CIK 320193  (Apple Inc.)        ✓
Tesla  → CIK 1318605 (Tesla Inc.)        ✓
```

- Stale APPLE data (Apple Hospitality REIT, 1,209 chunks) removed from ChromaDB
- Re-ingested as Apple Inc. → correct 10-K (2025-10-31) returned ✓
