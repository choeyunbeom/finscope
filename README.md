# finscope

> Analyse any public company's financial filings in seconds using a Multi-Agent RAG system powered by LangGraph.

Ask questions like:
- *"What are Tesla's key risk factors in 2024?"*
- *"Summarise Apple's latest 10-K filing"*
- *"What does HSBC's annual report say about credit exposure?"*

finscope retrieves filings directly from **SEC EDGAR** and **Companies House**, then routes them through a 3-agent pipeline that delivers cited, hallucination-checked analysis — entirely local, no API costs.

---

## How It Works

```
User Query (e.g. "AAPL" or "Apple")
    ↓
[Input Resolver] — ticker/name → CIK → latest 10-K PDF
    ↓
[Retriever Agent] — PDF → chunks → ChromaDB (dense + BM25 hybrid)
    ↓
[Analyzer Agent] — Risk / Growth / Competitor (runs in parallel)
    ↓
[Critic Agent] — hallucination check → retry if >30% uncited (max 2x)
    ↓
Final Report with source citations
```

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Agent Orchestration | LangGraph | StateGraph with conditional retry edges |
| LLM | qwen2.5:14b via Ollama | Local, no API cost |
| Embedding | nomic-embed-text via Ollama | Local |
| Vector DB | ChromaDB | Zero-infra, persistent |
| Retrieval | Dense + BM25 hybrid | Better recall on financial jargon |
| PDF Parsing | pdfplumber | Handles financial tables |
| Backend | FastAPI | |
| UI | Streamlit | |
| Monitoring | Langfuse | LLM tracing + eval signals |
| Data Sources | SEC EDGAR API, Companies House API | Free, legal, no scraping |

---

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Fill in: SEC_EDGAR_USER_AGENT, COMPANIES_HOUSE_API_KEY (optional), Langfuse keys

# 3. Pull Ollama models
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

---

## Run

```bash
# Ingest a company's latest 10-K
python -m src.ingestion.ingest --company "Apple" --source sec --filing 10-K

# Start the API
uvicorn src.api.main:app --reload

# Launch the UI
streamlit run ui/app.py
```

---

## Project Structure

```
src/
├── agents/
│   ├── graph.py          # LangGraph StateGraph (entry point)
│   ├── retriever.py      # SEC EDGAR / Companies House → ChromaDB
│   ├── analyzer.py       # Parallel Risk / Growth / Competitor analysis
│   └── critic.py         # Citation check + retry decision
├── ingestion/
│   ├── base.py           # BaseDocumentLoader
│   ├── sec_edgar.py      # SEC EDGAR API (10-K, 10-Q)
│   └── companies_house.py
├── retrieval/
│   ├── chunker.py        # 512-token chunks with financial metadata
│   └── hybrid_retriever.py
└── api/
    └── main.py
```

---

## Status

> Week 1 in progress — building SEC EDGAR & Companies House ingestion pipeline.

---

## Background

Extended from [arxiv_rag_system](https://github.com/choeyunbeom/arxiv_rag_system) — same hybrid retrieval pipeline, adapted for financial filings instead of academic papers.
