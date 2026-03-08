# Financial Report Analyst

Multi-Agent RAG system that retrieves and analyses financial reports (SEC EDGAR + Companies House) using LangGraph.

## Architecture

```
User Query
    ↓
[Orchestrator] — LangGraph StateGraph
    ├── [Retriever Agent]  — SEC EDGAR / Companies House API → ChromaDB
    ├── [Analyzer Agent]   — Risk / Growth / Competitor (parallel)
    └── [Critic Agent]     — Hallucination check → retry loop (max 2)
```

## Tech Stack

- **Agent Framework:** LangGraph
- **LLM:** qwen2.5:14b (Ollama, local)
- **Embedding:** nomic-embed-text (Ollama, local)
- **Vector DB:** ChromaDB
- **PDF Parsing:** pdfplumber
- **Backend:** FastAPI
- **UI:** Streamlit
- **Monitoring:** Langfuse

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Copy and fill in env vars
cp .env.example .env

# 3. Pull Ollama models
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

## Run

```bash
# API
uvicorn src.api.main:app --reload

# UI
streamlit run ui/app.py

# Ingest a company
python -m src.ingestion.ingest --company "Apple" --source sec --filing 10-K
```

## Status

> Week 1 in progress — building SEC EDGAR & Companies House ingestion pipeline.
