"""FastAPI entry point."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agents.graph import build_graph
from src.ingestion.ingest import ingest_sec, ingest_companies_house
from monitoring.langfuse_config import trace_graph

app = FastAPI(title="Financial Report Analyst", version="0.1.0")


class QueryRequest(BaseModel):
    query: str
    company: str | None = None
    filing_type: str = "10-K"
    source: str = "sec"


class QueryResponse(BaseModel):
    report: str
    sources: list[str] = []
    retry_count: int = 0


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=QueryResponse)
async def analyze(req: QueryRequest):
    # Ingest if company provided
    if req.company:
        try:
            if req.source == "sec":
                ingest_sec(company=req.company, filing_type=req.filing_type)
            else:
                ingest_companies_house(company=req.company)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    # Run multi-agent graph
    graph = build_graph()
    with trace_graph(req.query) as trace:
        result = await graph.ainvoke({
            "query": req.query,
            "company": req.company or "",
            "documents": [],
            "analysis": "",
            "critique": "",
            "final_report": "",
            "retry_count": 0,
        })
        if trace:
            trace.update(output={"report": result["final_report"], "retry_count": result["retry_count"]})

    sources = [
        f"{d['metadata'].get('filing_type', '')} {d['metadata'].get('filing_date', '')} — {d['text'][:80]}..."
        for d in result.get("documents", [])
    ]

    return QueryResponse(
        report=result["final_report"],
        sources=sources,
        retry_count=result["retry_count"],
    )
