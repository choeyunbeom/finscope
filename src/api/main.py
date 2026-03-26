"""FastAPI entry point."""

import uuid
from enum import Enum

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from src.agents.graph import build_graph
from src.ingestion.ingest import ingest_sec, ingest_companies_house
from monitoring.langfuse_config import trace_graph

try:
    from langfuse import observe as _observe
except Exception:
    def _observe(fn):  # no-op if langfuse unavailable
        return fn

app = FastAPI(title="Financial Report Analyst", version="0.1.0")


# ---------------------------------------------------------------------------
# In-memory job store (swap for Redis/DB in production)
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    pending = "pending"
    ingesting = "ingesting"
    analyzing = "analyzing"
    completed = "completed"
    failed = "failed"


class JobState(BaseModel):
    status: JobStatus = JobStatus.pending
    report: str = ""
    sources: list[str] = []
    retry_count: int = 0
    error: str = ""


_jobs: dict[str, JobState] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    company: str | None = None
    filing_type: str = "10-K"
    source: str = "sec"


class QueryResponse(BaseModel):
    report: str
    sources: list[str] = []
    retry_count: int = 0


class JobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus


# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(job_id: str, req: QueryRequest):
    """Run ingestion + multi-agent analysis in background."""
    job = _jobs[job_id]

    # Stage 1: Ingest
    if req.company:
        job.status = JobStatus.ingesting
        try:
            if req.source == "sec":
                ingest_sec(company=req.company, filing_type=req.filing_type)
            else:
                ingest_companies_house(company=req.company)
        except Exception as e:
            job.status = JobStatus.failed
            job.error = f"Ingestion failed: {e}"
            return

    # Stage 2: Analyze
    job.status = JobStatus.analyzing
    try:
        graph = build_graph()
        import asyncio
        result = asyncio.run(graph.ainvoke({
            "query": req.query,
            "company": req.company or "",
            "documents": [],
            "analysis": "",
            "critique": "",
            "critique_feedback": "",
            "final_report": "",
            "retry_count": 0,
        }))

        job.report = result["final_report"]
        job.sources = [
            f"{d['metadata'].get('filing_type', '')} {d['metadata'].get('filing_date', '')} — {d['text'][:80]}..."
            for d in result.get("documents", [])
        ]
        job.retry_count = result["retry_count"]
        job.status = JobStatus.completed
    except Exception as e:
        job.status = JobStatus.failed
        job.error = f"Analysis failed: {e}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze/async", response_model=JobCreatedResponse)
async def analyze_async(req: QueryRequest, background_tasks: BackgroundTasks):
    """Submit an analysis job. Poll /analyze/status/{job_id} for results."""
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = JobState()
    background_tasks.add_task(_run_pipeline, job_id, req)
    return JobCreatedResponse(job_id=job_id, status=JobStatus.pending)


@app.get("/analyze/status/{job_id}")
async def analyze_status(job_id: str):
    """Poll job status. Returns full result when completed."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/analyze", response_model=QueryResponse)
@_observe(name="financial-analysis")
async def analyze(req: QueryRequest):
    """Synchronous endpoint (kept for backward compatibility and Streamlit UI)."""
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
    with trace_graph(req.query):
        result = await graph.ainvoke({
            "query": req.query,
            "company": req.company or "",
            "documents": [],
            "analysis": "",
            "critique": "",
            "critique_feedback": "",
            "final_report": "",
            "retry_count": 0,
        })

    sources = [
        f"{d['metadata'].get('filing_type', '')} {d['metadata'].get('filing_date', '')} — {d['text'][:80]}..."
        for d in result.get("documents", [])
    ]

    return QueryResponse(
        report=result["final_report"],
        sources=sources,
        retry_count=result["retry_count"],
    )
