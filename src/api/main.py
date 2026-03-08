"""FastAPI entry point."""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Financial Report Analyst", version="0.1.0")


class QueryRequest(BaseModel):
    query: str
    company: str | None = None
    filing_type: str = "10-K"


class QueryResponse(BaseModel):
    report: str
    sources: list[str] = []


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=QueryResponse)
async def analyze(req: QueryRequest):
    # TODO Week 2: wire up LangGraph
    raise NotImplementedError
