"""Pydantic output schemas for agent nodes.

Each schema validates the dict returned by a node before it enters
the next stage of the LangGraph state. This catches malformed LLM
responses (missing keys, wrong types) at the boundary rather than
letting them propagate silently and cause cryptic errors downstream.
"""

from pydantic import BaseModel, field_validator


class AnalyzerOutput(BaseModel):
    analysis: str

    @field_validator("analysis")
    @classmethod
    def must_contain_sections(cls, v: str) -> str:
        required = ["## Risk Analysis", "## Growth Analysis", "## Competitive Analysis"]
        missing = [h for h in required if h not in v]
        if missing:
            raise ValueError(f"Analysis missing sections: {missing}")
        return v


class CriticOutput(BaseModel):
    critique: str
    critique_feedback: str
    final_report: str

    @field_validator("critique")
    @classmethod
    def must_be_valid_verdict(cls, v: str) -> str:
        if v not in ("sufficient", "insufficient"):
            raise ValueError(f"critique must be 'sufficient' or 'insufficient', got: {v!r}")
        return v


class QueryRewriterOutput(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Rewritten query must not be empty.")
        return v
