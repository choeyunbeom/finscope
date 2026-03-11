"""Langfuse tracing configuration."""

import os
from contextlib import contextmanager

_client = None


def get_langfuse():
    """Return Langfuse client, or None if keys not configured."""
    global _client
    if _client is not None:
        return _client

    pub = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec = os.getenv("LANGFUSE_SECRET_KEY")
    if not pub or not sec:
        return None

    try:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=pub,
            secret_key=sec,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        return _client
    except Exception:
        return None


@contextmanager
def trace_graph(query: str):
    """Context manager that creates a Langfuse trace for a graph run."""
    lf = get_langfuse()
    if lf is None:
        yield None
        return

    trace = lf.trace(name="financial-analysis", input={"query": query})
    try:
        yield trace
    finally:
        lf.flush()


def trace_span(trace, name: str, input_data: dict):
    """Create a span on an existing trace. No-op if trace is None."""
    if trace is None:
        return None
    return trace.span(name=name, input=input_data)
