"""Langfuse tracing configuration (v3 / OpenTelemetry-based)."""

import os
from contextlib import contextmanager

_langfuse = None


def get_langfuse():
    """Return Langfuse client, or None if keys not configured."""
    global _langfuse
    if _langfuse is not None:
        return _langfuse

    pub = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec = os.getenv("LANGFUSE_SECRET_KEY")
    if not pub or not sec:
        return None

    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=pub,
            secret_key=sec,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        return _langfuse
    except Exception:
        return None


@contextmanager
def trace_graph(query: str):
    """Context manager that wraps a graph run in a Langfuse trace (v3)."""
    lf = get_langfuse()
    if lf is None:
        yield None
        return

    yield lf
    lf.flush()


def trace_span(trace, name: str, input_data: dict):
    """No-op in v3 — spans are handled via @observe decorator."""
    return None
