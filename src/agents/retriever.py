"""Retriever agent node — hybrid search (dense + BM25 + rerank)."""

import chromadb

from src.api.core.config import settings
from src.agents.graph import AgentState
from src.ingestion.base import Document
from src.retrieval.hybrid_retriever import HybridRetriever

try:
    from langfuse import observe
except ImportError:
    def observe(fn=None, **kwargs):
        return fn if fn is not None else lambda f: f


# ---------------------------------------------------------------------------
# Per-company HybridRetriever cache
# Avoids rebuilding BM25 index + loading CrossEncoder on every retriever call.
# Invalidated when chunk count changes (i.e. new ingestion occurred).
# ---------------------------------------------------------------------------
_retriever_cache: dict[str, tuple[int, HybridRetriever]] = {}


def _load_all_documents(company: str | None = None) -> list[Document]:
    """Load documents from ChromaDB for BM25 corpus.

    Filters by company metadata if provided to avoid cross-company contamination.
    """
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    collection = client.get_collection(settings.CHROMA_COLLECTION)

    if company:
        results = collection.get(
            where={"company": company.upper()},
            include=["documents", "metadatas"],
        )
    else:
        results = collection.get(include=["documents", "metadatas"])

    return [
        Document(content=doc, metadata={**meta, "chunk_id": cid})
        for doc, meta, cid in zip(results["documents"], results["metadatas"], results["ids"])
    ]


def _get_retriever(company: str, docs: list[Document]) -> HybridRetriever:
    """Return a cached HybridRetriever or build a new one.

    Cache key: company name (uppercased).
    Invalidation: if the number of chunks changed since last cache entry.
    """
    cache_key = company.upper() if company else "__all__"
    cached = _retriever_cache.get(cache_key)

    if cached and cached[0] == len(docs):
        return cached[1]

    retriever = HybridRetriever(docs)
    _retriever_cache[cache_key] = (len(docs), retriever)
    return retriever


def invalidate_cache(company: str | None = None):
    """Invalidate retriever cache after new ingestion."""
    if company:
        _retriever_cache.pop(company.upper(), None)
    else:
        _retriever_cache.clear()


@observe(name="retriever-node")
async def retriever_node(state: AgentState) -> dict:
    query = state["query"]
    company = state.get("company", "")

    all_docs = _load_all_documents(company=company)
    if not all_docs:
        return {
            "documents": [],
            "retry_count": state.get("retry_count", 0) + 1,
        }

    retriever = _get_retriever(company, all_docs)
    results = retriever.retrieve(query, top_k=8)

    documents = [
        {"text": doc.content, "metadata": doc.metadata}
        for doc in results
    ]

    return {
        "documents": documents,
        "retry_count": state.get("retry_count", 0) + 1,
    }
