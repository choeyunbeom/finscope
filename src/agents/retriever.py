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


def _load_all_documents(company: str | None = None) -> list[Document]:
    """Load documents from ChromaDB for BM25 corpus.

    Filters by company metadata if provided to avoid cross-company contamination.

    Note: loads full collection on every call to build BM25 index + CrossEncoder.
    Acceptable for prototype scale (~500-2000 chunks).
    Production improvement: cache HybridRetriever instance per collection,
    invalidate on new ingestion.
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
    retriever = HybridRetriever(all_docs)
    results = retriever.retrieve(query, top_k=8)

    documents = [
        {"text": doc.content, "metadata": doc.metadata}
        for doc in results
    ]

    return {
        "documents": documents,
        "retry_count": state.get("retry_count", 0) + 1,
    }
