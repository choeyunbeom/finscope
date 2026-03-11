"""Retriever agent node — hybrid search (dense + BM25 + rerank)."""

import chromadb

from src.api.core.config import settings
from src.agents.graph import AgentState
from src.ingestion.base import Document
from src.retrieval.hybrid_retriever import HybridRetriever


def _load_all_documents() -> list[Document]:
    """Load all documents from ChromaDB for BM25 corpus."""
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    collection = client.get_collection(settings.CHROMA_COLLECTION)

    results = collection.get(include=["documents", "metadatas"])
    return [
        Document(content=doc, metadata={**meta, "chunk_id": cid})
        for doc, meta, cid in zip(results["documents"], results["metadatas"], results["ids"])
    ]


async def retriever_node(state: AgentState) -> dict:
    query = state["query"]

    all_docs = _load_all_documents()
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
