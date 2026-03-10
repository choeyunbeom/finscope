"""Retriever agent node — ChromaDB vector search."""

import httpx
import chromadb

from src.api.core.config import settings
from src.agents.graph import AgentState


def _embed(query: str) -> list[float]:
    resp = httpx.post(
        f"{settings.OLLAMA_BASE_URL}/api/embed",
        json={"model": settings.OLLAMA_EMBED_MODEL, "input": [query]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


async def retriever_node(state: AgentState) -> dict:
    query = state["query"]

    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    collection = client.get_collection(settings.CHROMA_COLLECTION)

    embedding = _embed(query)
    results = collection.query(query_embeddings=[embedding], n_results=8)

    documents = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

    return {
        "documents": documents,
        "retry_count": state.get("retry_count", 0) + 1,
    }
