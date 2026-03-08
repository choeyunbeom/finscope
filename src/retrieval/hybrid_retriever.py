"""Hybrid retriever — ChromaDB dense + BM25 sparse."""
# TODO Week 1: wire up ChromaDB collection + BM25 index
from rank_bm25 import BM25Okapi
from src.ingestion.base import Document


class HybridRetriever:
    """Dense (ChromaDB cosine) + Sparse (BM25) retrieval with score fusion."""

    def __init__(self, collection, documents: list[Document], alpha: float = 0.5):
        self.collection = collection  # chromadb.Collection
        self.documents = documents
        self.alpha = alpha  # weight for dense score
        corpus = [doc.content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        raise NotImplementedError
