"""Hybrid retriever — ChromaDB dense + BM25 sparse with RRF fusion and reranking."""

import re

import chromadb
import httpx
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.api.core.config import settings
from src.ingestion.base import Document

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class HybridRetriever:
    """Dense (ChromaDB cosine) + Sparse (BM25) retrieval with RRF fusion and cross-encoder reranking."""

    def __init__(self, documents: list[Document], collection_name: str | None = None):
        self.documents = documents

        # ChromaDB (local persistent)
        chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        col_name = collection_name or settings.CHROMA_COLLECTION
        self.collection = chroma_client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 on financial keywords
        self.corpus_ids = [doc.metadata.get("chunk_id", str(i)) for i, doc in enumerate(documents)]
        tokenized = [self._tokenize(doc.content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Reranker
        self.reranker = CrossEncoder(RERANKER_MODEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Hybrid search: vector + BM25 → RRF fusion → rerank → top_k."""
        fetch_k = top_k * 6  # broad candidates for fusion

        # Stage 1: both searches
        vector_ranks = self._vector_search(query, fetch_k)
        bm25_ranks = self._bm25_search(query, fetch_k)

        # Stage 2: RRF fusion
        fused = self._rrf_fusion(vector_ranks, bm25_ranks)[:fetch_k]

        # Stage 3: rerank
        reranked = self._rerank(query, fused)

        return reranked[:top_k]

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        # Try new /api/embed first, fall back to /api/embeddings for older Ollama
        try:
            resp = httpx.post(
                f"{settings.OLLAMA_BASE_URL}/api/embed",
                json={"model": settings.OLLAMA_EMBED_MODEL, "input": [text]},
                timeout=30,
            )
            if resp.status_code == 404:
                raise ValueError("404")
            resp.raise_for_status()
            return resp.json()["embeddings"][0]
        except (ValueError, httpx.HTTPStatusError):
            resp = httpx.post(
                f"{settings.OLLAMA_BASE_URL}/api/embeddings",
                json={"model": settings.OLLAMA_EMBED_MODEL, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]

    # ------------------------------------------------------------------
    # Search stages
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)
        return [w for w in text.split() if len(w) > 1]

    def _vector_search(self, query: str, top_k: int) -> dict[str, int]:
        """Returns {chunk_id: rank}."""
        embedding = self._embed(query)
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        return {cid: rank + 1 for rank, cid in enumerate(results["ids"][0])}

    def _bm25_search(self, query: str, top_k: int) -> dict[str, int]:
        """Returns {chunk_id: rank}."""
        scores = self.bm25.get_scores(self._tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return {
            self.corpus_ids[i]: rank + 1
            for rank, i in enumerate(top_indices)
            if scores[i] > 0
        }

    def _rrf_fusion(self, vector_ranks: dict, bm25_ranks: dict, k: int = 60) -> list[str]:
        """Reciprocal Rank Fusion."""
        all_ids = set(vector_ranks) | set(bm25_ranks)
        scores = {}
        for cid in all_ids:
            score = 0.0
            if cid in vector_ranks:
                score += 1.0 / (k + vector_ranks[cid])
            if cid in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[cid])
            scores[cid] = score
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    def _rerank(self, query: str, chunk_ids: list[str]) -> list[Document]:
        """Cross-encoder reranking. Returns Documents sorted by score."""
        id_to_doc = {
            doc.metadata.get("chunk_id", str(i)): doc
            for i, doc in enumerate(self.documents)
        }

        candidates = [(cid, id_to_doc[cid]) for cid in chunk_ids if cid in id_to_doc]
        if not candidates:
            return []

        pairs = [[query, " ".join(doc.content.split()[:200])] for _, doc in candidates]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for (_, doc), _ in ranked]
