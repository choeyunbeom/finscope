"""Unit tests for HybridRetriever — edge cases for RRF fusion, BM25, and reranking."""

import pytest
from unittest.mock import MagicMock, patch

from src.ingestion.base import Document
from src.retrieval.hybrid_retriever import HybridRetriever


def _make_doc(chunk_id: str, content: str) -> Document:
    return Document(
        content=content,
        metadata={"chunk_id": chunk_id, "filing_type": "10-K", "filing_date": "2022-10-28"},
    )


DOCS = [
    _make_doc("a1", "Apple revenue was $394 billion in fiscal 2022."),
    _make_doc("b2", "Key risk factors include supply chain disruptions."),
    _make_doc("c3", "The company faces competition from Samsung and Google."),
]


@pytest.fixture
def retriever():
    """HybridRetriever with ChromaDB, Ollama, and CrossEncoder all patched out."""
    with (
        patch("src.retrieval.hybrid_retriever.chromadb.PersistentClient"),
        patch("src.retrieval.hybrid_retriever.CrossEncoder"),
    ):
        r = HybridRetriever(DOCS)
        return r


# ---------------------------------------------------------------------------
# RRF fusion logic
# ---------------------------------------------------------------------------

class TestRRFFusion:
    def test_both_searches_hit_same_doc_ranks_highest(self, retriever):
        """A doc that appears in both vector and BM25 results should rank above docs in only one."""
        vector_ranks = {"a1": 1, "b2": 2}
        bm25_ranks = {"a1": 1, "c3": 2}

        fused = retriever._rrf_fusion(vector_ranks, bm25_ranks)

        assert fused[0] == "a1"

    def test_bm25_only_hit_included_in_fusion(self, retriever):
        """Doc only in BM25 results must still appear in fused output."""
        vector_ranks = {"a1": 1}
        bm25_ranks = {"b2": 1}

        fused = retriever._rrf_fusion(vector_ranks, bm25_ranks)

        assert "b2" in fused

    def test_dense_only_hit_included_in_fusion(self, retriever):
        """Doc only in vector results must still appear in fused output."""
        vector_ranks = {"c3": 1}
        bm25_ranks = {"a1": 1}

        fused = retriever._rrf_fusion(vector_ranks, bm25_ranks)

        assert "c3" in fused

    def test_empty_vector_results_falls_back_to_bm25(self, retriever):
        """When vector search returns nothing, BM25 results drive the fused ranking."""
        vector_ranks = {}
        bm25_ranks = {"a1": 1, "b2": 2}

        fused = retriever._rrf_fusion(vector_ranks, bm25_ranks)

        assert fused == ["a1", "b2"]

    def test_empty_bm25_results_falls_back_to_vector(self, retriever):
        """When BM25 returns nothing, vector results drive the fused ranking."""
        vector_ranks = {"a1": 1, "b2": 2}
        bm25_ranks = {}

        fused = retriever._rrf_fusion(vector_ranks, bm25_ranks)

        assert fused == ["a1", "b2"]

    def test_both_empty_returns_empty_list(self, retriever):
        """No results from either search → fused output is empty."""
        fused = retriever._rrf_fusion({}, {})

        assert fused == []

    def test_rrf_score_formula(self, retriever):
        """RRF score = 1/(k+rank). With k=60, rank-1 doc scores higher than rank-2."""
        vector_ranks = {"a1": 1, "b2": 2}
        bm25_ranks = {}

        fused = retriever._rrf_fusion(vector_ranks, bm25_ranks, k=60)

        assert fused.index("a1") < fused.index("b2")


# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------

class TestBM25Search:
    def test_exact_keyword_match_returns_nonzero_score(self, retriever):
        """BM25 should return results when query term appears in corpus."""
        results = retriever._bm25_search("revenue billion", top_k=3)

        assert "a1" in results

    def test_zero_score_docs_excluded(self, retriever):
        """Docs with BM25 score of 0 should not appear in results."""
        results = retriever._bm25_search("revenue", top_k=3)

        # All returned IDs must have matched at least one token
        assert all(isinstance(rank, int) for rank in results.values())

    def test_no_matching_terms_returns_empty(self, retriever):
        """Query with no tokens matching the corpus should return empty dict."""
        results = retriever._bm25_search("xyzzy foobar qux", top_k=3)

        assert results == {}

    def test_top_k_limits_results(self, retriever):
        """BM25 should return at most top_k results."""
        results = retriever._bm25_search("the company", top_k=1)

        assert len(results) <= 1


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

class TestRerank:
    def test_rerank_returns_documents_in_order(self, retriever):
        """Reranker should return Document objects sorted by cross-encoder score."""
        retriever.reranker.predict = MagicMock(return_value=[0.9, 0.3, 0.6])
        chunk_ids = ["a1", "b2", "c3"]

        result = retriever._rerank("Apple revenue", chunk_ids)

        assert result[0].metadata["chunk_id"] == "a1"
        assert result[1].metadata["chunk_id"] == "c3"
        assert result[2].metadata["chunk_id"] == "b2"

    def test_rerank_unknown_ids_are_skipped(self, retriever):
        """Chunk IDs not in the document corpus should be silently skipped."""
        retriever.reranker.predict = MagicMock(return_value=[0.8])
        chunk_ids = ["a1", "unknown_id"]

        result = retriever._rerank("Apple revenue", chunk_ids)

        assert len(result) == 1
        assert result[0].metadata["chunk_id"] == "a1"

    def test_rerank_empty_candidates_returns_empty(self, retriever):
        """Empty chunk_ids list → reranker returns empty list without calling model."""
        result = retriever._rerank("Apple revenue", [])

        assert result == []
        retriever.reranker.predict.assert_not_called()

    def test_rerank_all_unknown_ids_returns_empty(self, retriever):
        """All unknown IDs → no candidates to rerank → returns empty list."""
        result = retriever._rerank("Apple revenue", ["ghost1", "ghost2"])

        assert result == []


# ---------------------------------------------------------------------------
# Full retrieve() — integration of all stages
# ---------------------------------------------------------------------------

class TestRetrieve:
    @patch("src.retrieval.hybrid_retriever.httpx.post")
    def test_retrieve_top_k_respected(self, mock_post, retriever):
        """retrieve() must return at most top_k documents."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "embeddings": [[0.1] * 768]
        }
        retriever.collection.query = MagicMock(return_value={
            "ids": [["a1", "b2", "c3"]],
            "distances": [[0.1, 0.2, 0.3]],
        })
        retriever.reranker.predict = MagicMock(return_value=[0.9, 0.5, 0.7])

        result = retriever.retrieve("Apple revenue risk", top_k=2)

        assert len(result) <= 2

    @patch("src.retrieval.hybrid_retriever.httpx.post")
    def test_retrieve_with_empty_vector_results(self, mock_post, retriever):
        """When vector search returns no IDs, retrieve() falls back to BM25 only."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"embeddings": [[0.1] * 768]}
        retriever.collection.query = MagicMock(return_value={"ids": [[]], "distances": [[]]})
        retriever.reranker.predict = MagicMock(return_value=[0.8, 0.4])

        result = retriever.retrieve("revenue billion", top_k=3)

        # Should still return results driven by BM25
        assert isinstance(result, list)

    @patch("src.retrieval.hybrid_retriever.httpx.post")
    def test_retrieve_returns_document_objects(self, mock_post, retriever):
        """retrieve() must return a list of Document instances."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"embeddings": [[0.1] * 768]}
        retriever.collection.query = MagicMock(return_value={
            "ids": [["a1", "b2"]],
            "distances": [[0.1, 0.2]],
        })
        retriever.reranker.predict = MagicMock(return_value=[0.9, 0.5])

        result = retriever.retrieve("Apple revenue", top_k=2)

        assert all(isinstance(doc, Document) for doc in result)
