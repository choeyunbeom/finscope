"""Integration tests for the LangGraph multi-agent workflow.

These tests verify agent collaboration: state is correctly passed between
nodes, conditional edges route as expected, and end-to-end scenarios
produce the right final_report outcome.

All external I/O (ChromaDB, Groq API) is patched at the module boundary,
so tests run fully offline with no credentials required.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.graph import AgentState, build_graph, should_retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_groq_response(content: str) -> MagicMock:
    """Build a minimal Groq API response mock."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp


SAMPLE_DOCS = [
    {
        "text": "Apple Inc. reported revenue of $394 billion in fiscal 2022.",
        "metadata": {"filing_type": "10-K", "filing_date": "2022-10-28", "source": "sec_edgar"},
    },
    {
        "text": "Key risk factors include supply chain disruptions and geopolitical tensions.",
        "metadata": {"filing_type": "10-K", "filing_date": "2022-10-28", "source": "sec_edgar"},
    },
]

ANALYSIS_TEXT = (
    "## Risk Analysis\nsupply chain disruptions [10-K 2022-10-28]\n\n"
    "## Growth Analysis\nrevenue $394 billion [10-K 2022-10-28]\n\n"
    "## Competitive Analysis\ncompetes with Samsung and Google [10-K 2022-10-28]"
)

SUFFICIENT_CRITIC = (
    "CITED_COUNT: 3\nUNCITED_COUNT: 0\nVERDICT: sufficient\nFEEDBACK: All claims are well cited."
)

INSUFFICIENT_CRITIC = (
    "CITED_COUNT: 1\nUNCITED_COUNT: 4\nVERDICT: insufficient\nFEEDBACK: Revenue figure lacks citation."
)


# ---------------------------------------------------------------------------
# Unit: conditional edge logic (no graph execution needed)
# ---------------------------------------------------------------------------

class TestShouldRetry:
    def test_insufficient_under_max_retries_routes_to_retry(self):
        state = AgentState(
            query="test", company="AAPL", documents=[],
            analysis="", critique="insufficient", critique_feedback="",
            final_report="", retry_count=1,
        )
        assert should_retry(state) == "retry"

    def test_insufficient_at_max_retries_routes_to_done(self):
        state = AgentState(
            query="test", company="AAPL", documents=[],
            analysis="", critique="insufficient", critique_feedback="",
            final_report="", retry_count=2,
        )
        assert should_retry(state) == "done"

    def test_sufficient_routes_to_done(self):
        state = AgentState(
            query="test", company="AAPL", documents=[],
            analysis="", critique="sufficient", critique_feedback="",
            final_report="analysis here", retry_count=0,
        )
        assert should_retry(state) == "done"

    def test_first_call_insufficient_routes_to_retry(self):
        """retry_count=0 with insufficient → still routes to retry (< 2)."""
        state = AgentState(
            query="test", company="AAPL", documents=[],
            analysis="", critique="insufficient", critique_feedback="",
            final_report="", retry_count=0,
        )
        assert should_retry(state) == "retry"


# ---------------------------------------------------------------------------
# Integration: full graph scenarios
# ---------------------------------------------------------------------------

@pytest.fixture
def graph():
    return build_graph()


class TestHappyPath:
    """retriever → analyzer → critic(sufficient) → END"""

    @patch("src.agents.retriever._load_all_documents")
    @patch("src.agents.retriever.HybridRetriever")
    @patch("src.agents.analyzer.asyncio.to_thread")
    @patch("src.agents.critic.Groq")
    async def test_final_report_is_set(
        self,
        mock_groq_cls,
        mock_to_thread,
        mock_retriever_cls,
        mock_load,
        graph,
    ):
        mock_load.return_value = [
            MagicMock(content=d["text"], metadata=d["metadata"]) for d in SAMPLE_DOCS
        ]
        mock_retriever_cls.return_value.retrieve.return_value = [
            MagicMock(content=d["text"], metadata=d["metadata"]) for d in SAMPLE_DOCS
        ]
        mock_to_thread.side_effect = [
            "supply chain disruptions [10-K 2022-10-28]",
            "revenue $394 billion [10-K 2022-10-28]",
            "competes with Samsung [10-K 2022-10-28]",
        ]
        mock_groq_cls.return_value.chat.completions.create.return_value = (
            _make_groq_response(SUFFICIENT_CRITIC)
        )

        initial: AgentState = {
            "query": "What are Apple's main risks?",
            "company": "AAPL",
            "documents": [],
            "analysis": "",
            "critique": "",
            "critique_feedback": "",
            "final_report": "",
            "retry_count": 0,
        }
        result = await graph.ainvoke(initial)

        assert result["critique"] == "sufficient"
        assert result["final_report"] != ""
        assert "## Risk Analysis" in result["final_report"]
        assert "## Growth Analysis" in result["final_report"]
        assert "## Competitive Analysis" in result["final_report"]

    @patch("src.agents.retriever._load_all_documents")
    @patch("src.agents.retriever.HybridRetriever")
    @patch("src.agents.analyzer.asyncio.to_thread")
    @patch("src.agents.critic.Groq")
    async def test_retry_count_increments_once(
        self,
        mock_groq_cls,
        mock_to_thread,
        mock_retriever_cls,
        mock_load,
        graph,
    ):
        mock_load.return_value = [
            MagicMock(content=d["text"], metadata=d["metadata"]) for d in SAMPLE_DOCS
        ]
        mock_retriever_cls.return_value.retrieve.return_value = [
            MagicMock(content=d["text"], metadata=d["metadata"]) for d in SAMPLE_DOCS
        ]
        mock_to_thread.side_effect = ["risk", "growth", "competitors"]
        mock_groq_cls.return_value.chat.completions.create.return_value = (
            _make_groq_response(SUFFICIENT_CRITIC)
        )

        initial: AgentState = {
            "query": "Apple revenue?", "company": "AAPL",
            "documents": [], "analysis": "", "critique": "",
            "critique_feedback": "", "final_report": "", "retry_count": 0,
        }
        result = await graph.ainvoke(initial)

        # Retriever runs once → retry_count increments to 1
        assert result["retry_count"] == 1


class TestRetryPath:
    """retriever → analyzer → critic(insufficient) → query_rewriter → retriever → analyzer → critic(sufficient) → END"""

    @patch("src.agents.retriever._load_all_documents")
    @patch("src.agents.retriever.HybridRetriever")
    @patch("src.agents.analyzer.asyncio.to_thread")
    @patch("src.agents.critic.Groq")
    @patch("src.agents.query_rewriter.Groq")
    async def test_query_is_rewritten_on_retry(
        self,
        mock_rewriter_groq_cls,
        mock_critic_groq_cls,
        mock_to_thread,
        mock_retriever_cls,
        mock_load,
        graph,
    ):
        mock_load.return_value = [
            MagicMock(content=d["text"], metadata=d["metadata"]) for d in SAMPLE_DOCS
        ]
        mock_retriever_cls.return_value.retrieve.return_value = [
            MagicMock(content=d["text"], metadata=d["metadata"]) for d in SAMPLE_DOCS
        ]
        # analyzer runs twice (initial + after retry) → 6 side_effect values
        mock_to_thread.side_effect = [
            "risk v1", "growth v1", "competitors v1",
            "risk v2", "growth v2", "competitors v2",
        ]

        # critic: first call insufficient, second call sufficient
        mock_critic_groq_cls.return_value.chat.completions.create.side_effect = [
            _make_groq_response(INSUFFICIENT_CRITIC),
            _make_groq_response(SUFFICIENT_CRITIC),
        ]

        # query rewriter returns a refined query
        mock_rewriter_groq_cls.return_value.chat.completions.create.return_value = (
            _make_groq_response("Apple net sales uncertainties fiscal 2022")
        )

        initial: AgentState = {
            "query": "What are Apple's main risks?",
            "company": "AAPL",
            "documents": [], "analysis": "", "critique": "",
            "critique_feedback": "", "final_report": "", "retry_count": 0,
        }
        result = await graph.ainvoke(initial)

        assert result["final_report"] != ""
        assert result["critique"] == "sufficient"
        # Query was rewritten (no longer equals the original)
        assert result["query"] != "What are Apple's main risks?"
        # Retriever ran twice → retry_count == 2
        assert result["retry_count"] == 2

    @patch("src.agents.retriever._load_all_documents")
    @patch("src.agents.retriever.HybridRetriever")
    @patch("src.agents.analyzer.asyncio.to_thread")
    @patch("src.agents.critic.Groq")
    @patch("src.agents.query_rewriter.Groq")
    async def test_state_documents_refreshed_after_retry(
        self,
        mock_rewriter_groq_cls,
        mock_critic_groq_cls,
        mock_to_thread,
        mock_retriever_cls,
        mock_load,
        graph,
    ):
        """Retriever must return documents on both the initial call and the retry call."""
        second_doc = {
            "text": "Apple operating income grew 15% year over year.",
            "metadata": {"filing_type": "10-K", "filing_date": "2022-10-28", "source": "sec_edgar"},
        }
        mock_load.return_value = [
            MagicMock(content=d["text"], metadata=d["metadata"])
            for d in SAMPLE_DOCS + [second_doc]
        ]
        mock_retriever_cls.return_value.retrieve.return_value = [
            MagicMock(content=d["text"], metadata=d["metadata"])
            for d in SAMPLE_DOCS + [second_doc]
        ]
        mock_to_thread.side_effect = [
            "risk v1", "growth v1", "competitors v1",
            "risk v2", "growth v2", "competitors v2",
        ]
        mock_critic_groq_cls.return_value.chat.completions.create.side_effect = [
            _make_groq_response(INSUFFICIENT_CRITIC),
            _make_groq_response(SUFFICIENT_CRITIC),
        ]
        mock_rewriter_groq_cls.return_value.chat.completions.create.return_value = (
            _make_groq_response("Apple net sales operating income")
        )

        initial: AgentState = {
            "query": "Apple revenue growth",
            "company": "AAPL",
            "documents": [], "analysis": "", "critique": "",
            "critique_feedback": "", "final_report": "", "retry_count": 0,
        }
        result = await graph.ainvoke(initial)

        assert len(result["documents"]) == 3


class TestMaxRetryForcedDone:
    """After 2 retries, critic must set final_report even with insufficient verdict."""

    @patch("src.agents.retriever._load_all_documents")
    @patch("src.agents.retriever.HybridRetriever")
    @patch("src.agents.analyzer.asyncio.to_thread")
    @patch("src.agents.critic.Groq")
    @patch("src.agents.query_rewriter.Groq")
    async def test_final_report_set_at_max_retries(
        self,
        mock_rewriter_groq_cls,
        mock_critic_groq_cls,
        mock_to_thread,
        mock_retriever_cls,
        mock_load,
        graph,
    ):
        mock_load.return_value = [
            MagicMock(content=d["text"], metadata=d["metadata"]) for d in SAMPLE_DOCS
        ]
        mock_retriever_cls.return_value.retrieve.return_value = [
            MagicMock(content=d["text"], metadata=d["metadata"]) for d in SAMPLE_DOCS
        ]
        # analyzer runs twice (initial + 1 retry); should_retry fires "done" at retry_count==2
        mock_to_thread.side_effect = [
            "risk", "growth", "competitors",
            "risk", "growth", "competitors",
        ]
        # critic always returns insufficient → forces done at retry_count == 2
        mock_critic_groq_cls.return_value.chat.completions.create.return_value = (
            _make_groq_response(INSUFFICIENT_CRITIC)
        )
        mock_rewriter_groq_cls.return_value.chat.completions.create.return_value = (
            _make_groq_response("alternative query")
        )

        initial: AgentState = {
            "query": "Apple financials",
            "company": "AAPL",
            "documents": [], "analysis": "", "critique": "",
            "critique_feedback": "", "final_report": "", "retry_count": 0,
        }
        result = await graph.ainvoke(initial)

        # Forced done: final_report must be non-empty despite insufficient verdict
        assert result["final_report"] != ""
        assert result["critique"] == "insufficient"
        # retriever: initial(→1) + retry1(→2); critic fires done at retry_count==2
        assert result["retry_count"] == 2

    @patch("src.agents.retriever._load_all_documents")
    @patch("src.agents.retriever.HybridRetriever")
    @patch("src.agents.analyzer.asyncio.to_thread")
    @patch("src.agents.critic.Groq")
    @patch("src.agents.query_rewriter.Groq")
    async def test_graph_does_not_loop_past_max_retries(
        self,
        mock_rewriter_groq_cls,
        mock_critic_groq_cls,
        mock_to_thread,
        mock_retriever_cls,
        mock_load,
        graph,
    ):
        """Graph must terminate and not keep cycling after max retries."""
        mock_load.return_value = [
            MagicMock(content=d["text"], metadata=d["metadata"]) for d in SAMPLE_DOCS
        ]
        mock_retriever_cls.return_value.retrieve.return_value = [
            MagicMock(content=d["text"], metadata=d["metadata"]) for d in SAMPLE_DOCS
        ]
        mock_to_thread.side_effect = [
            "risk", "growth", "competitors",
            "risk", "growth", "competitors",
        ]
        mock_critic_groq_cls.return_value.chat.completions.create.return_value = (
            _make_groq_response(INSUFFICIENT_CRITIC)
        )
        mock_rewriter_groq_cls.return_value.chat.completions.create.return_value = (
            _make_groq_response("alternative query")
        )

        initial: AgentState = {
            "query": "Apple financials",
            "company": "AAPL",
            "documents": [], "analysis": "", "critique": "",
            "critique_feedback": "", "final_report": "", "retry_count": 0,
        }
        result = await graph.ainvoke(initial)

        # retriever: initial(→1) + retry1(→2); should_retry returns "done" at retry_count==2
        assert result["retry_count"] == 2


class TestEmptyDocumentHandling:
    """Retriever returns empty docs → graph should still terminate gracefully."""

    @patch("src.agents.retriever._load_all_documents")
    @patch("src.agents.analyzer.asyncio.to_thread")
    @patch("src.agents.critic.Groq")
    @patch("src.agents.query_rewriter.Groq")
    async def test_empty_docs_reach_end(
        self,
        mock_rewriter_groq_cls,
        mock_critic_groq_cls,
        mock_to_thread,
        mock_load,
        graph,
    ):
        mock_load.return_value = []  # nothing in ChromaDB
        # Analyzer still runs with empty doc list
        mock_to_thread.side_effect = [
            "no risk data", "no growth data", "no competitor data",
            "no risk data", "no growth data", "no competitor data",
            "no risk data", "no growth data", "no competitor data",
        ]
        mock_critic_groq_cls.return_value.chat.completions.create.return_value = (
            _make_groq_response(INSUFFICIENT_CRITIC)
        )
        mock_rewriter_groq_cls.return_value.chat.completions.create.return_value = (
            _make_groq_response("refined query")
        )

        initial: AgentState = {
            "query": "Apple financials",
            "company": "AAPL",
            "documents": [], "analysis": "", "critique": "",
            "critique_feedback": "", "final_report": "", "retry_count": 0,
        }
        result = await graph.ainvoke(initial)

        # Graph must terminate (not hang), and final_report must be set (forced done)
        assert "final_report" in result
        assert result["final_report"] != ""
