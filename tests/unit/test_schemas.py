"""Unit tests for agent output Pydantic schemas."""

import pytest
from pydantic import ValidationError

from src.agents.schemas import AnalyzerOutput, CriticOutput, QueryRewriterOutput


VALID_ANALYSIS = (
    "## Risk Analysis\nsupply chain issues\n\n"
    "## Growth Analysis\nrevenue up 8%\n\n"
    "## Competitive Analysis\ncompetes with Samsung"
)


class TestAnalyzerOutput:
    def test_valid_analysis_passes(self):
        out = AnalyzerOutput(analysis=VALID_ANALYSIS)
        assert out.analysis == VALID_ANALYSIS

    def test_missing_risk_section_raises(self):
        bad = VALID_ANALYSIS.replace("## Risk Analysis\n", "")
        with pytest.raises(ValidationError, match="missing sections"):
            AnalyzerOutput(analysis=bad)

    def test_missing_growth_section_raises(self):
        bad = VALID_ANALYSIS.replace("## Growth Analysis\n", "")
        with pytest.raises(ValidationError, match="missing sections"):
            AnalyzerOutput(analysis=bad)

    def test_missing_competitive_section_raises(self):
        bad = VALID_ANALYSIS.replace("## Competitive Analysis\n", "")
        with pytest.raises(ValidationError, match="missing sections"):
            AnalyzerOutput(analysis=bad)

    def test_model_dump_returns_dict_with_analysis_key(self):
        out = AnalyzerOutput(analysis=VALID_ANALYSIS).model_dump()
        assert "analysis" in out


class TestCriticOutput:
    def test_sufficient_verdict_passes(self):
        out = CriticOutput(critique="sufficient", critique_feedback="All cited.", final_report="report")
        assert out.critique == "sufficient"

    def test_insufficient_verdict_passes(self):
        out = CriticOutput(critique="insufficient", critique_feedback="Missing citations.", final_report="")
        assert out.critique == "insufficient"

    def test_invalid_verdict_raises(self):
        with pytest.raises(ValidationError, match="sufficient.*insufficient"):
            CriticOutput(critique="unknown", critique_feedback="", final_report="")

    def test_empty_final_report_allowed_when_insufficient(self):
        """final_report can be empty string when verdict is insufficient."""
        out = CriticOutput(critique="insufficient", critique_feedback="bad", final_report="")
        assert out.final_report == ""

    def test_model_dump_contains_all_keys(self):
        out = CriticOutput(
            critique="sufficient", critique_feedback="ok", final_report="done"
        ).model_dump()
        assert {"critique", "critique_feedback", "final_report"} <= out.keys()


class TestQueryRewriterOutput:
    def test_valid_query_passes(self):
        out = QueryRewriterOutput(query="Apple net sales fiscal 2022")
        assert out.query == "Apple net sales fiscal 2022"

    def test_empty_string_raises(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            QueryRewriterOutput(query="")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            QueryRewriterOutput(query="   ")

    def test_model_dump_returns_query_key(self):
        out = QueryRewriterOutput(query="Apple revenue risk").model_dump()
        assert "query" in out
