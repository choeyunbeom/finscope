"""Critic agent node — citation check + retry decision."""

import re

from groq import Groq

from src.api.core.config import settings
from src.agents.graph import AgentState

CRITIC_PROMPT = """You are a financial analysis quality reviewer.

Review the following analysis and check whether each factual claim is supported by the provided source excerpts.

Source excerpts:
{context}

Analysis to review:
{analysis}

For each claim in the analysis, determine if it is:
- CITED: directly supported by the source excerpts
- UNCITED: not supported by the source excerpts

Respond in this exact format:
CITED_COUNT: <number>
UNCITED_COUNT: <number>
VERDICT: <sufficient|insufficient>
FEEDBACK: <one sentence explaining your verdict>

A verdict is "insufficient" if more than 30% of claims are uncited."""


def _build_context(documents: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{d['metadata'].get('filing_type', 'filing')} {d['metadata'].get('filing_date', '')}]\n{d['text']}"
        for d in documents
    )


def _parse_verdict(response: str) -> tuple[str, str]:
    """Extract verdict and feedback from critic response."""
    verdict_match = re.search(r"VERDICT:\s*(sufficient|insufficient)", response, re.IGNORECASE)
    feedback_match = re.search(r"FEEDBACK:\s*(.+)", response)

    verdict = verdict_match.group(1).lower() if verdict_match else "sufficient"
    feedback = feedback_match.group(1).strip() if feedback_match else response.strip()
    return verdict, feedback


async def critic_node(state: AgentState) -> dict:
    context = _build_context(state["documents"])

    client = Groq(api_key=settings.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[{
            "role": "user",
            "content": CRITIC_PROMPT.format(context=context, analysis=state["analysis"]),
        }],
        temperature=0.0,
    )

    raw = response.choices[0].message.content
    verdict, feedback = _parse_verdict(raw)

    # If sufficient or max retries reached, set final report
    final_report = ""
    if verdict == "sufficient" or state.get("retry_count", 0) >= 2:
        final_report = state["analysis"]

    return {
        "critique": verdict,
        "final_report": final_report,
    }
