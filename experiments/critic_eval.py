"""
Critic Agent Evaluation — Sensitivity / Specificity test.

Pulls real AAPL chunks from ChromaDB, generates synthetic analyses
(clean vs hallucinated), and measures how well the Critic detects them.

Usage:
    uv run python -m experiments.critic_eval
"""

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()  # export .env vars to OS env BEFORE langfuse import

import chromadb
from groq import Groq

from src.api.core.config import settings
try:
    from langfuse import observe, Langfuse
    _langfuse_available = True
except ImportError:
    def observe(fn=None, **kwargs):
        return fn if fn is not None else lambda f: f
    _langfuse_available = False

# ---------------------------------------------------------------------------
# 1. Load real chunks from ChromaDB
# ---------------------------------------------------------------------------

def load_real_chunks(n: int = 8) -> list[dict]:
    """Pull n chunks from the AAPL collection in ChromaDB."""
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(name=settings.CHROMA_COLLECTION)

    results = collection.peek(limit=n)
    chunks = []
    for i, doc_text in enumerate(results["documents"]):
        meta = results["metadatas"][i] if results["metadatas"] else {}
        chunks.append({
            "text": doc_text,
            "metadata": meta,
        })
    return chunks


# ---------------------------------------------------------------------------
# 2. Synthetic analysis generator
# ---------------------------------------------------------------------------

GENERATE_CLEAN_PROMPT = """Based ONLY on the following source excerpts, write a short financial analysis (4-6 claims).
Every claim must cite a specific number or fact from the excerpts.

Source excerpts:
{context}

Write the analysis now. Each sentence should reference data from the sources."""

GENERATE_HALLUCINATED_PROMPT = """Based on the following source excerpts, write a short financial analysis (4-6 claims).
However, you MUST inject exactly {n_hallucinations} fabricated claims that contain numbers or facts NOT present in the sources.
Make the fabricated claims sound plausible but they must not be supported by the excerpts.
Mark each fabricated claim with [FAB] at the end (this marker is for evaluation only).

Source excerpts:
{context}

Write the analysis now, mixing real and fabricated claims."""

GENERATE_BORDERLINE_PROMPT = """Based on the following source excerpts, write a short financial analysis with exactly 10 claims.
Exactly 3 of the 10 claims should contain slightly altered numbers (e.g. rounding differently, off by 5-15%).
The other 7 should be accurately cited from the sources.
Mark altered claims with [ALT] at the end.

Source excerpts:
{context}

Write the analysis now."""


def _build_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{c['metadata'].get('filing_type', 'filing')} {c['metadata'].get('filing_date', '')}]\n{c['text']}"
        for c in chunks
    )


@observe(name="critic-eval-llm-call")
def _call_llm(prompt: str) -> str:
    client = Groq(api_key=settings.GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def generate_test_cases(chunks: list[dict]) -> list[dict]:
    """Generate synthetic test cases: clean, hallucinated, and borderline."""
    context = _build_context(chunks)
    cases = []

    # --- Clean analyses (expected: sufficient) ---
    for i in range(3):
        print(f"  Generating clean case {i+1}/3...")
        analysis = _call_llm(GENERATE_CLEAN_PROMPT.format(context=context))
        cases.append({
            "id": f"clean_{i+1}",
            "type": "clean",
            "expected_verdict": "sufficient",
            "analysis": analysis,
        })
        time.sleep(1)  # rate limit

    # --- Fully hallucinated (expected: insufficient) ---
    for i in range(3):
        print(f"  Generating hallucinated case {i+1}/3...")
        analysis = _call_llm(GENERATE_HALLUCINATED_PROMPT.format(
            context=context, n_hallucinations=4
        ))
        # Strip [FAB] markers before sending to critic
        cases.append({
            "id": f"hallucinated_{i+1}",
            "type": "hallucinated",
            "expected_verdict": "insufficient",
            "analysis": re.sub(r"\s*\[FAB\]", "", analysis),
            "analysis_with_markers": analysis,
        })
        time.sleep(1)

    # --- Borderline ~30% hallucinated (expected: could go either way) ---
    for i in range(3):
        print(f"  Generating borderline case {i+1}/3...")
        analysis = _call_llm(GENERATE_BORDERLINE_PROMPT.format(context=context))
        cases.append({
            "id": f"borderline_{i+1}",
            "type": "borderline",
            "expected_verdict": "borderline",
            "analysis": re.sub(r"\s*\[ALT\]", "", analysis),
            "analysis_with_markers": analysis,
        })
        time.sleep(1)

    return cases


# ---------------------------------------------------------------------------
# 3. Run Critic on each case
# ---------------------------------------------------------------------------

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


@observe(name="critic-eval-judge")
def run_critic(context: str, analysis: str) -> dict:
    raw = _call_llm(CRITIC_PROMPT.format(context=context, analysis=analysis))

    cited = re.search(r"CITED_COUNT:\s*(\d+)", raw)
    uncited = re.search(r"UNCITED_COUNT:\s*(\d+)", raw)
    verdict_match = re.search(r"VERDICT:\s*(sufficient|insufficient)", raw, re.IGNORECASE)
    feedback_match = re.search(r"FEEDBACK:\s*(.+)", raw)

    cited_count = int(cited.group(1)) if cited else 0
    uncited_count = int(uncited.group(1)) if uncited else 0
    verdict = verdict_match.group(1).lower() if verdict_match else "unknown"
    feedback = feedback_match.group(1).strip() if feedback_match else raw[:200]

    return {
        "cited_count": cited_count,
        "uncited_count": uncited_count,
        "verdict": verdict,
        "feedback": feedback,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# 4. Evaluate & report
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    case_id: str
    case_type: str
    expected: str
    actual_verdict: str
    cited: int
    uncited: int
    feedback: str
    correct: bool


def evaluate(cases: list[dict], chunks: list[dict]) -> list[EvalResult]:
    context = _build_context(chunks)
    results = []

    for case in cases:
        print(f"  Evaluating {case['id']}...")
        critic_out = run_critic(context, case["analysis"])
        time.sleep(1)

        if case["expected_verdict"] == "borderline":
            correct = True  # borderline cases are informational
        else:
            correct = critic_out["verdict"] == case["expected_verdict"]

        results.append(EvalResult(
            case_id=case["id"],
            case_type=case["type"],
            expected=case["expected_verdict"],
            actual_verdict=critic_out["verdict"],
            cited=critic_out["cited_count"],
            uncited=critic_out["uncited_count"],
            feedback=critic_out["feedback"],
            correct=correct,
        ))

    return results


def print_report(results: list[EvalResult]):
    print("\n" + "=" * 70)
    print("CRITIC AGENT EVALUATION REPORT")
    print("=" * 70)

    # Per-case results
    for r in results:
        status = "✓" if r.correct else "✗"
        print(f"\n  [{status}] {r.case_id}")
        print(f"      Type:     {r.case_type}")
        print(f"      Expected: {r.expected}")
        print(f"      Actual:   {r.actual_verdict}")
        print(f"      Cited/Uncited: {r.cited}/{r.uncited}")
        print(f"      Feedback: {r.feedback[:100]}")

    # Confusion matrix (excluding borderline)
    clean_results = [r for r in results if r.case_type == "clean"]
    halluc_results = [r for r in results if r.case_type == "hallucinated"]
    borderline_results = [r for r in results if r.case_type == "borderline"]

    tp = sum(1 for r in halluc_results if r.actual_verdict == "insufficient")
    fn = sum(1 for r in halluc_results if r.actual_verdict == "sufficient")
    tn = sum(1 for r in clean_results if r.actual_verdict == "sufficient")
    fp = sum(1 for r in clean_results if r.actual_verdict == "insufficient")

    print("\n" + "-" * 70)
    print("CONFUSION MATRIX (clean vs hallucinated only)")
    print("-" * 70)
    print(f"                    Predicted")
    print(f"                    sufficient  insufficient")
    print(f"  Actual clean      {tn:>6}      {fp:>6}")
    print(f"  Actual halluc     {fn:>6}      {tp:>6}")

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    print(f"\n  Sensitivity (hallucination detection): {sensitivity:.0%} ({tp}/{tp+fn})")
    print(f"  Specificity (clean pass-through):      {specificity:.0%} ({tn}/{tn+fp})")
    print(f"  Accuracy:                              {accuracy:.0%}")

    # Borderline behavior
    print(f"\n  Borderline cases (~30% altered):")
    for r in borderline_results:
        print(f"    {r.case_id}: {r.actual_verdict} (cited={r.cited}, uncited={r.uncited})")

    print("\n" + "=" * 70)

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

@observe(name="critic-eval-experiment")
def main():
    print("=" * 70)
    print("Critic Agent Evaluation")
    print("LLM-as-Judge Hallucination Detection Test")
    print("=" * 70)

    print("\n[1/3] Loading chunks from ChromaDB...")
    chunks = load_real_chunks(n=8)
    print(f"  Loaded {len(chunks)} chunks")

    print("\n[2/3] Generating synthetic test cases...")
    cases = generate_test_cases(chunks)
    print(f"  Generated {len(cases)} test cases")

    # Save test cases for reproducibility
    with open("experiments/critic_eval_cases.json", "w") as f:
        json.dump(cases, f, indent=2)
    print("  Saved to experiments/critic_eval_cases.json")

    print("\n[3/3] Running Critic on each case...")
    results = evaluate(cases, chunks)

    metrics = print_report(results)

    # Save full results
    output = {
        "metrics": metrics,
        "results": [
            {
                "case_id": r.case_id,
                "case_type": r.case_type,
                "expected": r.expected,
                "actual": r.actual_verdict,
                "cited": r.cited,
                "uncited": r.uncited,
                "correct": r.correct,
                "feedback": r.feedback,
            }
            for r in results
        ],
    }
    with open("experiments/critic_eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nFull results saved to experiments/critic_eval_results.json")


if __name__ == "__main__":
    main()
    # Flush traces before exit
    if _langfuse_available:
        Langfuse().flush()
