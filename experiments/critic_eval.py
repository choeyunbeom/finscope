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
    """Pull chunks that contain real financial figures (not XBRL metadata tags)."""
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(name=settings.CHROMA_COLLECTION)

    # Scan up to 600 chunks and filter for ones with actual numbers
    results = collection.get(limit=600, include=["documents", "metadatas"])
    chunks = []
    for doc_text, meta in zip(results["documents"], results["metadatas"]):
        has_figures = re.search(r"\$[\d,]+|[\d,]+\s*(billion|million|%)", doc_text, re.I)
        no_xbrl = "http" not in doc_text and len(doc_text) > 200
        if has_figures and no_xbrl:
            chunks.append({"text": doc_text, "metadata": meta})
        if len(chunks) >= n:
            break
    return chunks


# ---------------------------------------------------------------------------
# 2. Synthetic analysis generator
# ---------------------------------------------------------------------------

GENERATE_CLEAN_PROMPT = """Based ONLY on the following source excerpts, write a financial analysis with exactly 8 claims.
Every claim must cite a specific number or fact directly from the excerpts.
Paraphrasing the source is acceptable, but do not invent any numbers.

Source excerpts:
{context}

Write the analysis now."""

# Tier 1: obvious hallucination — numbers completely fabricated (not in source at all)
GENERATE_HALLUCINATED_OBVIOUS_PROMPT = """Based on the following source excerpts, write a financial analysis with exactly 8 claims.
Exactly {n_fab} of the claims must contain numbers that are completely fabricated and NOT present anywhere in the source.
The fabricated numbers should be plausible-sounding but wrong (e.g. wrong revenue figure, invented margin %).
The other claims should be grounded in the source.
Mark each fabricated claim with [FAB].

Source excerpts:
{context}

Write the analysis now."""

# Tier 2: subtle hallucination — paraphrased citation that alters the meaning
GENERATE_HALLUCINATED_SUBTLE_PROMPT = """Based on the following source excerpts, write a financial analysis with exactly 8 claims.
Exactly {n_fab} of the claims should cite a real figure from the source but change it slightly
(e.g. report $94.83B as $98B, flip a YoY increase to a decrease, or report the wrong year).
These should be hard to detect without checking the source carefully.
The other claims should be accurately cited.
Mark each subtly altered claim with [FAB].

Source excerpts:
{context}

Write the analysis now."""

# Tier 3: borderline — ~30% altered, near the 35% threshold
GENERATE_BORDERLINE_PROMPT = """Based on the following source excerpts, write a financial analysis with exactly 8 claims.
Exactly 2 of the 8 claims should contain slightly altered numbers (off by 5-15% or minor rounding).
The other 6 claims should be accurately cited from the source.
Mark altered claims with [ALT].

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
    """Generate 27 synthetic test cases across 3 difficulty tiers.

    Clean (9): source-grounded analyses — expected sufficient
    Hallucinated obvious (3): completely fabricated numbers — clearly insufficient
    Hallucinated subtle (3): real figures slightly altered — harder to catch
    Borderline (3): ~25% altered, near the 35% threshold
    """
    context = _build_context(chunks)
    cases = []

    # --- Clean (9 cases) ---
    for i in range(9):
        print(f"  Generating clean case {i+1}/9...")
        analysis = _call_llm(GENERATE_CLEAN_PROMPT.format(context=context))
        cases.append({
            "id": f"clean_{i+1}",
            "type": "clean",
            "expected_verdict": "sufficient",
            "analysis": analysis,
        })
        time.sleep(2)

    # --- Hallucinated obvious (3 cases): 4-5 of 8 claims fabricated → clearly >35% ---
    for i in range(3):
        print(f"  Generating hallucinated_obvious case {i+1}/3...")
        analysis = _call_llm(GENERATE_HALLUCINATED_OBVIOUS_PROMPT.format(
            context=context, n_fab=5
        ))
        cases.append({
            "id": f"hallucinated_obvious_{i+1}",
            "type": "hallucinated_obvious",
            "expected_verdict": "insufficient",
            "analysis": re.sub(r"\s*\[FAB\]", "", analysis),
            "analysis_with_markers": analysis,
        })
        time.sleep(2)

    # --- Hallucinated subtle (3 cases): 3-4 of 8 claims subtly altered → >35%, harder ---
    for i in range(3):
        print(f"  Generating hallucinated_subtle case {i+1}/3...")
        analysis = _call_llm(GENERATE_HALLUCINATED_SUBTLE_PROMPT.format(
            context=context, n_fab=4
        ))
        cases.append({
            "id": f"hallucinated_subtle_{i+1}",
            "type": "hallucinated_subtle",
            "expected_verdict": "insufficient",
            "analysis": re.sub(r"\s*\[FAB\]", "", analysis),
            "analysis_with_markers": analysis,
        })
        time.sleep(2)

    # --- Borderline (3 cases): 2 of 8 claims altered (25%) → near threshold ---
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
        time.sleep(2)

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

    clean_results = [r for r in results if r.case_type == "clean"]
    halluc_obvious = [r for r in results if r.case_type == "hallucinated_obvious"]
    halluc_subtle  = [r for r in results if r.case_type == "hallucinated_subtle"]
    halluc_all     = halluc_obvious + halluc_subtle
    borderline_results = [r for r in results if r.case_type == "borderline"]

    tp = sum(1 for r in halluc_all if r.actual_verdict == "insufficient")
    fn = sum(1 for r in halluc_all if r.actual_verdict == "sufficient")
    tn = sum(1 for r in clean_results if r.actual_verdict == "sufficient")
    fp = sum(1 for r in clean_results if r.actual_verdict == "insufficient")

    tp_ob = sum(1 for r in halluc_obvious if r.actual_verdict == "insufficient")
    tp_su = sum(1 for r in halluc_subtle  if r.actual_verdict == "insufficient")

    print("\n" + "-" * 70)
    print("CONFUSION MATRIX (clean vs hallucinated, borderline excluded)")
    print("-" * 70)
    print(f"                    Predicted")
    print(f"                    sufficient  insufficient")
    print(f"  Actual clean      {tn:>6}      {fp:>6}   (n={len(clean_results)})")
    print(f"  Actual halluc     {fn:>6}      {tp:>6}   (n={len(halluc_all)})")

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    print(f"\n  Sensitivity overall:          {sensitivity:.0%} ({tp}/{tp+fn})")
    print(f"    — Obvious hallucination:    {tp_ob/len(halluc_obvious):.0%} ({tp_ob}/{len(halluc_obvious)})")
    print(f"    — Subtle hallucination:     {tp_su/len(halluc_subtle):.0%} ({tp_su}/{len(halluc_subtle)})")
    print(f"  Specificity (clean):          {specificity:.0%} ({tn}/{tn+fp})")
    print(f"  Accuracy:                     {accuracy:.0%}")

    print(f"\n  Borderline cases (~25% altered, near threshold):")
    for r in borderline_results:
        print(f"    {r.case_id}: {r.actual_verdict} (cited={r.cited}, uncited={r.uncited})")

    print("\n" + "=" * 70)

    return {
        "sensitivity": sensitivity,
        "sensitivity_obvious": tp_ob / len(halluc_obvious) if halluc_obvious else 0,
        "sensitivity_subtle": tp_su / len(halluc_subtle) if halluc_subtle else 0,
        "specificity": specificity,
        "accuracy": accuracy,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "n_clean": len(clean_results),
        "n_hallucinated": len(halluc_all),
        "n_borderline": len(borderline_results),
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
