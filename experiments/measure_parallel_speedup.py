"""Measure parallel vs sequential analyzer latency using real Groq API calls.

Uses actual ChromaDB chunks (8 retrieved docs, ~1000 token context) to match
production workload. Run from finscope/ project root.
"""

import asyncio
import time

import chromadb
from dotenv import load_dotenv

load_dotenv()

from groq import Groq
from src.api.core.config import settings

RISK_PROMPT = """You are a financial risk analyst. Based on the following excerpts from a financial filing,
identify and summarise the key risk factors. Be specific and cite the source text.

Excerpts:
{context}

Provide a concise risk analysis with citations (aim for 200-300 words)."""

GROWTH_PROMPT = """You are a financial growth analyst. Based on the following excerpts from a financial filing,
identify and summarise the key growth drivers, revenue trends, and future outlook. Cite the source text.

Excerpts:
{context}

Provide a concise growth analysis with citations (aim for 200-300 words)."""

COMPETITOR_PROMPT = """You are a competitive intelligence analyst. Based on the following excerpts from a financial filing,
identify mentions of competitors, market position, and competitive advantages or threats. Cite the source text.

Excerpts:
{context}

Provide a concise competitive analysis with citations (aim for 200-300 words)."""


def load_real_context() -> str:
    client = chromadb.PersistentClient(path="./chroma_db")
    col = client.get_collection("financial_filings")
    results = col.get(limit=8, include=["documents", "metadatas"])
    return "\n\n---\n\n".join(results["documents"])


def call_groq_sync(prompt: str) -> str:
    client = Groq(api_key=settings.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=400,
    )
    return response.choices[0].message.content


async def call_groq_async(label: str, prompt: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    result = await asyncio.to_thread(call_groq_sync, prompt)
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] {elapsed:.2f}s  ({len(result)} chars output)")
    return result, elapsed


async def run_parallel(context: str) -> float:
    print("--- Parallel run (asyncio.gather) ---")
    t0 = time.perf_counter()
    await asyncio.gather(
        call_groq_async("risk", RISK_PROMPT.format(context=context)),
        call_groq_async("growth", GROWTH_PROMPT.format(context=context)),
        call_groq_async("competitors", COMPETITOR_PROMPT.format(context=context)),
    )
    total = time.perf_counter() - t0
    print(f"  → Total parallel wall-clock: {total:.2f}s\n")
    return total


async def run_sequential(context: str) -> float:
    print("--- Sequential run (await each) ---")
    t0 = time.perf_counter()
    await call_groq_async("risk", RISK_PROMPT.format(context=context))
    await call_groq_async("growth", GROWTH_PROMPT.format(context=context))
    await call_groq_async("competitors", COMPETITOR_PROMPT.format(context=context))
    total = time.perf_counter() - t0
    print(f"  → Total sequential wall-clock: {total:.2f}s\n")
    return total


async def main():
    context = load_real_context()
    print(f"Context: {len(context)} chars (~{len(context)//4} tokens), 8 retrieved chunks\n")

    print("Warming up (avoid cold-start skew)...")
    call_groq_sync("Reply with one word: ready")
    print("Done.\n")

    p_times = []
    s_times = []
    rounds = 3

    for i in range(rounds):
        print(f"=== Round {i + 1}/{rounds} ===")
        p = await run_parallel(context)
        await asyncio.sleep(3)
        s = await run_sequential(context)
        p_times.append(p)
        s_times.append(s)
        if i < rounds - 1:
            await asyncio.sleep(3)

    p_avg = sum(p_times) / len(p_times)
    s_avg = sum(s_times) / len(s_times)
    p_min = min(p_times)
    s_min = min(s_times)

    print("=" * 50)
    print(f"RESULTS ({rounds} rounds, real Groq API, {settings.GROQ_MODEL})")
    print(f"  Parallel   — avg: {p_avg:.1f}s  min: {p_min:.1f}s  all: {[f'{t:.1f}' for t in p_times]}")
    print(f"  Sequential — avg: {s_avg:.1f}s  min: {s_min:.1f}s  all: {[f'{t:.1f}' for t in s_times]}")
    print(f"  Speedup (avg): {s_avg/p_avg:.1f}x")
    print(f"  Speedup (min): {s_min/p_min:.1f}x")
    print(f"  Time saved:    {s_avg - p_avg:.1f}s avg")
    print()
    print("→ README update:")
    print(f'  End-to-end latency | ~{p_avg:.0f}s parallel vs ~{s_avg:.0f}s sequential ({s_avg/p_avg:.1f}x speedup, {settings.GROQ_MODEL}) |')
