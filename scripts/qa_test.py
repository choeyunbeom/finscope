"""
Single-agent Q&A smoke test.
Retrieves chunks from ChromaDB and answers using Groq.

Usage:
  uv run python scripts/qa_test.py "What are Apple's main risk factors?"
  uv run python scripts/qa_test.py "What was Apple's revenue in 2025?"
"""

import sys

import chromadb
import httpx
from groq import Groq

sys.path.insert(0, ".")
from src.api.core.config import settings

SYSTEM_PROMPT = """You are a financial analyst. Answer the question using ONLY the provided excerpts from SEC filings.
Cite the source for every claim. If the answer is not in the excerpts, say so."""


def embed_query(query: str) -> list[float]:
    resp = httpx.post(
        f"{settings.OLLAMA_BASE_URL}/api/embed",
        json={"model": settings.OLLAMA_EMBED_MODEL, "input": [query]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    collection = client.get_collection(settings.CHROMA_COLLECTION)

    embedding = embed_query(query)
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "metadata": meta})
    return chunks


def answer(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['metadata'].get('filing_type', 'unknown')} {c['metadata'].get('filing_date', '')}]\n{c['text']}"
        for c in chunks
    )

    client = Groq(api_key=settings.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are Apple's main risk factors?"
    print(f"\nQuery: {query}\n")

    print("Retrieving chunks...")
    chunks = retrieve(query)
    print(f"Retrieved {len(chunks)} chunks\n")

    for i, c in enumerate(chunks):
        print(f"[{i+1}] {c['metadata'].get('filing_date', '')} | {c['text'][:120]}...")

    print("\nGenerating answer...\n")
    result = answer(query, chunks)
    print("=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
