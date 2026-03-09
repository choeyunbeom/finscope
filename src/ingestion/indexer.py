"""ChromaDB indexer — embeds chunks and stores them in a persistent collection."""

import hashlib

import chromadb
import httpx

from src.api.core.config import settings
from src.ingestion.base import Document

BATCH_SIZE = 32


def _generate_chunk_id(content: str, metadata: dict, index: int) -> str:
    raw = f"{metadata.get('source','')}:{metadata.get('filing_date','')}:{index}:{content[:80]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _embed_batch(texts: list[str]) -> list[list[float]] | None:
    try:
        resp = httpx.post(
            f"{settings.OLLAMA_BASE_URL}/api/embed",
            json={"model": settings.OLLAMA_EMBED_MODEL, "input": texts},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()["embeddings"]
    except Exception as e:
        print(f"[indexer] embed_batch failed: {e}")
    return None


def _embed_single(text: str) -> list[float] | None:
    try:
        resp = httpx.post(
            f"{settings.OLLAMA_BASE_URL}/api/embed",
            json={"model": settings.OLLAMA_EMBED_MODEL, "input": [text]},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()["embeddings"][0]
    except Exception as e:
        print(f"[indexer] embed_single failed: {e}")
    return None


def index_documents(chunks: list[Document], collection_name: str | None = None) -> int:
    """Embed and index a list of Document chunks into ChromaDB.

    Returns the number of successfully indexed chunks.
    """
    col_name = collection_name or settings.CHROMA_COLLECTION
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

    try:
        collection = client.get_collection(col_name)
    except Exception:
        collection = client.create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )

    total = len(chunks)
    indexed = 0
    skipped = 0
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)
        batch = chunks[start:end]

        texts = [c.content.strip() for c in batch]
        ids = [_generate_chunk_id(c.content, c.metadata, start + i) for i, c in enumerate(batch)]
        metadatas = [
            {k: str(v) for k, v in c.metadata.items()}  # ChromaDB requires str values
            for c in batch
        ]

        embeddings = _embed_batch(texts)

        if embeddings:
            collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
            indexed += len(batch)
            print(f"  Batch {batch_idx + 1}/{total_batches} — {indexed}/{total} indexed")
        else:
            # Fallback: embed individually
            ok = 0
            for i in range(len(batch)):
                emb = _embed_single(texts[i])
                if emb:
                    collection.add(
                        ids=[ids[i]], embeddings=[emb],
                        documents=[texts[i]], metadatas=[metadatas[i]],
                    )
                    indexed += 1
                    ok += 1
                else:
                    skipped += 1
            print(f"  Batch {batch_idx + 1}/{total_batches} fallback — {ok} ok, {len(batch)-ok} skipped")

    print(f"\n  Indexed {indexed}/{total} chunks into '{col_name}' (skipped {skipped})")
    return indexed
