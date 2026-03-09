"""Document chunker — 512-token chunks with financial metadata."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.base import Document


def chunk_documents(docs: list[Document], chunk_size: int = 512, chunk_overlap: int = 64) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for doc in docs:
        texts = splitter.split_text(doc.content)
        for i, text in enumerate(texts):
            chunks.append(Document(content=text, metadata={**doc.metadata, "chunk_index": i}))
    return chunks
