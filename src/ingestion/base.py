"""BaseDocumentLoader — base class for all document loaders."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDocumentLoader(ABC):
    """Abstract base for SEC EDGAR and Companies House loaders."""

    @abstractmethod
    def fetch(self, **kwargs) -> list[Document]:
        """Fetch raw documents from the source."""
        ...

    @abstractmethod
    def parse(self, raw: Any) -> list[Document]:
        """Parse raw data into Document objects."""
        ...

    def load(self, **kwargs) -> list[Document]:
        """fetch + parse pipeline."""
        raw = self.fetch(**kwargs)
        return self.parse(raw)
