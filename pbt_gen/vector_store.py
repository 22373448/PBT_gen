from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class VectorStore(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:  # pragma: no cover - interface
        ...


@dataclass
class DummyVectorStore:
    """
    A placeholder in-memory vector store for development.
    You should replace this with your own implementation that calls
    Pinecone/Faiss/Elastic/etc. and returns hits with rich metadata.
    """

    documents: list[dict[str, Any]]

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        # Naive stub: just return first top_k docs.
        # Real implementation should do embedding + ANN search.
        return self.documents[:top_k]


