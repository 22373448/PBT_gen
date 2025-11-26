from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, List

from .embedding import EMBEDDER, CHROMA_CLIENT, HuggingFaceEmbedder


class VectorStore(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:  # pragma: no cover - interface
        ...


@dataclass
class ChromaVectorStore:
    """
    VectorStore implementation backed by chromadb + HuggingFace embeddings.

    Each document is expected to have:
    - "id": unique string id
    - "content": text (e.g. function source code or docstring)
    - "metadata": dict, MUST include at least:
        - "module_path": python module path of the function/file
        - "rel_path": relative file path (for context & imports)
    """

    collection_name: str
    embedder: HuggingFaceEmbedder = EMBEDDER
    client: Any = CHROMA_CLIENT

    def __post_init__(self) -> None:
        self.collection = self.client.create_collection(self.collection_name)

    def index_documents(self, documents: List[dict[str, Any]]) -> None:
        if not documents:
            return

        ids: List[str] = [str(doc["id"]) for doc in documents]
        texts: List[str] = [str(doc["content"]) for doc in documents]
        metadatas: List[dict[str, Any]] = [dict(doc.get("metadata") or {}) for doc in documents]

        vectors = self.embedder.embed_documents(texts)
        self.collection.add(
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas,
            documents=texts,
        )

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        query_vector = self.embedder.embed_query(query)
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
            )
        except RuntimeError:
            return []

        hits: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for idx, doc_id in enumerate(ids):
            hits.append(
                {
                    "id": doc_id,
                    "content": documents[idx] if idx < len(documents) else "",
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                }
            )

        return hits


