from __future__ import annotations

import os
from typing import List

from transformers import AutoTokenizer, AutoModel
import torch
import chromadb


class HuggingFaceEmbedder:
    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding as a list of floats.
        """
        text = self.embed_instruction + text.replace("\n", " ")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            **self.encode_kwargs,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()

        return embedding.tolist()

    def __init__(
        self,
        model_name_or_path: str,
        embed_instruction: str = "",
        show_progress: bool = False,
        encode_kwargs=None,
    ):
        if encode_kwargs is None:
            encode_kwargs = {}
        self.embed_instruction = embed_instruction
        self.show_progress = show_progress
        self.encode_kwargs = encode_kwargs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=True,
        ).to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [self.embed_instruction + t.replace("\n", " ") for t in texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
            **self.encode_kwargs,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = torch.mean(outputs.last_hidden_state, dim=1)

        return embeddings.tolist()


def create_default_embedder() -> HuggingFaceEmbedder:
    """
    Create a default HuggingFaceEmbedder using TRANSFORMER_PATH from environment.
    """
    model_path = os.getenv("TRANSFORMER_PATH") or ""
    return HuggingFaceEmbedder(model_name_or_path=model_path)


EMBEDDER: HuggingFaceEmbedder = create_default_embedder()
CHROMA_CLIENT = chromadb.Client()

