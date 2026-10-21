"""Shared judge-embedding backend for RAGAS.

This keeps the RAGAS judge path usable without an OpenAI key by using a local
multilingual sentence-transformer for the embedding side of the metrics.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from ragas.embeddings import LangchainEmbeddingsWrapper

JUDGE_EMBED_MODEL = os.getenv(
    "ACTIVIITY_JUDGE_EMBED_MODEL",
    "ibm-granite/granite-embedding-311m-multilingual-r2",
)
JUDGE_EMBED_DEVICE = os.getenv("ACTIVIITY_JUDGE_EMBED_DEVICE", "cpu")


class _HFInferenceEmbeddings:
    """Judge embedding backend using Hugging Face Inference API."""

    def __init__(self, model_name: str):
        from huggingface_hub import InferenceClient
        token = os.getenv("HF_TOKEN")
        self.model_name = model_name
        self._client = InferenceClient(
            provider="hf-inference",
            api_key=token or None,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Hugging Face Inference API for feature extraction
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        res = self._client.feature_extraction(text, model=self.model_name)
        return res.tolist() if hasattr(res, "tolist") else list(res)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class _SentenceTransformerEmbeddings:
    def __init__(self, model_name: str, device: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


@lru_cache(maxsize=1)
def _backend(model_name: str, device: str) -> Any:
    # Use the HF Inference API only when explicitly requested: the default
    # is a local sentence-transformer. Routing on HF_TOKEN alone broke the
    # RAGAS embedding side because granite-embedding is not served by the
    # Inference API (returns NaN for answer_relevancy / answer_correctness).
    embed_provider = os.getenv("EMBED_PROVIDER", "").lower()
    if embed_provider == "hf-inference":
        return _HFInferenceEmbeddings(model_name)
    return _SentenceTransformerEmbeddings(model_name, device)


def build_judge_embeddings():
    return LangchainEmbeddingsWrapper(_backend(JUDGE_EMBED_MODEL, JUDGE_EMBED_DEVICE))