"""Minimal provider factory for the naive baseline.

Not meant to be as flexible as the Agentic service's factory — just enough
to switch between OpenRouter/OpenAI (cloud) and Ollama (on-prem) via env.
"""
from __future__ import annotations

from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr, Field

from baseline.config import CFG


def build_llm() -> LLM:
    if CFG.llm_provider == "openrouter":
        from llama_index.llms.openrouter import OpenRouter
        try:
            return OpenRouter(
                api_key=CFG.openrouter_api_key or None,
                model=CFG.openrouter_model,
                temperature=CFG.llm_temperature,
                max_tokens=1024,
                streaming=True,
            )
        except TypeError:
            return OpenRouter(
                api_key=CFG.openrouter_api_key or None,
                model=CFG.openrouter_model,
                temperature=CFG.llm_temperature,
                max_tokens=1024,
            )
    if CFG.llm_provider == "ollama":
        from llama_index.llms.ollama import Ollama
        try:
            return Ollama(
                base_url=CFG.ollama_base_url,
                model=CFG.ollama_llm_model,
                temperature=CFG.llm_temperature,
                request_timeout=120.0,
                streaming=True,
            )
        except TypeError:
            return Ollama(
                base_url=CFG.ollama_base_url,
                model=CFG.ollama_llm_model,
                temperature=CFG.llm_temperature,
                request_timeout=120.0,
            )
    raise ValueError(f"Unknown llm_provider: {CFG.llm_provider}")


def build_embed() -> BaseEmbedding:
    if CFG.embed_provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding(
            api_key=CFG.openai_api_key or None,
            model=CFG.openai_embed_model,
        )
    if CFG.embed_provider == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding
        return OllamaEmbedding(
            base_url=CFG.ollama_base_url,
            model_name=CFG.ollama_embed_model,
        )
    if CFG.embed_provider in ("hf", "hf-inference"):
        from huggingface_hub import InferenceClient

        class HFInferenceEmbedding(BaseEmbedding):
            model_name: str = Field(default=CFG.st_embed_model)
            _client: InferenceClient = PrivateAttr()

            def model_post_init(self, __context) -> None:
                self._client = InferenceClient(
                    provider="hf-inference",
                    api_key=CFG.hf_token or None,
                )

            def _get_text_embedding(self, text: str) -> list[float]:
                res = self._client.feature_extraction(text, model=self.model_name)
                return res.tolist() if hasattr(res, "tolist") else list(res)

            def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
                return [self._get_text_embedding(t) for t in texts]

            def _get_query_embedding(self, query: str) -> list[float]:
                return self._get_text_embedding(query)

            async def _aget_query_embedding(self, query: str) -> list[float]:
                import asyncio
                return await asyncio.to_thread(self._get_query_embedding, query)

        return HFInferenceEmbedding()
    if CFG.embed_provider == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        class SentenceTransformerEmbedding(BaseEmbedding):
            model_name: str = Field(default=CFG.st_embed_model)
            device: str = Field(default=CFG.st_embed_device)
            _model: SentenceTransformer = PrivateAttr()

            def model_post_init(self, __context) -> None:
                self._model = SentenceTransformer(self.model_name, device=self.device)

            def _get_text_embedding(self, text: str) -> list[float]:
                return self._model.encode(
                    text,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).tolist()

            def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
                return self._model.encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).tolist()

            def _get_query_embedding(self, query: str) -> list[float]:
                return self._get_text_embedding(query)

            async def _aget_query_embedding(self, query: str) -> list[float]:
                return self._get_query_embedding(query)

        return SentenceTransformerEmbedding()
    raise ValueError(f"Unknown embed_provider: {CFG.embed_provider}")
