"""LLM / Embedding provider factory.

Same env knobs as the naive baseline so target swaps are apples-to-apples.
"""
from __future__ import annotations

from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field, PrivateAttr

from activiity.config import CFG


def _set_context_window(llm: LLM) -> None:
    """Fix the context window reported by the provider wrapper.

    llama-index's OpenRouter/Ollama wrappers default to 3900 tokens, which
    under-reports the real model context (e.g. 128k for gemma/ministral).
    A wrong (too small) window makes the ReAct prompt builder truncate the
    system prompt and makes the compact synthesizer split the retrieved
    nodes into several tiny chunks, forcing the REFINE path instead of one
    QA_FR pass.
    """
    try:
        llm.context_window = CFG.context_window
    except Exception:  # noqa: BLE001
        pass


def build_llm() -> LLM:
    if CFG.llm_provider == "openrouter":
        from llama_index.llms.openrouter import OpenRouter
        if not CFG.openrouter_api_key:
            raise ValueError("OpenRouter API key is required .")
        try:
            llm = OpenRouter(
                api_key=CFG.openrouter_api_key,
                model=CFG.openrouter_model,
                temperature=CFG.llm_temperature,
                max_tokens=CFG.router_max_tokens,
                streaming=True,
            )
        except TypeError:
            llm = OpenRouter(
                api_key=CFG.openrouter_api_key,
                model=CFG.openrouter_model,
                temperature=CFG.llm_temperature,
                max_tokens=CFG.router_max_tokens,
            )
        _set_context_window(llm)
        return llm
    if CFG.llm_provider == "ollama":
        from llama_index.llms.ollama import Ollama
        try:
            llm = Ollama(
                base_url=CFG.ollama_base_url,
                model=CFG.ollama_llm_model,
                temperature=CFG.llm_temperature,
                request_timeout=120.0,
                streaming=True,
            )
        except TypeError:
            llm = Ollama(
                base_url=CFG.ollama_base_url,
                model=CFG.ollama_llm_model,
                temperature=CFG.llm_temperature,
                request_timeout=120.0,
            )
        _set_context_window(llm)
        return llm
    raise ValueError(f"Unknown llm_provider: {CFG.llm_provider}")


def build_synth_llm(synth_model_override: str | None = None) -> LLM:
    """Build the synthesis LLM (Haiku by default, falls back to the same
    LLM as the agent if USE_HAIKU_SYNTH=0).

    Two-LLM split: Sonnet 4.6 reasons inside the ReAct loop (good at picking
    the right tool from FR descriptions); Haiku produces the final answer
    inside each per-UA query engine (3-5x cheaper, 2-3x faster, negligible
    quality loss on extractive synthesis)."""
    # Custom synth model override takes precedence over config.
    custom_model = synth_model_override
    use_custom = bool(custom_model) or getattr(CFG, "use_synth_custom", False)
    if not custom_model and use_custom:
        custom_model = getattr(CFG, "synth_model", None)

    if not CFG.use_haiku_synth and not custom_model:
        return build_llm()

    synth_model = custom_model or CFG.haiku_model
    max_tokens = 2048 if custom_model else 900

    if CFG.llm_provider == "openrouter":
        from llama_index.llms.openrouter import OpenRouter
        if not CFG.openrouter_api_key:
            raise ValueError("OpenRouter API key is required.")
        try:
            llm = OpenRouter(
                api_key=CFG.openrouter_api_key,
                model=synth_model,
                temperature=CFG.llm_temperature,
                max_tokens=max_tokens,
                streaming=True,
            )
        except TypeError:
            llm = OpenRouter(
                api_key=CFG.openrouter_api_key,
                model=synth_model,
                temperature=CFG.llm_temperature,
                max_tokens=max_tokens,
            )
        _set_context_window(llm)
        return llm
    if CFG.llm_provider == "ollama":
        # On-prem: use the same Ollama LLM (no Haiku equivalent).
        llm = build_llm()
        _set_context_window(llm)
        return llm
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
            base_url=CFG.ollama_base_url, model_name=CFG.ollama_embed_model,
        )
    if CFG.embed_provider in ("sentence_transformers", "huggingface", "local"):
        # Local embeddings (sentence-transformers) — no network at query time.
        # Loads CFG.st_embed_model from the HF cache / local disk.
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(model_name=CFG.st_embed_model)
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
                # InferenceClient.feature_extraction returns a numpy array or list
                res = self._client.feature_extraction(text, model=self.model_name)
                return res.tolist() if hasattr(res, "tolist") else list(res)

            def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
                # Note: Inference API often prefers single calls or handles batches
                # depending on the endpoint configuration. We map it here.
                return [self._get_text_embedding(t) for t in texts]

            def _get_query_embedding(self, query: str) -> list[float]:
                return self._get_text_embedding(query)

            async def _aget_query_embedding(self, query: str) -> list[float]:
                return self._get_query_embedding(query)

        return HFInferenceEmbedding()
    raise ValueError(f"Unknown embed_provider: {CFG.embed_provider}")
