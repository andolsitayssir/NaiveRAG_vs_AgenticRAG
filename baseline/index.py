"""Build or load the single flat vector index over all UA buckets.

Uses LlamaIndex's default in-memory `SimpleVectorStore` — persisted to disk
so repeated runs don't re-embed. No Qdrant, no Docker, no MCP for the naive
baseline: "one index, top-k, done".
"""
from __future__ import annotations
from pathlib import Path

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    Settings as LISettings,
    load_index_from_storage,
)
from activiity.ingest.structure_splitter import StructureAwareSplitter

from baseline.config import CFG, DATA_DIR, PERSIST_DIR
from baseline.loader import load_corpus
from baseline.providers import build_llm, build_embed


def _attach_providers_to_global_settings():
    """LlamaIndex uses a global `Settings` singleton so sub-engines inherit
    the same LLM + embedding model."""
    LISettings.llm = build_llm()
    LISettings.embed_model = build_embed()
    # Force a large context window so LlamaIndex doesn't split the synthesis into multiple calls
    LISettings.context_window = 128000
    # Using StructureAwareSplitter to keep protocols and sections together
    LISettings.node_parser = StructureAwareSplitter(
        fallback_chunk_size=CFG.chunk_size, 
        fallback_overlap=CFG.chunk_overlap,
    )


def build_index(data_dir: Path = DATA_DIR, persist_dir: Path = PERSIST_DIR,
                show_progress: bool = True) -> VectorStoreIndex:
    """Embed everything and persist to disk. Overwrites any prior index."""
    _attach_providers_to_global_settings()
    persist_dir.mkdir(parents=True, exist_ok=True)

    docs = load_corpus(data_dir)
    if not docs:
        raise RuntimeError(f"No documents found under {data_dir}")

    index = VectorStoreIndex.from_documents(docs, show_progress=show_progress)
    index.storage_context.persist(persist_dir=str(persist_dir))
    return index


def load_index(persist_dir: Path = PERSIST_DIR) -> VectorStoreIndex:
    _attach_providers_to_global_settings()
    storage = StorageContext.from_defaults(persist_dir=str(persist_dir))
    return load_index_from_storage(storage)


def build_or_load(data_dir: Path = DATA_DIR,
                  persist_dir: Path = PERSIST_DIR) -> VectorStoreIndex:
    """Convenience: load if present, else build."""
    if (persist_dir / "docstore.json").exists():
        return load_index(persist_dir)
    return build_index(data_dir, persist_dir)
