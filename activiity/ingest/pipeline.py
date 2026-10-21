"""Ingestion pipeline → Qdrant.

Walks data/UA-*/*.{docx,pdf}, chunks, embeds, upserts to a per-embed-model
Qdrant collection with a payload index on `ua_id` for fast pre-filtering.
"""
from __future__ import annotations
from pathlib import Path

from llama_index.core import Settings as LISettings, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models as qm

from activiity.config import CFG, DATA_DIR, collection_name, embed_model_name
from activiity.providers.factory import build_embed
# Reuse the loader from the naive baseline — same parsed text + metadata,
# guaranteeing apples-to-apples comparison with the naive run.
from baseline.loader import load_corpus


def ensure_collection(client: QdrantClient, name: str, dim: int):
    """Create the collection + payload index on ua_id if missing."""
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        client.create_payload_index(
            name, field_name="ua_id",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )


def build_index(force: bool = False) -> dict:
    """Build the Qdrant index. Idempotent unless force=True."""
    coll = collection_name(embed_model_name())
    client = QdrantClient(url=CFG.qdrant_url, timeout=60)

    if force and client.collection_exists(coll):
        client.delete_collection(coll)

    # Quick skip: if collection already has vectors and force is False,
    # avoid the embed-initialisation network call entirely.
    if not force and client.collection_exists(coll):
        info = client.get_collection(coll)
        if info.points_count:
            print(
                f"[ingest] collection {coll!r} already has "
                f"{info.points_count} points — skipping (force=False)"
            )
            return {"collection": coll, "points": info.points_count, "skipped": True}

    embed = build_embed()
    LISettings.embed_model = embed

    dim = len(embed.get_text_embedding("ping"))

    ensure_collection(client, coll, dim)

    info = client.get_collection(coll)
    if info.points_count and not force:
        print(f"[ingest] collection {coll!r} already has "
              f"{info.points_count} points — skipping (force=False)")
        return {"collection": coll, "points": info.points_count, "skipped": True}

    vstore = QdrantVectorStore(client=client, collection_name=coll)

    # Choose the splitter based on config flag (RAG optimization #1)
    if CFG.use_structure_splitter:
        from activiity.ingest.structure_splitter import StructureAwareSplitter
        splitter = StructureAwareSplitter(
            target_chunk_chars=CFG.structure_target_chars,
            min_chunk_chars=CFG.structure_min_chars,
            max_chunk_chars=CFG.structure_max_chars,
        )
        print(
            f"[ingest] using StructureAwareSplitter (Fiche Protocole markers) "
            f"target={CFG.structure_target_chars} "
            f"min={CFG.structure_min_chars} max={CFG.structure_max_chars}"
        )
    else:
        splitter = SentenceSplitter(
            chunk_size=CFG.chunk_size, chunk_overlap=CFG.chunk_overlap,
        )
        print(f"[ingest] using SentenceSplitter chunk={CFG.chunk_size}/{CFG.chunk_overlap}")

    pipeline = IngestionPipeline(
        transformations=[splitter, embed],
        vector_store=vstore,
    )
    docs = load_corpus(DATA_DIR)
    if not docs:
        raise RuntimeError(f"no documents under {DATA_DIR}")

    nodes = pipeline.run(documents=docs, show_progress=True)
    info = client.get_collection(coll)
    return {"collection": coll, "documents": len(docs),
            "nodes": len(nodes), "points": info.points_count, "skipped": False}


def open_index() -> tuple[VectorStoreIndex, QdrantClient, str]:
    """Open the existing Qdrant collection as a LlamaIndex VectorStoreIndex."""
    embed = build_embed()
    LISettings.embed_model = embed

    coll = collection_name(embed_model_name())

    client = QdrantClient(url=CFG.qdrant_url, timeout=60)
    if not client.collection_exists(coll):
        raise RuntimeError(f"Qdrant collection {coll!r} not found — "
                           f"run `python -m activiity.ingest.cli` first")

    vstore = QdrantVectorStore(client=client, collection_name=coll)
    index = VectorStoreIndex.from_vector_store(vstore)
    return index, client, coll
