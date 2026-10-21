"""Activiity — naive RAG baseline.

Deliberately minimal:
- ONE flat index over all UA-1..UA-10 documents
- top-k vector retrieval, no router, no reranker, no agent
- a single `query_engine.query()` synthesis call per question

Used as a baseline to measure the lift provided by the Agentic RAG.
"""
