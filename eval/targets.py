"""Factory that returns the RAG service the eval should evaluate.

Selected via CLI flag --target or env var ACTIVIITY_TARGET:

    naive     → baseline.service.NaiveRagService
    agentic   → activiity.rag.service.AgenticRagService  (when implemented)
    mock      → eval.runner.MockSvc (deterministic fake, handled in run_eval)
"""
from __future__ import annotations
import os


def build_target(name: str | None = None):
    name = (name or os.getenv("ACTIVIITY_TARGET", "naive")).lower()

    if name == "naive":
        from baseline.service import NaiveRagService
        return NaiveRagService()

    if name == "agentic":
        from activiity.rag.service import AgenticRagService
        return AgenticRagService()

    raise ValueError(f"Unknown --target: {name!r}. Use 'naive' or 'agentic'.")
