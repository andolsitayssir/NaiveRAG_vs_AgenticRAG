"""Tiny CLI for the naive baseline.

Examples:
    # build the flat index once
    python -m baseline.cli ingest

    # ask an ad-hoc question
    python -m baseline.cli ask "Qu'est-ce que la délégation efficace ?"

    # wipe and rebuild
    python -m baseline.cli ingest --force
"""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path

from baseline.config import DATA_DIR, PERSIST_DIR
from baseline.index import build_index, build_or_load
from baseline.service import NaiveRagService


def cmd_ingest(args):
    if args.force and PERSIST_DIR.exists():
        print(f"[ingest] removing {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    idx = build_index(DATA_DIR, PERSIST_DIR)
    print(f"[ingest] indexed -> {PERSIST_DIR}")
    print(f"         # docs: {len(idx.docstore.docs)}")

def cmd_ask(args):
    svc = NaiveRagService()
    res = svc.ask(args.question)
    print("=== answer ===")
    print(res.answer)
    print("=== top sources ===")
    for n in res.retrieved_nodes:
        md = n.node.metadata if hasattr(n, "node") else n.metadata
        score = getattr(n, "score", None)
        print(f"  [{score:.3f}] {md.get('ua_id')}  {md.get('file_name')}"
              if score else f"  {md.get('ua_id')}  {md.get('file_name')}")
    print(f"\ntokens_in={res.tokens_in}  tokens_out={res.tokens_out}  "
          f"model={res.model_name}  embed={res.embed_model}  "
          f"ttft={res.ttft_s:.2f}s  iterations={res.iterations}")

def main():
    ap = argparse.ArgumentParser(prog="baseline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Build the flat index over data/")
    ing.add_argument("--force", action="store_true", help="Delete and rebuild")
    ing.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="Ask a question against the index")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
