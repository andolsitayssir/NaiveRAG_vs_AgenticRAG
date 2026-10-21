"""CLI for testing the Agentic RAG service (single question or interactive)."""
from __future__ import annotations
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG CLI")
    parser.add_argument("question", nargs="*", default=None, help="Question to ask")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--version", action="store_true", help="Show version")

    args = parser.parse_args()

    try:
        from activiity.rag.service import AgenticRagService
    except ImportError as e:
        print(f"[error] {e}", file=sys.stderr)
        print("[error] Is the venv activated? Try: source env/bin/activate", file=sys.stderr)
        print("[error] Is Qdrant running? Try: docker compose up -d qdrant", file=sys.stderr)
        sys.exit(1)

    svc = AgenticRagService()

    if args.version:
        print(f"Activiity Agentic RAG v3")
        print(f"  LLM: {svc._llm_name}")
        print(f"  Embed: {svc._embed_name}")
        return

    def handle(q):
        import time
        t0 = time.perf_counter()
        result = svc.ask(q)
        elapsed = time.perf_counter() - t0
        print(f"\n{'=' * 60}")
        print(f"ANSWER ({elapsed:.1f}s, {result.tokens_in + result.tokens_out} tokens)")
        print(f"{'=' * 60}")
        print(result.answer)
        print(f"\nTools: {result.tools_called}")
        print(f"Units: {result.units_called}")
        print(f"Iterations: {result.iterations}")
        print(f"Chunks: {len(result.retrieved_nodes)}")

    if args.question and args.question[0].lower() == "ask":
        args.question = args.question[1:]

    if args.question:
        handle(" ".join(args.question))
    elif args.interactive:
        print("Activiity Agentic RAG — interactive mode (type :quit to stop)")
        print("=" * 50)
        while True:
            try:
                q = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q == ":quit":
                break
            handle(q)
    else:
        print("Usage: py -m activiity.cli <question> | py -m activiity.cli -i")
        return 1


if __name__ == "__main__":
    main()
