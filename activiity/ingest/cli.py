"""CLI: python -m activiity.ingest.cli [--force]"""
from __future__ import annotations
import argparse, json
from activiity.ingest.pipeline import build_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Drop existing collection and re-embed everything")
    args = ap.parse_args()

    info = build_index(force=args.force)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
