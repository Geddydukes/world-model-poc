"""CLI wrapper for rebuilding ANN indexes."""

from __future__ import annotations

import argparse

from src.index.rebuild import rebuild_index


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild ANN index from embeddings")
    ap.add_argument("--modality", required=True, choices=["vision", "audio", "object"])
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--sqlite-path", default="memory/episodic.sqlite")
    ap.add_argument("--output-dir", default="memory/indexes")
    ap.add_argument("--use-cosine", action="store_true")
    args = ap.parse_args()

    index_path = rebuild_index(
        modality=args.modality,
        model_tag=args.model_tag,
        sqlite_path=args.sqlite_path,
        output_dir=args.output_dir,
        use_cosine=args.use_cosine,
    )
    print(f"Index rebuilt at {index_path}")


if __name__ == "__main__":
    main()
