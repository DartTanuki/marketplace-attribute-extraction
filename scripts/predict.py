"""Command-line inference example."""

from __future__ import annotations

import argparse
import json

from marketplace_attribute_extraction import ProductAttributeExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Marketplace search query")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--device", default=None)
    parser.add_argument("--shortlist-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    extractor = ProductAttributeExtractor.from_artifacts(
        args.artifacts,
        device=args.device,
        shortlist_size=args.shortlist_size,
        gliner_threshold=args.threshold,
    )
    result = extractor.predict(args.query)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
