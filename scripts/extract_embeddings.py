"""Extract frozen first-token emotion embeddings for selected posts."""

from __future__ import annotations

import argparse

from eot.config import load_config
from eot.embeddings import encode_posts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eot_main.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    embedding_path, metadata_path = encode_posts(config, args.device, args.batch_size)
    print(f"embeddings={embedding_path}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
