#!/usr/bin/env python3
"""Prepend PNA-specific interpretation to a compare_representation_source markdown note."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--pre_dim", type=int, required=True)
    parser.add_argument("--post_dim", type=int, required=True)
    args = parser.parse_args()
    path = Path(args.path)
    body = path.read_text(encoding="utf-8")
    preamble = (
        f"**PNA interpretation:** `pre_embedding_3h` is **{args.pre_dim}-d** and `post_embedding` is "
        f"**{args.post_dim}-d**. The embedding head is `Linear({args.pre_dim}, {args.post_dim})` — an "
        "**expansion**, not the GIN-style 198→128 compression. Do not assume the GIN pre-3h ranking "
        "advantage transfers.\n\n"
    )
    if preamble.strip() in body:
        return
    path.write_text(preamble + body, encoding="utf-8")
    print(f"Updated {path}")


if __name__ == "__main__":
    main()
