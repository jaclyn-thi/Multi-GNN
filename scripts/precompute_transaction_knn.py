#!/usr/bin/env python
"""Precompute sparse feature-KNN neighbors for transaction contrastive filtering.

Supports CPU sklearn, FAISS GPU exact search, PyTorch CUDA batched top-k, optional
FAISS IVF approximate search, and resumable shard output.

Examples:

  # 100k-row GPU smoke
  python scripts/precompute_transaction_knn.py precompute \\
    --data Small-HI --feature_set edge_native+degree_fan --k 15 \\
    --backend auto --max_rows 100000 \\
    --output morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15_smoke100k.npz

  # Full train GPU job with periodic shards
  python scripts/precompute_transaction_knn.py precompute \\
    --data Small-HI --feature_set edge_native+degree_fan --k 15 \\
    --backend auto --query_batch_size 8192 --shard_rows 250000 \\
    --shard_dir morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15_shards \\
    --output morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15.npz

  # Merge shards only
  python scripts/precompute_transaction_knn.py merge \\
    --shard_dir morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15_shards \\
    --output morphology_cache/Small-HI/transaction_knn_edge_native_degree_fan_k15.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from morphology.target_registry import morph_target_group
from transaction_knn.backends import build_backend, build_exact_reference_backend, list_backends
from transaction_knn.features import (
    build_features_detailed,
    dataset_metadata,
    feature_set_metadata,
    load_train_frame,
    standardize_features,
)
from transaction_knn.shards import (
    merge_shards,
    print_sanity_report,
    shard_path,
    validate_cache,
    write_shard,
)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", default="Small-HI", help="Dataset folder under aml-data")
    parser.add_argument("--data_config", default="data_config.json", help="Path to data_config.json")
    parser.add_argument(
        "--feature_set",
        default="edge_native",
        help="Label-free feature family for KNN (e.g. edge_native+degree_fan, richer_v1).",
    )
    parser.add_argument(
        "--categorical_encoding",
        default="ordinal",
        choices=["ordinal", "one_hot"],
        help="Encoding for currency/payment format when edge_native is included.",
    )
    parser.add_argument(
        "--scaling",
        default="legacy_standard",
        choices=["legacy_standard", "standard", "robust", "none"],
        help="legacy_standard = global StandardScaler; richer sets often use robust per-group.",
    )
    parser.add_argument("--k", type=int, default=50, help="Top-k neighbors per train transaction")
    parser.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "euclidean"],
        help="KNN metric after preprocessing/standardization",
    )
    parser.add_argument("--log_level", default="INFO", help="Python logging level")


def _add_precompute_args(parser: argparse.ArgumentParser) -> None:
    _add_common_args(parser)
    parser.add_argument("--output", required=True, help="Final merged .npz path")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=list_backends(),
        help="KNN backend (auto prefers FAISS GPU, then torch GPU, then CPU sklearn)",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=8192,
        help="Rows queried per KNN batch; lowers peak GPU memory.",
    )
    parser.add_argument(
        "--shard_rows",
        type=int,
        default=250_000,
        help="Write a shard every N query rows (0 = only final output).",
    )
    parser.add_argument(
        "--shard_dir",
        default="",
        help="Directory for shard .npz files (default: <output_stem>_shards).",
    )
    parser.add_argument("--resume", action="store_true", help="Skip shards that already exist.")
    parser.add_argument("--max_rows", type=int, default=0, help="Optional cap for smoke tests.")
    parser.add_argument(
        "--approx_recall_subset",
        type=int,
        default=0,
        help="If >0 and backend is approximate, compare recall@k vs exact on this many queries.",
    )
    parser.add_argument("--faiss_nlist", type=int, default=4096, help="IVF nlist (faiss_ivf only).")
    parser.add_argument("--faiss_nprobe", type=int, default=64, help="IVF nprobe (faiss_ivf only).")
    parser.add_argument("--faiss_train_size", type=int, default=200_000, help="IVF train sample size.")
    parser.add_argument("--no_merge", action="store_true", help="Write shards only; skip final merge.")


def cmd_precompute(args: argparse.Namespace) -> None:
    if args.k <= 0:
        raise ValueError("--k must be positive")

    df, df_train, split, spec = load_train_frame(
        args.data,
        args.data_config,
        max_rows=int(args.max_rows),
    )
    logging.info("Train-only KNN rows: %d / %d", len(df_train), len(df))

    detail = build_features_detailed(
        df_train,
        args.feature_set,
        categorical_encoding=args.categorical_encoding,
        scaling="none" if args.scaling == "legacy_standard" else args.scaling,
    )
    features = detail.features
    feature_names = detail.names
    if args.scaling == "legacy_standard":
        features = standardize_features(features)
    elif args.scaling in {"standard", "robust"}:
        pass  # already scaled in build_features_detailed
    elif args.scaling == "none":
        pass
    if features.shape[0] < 2:
        raise ValueError("Need at least two train transactions to build KNN")
    k = min(int(args.k), features.shape[0] - 1)
    query_batch_size = max(1, int(args.query_batch_size))
    shard_rows = max(0, int(args.shard_rows))

    backend = build_backend(
        args.backend,
        metric=args.metric,
        faiss_nlist=args.faiss_nlist,
        faiss_nprobe=args.faiss_nprobe,
        faiss_train_size=args.faiss_train_size,
    )
    logging.info(
        "Fitting KNN backend=%s feature_set=%s dim=%d k=%d metric=%s rows=%d",
        backend.name,
        args.feature_set,
        features.shape[1],
        k,
        args.metric,
        features.shape[0],
    )
    backend.fit(features, k=k)

    if int(args.approx_recall_subset) > 0 and args.backend == "faiss_ivf":
        n = min(int(args.approx_recall_subset), features.shape[0])
        rng = np.random.default_rng(0)
        q_idx = np.sort(rng.choice(features.shape[0], size=n, replace=False))
        exact = build_exact_reference_backend(args.metric)
        exact.fit(features, k=k)
        recall = backend.recall_at_k(q_idx, k, exact)
        logging.info("Approximate recall@k on %d queries vs exact backend: %.4f", n, recall)

    output = Path(args.output)
    shard_dir = Path(args.shard_dir) if args.shard_dir else output.with_suffix("").with_name(output.stem + "_shards")
    use_shards = shard_rows > 0
    if use_shards:
        shard_dir.mkdir(parents=True, exist_ok=True)

    metadata = dataset_metadata(
        args.data,
        spec,
        args.feature_set,
        feature_names,
        k,
        args.metric,
        query_batch_size,
        backend.backend_id,
        features.shape[0],
        split,
        shard_dir=str(shard_dir) if use_shards else None,
        shard_rows=shard_rows,
        max_rows=int(args.max_rows),
        requested_backend=args.backend,
        **feature_set_metadata(detail),
    )

    n_rows = features.shape[0]
    csv_edge_ids = df_train["EdgeID"].astype(np.int64).to_numpy()

    for shard_start in range(0, n_rows, shard_rows if use_shards else n_rows):
        shard_end = min(shard_start + (shard_rows if use_shards else n_rows), n_rows)
        if use_shards:
            out = shard_path(shard_dir, shard_start, shard_end)
            if args.resume and out.is_file():
                logging.info("Skipping existing shard %s", out)
                continue

        neighbor_ids = np.full((shard_end - shard_start, k), -1, dtype=np.int64)
        neighbor_sims = np.full((shard_end - shard_start, k), np.nan, dtype=np.float32)
        query_positions = np.arange(shard_start, shard_end, dtype=np.int64)

        for batch_start in range(0, query_positions.shape[0], query_batch_size):
            batch_end = min(batch_start + query_batch_size, query_positions.shape[0])
            batch_idx = query_positions[batch_start:batch_end]
            logging.info("Querying rows %d:%d (backend=%s)", int(batch_idx[0]), int(batch_idx[-1] + 1), backend.name)
            idx, sims = backend.query(batch_idx, k)
            neighbor_ids[batch_start:batch_end] = idx
            neighbor_sims[batch_start:batch_end] = sims

        edge_ids = np.arange(shard_start, shard_end, dtype=np.int64)
        if use_shards:
            write_shard(
                shard_dir,
                shard_start,
                shard_end,
                edge_ids=edge_ids,
                csv_edge_ids=csv_edge_ids[shard_start:shard_end],
                neighbor_ids=neighbor_ids,
                neighbor_sims=neighbor_sims,
                metadata=metadata,
            )

    if use_shards and not args.no_merge:
        final_path = merge_shards(shard_dir, output)
    elif not use_shards:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            edge_ids=np.arange(n_rows, dtype=np.int64),
            csv_edge_ids=csv_edge_ids,
            neighbor_ids=neighbor_ids,
            neighbor_sims=neighbor_sims,
            feature_names=np.asarray(feature_names, dtype=object),
            k=np.asarray(k, dtype=np.int64),
            feature_set=np.asarray(args.feature_set),
            metadata_json=np.asarray(__import__("json").dumps(metadata, sort_keys=True)),
        )
        logging.info("Wrote sparse KNN cache: %s", output)
        final_path = output
    else:
        logging.info("Shard-only run complete; merge skipped (%s)", shard_dir)
        return

    report = validate_cache(final_path)
    print_sanity_report(report)
    logging.info(
        "Feature groups in cache: %s",
        sorted({morph_target_group(n) for n in feature_names}),
    )


def cmd_merge(args: argparse.Namespace) -> None:
    output = merge_shards(Path(args.shard_dir), Path(args.output))
    report = validate_cache(output)
    print_sanity_report(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_level", default="INFO", help="Python logging level")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("precompute", help="Build KNN cache (optionally sharded).")
    _add_precompute_args(p_pre)
    p_pre.set_defaults(func=cmd_precompute)

    p_merge = sub.add_parser("merge", help="Merge shard directory into final .npz.")
    p_merge.add_argument("--shard_dir", required=True)
    p_merge.add_argument("--output", required=True)
    p_merge.set_defaults(func=cmd_merge)
    return parser


def main() -> None:
    # Backward compatibility: older invocations omitted the ``precompute`` subcommand.
    if len(sys.argv) > 1 and sys.argv[1] not in ("precompute", "merge", "-h", "--help"):
        sys.argv.insert(1, "precompute")

    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
