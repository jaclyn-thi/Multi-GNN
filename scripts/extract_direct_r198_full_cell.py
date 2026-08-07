#!/usr/bin/env python3
"""One (run, epoch, splits) full R198 extract via embedding_extraction.run_embedding_extraction.

Resume-safe for DIRECT_H scheduled val analysis and frozen-transfer scouts:
  - single cell per process (host RAM released on exit)
  - loader_num_workers=0
  - --skip_test_eval (no test graph / emptied te_inds)
  - model.eval() + frozen params + torch.inference_mode()
  - refuse overwrite of already-valid completed split .npz files
  - write under a staging dir; atomic rename after integrity checks

Default --data Small-HI preserves prior AMLWorld extract behavior.
PaySim / SAML-D require train-fit z-norm and transfer feature contracts.

Never trains. Never writes test.npz. Never calls the seed-only extract path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch

from data_loading import get_data
from embedding_extraction import run_embedding_extraction
from util import create_parser, logger_setup, set_seed

ROOT = Path(__file__).resolve().parents[1]

ALLOWED_DATA = ("Small-HI", "PaySim", "SAML-D")
SCOUT_SPLIT_ALLOWLIST = frozenset({"train", "val"})
SEED_ONLY_SCRIPT = "extract_direct_r198_seed_only_cell.py"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refuse_seed_only_path() -> None:
    """Hard refuse any attempt to route through the invalid seed-only extractor."""
    raise RuntimeError(
        f"Refusing seed-only R198 extraction path ({SEED_ONLY_SCRIPT}). "
        "Use embedding_extraction.run_embedding_extraction (full-subgraph) only."
    )


def parse_extract_splits(splits: str) -> List[str]:
    split_list = [s.strip().lower() for s in str(splits).split(",") if s.strip()]
    if not split_list:
        raise SystemExit("Empty --splits")
    if any(s == "test" for s in split_list) or "test" in str(splits).lower():
        raise SystemExit("Refusing test extraction / test in --splits")
    if any(s not in SCOUT_SPLIT_ALLOWLIST for s in split_list):
        raise SystemExit(
            f"Invalid --splits={splits!r}; hard allowlist is train,val only"
        )
    return split_list


def _validate_npz(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"ok": False, "reason": "missing"}
    try:
        d = np.load(path)
        Z = np.asarray(d["Z"])
        y = np.asarray(d["y"]).reshape(-1)
        eid = np.asarray(d["edge_id"]).reshape(-1)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"unreadable: {e}"}
    if Z.ndim != 2 or Z.shape[1] != 198:
        return {"ok": False, "reason": f"bad dim {getattr(Z, 'shape', None)}"}
    if Z.shape[0] != y.shape[0] or Z.shape[0] != eid.shape[0]:
        return {"ok": False, "reason": "row mismatch"}
    if not np.isfinite(Z).all():
        return {"ok": False, "reason": "non-finite"}
    if eid.size != np.unique(eid).size:
        return {"ok": False, "reason": "duplicate edge_id"}
    return {"ok": True, "n": int(Z.shape[0]), "dim": 198}


def _checksum_from_npz(path: Path) -> Dict[str, Any]:
    d = np.load(path)
    eid = np.asarray(d["edge_id"]).reshape(-1)
    y = np.asarray(d["y"]).reshape(-1)
    return {
        "num_rows": int(eid.shape[0]),
        "num_positives": int((y == 1).sum()),
        "positive_rate": float(y.mean()) if eid.shape[0] else float("nan"),
        "edge_id_sum": int(eid.astype("int64").sum()),
        "edge_id_first": int(eid[0]) if eid.shape[0] else None,
        "edge_id_last": int(eid[-1]) if eid.shape[0] else None,
    }


def _build_args(
    run: str,
    epoch: int,
    splits: str,
    embeddings_dir: str,
    *,
    data: str = "Small-HI",
    feature_contract: Optional[str] = None,
    train_fit_edge_znorm: bool = False,
    random_init: bool = False,
    embeddings_subdir: Optional[str] = None,
    extract_max_batches: Optional[int] = None,
    seed: int = 2,
) -> argparse.Namespace:
    if data not in ALLOWED_DATA:
        raise SystemExit(f"Unsupported --data={data!r}; allowed={ALLOWED_DATA}")
    if "test" in splits.lower():
        raise SystemExit("Refusing test in extract_splits")
    if data == "PaySim":
        if feature_contract != "paysim_legacy_duplicate_v1":
            raise SystemExit(
                "PaySim full R198 extract requires "
                "--feature_contract=paysim_legacy_duplicate_v1 "
                "(balance/native contracts prohibited)"
            )
        if not train_fit_edge_znorm:
            raise SystemExit("PaySim extract requires --train_fit_edge_znorm")
    if data == "SAML-D":
        if feature_contract is not None:
            raise SystemExit(
                "SAML-D frozen transfer uses protocol-B geometry without "
                "PaySim --feature_contract"
            )
        if not train_fit_edge_znorm:
            raise SystemExit("SAML-D extract requires --train_fit_edge_znorm")

    parser = create_parser()
    parser.add_argument("--embeddings_dir", type=str, default="embeddings")
    parser.add_argument("--random_init", action="store_true")
    parser.add_argument("--checkpoint_suffix", type=str, default="")
    parser.add_argument("--embeddings_subdir", type=str, default=None)
    parser.add_argument(
        "--representation_source",
        type=str,
        default="pre_embedding_3h",
        choices=["post_embedding", "pre_embedding_3h"],
    )
    parser.add_argument("--extract_splits", type=str, default="train,val")
    parser.add_argument("--extract_max_batches", type=int, default=None)
    argv = [
        "--data",
        data,
        "--model",
        "gin",
        "--batch_size",
        "8192",
        "--num_neighs",
        "100",
        "100",
        "--loader_num_workers",
        "0",
        "--reverse_mp",
        "--ego",
        "--ports",
        "--emlps",
        "--tds",
        "--correct_reverse_edge_features",
        "--seed",
        str(int(seed)),
        "--tqdm",
        "--testing",
        "--skip_test_eval",
        "--direct_r198_infonce",
        "--representation_source",
        "pre_embedding_3h",
        "--extract_splits",
        splits,
        "--objective",
        "contrastive",
        "--unique_name",
        run,
        "--checkpoint_suffix",
        f"_epoch{epoch:02d}",
        "--embeddings_subdir",
        embeddings_subdir or f"{run}_epoch{epoch:02d}",
        "--embeddings_dir",
        embeddings_dir,
    ]
    if train_fit_edge_znorm:
        argv.append("--train_fit_edge_znorm")
    if feature_contract:
        argv.extend(["--feature_contract", feature_contract])
    if random_init:
        argv.append("--random_init")
    if extract_max_batches is not None:
        argv.extend(["--extract_max_batches", str(int(extract_max_batches))])
    args = parser.parse_args(argv)
    # Hard locks independent of argparse defaults.
    args.include_temporal_flow_edge_features = False
    args.preserve_seed_edges = False
    args.skip_test_eval = True
    args.ports = True
    args.tds = True
    args.correct_reverse_edge_features = True
    args.emlps = True
    args.reverse_mp = True
    args.ego = True
    args.embedding_dim = 198
    return args


def assert_transfer_geometry(args: argparse.Namespace) -> None:
    if not bool(args.ports) or not bool(args.tds):
        raise RuntimeError("ports and tds must be True (edge_dim=8)")
    if not bool(args.correct_reverse_edge_features):
        raise RuntimeError("correct_reverse_edge_features must be True")
    if bool(getattr(args, "include_temporal_flow_edge_features", False)):
        raise RuntimeError("include_temporal_flow_edge_features must be False")
    if bool(getattr(args, "preserve_seed_edges", False)):
        raise RuntimeError("preserve_seed_edges must be False")
    if "test" in str(getattr(args, "extract_splits", "")).lower():
        raise RuntimeError("extract_splits must not contain test")


def main() -> int:
    logger_setup()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument(
        "--splits",
        type=str,
        required=True,
        help="Comma-separated subset of {train,val} to generate (never test).",
    )
    ap.add_argument("--embeddings_dir", type=str, default="embeddings")
    ap.add_argument("--data", type=str, default="Small-HI", choices=list(ALLOWED_DATA))
    ap.add_argument(
        "--feature_contract",
        type=str,
        default=None,
        help="PaySim: paysim_legacy_duplicate_v1 required.",
    )
    ap.add_argument(
        "--train_fit_edge_znorm",
        action="store_true",
        help="Required for PaySim/SAML-D transfer extracts.",
    )
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--embeddings_subdir", type=str, default=None)
    ap.add_argument(
        "--extract_max_batches",
        type=int,
        default=None,
        help="Smoke only: limit batches per split (does not write full coverage).",
    )
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument(
        "--expected_checkpoint_sha256",
        type=str,
        default=None,
        help="If set, refuse when checkpoint SHA256 mismatches.",
    )
    ap.add_argument(
        "--allow_seed_only",
        action="store_true",
        help="Rejected: seed-only path is never allowed.",
    )
    cli = ap.parse_args()

    if cli.allow_seed_only:
        refuse_seed_only_path()

    split_list = parse_extract_splits(cli.splits)

    subdir = cli.embeddings_subdir or f"{cli.run}_epoch{cli.epoch:02d}"
    final_dir = ROOT / cli.embeddings_dir / subdir / "pre_embedding_3h"
    final_dir.mkdir(parents=True, exist_ok=True)
    if (final_dir / "test.npz").is_file():
        raise SystemExit(f"test.npz present — abort: {final_dir}")

    already: Set[str] = set()
    need: List[str] = []
    for s in split_list:
        st = _validate_npz(final_dir / f"{s}.npz")
        if st.get("ok") and cli.extract_max_batches is None:
            already.add(s)
            logging.info("Reuse valid %s.npz n=%s (skip regenerate)", s, st.get("n"))
        else:
            need.append(s)
    if not need:
        meta_path = final_dir / "meta.json"
        if not meta_path.is_file():
            raise SystemExit(f"All splits present but meta.json missing: {final_dir}")
        logging.info("Nothing to extract; all requested splits already valid")
        print(
            json.dumps(
                {
                    "status": "reuse",
                    "run": cli.run,
                    "epoch": cli.epoch,
                    "data": cli.data,
                    "splits": split_list,
                }
            )
        )
        return 0

    ckpt = ROOT / "saved-models" / f"checkpoint_{cli.run}_epoch{cli.epoch:02d}.tar"
    if not cli.random_init:
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        sha = _sha256_file(ckpt)
        if cli.expected_checkpoint_sha256 and sha != cli.expected_checkpoint_sha256:
            raise SystemExit(
                f"Checkpoint SHA256 mismatch: got {sha} expected {cli.expected_checkpoint_sha256}"
            )
    else:
        sha = None

    with open(ROOT / "data_config.json", "r", encoding="utf-8") as f:
        data_config = json.load(f)

    args = _build_args(
        cli.run,
        cli.epoch,
        ",".join(need),
        cli.embeddings_dir,
        data=cli.data,
        feature_contract=cli.feature_contract,
        train_fit_edge_znorm=bool(cli.train_fit_edge_znorm),
        random_init=bool(cli.random_init),
        embeddings_subdir=subdir,
        extract_max_batches=cli.extract_max_batches,
        seed=int(cli.seed),
    )
    assert_transfer_geometry(args)
    set_seed(int(args.seed))

    staging_root = final_dir / f".staging_{os.getpid()}_{int(time.time())}"
    staging_embeddings_dir = staging_root
    staging_embeddings_dir.mkdir(parents=True, exist_ok=True)
    args.embeddings_dir = str(staging_embeddings_dir)

    logging.info(
        "Full R198 cell extract data=%s run=%s epoch=%s need=%s reuse=%s "
        "workers=0 skip_test=1 random_init=%s max_batches=%s",
        cli.data,
        cli.run,
        cli.epoch,
        need,
        sorted(already),
        bool(cli.random_init),
        cli.extract_max_batches,
    )
    t0 = time.perf_counter()
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    logging.info(
        "Retrieved data in %.2fs (te_inds=%s)", time.perf_counter() - t0, int(te_inds.numel())
    )
    if int(te_inds.numel()) != 0:
        raise RuntimeError("te_inds non-empty despite skip_test_eval")

    # Edge-dim gate (ports+tds => 8). Hetero stores attrs under edge types.
    def _edge_dim(g) -> int:
        if hasattr(g, "edge_attr") and g.edge_attr is not None:
            return int(g.edge_attr.shape[1])
        store = g["node", "to", "node"]
        return int(store.edge_attr.shape[1])

    edim = _edge_dim(tr_data)
    if edim != 8:
        raise RuntimeError(
            f"edge_dim={edim} != 8; refuse protocol-A / missing TDS-or-ports geometry"
        )

    with torch.inference_mode():
        out = run_embedding_extraction(
            tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config
        )
    out = Path(out)
    logging.info("Staged extract at %s", out)

    promoted = []
    for s in need:
        src = out / f"{s}.npz"
        st = _validate_npz(src)
        if not st.get("ok"):
            raise RuntimeError(f"Staged {s}.npz failed integrity: {st}")
        if st["dim"] != 198:
            raise RuntimeError(f"Expected R198, got {st}")
        dst = final_dir / f"{s}.npz"
        if (
            cli.extract_max_batches is None
            and dst.is_file()
            and _validate_npz(dst).get("ok")
        ):
            raise RuntimeError(f"Refusing overwrite of valid {dst}")
        tmp_dst = final_dir / f"{s}.npz.promoting"
        if tmp_dst.is_file():
            tmp_dst.unlink()
        shutil.copy2(src, tmp_dst)
        st2 = _validate_npz(tmp_dst)
        if not st2.get("ok"):
            tmp_dst.unlink(missing_ok=True)
            raise RuntimeError(f"Promoted copy failed integrity: {st2}")
        os.replace(tmp_dst, dst)
        promoted.append({"split": s, **st2})
        logging.info("Atomically promoted %s -> %s", s, dst)

    if (final_dir / "test.npz").is_file():
        raise RuntimeError("test.npz appeared — abort")

    meta_src = out / "meta.json"
    meta: Dict[str, Any] = (
        json.loads(meta_src.read_text(encoding="utf-8")) if meta_src.is_file() else {}
    )
    checksums = dict(meta.get("split_checksums") or {})
    for s in ("train", "val"):
        p = final_dir / f"{s}.npz"
        if _validate_npz(p).get("ok"):
            checksums[s] = _checksum_from_npz(p)
    meta["split_checksums"] = checksums
    meta["extractor"] = "embedding_extraction.run_embedding_extraction"
    meta["extractor_script"] = "extract_direct_r198_full_cell"
    meta["seed_only_r198"] = False
    meta["skip_test_eval"] = True
    meta["loader_num_workers"] = 0
    meta["inference_mode"] = True
    meta["data"] = cli.data
    meta["edge_dim"] = 8
    meta["checkpoint_sha256"] = sha
    meta["promoted_splits"] = [p["split"] for p in promoted]
    meta["reused_splits"] = sorted(already)
    meta["partial_batches"] = cli.extract_max_batches is not None
    meta_tmp = final_dir / "meta.json.tmp"
    meta_tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(meta_tmp, final_dir / "meta.json")

    shutil.rmtree(staging_root, ignore_errors=True)

    result = {
        "status": "ok",
        "run": cli.run,
        "epoch": cli.epoch,
        "data": cli.data,
        "promoted": promoted,
        "reused": sorted(already),
        "embedding_dir": str(final_dir),
        "checkpoint": str(ckpt) if not cli.random_init else None,
        "checkpoint_sha256": sha,
        "test_extracted": False,
        "seed_only_r198": False,
        "partial_batches": cli.extract_max_batches is not None,
    }
    print(json.dumps(result, indent=2))
    logging.info("Cell complete: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
