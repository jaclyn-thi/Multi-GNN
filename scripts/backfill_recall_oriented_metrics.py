#!/usr/bin/env python3
"""CPU-only backfill of recall-oriented metrics via probe/eval re-runs.

Does NOT retrain GNNs, does NOT regenerate embeddings, does NOT use GPUs.
Writes enriched outputs under results/diagnostics/enriched/ (never overwrites
original probe/eval JSONs).

Scores are not persisted in existing diagnostics, so metrics are recomputed by
re-fitting the frozen linear probe (or supervised eval) from existing embeddings.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from util import logger_setup, set_seed

ENRICHED_DIR = _ROOT / "results" / "diagnostics" / "enriched"


@dataclass
class ProbeTarget:
    tag: str
    data: str
    run_name: str
    embedding_dir: Path
    cache_dir: Path
    representation_source: str
    arms: str
    seed: int = 1
    max_iter: int = 5000
    original_json: Optional[Path] = None


def _ensure_cpu() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _emb_ok(d: Path) -> bool:
    return all((d / f"{s}.npz").is_file() for s in ("train", "val", "test"))


def _backup_if_exists(path: Path) -> Optional[Path]:
    if not path.is_file():
        return None
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.is_file():
        bak.write_bytes(path.read_bytes())
        return bak
    return bak


def default_targets() -> List[ProbeTarget]:
    cache_hi = _ROOT / "results/cache/temporal_flow_causal/Small-HI"
    cache_li = _ROOT / "results/cache/temporal_flow_causal/Small-LI"
    arms = "A_embedding,B_embedding_raw,D_embedding_raw_temporal_flow"
    out: List[ProbeTarget] = []

    # Temporal-flow aux scout (post + pre)
    for variant in ("tf_reg_w0.10", "tf_reg_w0.05", "tf_bins5_w0.10", "tf_bins10_w0.10"):
        run = f"hi_tf_aux_{variant}_gin_emlps_tds_asym_proj_8192neg_queue0_accum4_20ep_seed1"
        base = _ROOT / "embeddings" / run
        for rep, sub, tag_rep in (
            ("post_embedding_128", ".", "post128"),
            ("pre_embedding_3h", "pre_embedding_3h", "pre3h"),
        ):
            emb = base if sub == "." else base / sub
            out.append(
                ProbeTarget(
                    tag=f"tf_aux_{variant}_{tag_rep}",
                    data="Small-HI",
                    run_name=run,
                    embedding_dir=emb,
                    cache_dir=cache_hi,
                    representation_source=rep,
                    arms=arms,
                    original_json=_ROOT
                    / f"results/diagnostics/tf_aux_{variant}_{tag_rep}_seed1.json",
                )
            )

    # Baseline post-128 (20ep seed1 strong recipe) — no pre-3h embeddings available
    base_run = "hi_contrastive_gin_emlps_tds_proj_asym_8192neg_queue0_accum4_20ep"
    out.append(
        ProbeTarget(
            tag="baseline_hi_20ep_seed1_post128",
            data="Small-HI",
            run_name=base_run,
            embedding_dir=_ROOT / "embeddings" / base_run,
            cache_dir=cache_hi,
            representation_source="post_embedding_128",
            arms=arms,
        )
    )

    # Validated temporal-flow HI (40ep seed2 pre-3h)
    hi40 = "gin_emlps_tds_asym_proj_8192neg_queue0_40ep_seed2"
    out.append(
        ProbeTarget(
            tag="tf_validated_hi_40ep_seed2_pre3h",
            data="Small-HI",
            run_name=hi40,
            embedding_dir=_ROOT / "embeddings" / hi40 / "pre_embedding_3h",
            cache_dir=cache_hi,
            representation_source="pre_embedding_3h",
            arms="A_embedding,B_embedding_raw,C_embedding_temporal_flow,D_embedding_raw_temporal_flow",
            seed=2,
            original_json=_ROOT
            / "results/diagnostics/temporal_flow_ablation_small_hi_40ep_seed2_maxiter5000.json",
        )
    )

    # Small-LI multiseed temporal-flow validated
    for seed in (1, 2, 3):
        run = f"small_li_gin_emlps_tds_asym_proj_8192neg_queue0_20ep_seed{seed}"
        emb = _ROOT / "embeddings" / run / "pre_embedding_3h"
        out.append(
            ProbeTarget(
                tag=f"tf_validated_li_seed{seed}_pre3h",
                data="Small-LI",
                run_name=run,
                embedding_dir=emb,
                cache_dir=cache_li,
                representation_source="pre_embedding_3h",
                arms="A_embedding,B_embedding_raw,C_embedding_temporal_flow,D_embedding_raw_temporal_flow",
                seed=seed,
                original_json=_ROOT
                / f"results/diagnostics/temporal_flow_ablation_small_li_seed{seed}_maxiter5000.json",
            )
        )
    return out


def run_probe_target(t: ProbeTarget, *, n_jobs: int, force: bool) -> Dict[str, Any]:
    out_json = ENRICHED_DIR / f"{t.tag}_recall_metrics.json"
    out_md = ENRICHED_DIR / f"{t.tag}_recall_metrics.md"
    status: Dict[str, Any] = {
        "tag": t.tag,
        "run_name": t.run_name,
        "embedding_dir": str(t.embedding_dir),
        "output_json": str(out_json.relative_to(_ROOT)),
        "status": "pending",
    }
    if out_json.is_file() and not force:
        status["status"] = "skipped_exists"
        return status
    if not _emb_ok(t.embedding_dir):
        status["status"] = "skipped_missing_embeddings"
        return status
    if not (t.cache_dir / "features.npy").is_file():
        status["status"] = "skipped_missing_tf_cache"
        return status

    from scripts import probe_temporal_flow_ablation as ptf

    args = argparse.Namespace(
        data=t.data,
        data_config="data_config.json",
        run_name=t.run_name,
        embedding_dir=str(t.embedding_dir),
        temporal_flow_cache_dir=str(t.cache_dir),
        output_json=str(out_json),
        output_md=str(out_md),
        class_weight="model",
        class_weight_pos=None,
        model="gin",
        probe_C=1.0,
        probe_max_iter=t.max_iter,
        max_iter=t.max_iter,
        probe_n_jobs=n_jobs,
        seed=t.seed,
        categorical_encoding="ordinal",
        min_pairing_coverage=0.999,
        protocol_source=None,
        arms=t.arms,
        shuffle_temporal_features_within_split=False,
        shuffle_seed=1,
        diagnostic_tag=f"{t.tag}_recall_metrics",
        representation_source=t.representation_source,
        testing=True,
    )
    if getattr(args, "max_iter", None) is not None:
        args.probe_max_iter = int(args.max_iter)

    logging.info("Running probe backfill tag=%s emb=%s", t.tag, t.embedding_dir)
    set_seed(int(args.seed))
    payload = ptf.run_ablation(args)
    payload["recall_metrics_backfill"] = {
        "source": "cpu_probe_rerun_from_existing_embeddings",
        "original_json": str(t.original_json) if t.original_json else None,
        "no_gnn_retrain": True,
        "no_embedding_regeneration": True,
        "gpu_used": False,
    }
    # Ensure precision-constrained keys present on at least one arm
    sample_arm = next(iter(payload.get("arms") or {}), None)
    test = (payload["arms"][sample_arm]["test"] if sample_arm else {})
    status["has_recall_at_precision_ge_0.90"] = "recall_at_precision_ge_0.90" in test
    status["has_precision_at_100"] = "precision_at_100" in test

    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ptf.write_markdown(out_md, payload)
    status["status"] = "wrote"
    return status


def main() -> int:
    _ensure_cpu()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", default="all", help="Comma-separated tags or 'all'")
    ap.add_argument("--n_jobs", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--inventory_json",
        default="results/diagnostics/enriched/recall_metrics_backfill_inventory.json",
    )
    args = ap.parse_args()
    logger_setup()

    targets = default_targets()
    if args.tags != "all":
        want = {t.strip() for t in args.tags.split(",") if t.strip()}
        targets = [t for t in targets if t.tag in want]

    inventory: Dict[str, Any] = {
        "gpu_used": False,
        "gnn_trained": False,
        "embeddings_regenerated": False,
        "results": [],
    }
    for t in targets:
        inventory["results"].append(run_probe_target(t, n_jobs=int(args.n_jobs), force=bool(args.force)))

    inv_path = _ROOT / args.inventory_json
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    inv_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    logging.info("Wrote inventory %s", inv_path)

    n_wrote = sum(1 for r in inventory["results"] if r["status"] == "wrote")
    n_skip = sum(1 for r in inventory["results"] if str(r["status"]).startswith("skipped"))
    logging.info("Backfill done: wrote=%d skipped=%d total=%d", n_wrote, n_skip, len(inventory["results"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
