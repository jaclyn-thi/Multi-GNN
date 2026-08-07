#!/usr/bin/env python3
"""Option A: migrate Wave-1 frozen-eval embeddings SCRATCH → POOL with verified symlink cutover.

Gates (per tree):
  1) squeue empty (checked at start of each tree)
  2) rsync copy while source remains intact
  3) path set + file counts + total bytes + full SHA256 manifest match
  4) rsync --checksum --dry-run shows zero file diffs
  5) atomic rename source → backup; symlink at original path → POOL dest
  6) validate symlink + representative file SHA256s
  7) remove scratch backup only after validation
  8) restore from backup if any gate fails before backup deletion
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path("/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN")
EMB_ROOT = Path("/orcd/scratch/orcd/008/jthi/Multi-GNN/embeddings")
ARCHIVE = Path("/orcd/pool/007/jthi/Multi-GNN/embeddings_archive")
OUT_DIR = REPO / "results/diagnostics/storage_cleanup_20260803_option_a"
LOG = OUT_DIR / "migrate.log"

WAVE1 = [
    "financial_multidataset_shared_core_phase4b_objective_ablation_frozen_eval",
    "financial_multidataset_shared_core_phase4b_mixed_long_frozen_eval",
    "financial_multidataset_shared_core_phase4b_frozen_eval",
    "expert_only_frozen_transfer_samld_paysim",
    "smallhi_samld_mixed_ssl_phase3_frozen_eval",
]

GiB = 1024**3


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=check)


def squeue_empty() -> bool:
    try:
        r = subprocess.run(
            ["timeout", "30", "squeue", "-u", "jthi", "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
        return r.returncode == 0 and r.stdout.strip() == ""
    except Exception as e:
        log(f"WARN squeue check failed: {e}")
        return False


def du_bytes(path: Path) -> int:
    r = run(["du", "-sbP", str(path)])
    return int(r.stdout.strip().split()[0])


def du_sh(path: Path) -> str:
    r = run(["du", "-P", "-sh", str(path)])
    return r.stdout.strip().split()[0]


def list_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            files.append(Path(dirpath) / name)
    files.sort()
    return files


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_manifest(root: Path) -> Tuple[Dict[str, Any], str]:
    """Return {rel: {size, sha256}} and overall manifest digest."""
    files = list_files(root)
    entries: Dict[str, Any] = {}
    lines: List[str] = []
    for fp in files:
        rel = str(fp.relative_to(root))
        size = fp.stat().st_size
        digest = sha256_file(fp)
        entries[rel] = {"size": size, "sha256": digest}
        lines.append(f"{digest}  {size}  {rel}")
    manifest_blob = "\n".join(lines) + ("\n" if lines else "")
    manifest_sha = hashlib.sha256(manifest_blob.encode("utf-8")).hexdigest()
    return entries, manifest_sha


def rsync_copy(src: Path, dst: Path) -> Tuple[int, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # trailing slash: copy contents into dst
    cmd = ["rsync", "-a", "--info=stats2", f"{src}/", f"{dst}/"]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return r.returncode, (r.stderr or "")[-2000:]


def rsync_checksum_dry_run(src: Path, dst: Path) -> Tuple[int, List[str]]:
    cmd = [
        "rsync",
        "-a",
        "--checksum",
        "--dry-run",
        "--out-format=%n",
        f"{src}/",
        f"{dst}/",
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    diffs = [ln for ln in (r.stdout or "").splitlines() if ln.strip() and not ln.startswith("./")]
    # also ignore directory-only lines that end with /
    diffs = [d for d in diffs if not d.endswith("/")]
    return r.returncode, diffs


def sample_hash_checks(root: Path, entries: Dict[str, Any], n: int = 5) -> List[Dict[str, str]]:
    # prefer meta.json + largest files
    rels = list(entries.keys())
    preferred = [r for r in rels if r.endswith("meta.json")]
    by_size = sorted(rels, key=lambda r: entries[r]["size"], reverse=True)
    chosen: List[str] = []
    for r in preferred + by_size:
        if r not in chosen:
            chosen.append(r)
        if len(chosen) >= n:
            break
    out = []
    for rel in chosen:
        got = sha256_file(root / rel)
        out.append({"rel": rel, "sha256": got, "expected": entries[rel]["sha256"], "ok": got == entries[rel]["sha256"]})
    return out


def migrate_one(name: str) -> Dict[str, Any]:
    src = EMB_ROOT / name
    dst = ARCHIVE / name
    backup = EMB_ROOT / f"{name}.__migrating_to_pool__"
    rec: Dict[str, Any] = {
        "name": name,
        "source": str(src),
        "destination": str(dst),
        "tmp": str(backup),
        "status": "started",
        "errors": [],
        "restored": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    log(f"=== BEGIN {name} ===")

    try:
        if not squeue_empty():
            raise RuntimeError("squeue not empty; aborting tree")
        if src.is_symlink():
            raise RuntimeError(f"source already symlink: {src}")
        if not src.is_dir():
            raise RuntimeError(f"source missing or not dir: {src}")
        if dst.exists():
            raise RuntimeError(f"destination already exists: {dst}")
        if backup.exists():
            raise RuntimeError(f"backup path already exists: {backup}")

        rec["source_size_bytes"] = du_bytes(src)
        src_files = list_files(src)
        rec["source_file_count"] = len(src_files)
        log(f"source size={rec['source_size_bytes']/GiB:.4f} GiB files={rec['source_file_count']}")

        log("building source SHA256 manifest (full)")
        src_entries, src_manifest = build_manifest(src)
        rec["source_rel_count"] = len(src_entries)
        rec["source_manifest_sha256"] = src_manifest
        rec["source_hash_sample"] = [
            {"rel": k, "size": v["size"], "sha256": v["sha256"]}
            for k, v in sorted(src_entries.items(), key=lambda kv: -kv[1]["size"])[:4]
        ]
        # persist per-tree manifests
        man_dir = OUT_DIR / "manifests" / name
        man_dir.mkdir(parents=True, exist_ok=True)
        (man_dir / "source_manifest.json").write_text(json.dumps({"manifest_sha256": src_manifest, "files": src_entries}, indent=2))

        log("rsync copy → POOL (source remains intact)")
        rc, err_tail = rsync_copy(src, dst)
        rec["rsync_copy_rc"] = rc
        rec["rsync_copy_stderr_tail"] = err_tail
        if rc != 0:
            raise RuntimeError(f"rsync copy failed rc={rc}: {err_tail}")

        rec["dest_size_bytes"] = du_bytes(dst)
        dest_files = list_files(dst)
        rec["dest_file_count"] = len(dest_files)
        if rec["dest_file_count"] != rec["source_file_count"]:
            raise RuntimeError(f"file count mismatch src={rec['source_file_count']} dst={rec['dest_file_count']}")
        if rec["dest_size_bytes"] != rec["source_size_bytes"]:
            # du -sb can differ slightly on some FS; fall back to sum of file sizes
            src_sum = sum(p.stat().st_size for p in src_files)
            dst_sum = sum(p.stat().st_size for p in dest_files)
            rec["source_filesize_sum"] = src_sum
            rec["dest_filesize_sum"] = dst_sum
            if src_sum != dst_sum:
                raise RuntimeError(f"byte size mismatch du src={rec['source_size_bytes']} dst={rec['dest_size_bytes']} filesum src={src_sum} dst={dst_sum}")
            log("WARN du -sb differed but file size sums match; continuing")

        log("building dest SHA256 manifest (full)")
        dst_entries, dst_manifest = build_manifest(dst)
        rec["dest_rel_count"] = len(dst_entries)
        rec["dest_manifest_sha256"] = dst_manifest
        (man_dir / "dest_manifest.json").write_text(json.dumps({"manifest_sha256": dst_manifest, "files": dst_entries}, indent=2))

        if set(src_entries) != set(dst_entries):
            raise RuntimeError("path set mismatch between source and dest")
        for rel, meta in src_entries.items():
            if dst_entries[rel]["size"] != meta["size"] or dst_entries[rel]["sha256"] != meta["sha256"]:
                raise RuntimeError(f"manifest mismatch on {rel}")
        if src_manifest != dst_manifest:
            raise RuntimeError("overall manifest sha mismatch")

        log("rsync --checksum --dry-run")
        dry_rc, dry_diffs = rsync_checksum_dry_run(src, dst)
        rec["rsync_dryrun_rc"] = dry_rc
        rec["rsync_dryrun_file_diffs"] = dry_diffs
        if dry_rc != 0 or dry_diffs:
            raise RuntimeError(f"checksum dry-run failed rc={dry_rc} diffs={dry_diffs[:20]}")

        log("atomic cutover: rename source → backup; create symlink")
        os.rename(str(src), str(backup))
        os.symlink(str(dst), str(src))
        if not src.is_symlink():
            raise RuntimeError("symlink creation failed")
        real = os.path.realpath(src)
        rec["symlink"] = str(src)
        rec["symlink_target"] = str(dst)
        rec["realpath_through_symlink"] = real
        if Path(real) != dst.resolve():
            raise RuntimeError(f"realpath mismatch: {real} != {dst}")

        log("validate representative files through symlink")
        checks = sample_hash_checks(src, src_entries, n=min(5, len(src_entries)))
        rec["representative_checks"] = checks
        if not all(c["ok"] for c in checks):
            raise RuntimeError(f"representative check failed: {checks}")

        # also confirm repo logical path resolves
        repo_link = REPO / "embeddings" / name
        rec["repo_logical_realpath"] = os.path.realpath(repo_link)
        if Path(rec["repo_logical_realpath"]) != dst.resolve():
            raise RuntimeError(f"repo embeddings/{name} realpath mismatch: {rec['repo_logical_realpath']}")

        log("remove scratch backup after successful validation")
        shutil.rmtree(backup)
        if backup.exists():
            raise RuntimeError("backup still exists after rmtree")

        rec["status"] = "success"
        rec["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        log(f"=== SUCCESS {name} reclaimed≈{rec['source_size_bytes']/GiB:.4f} GiB ===")
        return rec

    except Exception as e:
        rec["status"] = "failed"
        rec["errors"].append(str(e))
        rec["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        log(f"ERROR {name}: {e}")

        # Restore if we renamed but symlink/validation failed and backup still exists
        try:
            if backup.exists() and (not src.exists() or src.is_symlink()):
                log("attempting restore from backup")
                if src.is_symlink() or src.exists():
                    if src.is_symlink():
                        src.unlink()
                    elif src.is_dir():
                        # unexpected
                        pass
                if not src.exists():
                    os.rename(str(backup), str(src))
                    rec["restored"] = True
                    log("restored source from backup")
            # If dest exists but migration failed before cutover, leave dest for inspection but do not delete source
            if rec.get("restored") is False and dst.exists() and src.is_dir() and not src.is_symlink():
                log("leaving POOL dest in place for inspection; source intact")
        except Exception as e2:
            rec["errors"].append(f"restore_failed: {e2}")
            log(f"RESTORE FAILED: {e2}")
        return rec


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    if not squeue_empty():
        log("ABORT: squeue not empty at start")
        return 2

    before = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "squeue_empty": True,
        "scratch_project_du": du_sh(Path("/orcd/scratch/orcd/008/jthi/Multi-GNN")),
        "scratch_project_bytes": du_bytes(Path("/orcd/scratch/orcd/008/jthi/Multi-GNN")),
        "pool_project_du": du_sh(Path("/orcd/pool/007/jthi/Multi-GNN")),
        "pool_project_bytes": du_bytes(Path("/orcd/pool/007/jthi/Multi-GNN")),
        "quota_file": Path("/home/jthi/orcd/.quota").read_text() if Path("/home/jthi/orcd/.quota").exists() else None,
    }
    (OUT_DIR / "preoperation.json").write_text(json.dumps(before, indent=2))
    log(f"preoperation scratch={before['scratch_project_du']} pool={before['pool_project_du']}")

    results: List[Dict[str, Any]] = []
    for name in WAVE1:
        # skip if already successfully migrated (idempotent resume)
        src = EMB_ROOT / name
        dst = ARCHIVE / name
        if src.is_symlink() and dst.is_dir() and os.path.realpath(src) == str(dst.resolve()):
            log(f"SKIP already migrated: {name}")
            results.append({"name": name, "status": "already_migrated", "destination": str(dst)})
            continue
        rec = migrate_one(name)
        results.append(rec)
        (OUT_DIR / "progress.json").write_text(json.dumps({"migrations": results}, indent=2, default=str))
        if rec["status"] != "success" and rec.get("status") != "already_migrated":
            log("STOPPING after failure")
            break

    after = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scratch_project_du": du_sh(Path("/orcd/scratch/orcd/008/jthi/Multi-GNN")),
        "scratch_project_bytes": du_bytes(Path("/orcd/scratch/orcd/008/jthi/Multi-GNN")),
        "pool_project_du": du_sh(Path("/orcd/pool/007/jthi/Multi-GNN")),
        "pool_project_bytes": du_bytes(Path("/orcd/pool/007/jthi/Multi-GNN")),
        "quota_file": Path("/home/jthi/orcd/.quota").read_text() if Path("/home/jthi/orcd/.quota").exists() else None,
    }
    (OUT_DIR / "postoperation.json").write_text(json.dumps(after, indent=2))

    ok = sum(1 for r in results if r.get("status") in ("success", "already_migrated"))
    reclaimed = sum(int(r.get("source_size_bytes") or 0) for r in results if r.get("status") == "success")
    summary = {
        "title": "storage_cleanup_20260803 Option A Wave-1 migration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preoperation": before,
        "postoperation": after,
        "migrations": results,
        "success_count": ok,
        "attempted_count": len(results),
        "reclaimed_scratch_bytes": reclaimed,
        "reclaimed_scratch_gib": reclaimed / GiB,
        "archive_root": str(ARCHIVE),
    }
    (OUT_DIR / "execution.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"DONE success={ok}/{len(results)} reclaimed_gib={reclaimed/GiB:.4f}")
    log(f"after scratch={after['scratch_project_du']} pool={after['pool_project_du']}")
    return 0 if ok == len(WAVE1) else 1


if __name__ == "__main__":
    sys.exit(main())
