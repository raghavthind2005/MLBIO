"""Download Vision-SR1-47K at a pinned revision, then PROVE what landed on disk.

The download is the easy half. The half that matters is the verification: every
characterisation of this dataset so far was made through the HuggingFace
datasets-server API, not from bytes on our filesystem. Building a training pool
on an API summary would violate the rule that prior artifacts are verified on
disk exactly as described, so this script re-derives the composition from the
parquet itself and fails loudly if it disagrees.

Checks performed, in order of how badly a failure would hurt:

  1. Row count is exactly 47,628.
  2. `problem_type` marginals match the API exactly. These are the numbers the
     subset design depends on -- `regression` is being dropped, so if that
     count is wrong the drop is wrong.
  3. The schema carries every column the pool builder will need.
  4. `path` partitions cleanly into top-level categories. This is our
     stratification key and it is only usable if it is total and unambiguous.
  5. Image reuse across rows is measured, not assumed. If one image backs
     several rows, every one of them must land on the same side of the
     train/eval split or the split leaks.

Nothing here writes a pool. This script only establishes that the substrate is
what we believe it is.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

# Measured via the HF datasets-server on 2026-08-23. Treated as a HYPOTHESIS
# about the artifact, which the parquet on disk must confirm.
EXPECTED_ROWS = 47_628
EXPECTED_PROBLEM_TYPE = {
    "multiple choice": 29_702,
    "numerical": 12_586,
    "regression": 5_340,
}
REQUIRED_COLUMNS = {
    "problem_id", "problem", "data_type", "problem_type",
    "options", "solution", "path", "data_source", "answer", "images",
}


def download(repo: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    print(f"[fetch] {repo} @ {revision}", flush=True)
    local = snapshot_download(repo_id=repo, revision=revision, repo_type="dataset")
    print(f"[fetch] snapshot at {local}", flush=True)
    return Path(local)


def verify(snapshot: Path) -> dict:
    import pyarrow.parquet as pq

    shards = sorted(snapshot.rglob("*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet under {snapshot}")
    print(f"[verify] {len(shards)} parquet shards", flush=True)

    # `images` is the heavy column and we never need its bytes here; reading
    # only the light columns keeps this to seconds instead of decoding 4.35 GB.
    light = [c for c in REQUIRED_COLUMNS if c != "images"]

    total = 0
    by_ptype: collections.Counter = collections.Counter()
    by_source: collections.Counter = collections.Counter()
    by_category: collections.Counter = collections.Counter()
    path_counts: collections.Counter = collections.Counter()
    schema_cols: set[str] = set()
    malformed_paths: list[str] = []

    for shard in shards:
        pf = pq.ParquetFile(shard)
        schema_cols |= set(pf.schema_arrow.names)
        for batch in pf.iter_batches(batch_size=2048, columns=light):
            d = batch.to_pydict()
            n = len(d["problem_id"])
            total += n
            by_ptype.update(d["problem_type"])
            by_source.update(d["data_source"])
            for p in d["path"]:
                path_counts[p] += 1
                # "./Chart/MapQA/images/map_437.png" -> "Chart"
                parts = str(p).lstrip("./").split("/")
                if len(parts) < 2 or not parts[0]:
                    malformed_paths.append(str(p))
                    by_category["<UNPARSEABLE>"] += 1
                else:
                    by_category[parts[0]] += 1

    print(f"\n[verify] rows on disk: {total:,}")

    missing = REQUIRED_COLUMNS - schema_cols
    if missing:
        raise AssertionError(f"schema is missing required columns: {sorted(missing)}")
    print(f"[verify] schema OK, all {len(REQUIRED_COLUMNS)} required columns present")

    if total != EXPECTED_ROWS:
        raise AssertionError(f"row count {total:,} != expected {EXPECTED_ROWS:,}")
    print(f"[verify] row count matches the pinned revision")

    print("\n[verify] problem_type (disk vs API):")
    for k, exp in sorted(EXPECTED_PROBLEM_TYPE.items()):
        got = by_ptype.get(k, 0)
        flag = "OK " if got == exp else "MISMATCH"
        print(f"   {flag} {k:<18} disk={got:>7,}  api={exp:>7,}")
    if dict(by_ptype) != EXPECTED_PROBLEM_TYPE:
        raise AssertionError(f"problem_type marginals disagree: disk={dict(by_ptype)}")

    if malformed_paths:
        raise AssertionError(
            f"{len(malformed_paths)} paths do not yield a category, e.g. "
            f"{malformed_paths[:5]} -- the stratification key is not total"
        )
    print("\n[verify] path -> category is total (every row classified):")
    for k, v in by_category.most_common():
        print(f"   {k:<14} {v:>7,}  {v/total*100:5.1f}%")

    # Image reuse. If a path backs several rows, those rows are NOT independent
    # and must not straddle the train/eval boundary.
    reused = {p: c for p, c in path_counts.items() if c > 1}
    rows_in_reused = sum(reused.values())
    print(f"\n[verify] distinct images: {len(path_counts):,}")
    print(f"[verify] images backing >1 row: {len(reused):,} "
          f"({rows_in_reused:,} rows = {rows_in_reused/total*100:.1f}%)")
    if reused:
        worst = sorted(reused.items(), key=lambda kv: -kv[1])[:3]
        print(f"[verify] most reused: {[(p[:52], c) for p, c in worst]}")
        print("[verify] => the train/eval split MUST group by image path")
    else:
        print("[verify] => one image per row; splitting by row is safe")

    print("\n[verify] data_source:")
    for k, v in by_source.most_common():
        print(f"   {k:<34} {v:>6,}  {v/total*100:5.1f}%")

    return {
        "rows": total,
        "n_shards": len(shards),
        "by_problem_type": dict(by_ptype),
        "by_category": dict(by_category),
        "by_data_source": dict(by_source),
        "distinct_images": len(path_counts),
        "images_backing_multiple_rows": len(reused),
        "rows_on_reused_images": rows_in_reused,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("CA21_DATASET", ""))
    ap.add_argument("--revision", default=os.environ.get("CA21_DATASET_REV", ""))
    ap.add_argument("--out", required=True, help="where to write the provenance json")
    args = ap.parse_args()

    if not args.repo or not args.revision:
        raise SystemExit("--repo and --revision required (normally from _env.sh)")

    snapshot = download(args.repo, args.revision)
    stats = verify(snapshot)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": args.repo,
        "revision": args.revision,
        "snapshot_path": str(snapshot),
        "code_git_sha": os.environ.get("CA21_GIT_SHA", "unknown"),
        "verified": stats,
    }, indent=2))
    print(f"\n[done] provenance -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
