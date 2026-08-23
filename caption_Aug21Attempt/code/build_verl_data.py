"""Materialise the pool manifest as parquet files verl can load directly.

WHY THIS SHAPE. verl resolves `data.train_files` in three branches
(`verl/utils/dataset.py:129-143`): an `@` splits off the split name, then `isdir` ->
`load_dataset(type, data_dir=...)`, `isfile` -> `load_dataset(type, data_files=...)`,
else a HuggingFace repo id. We emit **one parquet file per split** and reference it by
full path with **no `@`**, so `data_split` defaults to `"train"` and the `isfile` branch
runs. The `isdir` branch is deliberately avoided: it infers the file type from
`os.listdir(path)[0]`, an arbitrary first entry, which is a stray-file hazard.

WHY WE SLICE RATHER THAN REBUILD. The rows are taken out of the source shards with
pyarrow and written back untouched, so the schema -- including the exact `images` feature
type -- is **inherited, never reconstructed**. This code therefore never needs to know
what shape `images` has. That is the point: job 3167490 died because a loader assumed
`images` was a list of structs when Vision-SR1-47K declares a singular `Image`. Code that
does not encode the assumption cannot get it wrong.

The manifest carries `shard` + `row_in_shard` locators. This module treats that alignment
as a claim to be **checked, not trusted**: gate 4 below re-reads the problem text and
answer from the written parquet and compares them to the manifest string-for-string, so a
locator that is off by even one row fails loudly instead of silently training on
mismatched (image, question) pairs -- which would look like a perception failure and would
be indistinguishable from one in every metric we log.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pool_io import extract_image_bytes, load_manifest  # noqa: E402


def manifest_selection_hash(splits: dict[str, list[dict]]) -> str:
    """Recompute build_pool's manifest_sha256 from arbitrary split dicts.

    Kept byte-identical to build_pool.py:327-330 so the two can be compared. If that
    changes, this must change with it -- a test pins them together.
    """
    payload = json.dumps(
        [[k, i["problem_id"], i["path"]] for k in sorted(splits)
         for i in splits[k]], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_split(snapshot: Path, items: list[dict], out_path: Path):
    """Slice `items` out of the source shards into one parquet file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    by_shard: dict[str, list[int]] = collections.defaultdict(list)
    for it in items:
        by_shard[it["shard"]].append(it["row_in_shard"])

    tables = []
    for shard in sorted(by_shard):
        rows = sorted(by_shard[shard])
        table = pq.read_table(snapshot / shard)
        if rows and rows[-1] >= table.num_rows:
            raise AssertionError(
                f"{shard}: manifest asks for row {rows[-1]} but the shard has "
                f"{table.num_rows}. The manifest and the snapshot disagree.")
        tables.append(table.take(rows))
        print(f"    {shard}: took {len(rows):,} of {table.num_rows:,}", flush=True)

    combined = pa.concat_tables(tables)
    # Deterministic order so a rebuild is byte-comparable. Training shuffles anyway
    # (config `data.shuffle: true`), so this costs nothing.
    combined = combined.sort_by("problem_id")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, out_path)
    return combined


def verify_split(table, items: list[dict], name: str) -> set[str]:
    """Gates. Every one of these has a specific silent failure behind it.

    Deliberately depends on nothing but ``to_pydict()``, so the gates are unit-testable
    off-cluster where pyarrow is not installed. A gate that can only run in the one
    place it is meant to protect is a gate that never gets exercised.
    """
    d = table.to_pydict()
    n = table.num_rows

    # 1. count
    if n != len(items):
        raise AssertionError(f"[{name}] wrote {n} rows for {len(items)} manifest items")

    # 2/3. identity: same problem_ids and paths, compared in a fixed order
    want = sorted(items, key=lambda r: r["problem_id"])
    got_ids = list(d["problem_id"])
    if got_ids != [r["problem_id"] for r in want]:
        raise AssertionError(f"[{name}] problem_id mismatch against the manifest")
    if list(d["path"]) != [r["path"] for r in want]:
        raise AssertionError(f"[{name}] path mismatch against the manifest")

    # 4. THE ALIGNMENT GATE. If `row_in_shard` were off by one, ids and paths could
    #    still line up by luck of sorting while the TEXT belonged to another row. A
    #    mismatched (image, question) pair is indistinguishable from a perception
    #    failure in every metric we log, so it is checked here rather than inferred.
    for field in ("problem", "answer"):
        got = [str(x) for x in d[field]]
        exp = [str(r[field]) for r in want]
        if got != exp:
            bad = next(i for i, (a, b) in enumerate(zip(got, exp)) if a != b)
            raise AssertionError(
                f"[{name}] {field} mismatch at row {bad} "
                f"(problem_id {got_ids[bad]}).\n  parquet:  {got[bad][:160]!r}\n"
                f"  manifest: {exp[bad][:160]!r}\n"
                f"The shard/row_in_shard locators do not point where the manifest says.")

    # 5. R8, re-checked on the written artifact rather than on the source
    for i, cell in enumerate(d["images"]):
        raw = extract_image_bytes(cell)          # raises on multi-image or bad struct
        if not raw:
            raise AssertionError(f"[{name}] row {i} carries empty image bytes")

    print(f"  [gate] {name}: {n:,} rows, ids/paths/problem/answer all match, "
          f"1 image each", flush=True)
    return set(d["path"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--provenance", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="trial,eval,dev")
    args = ap.parse_args()

    prov = json.loads(Path(args.provenance).read_text())
    snapshot = Path(prov["snapshot_path"])
    manifest = load_manifest(Path(args.pool))
    out = Path(args.out)

    declared = manifest.get("manifest_sha256")
    recomputed = manifest_selection_hash(manifest["splits"])
    if declared != recomputed:
        raise AssertionError(
            f"manifest_sha256 does not match its own contents.\n"
            f"  declared:   {declared}\n  recomputed: {recomputed}\n"
            f"The manifest was edited after it was written.")
    print(f"[prov] manifest {declared} verified against its own rows", flush=True)
    print(f"[prov] snapshot {snapshot}", flush=True)

    names = [s.strip() for s in args.splits.split(",") if s.strip()]
    paths_seen: dict[str, set[str]] = {}
    written = {}

    for name in names:
        items = manifest["splits"][name]
        dst = out / f"ca21_{name}.parquet"
        print(f"\n=== {name}: {len(items):,} items -> {dst.name} ===", flush=True)
        table = build_split(snapshot, items, dst)
        paths_seen[name] = verify_split(table, items, name)
        written[name] = {
            "path": str(dst),
            "rows": table.num_rows,
            "bytes": dst.stat().st_size,
        }

    # 6. S5.3, re-checked on the artifacts: one image lands in exactly one split.
    #    Verified at build time too, but this is the file the trainer actually reads.
    for a in names:
        for b in names:
            if a < b:
                overlap = paths_seen[a] & paths_seen[b]
                if overlap:
                    raise AssertionError(
                        f"S5.3 violated: {len(overlap)} images in BOTH {a} and {b}, "
                        f"e.g. {sorted(overlap)[:3]}")
    print(f"\n[gate] cross-split image disjointness holds over {names}", flush=True)

    meta = {
        "pool_manifest_sha256": declared,
        "dataset_revision": prov.get("revision"),
        "snapshot_path": str(snapshot),
        "splits": written,
        "verl_usage": {
            "note": "reference by FULL PATH with no '@' so data_split defaults to "
                    "'train' and verl takes the isfile branch "
                    "(verl/utils/dataset.py:129-143)",
            "train_files": written.get("trial", {}).get("path"),
            "val_files": written.get("eval", {}).get("path"),
            "prompt_key": "problem",
            "answer_key": "answer",
            "image_key": "images",
        },
    }
    (out / "verl_data_provenance.json").write_text(json.dumps(meta, indent=2))

    print("\n=== written ===")
    for k, v in written.items():
        print(f"  {k:<6} {v['rows']:>6,} rows  {v['bytes']/1e6:>8.1f} MB  {v['path']}")
    print(f"\n[done] -> {out}/verl_data_provenance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
