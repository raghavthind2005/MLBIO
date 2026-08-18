"""Run pool construction on the cluster, against the real grader.

Everything decided in ``docs/DECISIONS.md`` D5/D18/D20/D21/D22 is applied here.
The parsing and sampling logic lives in :mod:`virl_pool` and is unit-tested off
-cluster; this script only supplies real inputs (the parquet shards and the
container's ``mathruler``) and writes provenance-carrying artifacts.

Two modes:

``--report-only``
    Scan every row, report the gradeability and parse breakdown, write no pool.
    Use this first: it produces the retained-fraction number that D22 currently
    only estimates.

(default)
    Do the above and additionally draw the seeded pilot sample.

Images are **not** copied. The parquet embeds image bytes, so each pool item
records a locator ``"<shard>#<row>"`` and the generation stage re-reads the
image by index. Copying 2.7 GB to restate what the dataset already holds would
add a corruption surface for no benefit.

Run via ``runs/build_pool.sbatch`` -- it must execute inside the container,
because the login node has neither ``pyarrow`` nor ``mathruler``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from virl_pool import (  # noqa: E402
    DEFAULT_ALLOWED_FORMATS,
    ParseKind,
    build_pool,
    manifest_hash,
    parse_problem,
)

DEFAULT_SNAPSHOT = (
    "/iopsstor/scratch/cscs/raghavthind/hf_cache/hub/"
    "datasets--PAPOGalaxy--PAPO_ViRL39K_train/snapshots/"
    "ff6996d5cdd0e5fc12c01f3dab96f1af37453ceb/data"
)


def _git_sha() -> str:
    """Code provenance.

    Prefers ``CS1_GIT_SHA``, exported by the sbatch from the synced clone: the
    cluster code directory is a plain ``cp`` target, not a git repo, so asking
    git there yields nothing and provenance would silently record "unknown".
    """
    env = os.environ.get("CS1_GIT_SHA", "").strip()
    if env:
        return env
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def load_rows(snapshot_dir: Path) -> list[dict]:
    """Stream the shards, keeping text only.

    ``images`` is reduced to a list of locator strings so the image-count checks
    in :func:`virl_pool.build_pool` still work without holding image bytes in
    memory.
    """
    import pyarrow.parquet as pq

    shards = sorted(snapshot_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {snapshot_dir}")

    rows: list[dict] = []
    for shard in shards:
        pf = pq.ParquetFile(shard)
        row_in_shard = 0
        for batch in pf.iter_batches(batch_size=256, columns=["problem", "answer", "images"]):
            problems = batch.column("problem").to_pylist()
            answers = batch.column("answer").to_pylist()
            images = batch.column("images").to_pylist()
            for p, a, im in zip(problems, answers, images):
                n_img = len(im) if im is not None else 0
                rows.append(
                    {
                        "index": len(rows),
                        "problem": p or "",
                        "answer": a or "",
                        "images": [f"{shard.name}#{row_in_shard}" for _ in range(n_img)],
                    }
                )
                row_in_shard += 1
    return rows


def parse_breakdown(rows: list[dict]) -> tuple[Counter, Counter]:
    kinds: Counter = Counter()
    reasons: Counter = Counter()
    for row in rows:
        parsed = parse_problem(row["problem"])
        kinds[parsed.kind.value] += 1
        if parsed.kind is ParseKind.UNPARSEABLE:
            reasons[parsed.reason] += 1
    return kinds, reasons


def gradeability_breakdown(rows: list[dict], grade_fn) -> tuple[int, Counter, list[str]]:
    """Apply the D22 self-match rule and describe what it removes.

    Answers are bucketed only so a grader weakness confined to one answer shape
    cannot hide inside an aggregate.
    """
    import re

    letter = re.compile(r"^[A-E]$")
    numeric = re.compile(r"^[+-]?(\d+(\.\d+)?|\d+/\d+)$")

    kept = 0
    by_format: Counter = Counter()
    rejected_examples: list[str] = []
    for row in rows:
        ans = (row["answer"] or "").strip()
        shape = "letter" if letter.match(ans) else "numeric" if numeric.match(ans) else "other"
        try:
            ok = bool(ans) and bool(grade_fn(ans, ans))
        except Exception:
            ok = False
        if ok:
            kept += 1
            by_format[f"{shape}:kept"] += 1
        else:
            by_format[f"{shape}:dropped"] += 1
            if len(rejected_examples) < 300:
                rejected_examples.append(ans)
    return kept, by_format, rejected_examples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--n-subset", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument(
        "--allowed-formats",
        default=",".join(DEFAULT_ALLOWED_FORMATS),
        help="comma-separated answer shapes to keep (D22 default: letter,numeric)",
    )
    args = ap.parse_args()
    allowed = tuple(f.strip() for f in args.allowed_formats.split(",") if f.strip())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from mathruler.grader import grade_answer  # container-only import

    print(f"[1/4] loading shards from {args.snapshot}", flush=True)
    rows = load_rows(Path(args.snapshot))
    print(f"      rows loaded: {len(rows):,}", flush=True)

    print("[2/4] parse breakdown", flush=True)
    kinds, reasons = parse_breakdown(rows)
    for k, c in kinds.most_common():
        print(f"      {k:14s} {c:6d}  {c/len(rows):6.1%}", flush=True)
    print("      unparseable reasons:", flush=True)
    for r, c in reasons.most_common(10):
        print(f"        {c:5d}x  {r[:70]}", flush=True)

    print("[3/4] gradeability (D22: grade_answer(a, a) is True)", flush=True)
    kept, by_format, rejected = gradeability_breakdown(rows, grade_answer)
    print(f"      gradeable: {kept:,}/{len(rows):,} = {kept/len(rows):.1%}", flush=True)
    for k, c in sorted(by_format.items()):
        print(f"        {k:16s} {c:6d}", flush=True)

    report = {
        "n_rows": len(rows),
        "parse_kinds": dict(kinds),
        "parse_unparseable_reasons": dict(reasons),
        "gradeable_kept": kept,
        "gradeable_fraction": kept / len(rows),
        "gradeable_by_format": dict(by_format),
        "rejected_answer_examples": rejected,
        "snapshot": args.snapshot,
        "code_git_sha": _git_sha(),
        "allowed_formats": list(allowed),
    }
    (out / "pool_report.json").write_text(json.dumps(report, indent=2))
    print(f"      wrote {out/'pool_report.json'}", flush=True)

    if args.report_only:
        print("[4/4] --report-only: no pool drawn", flush=True)
        return 0

    print(f"[4/4] drawing pool n={args.n_items} subset={args.n_subset} seed={args.seed}", flush=True)
    items, subset, stats, rejects = build_pool(
        rows,
        grade_answer,
        n_items=args.n_items,
        n_subset=args.n_subset,
        seed=args.seed,
        allowed_formats=allowed,
    )

    (out / "pool_manifest.json").write_text(
        json.dumps(
            {
                "items": [asdict(it) for it in items],
                "m3_subset_indices": subset,
                "manifest_sha256": manifest_hash(items),
                "seed": args.seed,
                "snapshot": args.snapshot,
                "code_git_sha": _git_sha(),
                "stats": asdict(stats),
            },
            indent=2,
            default=list,
        )
    )
    with (out / "pool_rejects.jsonl").open("w") as fh:
        for rec in rejects:
            fh.write(json.dumps(rec) + "\n")

    print(f"      eligible={stats.n_eligible:,}  drawn={len(items)}  subset={len(subset)}", flush=True)
    print(f"      dropped: multi_image={stats.n_multi_image} unparseable={stats.n_unparseable} "
          f"ungradeable={stats.n_ungradeable} wrong_format={stats.n_wrong_format} "
          f"stem_leak={stats.n_stem_leak} "
          f"stem_reparse_mcq={stats.n_stem_reparse_mcq}", flush=True)
    print(f"      answer formats seen: {dict(stats.by_answer_format)}", flush=True)
    print(f"      drawn by format: "
          f"{ {f: sum(1 for it in items if it.answer_fmt == f) for f in allowed} }", flush=True)
    print(f"      manifest sha256 = {manifest_hash(items)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
