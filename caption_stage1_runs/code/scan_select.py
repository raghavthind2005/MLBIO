"""Aggregate scan shards, grade them, and emit the surviving candidate list.

Grading is delegated to :mod:`pilot_score` -- the Vision-SR1 rule and its
fallback are defined once and reused, so the pool is selected by exactly the
criterion the pilot was measured with. A second implementation here would be
free to drift, and the pool would then be filtered on a rule nobody measured.

Stage 1 (``--mode image --keep-min 0.8``)
    Keep rows the image policy answers correctly, D11/D32. Emits the candidate
    list stage 2 restricts to.

Stage 2 (``--mode text_only --keep-max 0.4``)
    Keep rows the *bare question* does NOT answer -- the vision-necessity
    filter. Pilot 0 found 34% of ViRL39K solvable from text alone, and on those
    the caption gap is structurally zero.

Both thresholds are expressed as pass-rate fractions of ``n_answers`` so they
stay meaningful if the draw count ever changes: 4/5 -> 0.8, 2/5 -> 0.4.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot_score import check_g_grade, grade_strict, wilson_ci  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--scan-dir", required=True)
    ap.add_argument("--mode", required=True, choices=("image", "text_only"))
    ap.add_argument("--out", required=True, help="json list of surviving indices")
    ap.add_argument("--keep-min", type=float, default=None,
                    help="keep rows with pass-rate >= this (stage 1: 0.8)")
    ap.add_argument("--keep-max", type=float, default=None,
                    help="keep rows with pass-rate <= this (stage 2: 0.4)")
    ap.add_argument("--report", default="", help="optional json report path")
    args = ap.parse_args()

    if (args.keep_min is None) == (args.keep_max is None):
        raise SystemExit("give exactly one of --keep-min / --keep-max")

    from mathruler.grader import extract_boxed_content, grade_answer

    manifest = json.loads((Path(args.pool) / "pool_manifest.json").read_text())
    gold = {it["index"]: it["answer"] for it in manifest["items"]}
    fmt = {it["index"]: it.get("answer_fmt", "other") for it in manifest["items"]}
    check_g_grade(gold, grade_answer)

    shards = sorted(Path(args.scan_dir).glob(f"scan_{args.mode}_shard*.jsonl"))
    if not shards:
        raise SystemExit(f"no {args.mode} shards in {args.scan_dir}")

    # Shard completeness is checked, not assumed: a missing shard would silently
    # shrink the pool and look like a low pass rate rather than a lost file.
    declared = {int(p.stem.split("of")[-1]) for p in shards}
    if len(declared) != 1:
        raise SystemExit(f"shards disagree on n_shards: {declared}")
    n_shards = declared.pop()
    seen = {int(p.stem.split("shard")[1].split("of")[0]) for p in shards}
    if seen != set(range(n_shards)):
        raise SystemExit(f"incomplete scan: have {sorted(seen)}, expected 0..{n_shards-1}")

    # Provenance: every shard must have been generated from THIS pool, by one
    # code version. Otherwise a stale shard left in the directory by an earlier
    # run would be silently folded into the pool -- and because it would still
    # carry valid indices and gradeable answers, nothing downstream would
    # notice. Cheap check; the failure it prevents is invisible.
    metas = []
    for p in shards:
        mp = p.parent / f"_meta_{p.stem[len('scan_'):]}.json"
        if not mp.exists():
            raise SystemExit(f"shard {p.name} has no _meta sidecar: provenance unverifiable")
        metas.append((p.name, json.loads(mp.read_text())))
    pool_shas = {m.get("pool_manifest_sha256") for _, m in metas}
    if len(pool_shas) != 1:
        raise SystemExit(f"shards came from different pools: {pool_shas}")
    shard_sha = pool_shas.pop()
    if shard_sha != manifest.get("manifest_sha256"):
        raise SystemExit(
            f"shards were generated from a DIFFERENT pool than --pool\n"
            f"  shards: {shard_sha}\n  --pool: {manifest.get('manifest_sha256')}")
    code_shas = {m.get("code_git_sha") for _, m in metas}
    if len(code_shas) != 1:
        print(f"  [warn] shards span multiple code versions: {code_shas}", flush=True)
    modes = {m.get("mode") for _, m in metas}
    if modes != {args.mode}:
        raise SystemExit(f"shard metas disagree with --mode {args.mode}: {modes}")

    per_item: dict[int, list[bool]] = defaultdict(list)
    truncated = 0
    total = 0
    for p in shards:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            i = r["index"]
            if i not in gold:
                raise KeyError(f"scan row {i} absent from pool manifest")
            per_item[i].append(
                grade_strict(r["answer"], gold[i], extract_boxed_content, grade_answer))
            truncated += r.get("finish_reason") == "length"
            total += 1

    rates = {i: sum(v) / len(v) for i, v in per_item.items()}
    if args.keep_min is not None:
        keep = sorted(i for i, r in rates.items() if r >= args.keep_min)
        rule = f"pass_rate >= {args.keep_min}"
    else:
        keep = sorted(i for i, r in rates.items() if r <= args.keep_max)
        rule = f"pass_rate <= {args.keep_max}"

    Path(args.out).write_text(json.dumps(keep))

    n = len(rates)
    lo, hi = wilson_ci(len(keep), n)
    hist = defaultdict(int)
    for v in per_item.values():
        hist[sum(v)] += 1
    by_fmt = defaultdict(lambda: [0, 0])
    for i in rates:
        by_fmt[fmt.get(i, "other")][1] += 1
    for i in keep:
        by_fmt[fmt.get(i, "other")][0] += 1

    print(f"=== scan_select mode={args.mode} rule={rule} ===")
    print(f"  shards           : {len(shards)} (complete)")
    print(f"  rows scanned     : {n:,}   answers {total:,}   truncated {truncated/max(total,1):.1%}")
    print(f"  SURVIVING        : {len(keep):,} / {n:,} = {len(keep)/max(n,1):.1%} "
          f"CI[{lo:.3f}, {hi:.3f}]")
    print(f"  pass-rate hist   : {dict(sorted(hist.items()))}")
    for f, (k, t) in sorted(by_fmt.items()):
        print(f"    {f:8s} {k:6,}/{t:6,} = {k/max(t,1):5.1%}")
    print(f"  wrote            : {args.out}")

    if args.report:
        Path(args.report).write_text(json.dumps({
            "mode": args.mode, "rule": rule, "n_scanned": n, "n_keep": len(keep),
            "keep_fraction": len(keep) / max(n, 1),
            "keep_ci95": [round(lo, 4), round(hi, 4)],
            "pass_rate_histogram": {str(k): v for k, v in sorted(hist.items())},
            "by_format": {f: {"keep": k, "total": t} for f, (k, t) in by_fmt.items()},
            "truncation_rate": round(truncated / max(total, 1), 4),
            "n_shards": n_shards,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
