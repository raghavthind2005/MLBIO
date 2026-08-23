"""Build the dev / trial / eval pools from the verified Vision-SR1-47K snapshot.

PIPELINE, in order. Order matters and is argued, not incidental.

    1. drop  problem_type == "regression"        (5,340 rows)
    2. drop  answers the grader cannot self-match
    3. collapse to ONE row per image, seeded
    4. stratified proportional draw -> dev / trial / eval

Filters run BEFORE the collapse so that an image is only lost if *every* row it
backs is ineligible. Collapsing first would pick a row at random and then
discard the whole image if that particular row happened to be ungradeable,
throwing away images for no reason.

WHY ONE ROW PER IMAGE
---------------------
The snapshot has 37,138 distinct images behind 47,628 rows: 8,886 images back
more than one row (TabMWP up to x5), so 40.7% of rows share an image with some
other row. Two consequences, and the second is why we go further than a
train/eval split rule:

  * across splits -- an image on both sides is straightforward eval leakage.
  * within a split -- rows sharing an image are NOT independent items. Group
    statistics, per-item pass rates, and any paired test would all be quietly
    clustered.

Taking exactly one row per image globally removes both at once, and costs
nothing here: we need ~6,300 images out of 37,138 available.

WHY grade_answer(a, a)
----------------------
Vision-SR1's own reward is, verbatim from `vision_sr1/reward_function/self_reward.py`::

    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

If `mathruler` cannot match an answer to *itself*, it can never credit a correct
response to that row, so accuracy on it is meaningless and `J_success` would
score it 0 regardless of what the model does. This is a property of the grader
we are actually using, not a regex proxy for one.

STRATIFICATION TARGET -- an interpretive choice, flagged
--------------------------------------------------------
"Preserve the distribution" admits two readings, and they are not close here.

  raw       the shipped 47,628-row composition -- what Vision-SR1 trained on.
  eligible  the composition of what survives our filters and the collapse to
            one row per image. DEFAULT.

They diverge whenever a filter is category-concentrated, and ours is: the
`regression` drop appears to fall almost entirely on Spatial, whose raw share
(~21.8%) is therefore close to double its eligible share. Targeting `raw` would
over-sample Spatial ~2x relative to what actually remains, and can exhaust it
outright. It is also incoherent as a claim -- a distribution over a population
cannot be "preserved" once a fifth of that population has been removed on
purpose. Hence `eligible` by default, `--target raw` available, and BOTH shares
printed next to the drawn share so the divergence is never invisible.

A category that cannot fill its quota fails loudly rather than being silently
under-filled.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Callable

# The five top-level categories of the dataset's own `path` column, verified on
# disk (job 3163760). This is the dataset's taxonomy, not one we invented --
# note CLEVR lives under ./Math/ and IconQA under ./General/, which is why a
# hand-grouping by data_source disagrees with it.
CATEGORIES = ("Knowledge", "Math", "Spatial", "Chart", "General")

DROP_PROBLEM_TYPE = "regression"


def category_of(path: str) -> str:
    """./Chart/MapQA/images/map_437.png -> Chart"""
    parts = str(path).lstrip("./").split("/")
    if len(parts) < 2 or not parts[0]:
        raise ValueError(f"path does not yield a category: {path!r}")
    return parts[0]


def load_rows(snapshot: Path) -> list[dict]:
    """Read the light columns. `images` is never touched -- we only need locators."""
    import pyarrow.parquet as pq

    cols = ["problem_id", "problem", "problem_type", "path", "data_source", "answer"]
    rows: list[dict] = []
    for shard in sorted(snapshot.rglob("*.parquet")):
        pf = pq.ParquetFile(shard)
        row_in_shard = 0
        for batch in pf.iter_batches(batch_size=2048, columns=cols):
            d = batch.to_pydict()
            for i in range(len(d["problem_id"])):
                rows.append({
                    "problem_id": d["problem_id"][i],
                    "problem_type": d["problem_type"][i],
                    "path": d["path"][i],
                    "data_source": d["data_source"][i],
                    "answer": d["answer"][i],
                    # Locator so the image can be fetched later without a scan.
                    "shard": shard.name,
                    "row_in_shard": row_in_shard,
                })
                row_in_shard += 1
    return rows


def build(rows: list[dict], grade_fn: Callable[[str, str], bool], seed: int,
          sizes: dict[str, int], target: str = "eligible") -> tuple[dict, dict]:
    rng = random.Random(seed)
    report: dict = {"seed": seed, "n_loaded": len(rows), "target_mode": target}

    raw_by_cat = collections.Counter(category_of(r["path"]) for r in rows)
    unknown = set(raw_by_cat) - set(CATEGORIES)
    if unknown:
        raise AssertionError(f"unexpected categories on disk: {sorted(unknown)}")
    report["raw_by_category"] = dict(raw_by_cat)

    # --- filter 1: regression -------------------------------------------------
    kept = [r for r in rows if r["problem_type"] != DROP_PROBLEM_TYPE]
    dropped_reg = len(rows) - len(kept)
    report["dropped_regression"] = dropped_reg

    # Crosstab of what the regression drop actually costs per category. This was
    # a PREDICTION ("it will fall mostly on Spatial") and predictions get checked.
    reg_by_cat = collections.Counter(
        category_of(r["path"]) for r in rows if r["problem_type"] == DROP_PROBLEM_TYPE)
    report["dropped_regression_by_category"] = dict(reg_by_cat)

    # --- filter 2: gradeability ----------------------------------------------
    gradeable, ungradeable = [], []
    for r in kept:
        a = r["answer"]
        try:
            ok = bool(grade_fn(a, a))
        except Exception:
            ok = False
        (gradeable if ok else ungradeable).append(r)
    report["dropped_ungradeable"] = len(ungradeable)
    report["dropped_ungradeable_by_category"] = dict(collections.Counter(
        category_of(r["path"]) for r in ungradeable))
    report["ungradeable_examples"] = [str(r["answer"])[:60] for r in ungradeable[:25]]

    # --- collapse: exactly one row per image ---------------------------------
    by_image: dict[str, list[dict]] = collections.defaultdict(list)
    for r in gradeable:
        by_image[r["path"]].append(r)
    # Sort inside each group so the seeded pick is reproducible regardless of
    # the order parquet happened to yield rows in.
    picked = []
    for path in sorted(by_image):
        group = sorted(by_image[path], key=lambda r: r["problem_id"])
        picked.append(group[rng.randrange(len(group))])
    report["eligible_rows_after_filters"] = len(gradeable)
    report["eligible_distinct_images"] = len(by_image)

    elig_by_cat: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for r in picked:
        elig_by_cat[category_of(r["path"])].append(r)
    report["eligible_by_category"] = {c: len(v) for c, v in elig_by_cat.items()}

    # --- which distribution are we preserving? -------------------------------
    # `eligible` (default): sample proportionally from the population we can
    # actually use. `raw`: mirror the shipped 47,628-row composition.
    #
    # These diverge sharply when a filter is category-concentrated -- and ours
    # is: `regression` appears to sit almost entirely in Spatial, so Spatial's
    # raw share (~21.8%) is close to double its eligible share. Targeting `raw`
    # while excluding regression would over-sample Spatial ~2x relative to what
    # remains, and is in any case incoherent: a distribution over a population
    # cannot be "preserved" once a fifth of that population is removed on
    # purpose. Both are reported so the divergence is never invisible.
    n_elig = sum(len(v) for v in elig_by_cat.values())
    if n_elig == 0:
        # Without this the next line dies with ZeroDivisionError, which sends
        # you looking at the arithmetic instead of at the filters that emptied
        # the pool -- the likely real cause being a grader that rejects
        # everything (wrong import, answers arriving as a non-string type).
        raise AssertionError(
            "no eligible rows after filtering: "
            f"loaded {len(rows):,}, dropped {dropped_reg:,} regression and "
            f"{len(ungradeable):,} ungradeable. Check the grader first -- "
            f"sample rejected answers: {report['ungradeable_examples'][:5]}")
    report["eligible_share"] = {c: len(elig_by_cat[c]) / n_elig for c in CATEGORIES}
    report["raw_share"] = {c: raw_by_cat[c] / len(rows) for c in CATEGORIES}
    if target == "eligible":
        target_share = report["eligible_share"]
    elif target == "raw":
        target_share = report["raw_share"]
    else:
        raise ValueError(f"target must be 'eligible' or 'raw', got {target!r}")

    # --- stratified proportional draw ----------------------------------------
    # Apportion PER SPLIT, not once globally then subdivided. Subdividing a
    # global quota rounds twice -- once into categories, once into splits -- and
    # the errors compound, so a requested trial of 5,000 silently returns 4,998
    # or 5,003. Doing largest-remainder independently for each split makes every
    # split's size exact by construction, because its per-category quotas are
    # built to sum to it.
    quota: dict[str, dict[str, int]] = {}
    for name, n in sizes.items():
        exact = {c: target_share[c] * n for c in CATEGORIES}
        q = {c: int(exact[c]) for c in CATEGORIES}
        for c in sorted(CATEGORIES, key=lambda c: -(exact[c] - int(exact[c])))[
                : n - sum(q.values())]:
            q[c] += 1
        assert sum(q.values()) == n, f"{name}: quota {sum(q.values())} != {n}"
        quota[name] = q
    report["quota"] = quota

    need = {c: sum(quota[s][c] for s in sizes) for c in CATEGORIES}
    short = {c: (need[c], len(elig_by_cat[c])) for c in CATEGORIES
             if need[c] > len(elig_by_cat[c])}
    if short:
        raise AssertionError(
            f"cannot fill quota from eligible images (need, have): {short}")

    # Shuffle within category, then hand out contiguous, non-overlapping blocks.
    # Because each image appears exactly once in `picked`, disjointness holds by
    # construction at both row and image level.
    splits: dict[str, list[dict]] = {k: [] for k in sizes}
    for c in CATEGORIES:
        pool = sorted(elig_by_cat[c], key=lambda r: r["path"])
        rng.shuffle(pool)
        cut = 0
        for name in sizes:
            n = quota[name][c]
            splits[name].extend(pool[cut: cut + n])
            cut += n

    for name, n in sizes.items():
        assert len(splits[name]) == n, f"{name}: got {len(splits[name])}, want {n}"

    return splits, report


def assert_invariants(splits: dict[str, list[dict]]) -> None:
    """Verified, not trusted. Each of these failing would be silent otherwise."""
    seen_images: dict[str, str] = {}
    for name, rows in splits.items():
        paths = [r["path"] for r in rows]
        if len(paths) != len(set(paths)):
            dup = [p for p, c in collections.Counter(paths).items() if c > 1][:3]
            raise AssertionError(f"{name}: image repeated within split, e.g. {dup}")
        for p in paths:
            if p in seen_images:
                raise AssertionError(
                    f"image {p!r} appears in both {seen_images[p]!r} and {name!r}")
            seen_images[p] = name
    ids = [r["problem_id"] for rows in splits.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("problem_id repeated across splits")
    print(f"[gate] invariants OK: {len(seen_images):,} images, each in exactly one split, "
          f"each exactly once")


def main() -> int:
    ap = argparse.ArgumentParser()
    # The snapshot is taken FROM the verification step's output rather than
    # retyped, so the pool is provably built on the artifact that passed its
    # gates. A hand-written path could silently point at an unverified copy.
    ap.add_argument("--provenance", required=True,
                    help="dataset_provenance.json written by fetch_dataset.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-trial", type=int, default=5000)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--target", choices=("eligible", "raw"), default="eligible",
                    help="which category distribution the draw preserves")
    args = ap.parse_args()

    from mathruler.grader import grade_answer

    prov = json.loads(Path(args.provenance).read_text())
    snapshot = Path(prov["snapshot_path"])
    verified_rows = prov["verified"]["rows"]
    print(f"[prov] {prov['dataset']} @ {prov['revision']}", flush=True)
    print(f"[prov] verified {verified_rows:,} rows at {snapshot}", flush=True)

    rows = load_rows(snapshot)
    print(f"[load] {len(rows):,} rows", flush=True)
    if len(rows) != verified_rows:
        raise AssertionError(
            f"snapshot changed under us: verification saw {verified_rows:,} rows, "
            f"this load sees {len(rows):,}")

    # Order defines how each category's block is carved; trial first so any
    # rounding remainder lands there rather than in the held-out set.
    sizes = {"trial": args.n_trial, "eval": args.n_eval, "dev": args.n_dev}
    splits, report = build(rows, grade_answer, args.seed, sizes, args.target)
    assert_invariants(splits)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": args.seed,
        "dataset": prov["dataset"],
        "dataset_revision": prov["revision"],
        "snapshot_path": str(snapshot),
        "code_git_sha": os.environ.get("CA21_GIT_SHA", "unknown"),
        "sizes": {k: len(v) for k, v in splits.items()},
        "splits": {k: [
            {"problem_id": r["problem_id"], "path": r["path"],
             "category": category_of(r["path"]), "data_source": r["data_source"],
             "problem_type": r["problem_type"], "answer": r["answer"],
             "shard": r["shard"], "row_in_shard": r["row_in_shard"]}
            for r in sorted(v, key=lambda r: r["path"])] for k, v in splits.items()},
    }
    payload = json.dumps(
        [[k, i["problem_id"], i["path"]] for k in sorted(manifest["splits"])
         for i in manifest["splits"][k]], sort_keys=True)
    manifest["manifest_sha256"] = hashlib.sha256(payload.encode()).hexdigest()

    (out / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "pool_report.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== filters ===")
    print(f"  loaded                 {report['n_loaded']:>7,}")
    print(f"  - regression           {report['dropped_regression']:>7,}   "
          f"by category: {report['dropped_regression_by_category']}")
    print(f"  - ungradeable          {report['dropped_ungradeable']:>7,}   "
          f"by category: {report['dropped_ungradeable_by_category']}")
    print(f"  = eligible rows        {report['eligible_rows_after_filters']:>7,}")
    print(f"  = eligible images      {report['eligible_distinct_images']:>7,}   "
          f"{report['eligible_by_category']}")
    if report["ungradeable_examples"]:
        print(f"  ungradeable examples: {report['ungradeable_examples'][:8]}")

    print(f"\n=== stratification (target = {report['target_mode']}) ===")
    drawn = collections.Counter(
        category_of(r["path"]) for v in splits.values() for r in v)
    n_drawn = sum(drawn.values())
    print(f"  {'category':<12} {'raw%':>7} {'eligible%':>10} {'drawn':>7} {'drawn%':>8}")
    for c in CATEGORIES:
        print(f"  {c:<12} {report['raw_share'][c]*100:6.1f}% "
              f"{report['eligible_share'][c]*100:9.1f}% "
              f"{drawn[c]:>7,} {drawn[c]/n_drawn*100:7.1f}%")
    print("  (raw vs eligible diverge where a filter is category-concentrated)")

    print(f"\n=== splits ===")
    for k, v in splits.items():
        by_c = collections.Counter(category_of(r["path"]) for r in v)
        print(f"  {k:<6} {len(v):>6,}  {dict(sorted(by_c.items()))}")
    print(f"\n[done] manifest sha256 {manifest['manifest_sha256']}")
    print(f"[done] -> {out}/pool_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
