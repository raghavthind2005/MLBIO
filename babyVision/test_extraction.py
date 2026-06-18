#!/usr/bin/env python3
"""
Extraction-faithfulness test: does OUR extract_boxed_answer agree with the
OFFICIAL benchmark's, on the answers we already saved? No re-inference.

Imports the official extractor straight from the benchmark repo and ours from
run_infer.py, applies both to the saved model outputs, and reports:
  - regex divergence on the FINAL answer (our re vs official regex)
  - how many records get an answer ONLY via our thinking-trace fallback
    (the official has no such fallback → those are where we are more lenient)
  - records the official considers unanswered (no \\boxed in final answer)

Run a small --limit first to eyeball, then full.

Usage:
  python test_extraction.py \
    --base /iopsstor/scratch/cscs/raghavthind/babyvision \
    --repo /iopsstor/scratch/cscs/raghavthind/code/babyvision \
    --limit 40
"""

import argparse
import json
import sys
from pathlib import Path

CONDS = [
    ("A0 no-think",    "results_a0_nothink",     1),
    ("standard",       "results_standard",        1),
    ("A3 forced-long", "results_a3_forced_long",  1),
]


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--repo", required=True, help="code dir containing run_infer.py and repo/")
    ap.add_argument("--limit", type=int, default=0, help="0 = all records")
    args = ap.parse_args()

    # OFFICIAL extractor (exact benchmark code)
    sys.path.insert(0, str(Path(args.repo) / "repo" / "babyvision_eval"))
    from utils import extract_boxed_answer as official_extract   # noqa: E402

    # OUR extractor
    sys.path.insert(0, str(args.repo))
    from run_infer import extract_boxed_answer as our_extract     # noqa: E402

    print("Loaded both extractors:")
    print(f"  official: {official_extract.__module__}")
    print(f"  ours:     {our_extract.__module__}\n")

    base = Path(args.base)
    grand = {"regex_diff": 0, "fallback_only": 0, "official_none": 0, "n": 0}

    for label, dirname, pi in CONDS:
        path = base / dirname / f"results_run{pi}.jsonl"
        print("=" * 72)
        print(f"  {label}  ({path.name})")
        print("=" * 72)
        if not path.exists():
            print(f"  MISSING {path}\n")
            continue
        recs = [r for r in load(path) if "error" not in r]
        if args.limit:
            recs = recs[:args.limit]

        regex_diff, fallback_only, official_none = [], [], 0
        for r in recs:
            ans   = r.get("answer_text") or ""
            think = r.get("thinking_trace") or ""

            our_final    = our_extract(ans)
            our_fallback = our_extract(ans) or our_extract(think)
            off_final    = official_extract(ans)

            # normalize for comparison (None vs "" and surrounding spaces)
            def norm(x):
                return None if x is None else str(x).strip()
            if norm(our_final) != norm(off_final):
                regex_diff.append((r, our_final, off_final))
            if our_final is None and our_fallback is not None:
                fallback_only.append((r, our_fallback))
            if off_final is None:
                official_none += 1

        n = len(recs)
        print(f"  records checked:                 {n}")
        print(f"  regex divergence (ours vs offcl):{len(regex_diff)}  "
              f"({len(regex_diff)/n*100:.1f}%)" if n else "")
        print(f"  answer only via thinking-fallback:{len(fallback_only)}  "
              f"({len(fallback_only)/n*100:.1f}%)  <- we are more lenient here" if n else "")
        print(f"  official sees NO box in final ans:{official_none}  "
              f"({official_none/n*100:.1f}%)" if n else "")

        for r, ours, off in regex_diff[:6]:
            print(f"    [regex-diff id={r.get('taskId')}] ours={ours!r}  official={off!r}")
        for r, fb in fallback_only[:6]:
            print(f"    [fallback id={r.get('taskId')}] recovered={fb!r}  "
                  f"gt={r.get('gt_answer')!r}  judge={r.get('judge_result')}")
        print()

        grand["regex_diff"]    += len(regex_diff)
        grand["fallback_only"] += len(fallback_only)
        grand["official_none"] += official_none
        grand["n"]             += n

    print("=" * 72)
    print("  TOTALS")
    print("=" * 72)
    n = grand["n"] or 1
    print(f"  regex divergence:     {grand['regex_diff']}/{grand['n']} ({grand['regex_diff']/n*100:.2f}%)")
    print(f"  fallback-only answers:{grand['fallback_only']}/{grand['n']} ({grand['fallback_only']/n*100:.2f}%)")
    print(f"""
  INTERPRETATION
  - regex divergence ~0%  → our extractor is already byte-faithful; no re-judge needed.
  - fallback-only > 0     → those records are scored leniently vs official; if we want
                            strict leaderboard parity, drop the thinking fallback and
                            re-judge (scores can only go DOWN, confirming they aren't
                            artificially depressed).
""")


if __name__ == "__main__":
    main()
