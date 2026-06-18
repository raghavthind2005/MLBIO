#!/usr/bin/env python3
"""
Robust re-scoring to remove the `\\boxed{Answer} (X)` format confound.

The model often echoes the prompt placeholder — it emits `\\boxed{Answer}` and
writes the REAL answer right after the box. The official extractor grabs the
junk "Answer" and the judge scores it wrong, even when the answer is correct.
This is faithful to the benchmark, but the junk rate differs by condition
(A0 ~17% vs standard ~5%), which confounds the A0-vs-standard comparison.

This script, on the SAVED outputs (no re-inference, no LLM judge):
  - faithful score   = official extractor + stored judge_result (leaderboard parity)
  - robust score     = recover the answer trailing `\\boxed{Answer}` when the box
                       content is junk; for CHOICE questions score DETERMINISTICALLY
                       against the gold letter (no judge needed).
It reports both per condition and the A0-vs-standard gap under each, so we can
see whether the format confound was masking a real effect.

Targets ONLY the observed failure mode (token immediately after the box) so it
cannot over-credit by fishing letters out of prose.

Usage:
  python robust_rescore.py \
    --base /iopsstor/scratch/cscs/raghavthind/babyvision \
    --repo /iopsstor/scratch/cscs/raghavthind/code/babyvision \
    --dump 10
"""

import argparse
import json
import re
import sys
from pathlib import Path

CONDS = [
    ("A0 no-think",    "results_a0_nothink",     1),
    ("standard",       "results_standard",        1),
    ("A3 forced-long", "results_a3_forced_long",  1),
]

JUNK_RE = re.compile(r"^\s*answer\s*:?\s*\(?\s*\)?\s*$", re.I)
# token(s) trailing the last \boxed{...}: an optional "(X)" or a bare letter/number
TRAIL_RE = re.compile(r"\\boxed\{[^{}]*\}\s*\(?\s*([A-Za-z0-9][^\n)]*)\)?")


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def is_junk(x):
    return x is None or not str(x).strip() or bool(JUNK_RE.match(str(x)))


def letter_of(s):
    if s is None:
        return None
    s = str(s)
    m = re.findall(r"\(([A-Ha-h])\)", s)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([A-Ha-h])\b", s)
    if m:
        return m[-1].upper()
    return None


def robust_answer(text, official_extract):
    """Official extraction, but if the box content is junk, recover the token
    written immediately AFTER the last \\boxed{...}. Returns (value, recovered?)."""
    e = official_extract(text)
    if e is not None and not is_junk(e):
        return e, False
    # box content is junk (or no box) → look for the trailing token after a box
    if text:
        m = list(TRAIL_RE.finditer(text))
        if m:
            return m[-1].group(1).strip(), True
    return e, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dump", type=int, default=10)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo) / "repo" / "babyvision_eval"))
    from utils import extract_boxed_answer as official_extract   # noqa: E402

    base = Path(args.base)
    summary = {}

    for label, dirname, pi in CONDS:
        path = base / dirname / f"results_run{pi}_judged.jsonl"
        print("=" * 74)
        print(f"  {label}")
        print("=" * 74)
        if not path.exists():
            print(f"  MISSING {path}\n")
            continue
        recs = [r for r in load(path) if "error" not in r]
        choice = [r for r in recs if r.get("ansType") == "choice"]

        # ── overall faithful accuracy (stored judge) ──
        faithful_all = sum(1 for r in recs if r.get("judge_result") is True) / len(recs)

        # ── CHOICE: faithful (judge) vs robust (deterministic gold letter) ──
        faith_c = sum(1 for r in choice if r.get("judge_result") is True) / len(choice)

        robust_correct = 0
        recovered = 0
        flips = []   # was wrong under faithful, correct under robust
        for r in choice:
            gold = letter_of(r.get("gt_answer"))
            val, rec = robust_answer(r.get("answer_text") or "", official_extract)
            if rec:
                recovered += 1
            pred = letter_of(val)
            ok = (pred is not None and pred == gold)
            if ok:
                robust_correct += 1
            if ok and r.get("judge_result") is not True:
                flips.append((r, val, pred, gold))
        robust_c = robust_correct / len(choice)

        print(f"  choice n={len(choice)}")
        print(f"    faithful (judge)        : {faith_c*100:5.1f}%")
        print(f"    robust  (det. gold letter): {robust_c*100:5.1f}%   "
              f"(+{(robust_c-faith_c)*100:.1f} pts; recovered trailing answer on {recovered})")
        print(f"    faithful→robust flips (wrong→right): {len(flips)}")
        for r, val, pred, gold in flips[:args.dump]:
            print(f"      id={r.get('taskId')} sub={r.get('subtype')!r} "
                  f"recovered={val!r}→{pred} gold={gold}")

        summary[label] = {"faithful_all": faithful_all,
                          "choice_faithful": faith_c, "choice_robust": robust_c,
                          "n_choice": len(choice)}
        print()

    # ── the comparison that matters ──
    print("=" * 74)
    print("  A0-vs-STANDARD GAP  (does the format confound change the story?)")
    print("=" * 74)
    if "A0 no-think" in summary and "standard" in summary:
        a0, st = summary["A0 no-think"], summary["standard"]
        print(f"  CHOICE accuracy:")
        print(f"    faithful: A0={a0['choice_faithful']*100:.1f}%  "
              f"standard={st['choice_faithful']*100:.1f}%  "
              f"gap={ (st['choice_faithful']-a0['choice_faithful'])*100:+.1f} pts (std−A0)")
        print(f"    robust  : A0={a0['choice_robust']*100:.1f}%  "
              f"standard={st['choice_robust']*100:.1f}%  "
              f"gap={ (st['choice_robust']-a0['choice_robust'])*100:+.1f} pts (std−A0)")
        print("""
  READ: if the std−A0 gap SHRINKS (or flips) under robust scoring, the raw
  "thinking ≈ no-thinking" result was partly a format-following artifact that
  hurt A0 more. If the gap is stable, the conclusion holds even after removing
  the confound. (Choice-only: blanks still need the LLM judge.)
""")


if __name__ == "__main__":
    main()
