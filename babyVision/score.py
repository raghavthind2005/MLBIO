#!/usr/bin/env python3
"""
BabyVision — single source of truth for scoring. Supersedes validate_results.py,
audit_judge.py, test_extraction.py, robust_rescore.py.

The official prompt says 'give your final answer in \\boxed{Answer} format', and
Gemma-4 often writes the placeholder literally: `\\boxed{Answer} (B)`, putting the
real choice OUTSIDE the box. The official extractor then grabs the junk "Answer".
This is a FORMATTING failure, not a perception failure, and it hits the no-think
condition far harder than the thinking one — so raw judge accuracy is a biased
estimator of perception.

This script reports TWO numbers per condition, side by side:
  - faithful : official extractor + stored LLM judge  (== leaderboard pipeline)
  - reliable : CHOICE scored deterministically vs the known gold letter (no judge,
               robust to the \\boxed{Answer} quirk); BLANK uses the stored judge
               (only ~3/253 blanks are affected by the quirk — negligible).

`--audit N` dumps N worked examples (model's own words → extracted letter → gold →
verdict) and every faithful↔reliable disagreement, so the extraction is verifiable
by eye. No re-inference, stdlib + the official extractor only.

Usage:
  python score.py --base <results_root> --repo <code_dir> [--audit 12]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# (key, dir, passes, label)
CONDS = [
    ("a0",       "results_a0_nothink",     [1],       "A0 no-think"),
    ("standard", "results_standard",       [1, 2, 3], "standard"),
    ("a3v1",     "results_a3_forced_long_v1", [1],    "A3 forced-long v1"),
    ("a3",       "results_a3_forced_long", [1],       "A3 forced-long v2"),
    ("b1",       "results_b1_reinject",    [1],       "B1 reinject"),
    ("b2",       "results_b2_noreinject",  [1],       "B2 no-reinject"),
]

JUNK_RE  = re.compile(r"^\s*answer\s*:?\s*\(?\s*\)?\s*$", re.I)
TRAIL_RE = re.compile(r"\\boxed\{[^{}]*\}\s*\(?\s*([A-Za-z0-9][^\n)]*)\)?")


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else None


def is_junk(x):
    return x is None or not str(x).strip() or bool(JUNK_RE.match(str(x)))


def letter_of(s):
    """Single choice letter A–H from a messy string, else None."""
    if s is None:
        return None
    s = str(s)
    m = re.findall(r"\(([A-Ha-h])\)", s)            # parenthesised first
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([A-Ha-h])\b", s)            # then a standalone letter
    if m:
        return m[-1].upper()
    return None


def reliable_choice_letter(answer_text, official_extract):
    """The letter the model actually committed to, robust to `\\boxed{Answer} (X)`.
    Order: clean box content → token trailing the box → None. Deliberately does NOT
    fish through prose, so it can't over-credit."""
    box = official_extract(answer_text)
    if box is not None and not is_junk(box):
        L = letter_of(box)
        if L:
            return L, "box"
    if answer_text:
        m = list(TRAIL_RE.finditer(answer_text))
        if m:
            L = letter_of(m[-1].group(1))
            if L:
                return L, "trailing"
    return None, "none"


def score_pass(recs, official_extract, audit_rows):
    choice = [r for r in recs if r.get("ansType") == "choice"]
    blank  = [r for r in recs if r.get("ansType") == "blank"]

    # faithful
    f_all = sum(1 for r in recs   if r.get("judge_result") is True)
    f_ch  = sum(1 for r in choice if r.get("judge_result") is True)
    f_bl  = sum(1 for r in blank  if r.get("judge_result") is True)

    # reliable: choice deterministic, blank = stored judge
    r_ch = box_fail = 0
    flips_up = flips_down = 0
    for r in choice:
        gold = letter_of(r.get("gt_answer"))
        pred, src = reliable_choice_letter(r.get("answer_text") or "", official_extract)
        ok = (pred is not None and pred == gold)
        if ok:
            r_ch += 1
        if is_junk(official_extract(r.get("answer_text") or "")):
            box_fail += 1
        faith_ok = r.get("judge_result") is True
        if ok and not faith_ok:
            flips_up += 1
            audit_rows["up"].append((r, pred, gold))
        if faith_ok and not ok:
            flips_down += 1
            audit_rows["down"].append((r, pred, gold))

    n, nc, nb = len(recs), len(choice), len(blank)
    return {
        "n": n, "n_choice": nc, "n_blank": nb,
        "faithful_overall": f_all / n,
        "faithful_choice":  f_ch / nc if nc else None,
        "faithful_blank":   f_bl / nb if nb else None,
        "reliable_choice":  r_ch / nc if nc else None,
        "reliable_blank":   f_bl / nb if nb else None,  # blanks unaffected → reuse judge
        "reliable_overall": (r_ch + f_bl) / n,
        "box_fail_choice":  box_fail / nc if nc else None,
        "flips_up": flips_up, "flips_down": flips_down,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--audit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo) / "repo" / "babyvision_eval"))
    from utils import extract_boxed_answer as official_extract     # noqa: E402

    base = Path(args.base)
    audit_rows = {"up": [], "down": []}
    summary = {}

    for key, d, passes, label in CONDS:
        per = []
        for pi in passes:
            recs = load(base / d / f"results_run{pi}_judged.jsonl")
            if recs is None:
                continue
            recs = [r for r in recs if "error" not in r]
            per.append(score_pass(recs, official_extract, audit_rows if pi == 1 else
                                   {"up": [], "down": []}))
        if per:
            summary[key] = (label, per)

    # ── main table ──
    def avg(per, k):
        vals = [p[k] for p in per if p.get(k) is not None]
        return sum(vals) / len(vals) if vals else None

    def pc(x):
        return " n/a " if x is None else f"{x*100:5.1f}"

    print("\n" + "=" * 92)
    print("  BabyVision SCORING — faithful (leaderboard) vs reliable (deterministic choice)")
    print("=" * 92)
    print(f"  {'condition':<20}{'n':>4}{'faith.all':>10}{'reliable':>10}"
          f"{'ch.faith':>9}{'ch.reliable':>12}{'box-fail%':>10}{'blank':>8}")
    for key, (label, per) in summary.items():
        n = per[0]["n"]
        print(f"  {label:<20}{n:>4}"
              f"{pc(avg(per,'faithful_overall')):>10}{pc(avg(per,'reliable_overall')):>10}"
              f"{pc(avg(per,'faithful_choice')):>9}{pc(avg(per,'reliable_choice')):>12}"
              f"{pc(avg(per,'box_fail_choice')):>10}{pc(avg(per,'faithful_blank')):>8}")

    # ── reliability commentary ──
    print("\n" + "-" * 92)
    print("  READ")
    print("-" * 92)
    print("""  - 'reliable' overall = deterministic choice (gold letter, no judge) + blank judge.
  - 'box-fail%' = share of CHOICE answers where the model failed to put the letter in
    the box (the \\boxed{Answer} quirk). High box-fail with flat reliable accuracy means
    reasoning improved FORMATTING, not perception.
  - faithful vs reliable on choice: the gap is pure format-following loss, not perception.""")

    if "a0" in summary and "standard" in summary:
        a0 = summary["a0"][1]; st = summary["standard"][1]
        g_f = (avg(st,'faithful_choice') - avg(a0,'faithful_choice')) * 100
        g_r = (avg(st,'reliable_choice') - avg(a0,'reliable_choice')) * 100
        print(f"\n  std − A0 (choice):  faithful {g_f:+.1f} pts   reliable {g_r:+.1f} pts"
              f"   (sign-stable ⇒ conclusion robust)")

    # ── per-condition flip counts ──
    print("\n  faithful↔reliable choice disagreements (per condition, pass 1):")
    for key, (label, per) in summary.items():
        print(f"    {label:<20} wrong→right(format-recovered): {per[0]['flips_up']:3d}"
              f"   right→wrong(judge-overcredit/misgrab): {per[0]['flips_down']:3d}")

    # ── audit dump ──
    if args.audit:
        print("\n" + "=" * 92)
        print(f"  AUDIT — verify extraction by eye (showing up to {args.audit} of each)")
        print("=" * 92)
        print("\n  [wrong→right] judge scored wrong, model actually gave the gold letter:")
        for r, pred, gold in audit_rows["up"][:args.audit]:
            tail = (r.get("answer_text") or "")[-200:].replace("\n", " ")
            print(f"    id={r.get('taskId')} sub={r.get('subtype')!r} pred={pred} gold={gold}")
            print(f"       ...{tail!r}")
        print("\n  [right→wrong] judge scored right, deterministic says wrong (INSPECT these):")
        for r, pred, gold in audit_rows["down"][:args.audit]:
            tail = (r.get("answer_text") or "")[-200:].replace("\n", " ")
            print(f"    id={r.get('taskId')} sub={r.get('subtype')!r} pred={pred} gold={gold} "
                  f"judge_raw={str(r.get('judge_raw'))[:20]!r}")
            print(f"       ...{tail!r}")

    print()


if __name__ == "__main__":
    main()
