#!/usr/bin/env python3
"""
Judge + extraction integrity audit for BabyVision (A0 / standard / A3).

The judge (run_judge.py) sees ONLY `extracted_answer`, never the full model
response. So a broken \\boxed{} extraction silently scores a correct answer as
wrong and depresses accuracy. This script quantifies that risk three ways:

  1. Extraction degeneracy: how many `extracted_answer` are None / empty / junk
     (e.g. the literal word "Answer"), split by judge True/False.
  2. Deterministic check on CHOICE questions: the gold letter is known, so we
     recompute correctness from the extracted letter and compare to the LLM judge.
     Disagreements = judge or extraction errors we can see without the judge.
  3. Recovery probe: for degenerate-extraction records, dump the full answer_text
     so we can eyeball whether a correct answer was actually present but missed.

Stdlib only — runs on the login node.

Usage:
  python audit_judge.py --base /iopsstor/scratch/cscs/raghavthind/babyvision
"""

import argparse
import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

CONDS = [
    ("A0 no-think",    "results_a0_nothink",     [1]),
    ("standard",       "results_standard",        [1, 2, 3]),
    ("A3 forced-long", "results_a3_forced_long",  [1]),
]

# extracted_answer values that carry no real content
JUNK_RE = re.compile(r"^\s*answer\s*:?\s*\(?\s*\)?\s*$", re.I)


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def is_junk(ext):
    if ext is None:
        return True
    s = str(ext).strip()
    if not s:
        return True
    return bool(JUNK_RE.match(s))


def letter_of(s):
    """Pull a single choice letter (A-H) out of a messy string, else None."""
    if s is None:
        return None
    s = str(s)
    # prefer a parenthesised letter: (A)
    m = re.findall(r"\(([A-Ha-h])\)", s)
    if m:
        return m[-1].upper()
    # else a standalone letter token
    m = re.findall(r"\b([A-Ha-h])\b", s)
    if m:
        return m[-1].upper()
    return None


def full_answer(r):
    for f in ("answer_text", "answer_raw", "response", "model_output"):
        if r.get(f):
            return str(r[f])
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", type=int, default=8, help="full examples to dump per condition")
    args = ap.parse_args()
    base = Path(args.base)
    random.seed(args.seed)

    for label, dirname, passes in CONDS:
        pi = passes[0]
        path = base / dirname / f"results_run{pi}_judged.jsonl"
        print("\n" + "=" * 74)
        print(f"  {label}   (pass {pi})")
        print("=" * 74)
        if not path.exists():
            print(f"  MISSING {path}")
            continue
        recs = [r for r in load(path) if "error" not in r]
        n = len(recs)
        choice = [r for r in recs if r.get("ansType") == "choice"]
        blank  = [r for r in recs if r.get("ansType") == "blank"]
        acc = sum(1 for r in recs if r.get("judge_result") is True) / n
        print(f"  valid={n}  choice={len(choice)}  blank={len(blank)}  acc={acc*100:.1f}%")

        # show the schema once so field names are explicit
        if recs:
            print(f"  record keys: {sorted(recs[0].keys())}")

        # ── 1. Extraction degeneracy ───────────────────────────────────────────
        junk      = [r for r in recs if is_junk(r.get("extracted_answer"))]
        junk_true = [r for r in junk if r.get("judge_result") is True]
        junk_false= [r for r in junk if r.get("judge_result") is False]
        print(f"\n  [1] EXTRACTION DEGENERACY")
        print(f"      junk/empty extracted_answer: {len(junk)}/{n} ({len(junk)/n*100:.1f}%)")
        print(f"        of those judged True:  {len(junk_true)}  (judge passed despite junk?!)")
        print(f"        of those judged False: {len(junk_false)}  (auto-wrong — recoverable?)")
        # how much would accuracy move if every junk-False actually had a right answer?
        ceiling = (sum(1 for r in recs if r.get("judge_result") is True) + len(junk_false)) / n
        print(f"      accuracy if ALL junk-False were really correct: "
              f"{acc*100:.1f}% → {ceiling*100:.1f}% (upper bound on extraction loss)")

        # ── 2. Deterministic check on choice questions ─────────────────────────
        print(f"\n  [2] DETERMINISTIC CHOICE CHECK (gold letter known)")
        det_ok = det_total = 0
        judge_false_letter_match = []   # judge said wrong but letters agree
        judge_true_letter_diff   = []   # judge said right but letters differ
        gt_no_letter = 0
        for r in choice:
            gt_l = letter_of(r.get("gt_answer"))
            ex_l = letter_of(r.get("extracted_answer"))
            if gt_l is None:
                gt_no_letter += 1
                continue
            det_total += 1
            det = (ex_l is not None and ex_l == gt_l)
            jr = r.get("judge_result") is True
            if det:
                det_ok += 1
            if jr and not det:
                judge_true_letter_diff.append((r, gt_l, ex_l))
            if det and not jr:
                judge_false_letter_match.append((r, gt_l, ex_l))
        if det_total:
            print(f"      choice w/ parseable gold letter: {det_total} "
                  f"(gt had no letter: {gt_no_letter})")
            print(f"      deterministic letter-match accuracy: {det_ok/det_total*100:.1f}%  "
                  f"(vs judge on same subset)")
            print(f"      judge=False but letters MATCH: {len(judge_false_letter_match)} "
                  f"(judge under-credited)")
            print(f"      judge=True  but letters DIFFER: {len(judge_true_letter_diff)} "
                  f"(judge over-credited / alias)")
            for r, gt_l, ex_l in judge_false_letter_match[:4]:
                print(f"        [under] id={r.get('taskId')} gt={r.get('gt_answer')!r}({gt_l}) "
                      f"ext={r.get('extracted_answer')!r}({ex_l})")
            for r, gt_l, ex_l in judge_true_letter_diff[:4]:
                print(f"        [over ] id={r.get('taskId')} gt={r.get('gt_answer')!r}({gt_l}) "
                      f"ext={r.get('extracted_answer')!r}({ex_l})")
        else:
            print("      (no choice records with parseable gold letter)")

        # ── 3. Recovery probe on junk-False ────────────────────────────────────
        print(f"\n  [3] RECOVERY PROBE — full answer_text for junk-extraction FALSE records")
        sample = random.sample(junk_false, min(args.dump, len(junk_false)))
        for r in sample:
            ans = full_answer(r)
            tail = ans[-260:].replace("\n", " ")
            print(f"      ── id={r.get('taskId')} sub={r.get('subtype')!r} "
                  f"ansType={r.get('ansType')!r}")
            print(f"         gt={r.get('gt_answer')!r}  extracted={r.get('extracted_answer')!r}")
            print(f"         answer_text tail: ...{tail!r}")

    print("\n" + "=" * 74)
    print("  READ-OUT GUIDE")
    print("=" * 74)
    print("""
  - If [1] junk% is small (<5%) and [3] shows the model genuinely had no
    correct answer in those cases → accuracy is HONEST, low score is real.
  - If [1] junk% is large OR [3] shows correct answers buried in answer_text
    that extraction missed → accuracy is ARTIFICIALLY DEPRESSED; fix
    extract_boxed_answer (or feed the judge the full answer_text) and re-judge.
  - In [2], judge=False-but-letters-match counts are pure judge/extraction
    error on the deterministic subset; large numbers there mean the LLM judge
    is unreliable and we should trust the deterministic letter check instead.
""")


if __name__ == "__main__":
    main()
