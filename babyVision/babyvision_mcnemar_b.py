#!/usr/bin/env python3
"""
Within-family paired McNemar for the corrected B' family (b1cot / b2cot).

analyze_conditions.py showed the key dissociation as raw family deltas:
  re-grounding+CoT (B1') helps PERCEPTION (+6.8) and hurts REASONING (-6.6) vs
  standard. The per-subtype n's (10-35) are too small to test individually; the
  PERCEPTION (n~191) and REASONING (n~197) family aggregates are the reliable unit.
This script runs the proper PAIRED test (exact-binomial McNemar on discordant
pairs, same items) for each family, for the comparisons that matter:

  B1'  vs standard   (does re-grounding+CoT move perception / reasoning?)
  B2'  vs standard   (CoT, no image re-show)
  B1'  vs B2'        (paired image-reshow effect, identical turn-1)

Standard is the MAJORITY of its 3 graded passes per item (matches analyze's 27.8%
fair baseline). Pure stdlib — runs on the login node, no GPU/server.

  python babyvision_mcnemar_b.py --base /iopsstor/.../babyvision
"""

import argparse
import json
from math import comb
from pathlib import Path

# Must match analyze_conditions.py's groupings.
PERCEPTION = {"Count 3D blocks", "Count Same Patterns", "Count Clusters", "Maze",
              "Connect the lines", "Metro map", "Lines Observation", "Find the same",
              "Find the different", "Find the shadow"}
REASONING  = {"3D Cube Unfold", "Paper Folding", "3D Views", "Rotation Patterns",
              "Mirroring Patterns", "2D Pattern Completion", "3D Pattern Completion",
              "Logic Patterns", "Overlay Patterns", "Reconstruction",
              "Recognize numbers and letters", "Pattern and Color Completion"}


def family_of(subtype):
    if subtype in PERCEPTION:
        return "perception"
    if subtype in REASONING:
        return "reasoning"
    return "other"


def load_graded(path):
    """taskId -> (grade bool, subtype). Skips error records."""
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            out[r["taskId"]] = (bool(r.get("grade")), r.get("subtype"))
    return out


def load_standard_majority(base):
    """taskId -> (majority-of-3-passes grade, subtype)."""
    per_task = {}
    sub = {}
    d = base / "results_standard"
    for pi in (1, 2, 3):
        p = d / f"results_run{pi}_graded.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            per_task.setdefault(r["taskId"], []).append(bool(r.get("grade")))
            sub[r["taskId"]] = r.get("subtype")
    return {t: (sum(v) >= (len(v) / 2 + 0.0001), sub[t]) for t, v in per_task.items()}


def mcnemar_exact(n01, n10):
    """Two-sided exact-binomial McNemar p over discordant pairs."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(p, 1.0)


def compare(A, B, nameA, nameB):
    """Paired comparison A vs B over shared taskIds, split by family."""
    shared = set(A) & set(B)
    print(f"\n{'='*72}\n{nameA}  vs  {nameB}   (paired, n={len(shared)})\n{'='*72}")
    print(f"  {'family':<12}{'n':>5}{nameA[:10]:>12}{nameB[:10]:>12}"
          f"{'Δ':>8}{'A+/B-':>8}{'A-/B+':>8}{'p(exact)':>11}")
    for fam in ("all", "perception", "reasoning"):
        ids = [t for t in shared if fam == "all" or family_of(A[t][1]) == fam]
        if not ids:
            continue
        a_correct = sum(1 for t in ids if A[t][0])
        b_correct = sum(1 for t in ids if B[t][0])
        n01 = sum(1 for t in ids if (not A[t][0]) and B[t][0])   # A wrong, B right
        n10 = sum(1 for t in ids if A[t][0] and (not B[t][0]))   # A right, B wrong
        accA = 100.0 * a_correct / len(ids)
        accB = 100.0 * b_correct / len(ids)
        p = mcnemar_exact(n01, n10)
        star = "  *" if p < 0.05 else ("  ." if p < 0.10 else "")
        print(f"  {fam:<12}{len(ids):>5}{accA:>11.1f}%{accB:>11.1f}%"
              f"{accA-accB:>+8.1f}{n10:>8}{n01:>8}{p:>11.4f}{star}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    args = ap.parse_args()
    base = Path(args.base)

    std   = load_standard_majority(base)
    b1cot = load_graded(base / "results_b1cot_reinject"   / "results_run1_graded.jsonl")
    b2cot = load_graded(base / "results_b2cot_noreinject" / "results_run1_graded.jsonl")

    print(f"Loaded: standard(maj3)={len(std)}  b1cot={len(b1cot)}  b2cot={len(b2cot)}")
    print("McNemar exact-binomial, two-sided.  * p<.05   . p<.10")
    print("Δ = accA - accB.  'A+/B-' = A right & B wrong (n10); 'A-/B+' = n01.")

    compare(b1cot, std,   "B1'", "std(maj3)")
    compare(b2cot, std,   "B2'", "std(maj3)")
    compare(b1cot, b2cot, "B1'", "B2'")


if __name__ == "__main__":
    main()
