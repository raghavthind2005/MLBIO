#!/usr/bin/env python3
"""BabyVision VALIDITY / SIGNIFICANCE checks — run before any writeup.

Point estimates (analyze_conditions.py) are not enough: with 388 items and
cross-pass noise of ±1.4 pts, a few-point delta can be pure sampling noise. This
script answers "is each cross-condition delta a true representative, or noise /
a fallback artifact?" using:

  1. PAIRED significance — McNemar exact two-sided p on discordant pairs + a
     paired bootstrap 95% CI on Δaccuracy, for every key contrast (same taskIds).
  2. STANDARD 3-pass noise band — the yardstick any 1-pass delta must clear.
  3. CHOICE self-validation per condition — does the judge's grade agree with the
     deterministic gold letter? (re-checked for B1/B2 after the T1-fallback patch).
  4. SEE-LESS curve, fallback-cleaned — the B curves binned on concluded-only
     records (fallback records have huge completion_tokens but a short T1 answer,
     so they contaminate the top quartile).

Reads *_graded.jsonl only (stdlib) — safe on the login node.
"""

import json
import random
import re
from math import factorial
from pathlib import Path


def comb(n, k):
    if k < 0 or k > n:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))

random.seed(0)
BASE = Path("/iopsstor/scratch/cscs/raghavthind/babyvision")


def load(d, p=1):
    f = BASE / d / f"results_run{p}_graded.jsonl"
    if not f.exists():
        return {}
    return {json.loads(l)["taskId"]: json.loads(l)
            for l in open(f) if l.strip() and "error" not in json.loads(l)}


def g(r):
    return r.get("grade") is True


def unc(r):
    """Turn-2 never concluded → graded on standing T1 answer (B-family only)."""
    return (r.get("finish_reason") == "length"
            and not (r.get("thinking_trace") or "").strip())


def letter_of(s):
    if s is None:
        return None
    s = str(s)
    m = re.findall(r"\(([A-Ha-h])\)", s) or re.findall(r"\b([A-Ha-h])\b", s)
    return m[-1].upper() if m else None


def mcnemar(d1, d2, ids):
    b = sum(1 for i in ids if g(d1[i]) and not g(d2[i]))      # d1 right, d2 wrong
    c = sum(1 for i in ids if not g(d1[i]) and g(d2[i]))      # d1 wrong, d2 right
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return b, c, p


def boot_ci(d1, d2, ids, reps=5000):
    arr = [(g(d1[i]), g(d2[i])) for i in ids]
    n = len(arr)
    diffs = []
    for _ in range(reps):
        s1 = s2 = 0
        for _ in range(n):
            a, b = arr[random.randrange(n)]
            s1 += a; s2 += b
        diffs.append((s1 - s2) / n)
    diffs.sort()
    return diffs[int(0.025 * reps)] * 100, diffs[int(0.975 * reps)] * 100


def acc(d, ids):
    return (sum(1 for i in ids if g(d[i])) / len(ids) * 100) if ids else float("nan")


def contrast(name, d1, d2, ids, concluded_filter=False):
    if concluded_filter:
        ids = [i for i in ids if not unc(d1[i]) and not unc(d2[i])]
    if not ids:
        print(f"  {name:<34} (no items)")
        return
    a1, a2 = acc(d1, ids), acc(d2, ids)
    b, c, p = mcnemar(d1, d2, ids)
    lo, hi = boot_ci(d1, d2, ids)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {name:<34} N={len(ids):<4} {a1:5.1f} vs {a2:5.1f}  "
          f"Δ={a1-a2:+5.1f}  CI[{lo:+5.1f},{hi:+5.1f}]  "
          f"McNemar b/c={b}/{c} p={p:.3f} {sig}")


def main():
    a0   = load("results_a0_nothink")
    s1   = load("results_standard", 1)
    s2   = load("results_standard", 2)
    s3   = load("results_standard", 3)
    a3   = load("results_a3_forced_long")
    b1   = load("results_b1_reinject")
    b2   = load("results_b2_noreinject")

    print("\n" + "=" * 84)
    print("BabyVision — VALIDITY / SIGNIFICANCE")
    print("=" * 84)

    # ── 1. standard 3-pass noise band (the yardstick) ──────────────────────────
    print("\n-- STANDARD 3-pass noise band (any 1-pass delta must clear this) --")
    paccs = [acc(s, list(s)) for s in (s1, s2, s3) if s]
    if paccs:
        print(f"  passes: " + "  ".join(f"{a:.1f}" for a in paccs) +
              f"   range={max(paccs)-min(paccs):.1f} pts   "
              f"min={min(paccs):.1f} max={max(paccs):.1f}")

    # ── 2. paired contrasts (full common set) ──────────────────────────────────
    print("\n-- PAIRED CONTRASTS, full common set (vs standard pass-1) --")
    print("   (Δ = first − second; CI = paired bootstrap 95%; sig from McNemar)")
    contrast("B1 vs B2  (re-grounding)", b1, b2, list(set(b1) & set(b2)))
    contrast("B1 vs standard",          b1, s1, list(set(b1) & set(s1)))
    contrast("B2 vs standard",          b2, s1, list(set(b2) & set(s1)))
    contrast("A3 vs standard",          a3, s1, list(set(a3) & set(s1)))
    contrast("A0 vs standard",          a0, s1, list(set(a0) & set(s1)))

    # ── 3. paired contrasts, concluded-only (B-family, no fallback blend) ──────
    print("\n-- PAIRED CONTRASTS, concluded-only (B drops loop-truncated items) --")
    contrast("B1 vs B2  (concluded)", b1, b2, list(set(b1) & set(b2)), True)

    # ── 4. choice self-validation (judge vs deterministic gold letter) ─────────
    print("\n-- CHOICE self-validation (grade vs deterministic gold letter) --")
    for nm, d in (("a0", a0), ("standard(p1)", s1), ("a3", a3),
                  ("b1", b1), ("b2", b2)):
        ch = [r for r in d.values() if r.get("ansType") == "choice"]
        agree = dis = 0
        for r in ch:
            det = (letter_of(r.get("extracted_answer")) is not None and
                   letter_of(r.get("extracted_answer")) == letter_of(r.get("gt_answer")))
            if det == g(r):
                agree += 1
            else:
                dis += 1
        rate = agree / len(ch) * 100 if ch else float("nan")
        print(f"  {nm:<14} choice_n={len(ch):<4} agree={agree:<4} disagree={dis:<3} "
              f"({rate:.1f}% agreement)")

    # ── 5. see-less curve, fallback-cleaned (B1/B2 concluded-only) ─────────────
    print("\n-- SEE-LESS curve, concluded-only (fallback removed) --")
    for nm, d in (("B1", b1), ("B2", b2)):
        rows = [(r.get("completion_tokens", 0), g(r))
                for r in d.values() if not unc(r) and r.get("completion_tokens")]
        rows.sort()
        n = len(rows)
        contaminated = sum(1 for r in d.values() if unc(r))
        print(f"  {nm} (concluded n={n}; {contaminated} fallback records excluded):")
        for qi in range(4):
            chunk = rows[qi * n // 4:(qi + 1) * n // 4]
            if chunk:
                a = sum(1 for _, ok in chunk if ok) / len(chunk) * 100
                print(f"    Q{qi+1} tok[{chunk[0][0]:>6}-{chunk[-1][0]:>6}] "
                      f"acc={a:5.1f}% (n={len(chunk)})")

    print("\n" + "=" * 84)
    print("READING IT: 'ns' = delta not distinguishable from 0 at this N. A delta whose")
    print("bootstrap CI spans 0, or that is smaller than the standard 3-pass range, is")
    print("NOT a true representative — report it as null / within-noise.")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()
