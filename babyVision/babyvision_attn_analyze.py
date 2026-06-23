#!/usr/bin/env python3
"""
Analyze b1cot / b2cot attention extraction (from extract_attention_b.py).

Answers, on the turn-2 reasoning:
  1. COVERAGE — how many extracted per condition, family balance, seq range.
  2. RE-INJECTED IMAGE IGNORED? (b1cot only) — paired v0 (original image) vs v1
     (re-injected image) attention. If v1 < v0, the model barely uses the fresh
     image → the attention correlate of the B1'-vs-B2' accuracy null.
  3. DECAY across turn-2 reasoning — image attention early vs late (by-thirds).
     "See less" = attention to the image falls as the reasoning runs on.
  4. PERCEPTION vs REASONING — does image attention (level + decay) differ by family,
     the attention correlate of the +6.8 / -6.6 accuracy dissociation?
  5. CORRECT vs WRONG — do correct turn-2 answers attend more to the image?

Stdlib only. NOTE: this runs on whatever seq<=MAX_SEQ_LEN subset extracted (the
long tail is memory-skipped), so it is biased toward shorter-reasoning items —
state that in any writeup.

  python babyvision_attn_analyze.py --base /iopsstor/.../babyvision
"""

import argparse
import json
from pathlib import Path

try:
    from math import comb
except ImportError:
    from math import factorial
    def comb(n, k):
        return 0 if k < 0 or k > n else factorial(n) // (factorial(k) * factorial(n - k))

PERCEPTION = {"Count 3D blocks", "Count Same Patterns", "Count Clusters", "Maze",
              "Connect the lines", "Metro map", "Lines Observation", "Find the same",
              "Find the different", "Find the shadow"}
REASONING  = {"3D Cube Unfold", "Paper Folding", "3D Views", "Rotation Patterns",
              "Mirroring Patterns", "2D Pattern Completion", "3D Pattern Completion",
              "Logic Patterns", "Overlay Patterns", "Reconstruction",
              "Recognize numbers and letters", "Pattern and Color Completion"}


def family_of(s):
    return "perception" if s in PERCEPTION else "reasoning" if s in REASONING else "other"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def sign_test(pos, neg):
    """Two-sided exact-binomial sign test over discordant pairs (pos vs neg)."""
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def load(path):
    recs = []
    if path.exists():
        for line in open(path):
            if line.strip():
                recs.append(json.loads(line))
    return recs


def coverage(name, recs):
    fams = {}
    for r in recs:
        fams[family_of(r.get("subtype"))] = fams.get(family_of(r.get("subtype")), 0) + 1
    seqs = [r.get("seq_len", 0) for r in recs]
    graded = [r for r in recs if r.get("grade") is not None]
    acc = mean([1.0 if r["grade"] else 0.0 for r in graded]) if graded else float("nan")
    print(f"  {name:<8} n={len(recs):<4} "
          f"perception={fams.get('perception',0):<3} reasoning={fams.get('reasoning',0):<3} "
          f"other={fams.get('other',0):<3} "
          f"seq[min={min(seqs) if seqs else 0},max={max(seqs) if seqs else 0}] "
          f"acc(graded n={len(graded)})={acc*100:.1f}%")


def v0_vs_v1(recs):
    """b1cot: original (turn0) vs re-injected (turn1) image attention, paired."""
    print("\n" + "=" * 72)
    print("2. RE-INJECTED IMAGE IGNORED?  (b1cot: v0=original vs v1=re-injected)")
    print("=" * 72)
    pairs = [(r["attn_visual_turn0_mean"], r["attn_visual_turn1_mean"]) for r in recs
             if r.get("attn_visual_turn0_mean") is not None
             and r.get("attn_visual_turn1_mean") is not None]
    if not pairs:
        print("  (no b1cot records with both visual blocks)"); return
    v0 = mean([a for a, _ in pairs]); v1 = mean([b for _, b in pairs])
    n_v0_gt = sum(1 for a, b in pairs if a > b)
    n_v1_gt = sum(1 for a, b in pairs if b > a)
    p = sign_test(n_v0_gt, n_v1_gt)
    print(f"  n={len(pairs)} paired")
    print(f"  mean attn to ORIGINAL image (v0)     : {v0:.4f}")
    print(f"  mean attn to RE-INJECTED image (v1)  : {v1:.4f}")
    print(f"  mean(v0 - v1)                        : {v0-v1:+.4f}   (v1/v0={v1/v0:.2f})")
    print(f"  samples v0>v1: {n_v0_gt}   v1>v0: {n_v1_gt}   sign-test p={p:.4f}"
          f"{'  *' if p<0.05 else ''}")
    print("  → v1<v0 means the model attends LESS to the freshly re-shown image than to")
    print("    the original — i.e. re-injection is largely ignored (matches B1'≈B2' null).")


def decay(recs, label, key="attn_visual_all_by_thirds"):
    """Early/mid/late image attention across the turn-2 reasoning."""
    e = mean([r[key]["early"] for r in recs if r.get(key)])
    m = mean([r[key]["mid"]   for r in recs if r.get(key)])
    l = mean([r[key]["late"]  for r in recs if r.get(key)])
    n = sum(1 for r in recs if r.get(key))
    drop = (l - e) / e * 100 if e else float("nan")
    print(f"  {label:<26} n={n:<4} early={e:.4f} mid={m:.4f} late={l:.4f}  "
          f"late-vs-early={drop:+.0f}%")


def by_family(recs, label, key="attn_visual_all_mean"):
    for fam in ("perception", "reasoning"):
        fr = [r for r in recs if family_of(r.get("subtype")) == fam]
        print(f"    {label} · {fam:<11} n={len(fr):<4} mean_img_attn={mean([r.get(key) for r in fr]):.4f}")


def correct_vs_wrong(recs, label, key="attn_visual_all_mean"):
    cor = [r.get(key) for r in recs if r.get("grade") is True]
    wro = [r.get(key) for r in recs if r.get("grade") is False]
    if cor or wro:
        print(f"    {label}: correct(n={len(cor)}) img_attn={mean(cor):.4f}   "
              f"wrong(n={len(wro)}) img_attn={mean(wro):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    args = ap.parse_args()
    base = Path(args.base)

    b1 = load(base / "results_b1cot_reinject"   / "attention_b.jsonl")
    b2 = load(base / "results_b2cot_noreinject" / "attention_b.jsonl")

    print("=" * 72)
    print("1. COVERAGE  (seq<=MAX_SEQ_LEN subset; biased toward shorter reasoning)")
    print("=" * 72)
    coverage("b1cot", b1)
    coverage("b2cot", b2)

    v0_vs_v1(b1)

    print("\n" + "=" * 72)
    print("3. IMAGE-ATTENTION DECAY across the turn-2 reasoning (early→late)")
    print("=" * 72)
    decay(b1, "b1cot (visual_all)")
    decay(b1, "b1cot (original v0 only)", key="attn_visual_turn0_by_thirds")
    decay(b2, "b2cot (single image)")

    print("\n" + "=" * 72)
    print("4. PERCEPTION vs REASONING — image-attention level + decay")
    print("=" * 72)
    print("  -- mean image attention by family --")
    by_family(b1, "b1cot"); by_family(b2, "b2cot")
    print("  -- decay by family (b1cot visual_all) --")
    for fam in ("perception", "reasoning"):
        decay([r for r in b1 if family_of(r.get("subtype")) == fam], f"b1cot {fam}")
    print("  -- decay by family (b2cot single image) --")
    for fam in ("perception", "reasoning"):
        decay([r for r in b2 if family_of(r.get("subtype")) == fam], f"b2cot {fam}")

    print("\n" + "=" * 72)
    print("5. CORRECT vs WRONG turn-2 answers — image attention")
    print("=" * 72)
    correct_vs_wrong(b1, "b1cot"); correct_vs_wrong(b2, "b2cot")

    print("\n" + "=" * 72)
    print("6. b1cot vs b2cot — image attention (note: b1cot visual_all sums 2 blocks)")
    print("=" * 72)
    print(f"  b1cot mean visual_all = {mean([r.get('attn_visual_all_mean') for r in b1]):.4f}"
          f"   (v0 only = {mean([r.get('attn_visual_turn0_mean') for r in b1]):.4f})")
    print(f"  b2cot mean visual_all = {mean([r.get('attn_visual_all_mean') for r in b2]):.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
