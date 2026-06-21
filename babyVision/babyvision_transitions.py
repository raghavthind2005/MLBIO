#!/usr/bin/env python3
"""BabyVision TRANSITION-STRUCTURE test — are the flips a TREND, or memoryless noise?

Churn showed cross-condition flips happen no more than re-sampling does, so the open
question is whether a flip is PREDICTABLE from the draw itself, or just an i.i.d.
coin-flip at the item's P(correct). We test it WITHIN each item (item held constant,
so difficulty is fully controlled), comparing its right-draws vs its wrong-draws on:

  1. LENGTH      — are wrong draws longer? (did the model talk itself out of it)
  2. CONFIDENCE  — are wrong draws less confident (entropy↑, logprob↓), or equally
                   confident (→ calibration failure, no internal tell)?
  3. ANSWER (choice) — do wrong draws converge on ONE distractor (structured
                   perceptual confusion) or scatter (noise)?

Split by difficulty band so we can look at the user's case: EASY items (high P) that
sometimes go wrong. Decisive read at the bottom. Stdlib, login-node safe.
"""

import json
import statistics as st
from collections import Counter
from pathlib import Path

BASE = Path("/iopsstor/scratch/cscs/raghavthind/babyvision")
DRAWS = [("results_a0_nothink", 1), ("results_standard", 1), ("results_standard", 2),
         ("results_standard", 3), ("results_a3_forced_long", 1),
         ("results_b1_reinject", 1), ("results_b2_noreinject", 1)]


def load(d, p):
    f = BASE / d / f"results_run{p}_graded.jsonl"
    if not f.exists():
        return {}
    return {json.loads(l)["taskId"]: json.loads(l)
            for l in open(f) if l.strip() and "error" not in json.loads(l)}


def g(r):
    return r.get("grade") is True


def within_item(items, field, lower_is_confident=False):
    """Per boundary item, mean(field|wrong) - mean(field|right). Returns mean delta
    and fraction of items where wrong-draws have the higher value."""
    deltas, frac_wrong_higher = [], 0
    for draws in items:
        rt = [d.get(field) for d in draws if g(d) and d.get(field) is not None]
        wr = [d.get(field) for d in draws if not g(d) and d.get(field) is not None]
        if rt and wr:
            delta = st.mean(wr) - st.mean(rt)
            deltas.append(delta)
            if delta > 0:
                frac_wrong_higher += 1
    if not deltas:
        return None
    return st.mean(deltas), frac_wrong_higher / len(deltas), len(deltas)


def main():
    draws = {f"{d}_{p}": load(d, p) for d, p in DRAWS}
    common = set.intersection(*[set(v) for v in draws.values()])
    byitem = {t: [draws[k][t] for k in draws] for t in common}
    pc = {t: sum(1 for d in byitem[t] if g(d)) / len(byitem[t]) for t in common}
    sub = {t: byitem[t][0].get("subtype") for t in common}
    anst = {t: byitem[t][0].get("ansType") for t in common}

    boundary = [t for t in common if 0 < pc[t] < 1]
    print("\n" + "=" * 74)
    print(f"BabyVision — TRANSITION STRUCTURE  ({len(boundary)} boundary items, 7 draws each)")
    print("=" * 74)

    bands = [("ALL boundary", lambda p: 0 < p < 1),
             ("HARD  (0<P≤.3)", lambda p: 0 < p <= 0.3),
             ("MID   (.3<P<.7)", lambda p: 0.3 < p < 0.7),
             ("EASY  (.7≤P<1)", lambda p: 0.7 <= p < 1)]

    # ── 1+2. length & confidence: within-item wrong vs right ───────────────────
    for fld, label in (("completion_tokens", "LENGTH (tokens)"),
                       ("entropy_mean", "ENTROPY (↑=less confident)"),
                       ("logprob_mean", "LOGPROB (↓=less confident)")):
        print(f"\n-- {label}: within-item  mean(WRONG) − mean(RIGHT) --")
        print(f"   {'band':<16}{'Δ(wrong−right)':>16}{'% items wrong>right':>22}{'n':>6}")
        for name, f in bands:
            items = [byitem[t] for t in boundary if f(pc[t])]
            res = within_item(items, fld)
            if res:
                d, frac, n = res
                print(f"   {name:<16}{d:>16.3f}{frac*100:>21.0f}%{n:>6}")

    # ── 3. choice answer structure: do wrong draws converge on one distractor? ──
    print("\n-- ANSWER STRUCTURE (choice boundary items): are errors systematic? --")
    print(f"   {'band':<16}{'mean #distinct wrong ans':>26}{'modal-wrong share':>20}{'n':>5}")
    for name, f in bands:
        ndist, modal, cnt = [], [], 0
        for t in boundary:
            if anst[t] != "choice" or not f(pc[t]):
                continue
            wrong = [str(d.get("extracted_answer")) for d in byitem[t]
                     if not g(d) and d.get("extracted_answer") is not None]
            if len(wrong) >= 2:
                c = Counter(wrong)
                ndist.append(len(c))
                modal.append(c.most_common(1)[0][1] / len(wrong))
                cnt += 1
        if cnt:
            print(f"   {name:<16}{st.mean(ndist):>26.2f}{st.mean(modal)*100:>19.0f}%{cnt:>5}")
    print("   (modal-share≈100% & #distinct≈1 ⇒ same wrong answer every time = STRUCTURED")
    print("    confusion; #distinct≈3 & low modal-share ⇒ scattered = NOISE. Choice has ~3 distractors.)")

    print("\n" + "-" * 74)
    print("DECISIVE READ: if wrong draws are clearly longer / less confident (% well")
    print("above 50) AND choice errors concentrate on one distractor → flips are")
    print("STRUCTURED, a real trend worth the attention pass. If everything sits ~50%")
    print("and errors scatter → flips are memoryless sampling noise, and this dataset")
    print("does NOT give room to study transitions beyond per-item P(correct).")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
