#!/usr/bin/env python3
"""BabyVision FLIP / STABILITY decomposition — find the narrowable research problem.

Churn proved cross-condition draws flip no more than re-samples, so we treat all 7
condition-runs of an item (a0, std×3, a3, b1, b2) as 7 SAMPLES of the model's
perception of that item. Each item then has an empirical P(correct), and the four
behavioural classes are:

    stable-correct   P≈1   reliably perceptible
    stable-wrong     P≈0   categorically invisible to the model
    boundary       0<P<1   unstable percept (the wrong↔right flippers)

PART A — ITEM LEVEL (what sets P(correct)?): population sizes, the per-subtype
         "solvability spectrum", and the subtype composition of the stable-wrong
         vs stable-correct sets. Tells us if the failure is narrowable (clusters
         in specific perceptual primitives) or diffuse.

PART B — DRAW LEVEL (what tips a boundary draw?): within standard's 3 same-seed
         passes, contrast right-draws vs wrong-draws of the SAME item on the
         internal signals we already capture (answer confidence / entropy / length).
         A preview of the within-item signature the attention pass will localize.

Reads *_graded.jsonl (stdlib, login-node safe).
"""

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

BASE = Path("/iopsstor/scratch/cscs/raghavthind/babyvision")

DRAWS = [  # (label, dir, pass) — 7 samples of each item's perception
    ("a0",  "results_a0_nothink",     1),
    ("s1",  "results_standard",       1),
    ("s2",  "results_standard",       2),
    ("s3",  "results_standard",       3),
    ("a3",  "results_a3_forced_long", 1),
    ("b1",  "results_b1_reinject",    1),
    ("b2",  "results_b2_noreinject",  1),
]


def load(d, p):
    f = BASE / d / f"results_run{p}_graded.jsonl"
    if not f.exists():
        return {}
    return {json.loads(l)["taskId"]: json.loads(l)
            for l in open(f) if l.strip() and "error" not in json.loads(l)}


def g(r):
    return r.get("grade") is True


def main():
    draws = {lbl: load(d, p) for lbl, d, p in DRAWS}
    # items present in all draws
    common = set.intersection(*[set(v) for v in draws.values()])
    meta = {tid: draws["s1"][tid] for tid in common}  # type/subtype/ansType reference

    # per-item correctness across the 7 draws
    pc = {}
    for tid in common:
        grades = [g(draws[lbl][tid]) for lbl in draws]
        pc[tid] = sum(grades) / len(grades)

    print("\n" + "=" * 80)
    print(f"BabyVision — FLIP / STABILITY decomposition  ({len(common)} items × {len(draws)} draws)")
    print("=" * 80)

    # ── PART A.1 — populations ──────────────────────────────────────────────────
    stable_wrong = [t for t in common if pc[t] == 0]
    stable_right = [t for t in common if pc[t] == 1]
    boundary     = [t for t in common if 0 < pc[t] < 1]
    print("\n-- A.1 POPULATIONS (P(correct) over 7 draws) --")
    print(f"  stable-correct (P=1.0): {len(stable_right):>4}  ({len(stable_right)/len(common)*100:4.1f}%)")
    print(f"  stable-wrong   (P=0.0): {len(stable_wrong):>4}  ({len(stable_wrong)/len(common)*100:4.1f}%)")
    print(f"  boundary     (0<P<1.0): {len(boundary):>4}  ({len(boundary)/len(common)*100:4.1f}%)")

    # ── PART A.2 — solvability spectrum by subtype ──────────────────────────────
    by_sub = defaultdict(list)
    for t in common:
        by_sub[meta[t]["subtype"]].append(pc[t])
    print("\n-- A.2 SOLVABILITY SPECTRUM by subtype (mean P(correct), sorted) --")
    print(f"  {'subtype':<32}{'n':>4}{'meanP':>8}{'%stbWrong':>11}{'%stbRight':>11}")
    rows = []
    for sub, ps in by_sub.items():
        n = len(ps)
        rows.append((st.mean(ps), sub, n,
                     sum(1 for x in ps if x == 0) / n * 100,
                     sum(1 for x in ps if x == 1) / n * 100))
    for mp, sub, n, sw, sr in sorted(rows):
        print(f"  {sub[:32]:<32}{n:>4}{mp*100:>7.1f}{sw:>11.0f}{sr:>11.0f}")

    # ── PART A.3 — what dominates the stable-wrong / stable-right sets ──────────
    def compose(ids, key):
        c = defaultdict(int)
        for t in ids:
            c[meta[t][key]] += 1
        return sorted(c.items(), key=lambda x: -x[1])
    print("\n-- A.3 COMPOSITION of stable-WRONG (categorically invisible) --")
    print("   by type:   " + ", ".join(f"{k}={v}" for k, v in compose(stable_wrong, "type")))
    print("   top subtypes: " + ", ".join(f"{k}={v}" for k, v in compose(stable_wrong, "subtype")[:8]))
    print("-- COMPOSITION of stable-CORRECT (reliably perceptible) --")
    print("   by type:   " + ", ".join(f"{k}={v}" for k, v in compose(stable_right, "type")))
    print("   top subtypes: " + ", ".join(f"{k}={v}" for k, v in compose(stable_right, "subtype")[:8]))

    # ── PART A.4 — choice vs blank ──────────────────────────────────────────────
    for at in ("choice", "blank"):
        ids = [t for t in common if meta[t]["ansType"] == at]
        if ids:
            print(f"\n  ansType={at:<7} n={len(ids):<4} meanP={st.mean([pc[t] for t in ids])*100:.1f}  "
                  f"stable-wrong={sum(1 for t in ids if pc[t]==0)}  "
                  f"stable-right={sum(1 for t in ids if pc[t]==1)}")

    # ── PART B — within-item draw signature (standard 3-pass, same seed) ────────
    s = {lbl: draws[lbl] for lbl in ("s1", "s2", "s3")}
    sc = set.intersection(*[set(v) for v in s.values()])
    # boundary within standard = flipped across the 3 passes
    flip = [t for t in sc if 0 < sum(g(s[lbl][t]) for lbl in s) < 3]
    print("\n" + "-" * 80)
    print(f"-- B. DRAW SIGNATURE: right vs wrong draws of the SAME item "
          f"(standard 3-pass; {len(flip)} flip items) --")

    fields = ["logprob_mean", "entropy_mean", "logprob_min", "completion_tokens"]
    present = [f for f in fields
               if any(s[lbl][t].get(f) is not None for lbl in s for t in flip)]
    print(f"  available signals: {present or '(none captured)'}")

    for fld in present:
        deltas, pos = [], 0
        for t in flip:
            right = [s[lbl][t].get(fld) for lbl in s if g(s[lbl][t]) and s[lbl][t].get(fld) is not None]
            wrong = [s[lbl][t].get(fld) for lbl in s if not g(s[lbl][t]) and s[lbl][t].get(fld) is not None]
            if right and wrong:
                d = st.mean(right) - st.mean(wrong)
                deltas.append(d)
                if d > 0:
                    pos += 1
        if deltas:
            print(f"  {fld:<18} mean(right−wrong)={st.mean(deltas):+10.4f}  "
                  f"items right>wrong: {pos}/{len(deltas)} ({pos/len(deltas)*100:.0f}%)")

    print("\n  reading B: a signal whose right−wrong delta is consistently signed (≫50% of")
    print("  items one way) is an internal signature of a correct draw — the thing the")
    print("  attention pass would try to localize spatially. Flat (~50%) ⇒ no internal")
    print("  tell from these signals ⇒ need attention / richer probes to explain the flip.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
