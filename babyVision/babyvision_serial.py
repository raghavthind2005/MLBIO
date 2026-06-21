#!/usr/bin/env python3
"""BabyVision SERIAL-vs-HOLISTIC test — harden the perception-deficit claim.

Hypothesis: Gemma-4 has holistic/parallel ("gist") vision but lacks serial/attentive
vision. Tasks needing step-by-step tracing / counting / element-by-element comparison
(SERIAL) should have far lower per-item P(correct) than single-glance recognition
(HOLISTIC).

P(correct) per item = fraction correct over all 7 condition-draws (licensed by the
churn result: cross-condition draws ≈ resamples). We test the grouping with:
  - group meanP / median / stable-wrong fraction
  - PERMUTATION test on the group-mean P difference (assumption-free, handles ties)
  - separation AUC = P(random holistic item scores > random serial item) + rank-biserial
  - OVERLAP (serial successes, holistic failures)
  - SENSITIVITY: drop borderline subtypes, re-test on high-confidence cases only.

Edit SERIAL / HOLISTIC / BORDERLINE below to change the grouping. Stdlib only.
"""

import json
import random
import statistics as st
from pathlib import Path

random.seed(0)
BASE = Path("/iopsstor/scratch/cscs/raghavthind/babyvision")

DRAWS = [("results_a0_nothink", 1), ("results_standard", 1), ("results_standard", 2),
         ("results_standard", 3), ("results_a3_forced_long", 1),
         ("results_b1_reinject", 1), ("results_b2_noreinject", 1)]

SERIAL = {
    "Maze", "Connect the lines", "Metro map", "Lines Observation",
    "Find the same", "Find the different", "Find the shadow",
    "Count 3D blocks", "Count Same Patterns", "Paper Folding", "3D Cube Unfold",
}
HOLISTIC = {
    "Rotation Patterns", "Recognize numbers and letters", "Overlay Patterns",
    "3D Views", "2D Pattern Completion", "3D Pattern Completion",
    "Mirroring Patterns", "Logic Patterns", "Reconstruction",
    "Count Clusters", "Pattern and Color Completion",
}
# subtypes whose serial/holistic membership is debatable → dropped in sensitivity run
BORDERLINE = {
    "Count Clusters", "3D Views", "Pattern and Color Completion",
    "2D Pattern Completion", "3D Pattern Completion",
    "Count Same Patterns", "Paper Folding", "3D Cube Unfold",
}


def load(d, p):
    f = BASE / d / f"results_run{p}_graded.jsonl"
    if not f.exists():
        return {}
    return {json.loads(l)["taskId"]: json.loads(l)
            for l in open(f) if l.strip() and "error" not in json.loads(l)}


def perm_test(xs, ys, reps=20000):
    """Two-sided permutation test on mean(ys) - mean(xs)."""
    obs = st.mean(ys) - st.mean(xs)
    pool = xs + ys
    n = len(xs)
    cnt = 0
    for _ in range(reps):
        random.shuffle(pool)
        d = st.mean(pool[n:]) - st.mean(pool[:n])
        if abs(d) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (reps + 1)


def auc(serial_ps, holistic_ps):
    """P(random holistic > random serial); ties count 0.5. = Mann-Whitney U / n1 n2."""
    wins = 0.0
    for h in holistic_ps:
        for s in serial_ps:
            wins += 1.0 if h > s else 0.5 if h == s else 0.0
    return wins / (len(serial_ps) * len(holistic_ps))


def report(name, pc, sub, serial_set, holistic_set):
    ser = [pc[t] for t in pc if sub[t] in serial_set]
    hol = [pc[t] for t in pc if sub[t] in holistic_set]
    if not ser or not hol:
        print(f"  {name}: empty group"); return
    obs, p = perm_test(ser, hol)
    a = auc(ser, hol)
    print(f"\n  [{name}]  serial n={len(ser)} meanP={st.mean(ser)*100:5.1f}%  "
          f"holistic n={len(hol)} meanP={st.mean(hol)*100:5.1f}%")
    print(f"     gap = {obs*100:+.1f} pts   permutation p = {p:.4f}   "
          f"AUC(separation) = {a:.3f}  (rank-biserial = {2*a-1:+.3f})")
    print(f"     overlap: serial with P>0.5 = {sum(1 for x in ser if x>0.5)}/{len(ser)}   "
          f"holistic with P<0.2 = {sum(1 for x in hol if x<0.2)}/{len(hol)}")


def main():
    draws = {f"{d}_{p}": load(d, p) for d, p in DRAWS}
    common = set.intersection(*[set(v) for v in draws.values()])
    any_d = next(iter(draws.values()))
    sub = {t: any_d[t]["subtype"] for t in common}
    pc = {t: sum(1 for v in draws.values() if v[t].get("grade") is True) / len(draws)
          for t in common}

    print("\n" + "=" * 76)
    print(f"BabyVision — SERIAL vs HOLISTIC  ({len(common)} items, P over {len(draws)} draws)")
    print("=" * 76)

    # subtype table sorted by meanP, with group label — visual confirmation
    print("\n-- subtype solvability with group label (sorted) --")
    bysub = {}
    for t in common:
        bysub.setdefault(sub[t], []).append(pc[t])
    print(f"  {'subtype':<32}{'grp':>5}{'n':>4}{'meanP':>8}")
    for s, ps in sorted(bysub.items(), key=lambda kv: st.mean(kv[1])):
        grp = "SER" if s in SERIAL else "HOL" if s in HOLISTIC else "?"
        print(f"  {s[:32]:<32}{grp:>5}{len(ps):>4}{st.mean(ps)*100:>7.1f}")

    # full grouping
    report("ALL subtypes", pc, sub, SERIAL, HOLISTIC)
    # sensitivity: high-confidence only (drop borderline)
    report("high-confidence (borderline dropped)", pc, sub,
           SERIAL - BORDERLINE, HOLISTIC - BORDERLINE)

    print("\n" + "-" * 76)
    print("  CLAIM holds if: large positive gap, permutation p<0.01, AUC≳0.75 (clear")
    print("  separation), little overlap — AND it survives the borderline-dropped run.")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
