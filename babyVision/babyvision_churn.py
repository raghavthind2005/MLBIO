#!/usr/bin/env python3
"""BabyVision CHURN test — does reasoning influence outcomes beyond sampling noise?

Reasoning manipulations DO flip individual items (~28% of items change right/wrong
when reasoning changes). The question is whether that churn is a real effect of
reasoning, or just sampling stochasticity that would happen anyway. We have the
control: standard ran 3 passes (identical prompt/condition, different seed) — its
pass-to-pass flip rate is the PURE sampling-noise baseline.

  WITHIN-condition churn (std p1/p2/p3)  = sampling noise floor
  CROSS-condition churn  (reasoning arms) = sampling + any reasoning effect

If CROSS ≈ WITHIN, changing the reasoning is statistically equivalent to re-rolling
the sampler — reasoning adds nothing beyond noise. We also report net directional
flip (b−c) per contrast: balanced ⇒ non-directional (no arm systematically wins).

Reads *_graded.jsonl (stdlib, login-node safe).
"""

import json
import statistics as st
from pathlib import Path

BASE = Path("/iopsstor/scratch/cscs/raghavthind/babyvision")


def load(d, p=1):
    f = BASE / d / f"results_run{p}_graded.jsonl"
    if not f.exists():
        return {}
    return {json.loads(l)["taskId"]: json.loads(l)
            for l in open(f) if l.strip() and "error" not in json.loads(l)}


def g(r):
    return r.get("grade") is True


def churn(d1, d2):
    ids = set(d1) & set(d2)
    b = sum(1 for i in ids if g(d1[i]) and not g(d2[i]))   # d1 right, d2 wrong
    c = sum(1 for i in ids if not g(d1[i]) and g(d2[i]))   # d1 wrong, d2 right
    dis = (b + c) / len(ids) * 100
    return dis, b, c, len(ids)


def main():
    a0 = load("results_a0_nothink")
    s1 = load("results_standard", 1)
    s2 = load("results_standard", 2)
    s3 = load("results_standard", 3)
    a3 = load("results_a3_forced_long")
    b1 = load("results_b1_reinject")
    b2 = load("results_b2_noreinject")

    print("\n" + "=" * 76)
    print("BabyVision — CHURN: reasoning effect vs sampling-noise floor")
    print("=" * 76)

    print("\n-- WITHIN standard (same condition, different seed) = SAMPLING FLOOR --")
    within = []
    for n, (x, y) in (("p1↔p2", (s1, s2)), ("p1↔p3", (s1, s3)), ("p2↔p3", (s2, s3))):
        if x and y:
            dis, b, c, nn = churn(x, y)
            within.append(dis)
            print(f"  std {n}:  disagree={dis:5.1f}%  (b/c={b}/{c}, net={b-c:+d})  N={nn}")
    floor = st.mean(within) if within else float("nan")
    print(f"  → SAMPLING-NOISE FLOOR (mean pairwise disagreement) = {floor:.1f}%")

    print("\n-- CROSS conditions (reasoning manipulation, vs standard p1) --")
    cross = []
    pairs = [("A0  ↔ std", a0, s1), ("A3  ↔ std", a3, s1),
             ("B1  ↔ std", b1, s1), ("B2  ↔ std", b2, s1),
             ("A0  ↔ A3 ", a0, a3), ("B1  ↔ B2 ", b1, b2)]
    for n, x, y in pairs:
        if x and y:
            dis, b, c, nn = churn(x, y)
            cross.append(dis)
            print(f"  {n}:  disagree={dis:5.1f}%  (b/c={b}/{c}, net={b-c:+d})  N={nn}")
    cm = st.mean(cross) if cross else float("nan")
    print(f"  → CROSS-condition mean disagreement = {cm:.1f}%")

    print("\n" + "-" * 76)
    print(f"  sampling floor      : {floor:.1f}%")
    print(f"  cross-cond churn    : {cm:.1f}%")
    print(f"  excess over sampling: {cm-floor:+.1f} pts")
    if abs(cm - floor) < 4:
        print("  ⇒ cross-condition churn ≈ sampling floor: changing the reasoning is")
        print("    equivalent to re-rolling the sampler. Reasoning perturbs answers but")
        print("    adds ~no churn beyond sampling, and the net flips are balanced ⇒")
        print("    reasoning is NON-DIRECTIONAL noise w.r.t. correctness, not a lever.")
    else:
        print("  ⇒ cross-condition churn EXCEEDS sampling floor by a clear margin:")
        print("    reasoning genuinely perturbs outcomes beyond sampling. Soften the")
        print("    claim to 'non-directional but real perturbation' (check net b−c).")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
