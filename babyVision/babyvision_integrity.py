#!/usr/bin/env python3
"""BabyVision DATA-INTEGRITY pass — prove the flat-accuracy result is real, not a bug.

"All conditions score ~the same" is suspicious until we rule out the boring
explanations: misaligned items, a manipulation that never took effect, a grading
collapse, or conditions that are secretly identical. This script proves, from the
graded jsonl alone (stdlib, login-node safe):

  1. ALIGNMENT      — every condition covers the identical taskId set, and each
                      item's gold answer + question is identical across conditions.
  2. SIGNATURES     — each condition's distinctive fields are present (A0 short /
                      no-think, A3 forced with n_forces, B1 two image-passes,
                      B2 zero T2 image) — i.e. the manipulation actually happened.
  3. LENGTH         — generation length genuinely differs across conditions
                      (if A0 and A3 had the same length, *that* would be the bug).
  4. CORRECTNESS    — cross-condition right/wrong structure: are the SAME items
                      right/wrong regardless of condition? (flat acc = item-intrinsic
                      difficulty, a mechanism) vs noise that happens to net equal.
  5. ANSWERS+GRADE  — do conditions produce different answers per item; is `grade`
                      actually varying (not a constant default); spot-check disagreements.
"""

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

BASE = Path("/iopsstor/scratch/cscs/raghavthind/babyvision")

# (key, dir, pass) — the five single-pass arms compared per-item; standard uses pass1
ARMS = [
    ("a0",  "results_a0_nothink",     1),
    ("std", "results_standard",       1),
    ("a3",  "results_a3_forced_long", 1),
    ("b1",  "results_b1_reinject",    1),
    ("b2",  "results_b2_noreinject",  1),
]


def load(d, p=1):
    f = BASE / d / f"results_run{p}_graded.jsonl"
    if not f.exists():
        return {}
    out = {}
    for l in open(f):
        if not l.strip():
            continue
        r = json.loads(l)
        if "error" not in r:
            out[r["taskId"]] = r
    return out


def g(r):
    return r.get("grade") is True


def main():
    D = {k: load(d, p) for k, d, p in ARMS}

    print("\n" + "=" * 84)
    print("BabyVision — DATA-INTEGRITY")
    print("=" * 84)

    # ── 1. ALIGNMENT ───────────────────────────────────────────────────────────
    print("\n-- 1. ALIGNMENT (identical items + identical gold across conditions) --")
    sets = {k: set(v) for k, v in D.items()}
    sizes = {k: len(v) for k, v in sets.items()}
    print("  taskId counts:", sizes)
    common = set.intersection(*sets.values())
    union = set.union(*sets.values())
    print(f"  common to all: {len(common)}   union: {len(union)}   "
          f"{'IDENTICAL ✓' if len(common)==len(union) else 'MISMATCH ✗'}")
    for k in D:
        extra = sets[k] - common
        if extra:
            print(f"    {k} has {len(extra)} not-in-all: {sorted(extra)[:10]}")

    gt_mismatch = q_mismatch = 0
    ref = "std" if "std" in D else list(D)[0]
    for tid in common:
        gts = {D[k][tid].get("gt_answer") for k in D}
        qs  = {D[k][tid].get("question_sent") for k in D}
        if len(gts) > 1:
            gt_mismatch += 1
        if len(qs) > 1:
            q_mismatch += 1
    print(f"  gold-answer mismatches across conditions: {gt_mismatch}/{len(common)} "
          f"{'✓' if gt_mismatch==0 else '✗ ALIGNMENT BUG'}")
    print(f"  question mismatches across conditions:    {q_mismatch}/{len(common)} "
          f"(note: B/A3 prompts may legitimately differ in suffix)")

    # ── 2. SIGNATURES (manipulation actually applied) ────────────────────────────
    print("\n-- 2. CONDITION SIGNATURES (proves each arm is what it claims) --")
    for k in D:
        recs = list(D[k].values())
        cond_lbls = {r.get("condition") for r in recs}
        sample = recs[0]
        sig = {fld: sample.get(fld) for fld in
               ("condition", "reinject_image", "n_image_tokens_turn2", "n_forces")}
        n_forced = sum(1 for r in recs if (r.get("n_forces") or 0) > 0)
        print(f"  {k:<4} condition-label={cond_lbls}  "
              f"reinject={sig['reinject_image']}  T2_img_tok={sig['n_image_tokens_turn2']}  "
              f"n_forces(sample)={sig['n_forces']}  arms-with-forces={n_forced}")

    # ── 3. LENGTH (generation genuinely differs) ─────────────────────────────────
    print("\n-- 3. REASONING LENGTH per condition (manipulation effect on generation) --")
    print(f"  {'cond':<5}{'n':>5}{'median':>9}{'mean':>9}{'min':>7}{'max':>8}")
    for k in D:
        lens = [r.get("completion_tokens") for r in D[k].values()
                if r.get("completion_tokens") is not None]
        if lens:
            print(f"  {k:<5}{len(lens):>5}{st.median(lens):>9.0f}{st.mean(lens):>9.0f}"
                  f"{min(lens):>7}{max(lens):>8}")
    print("  → if these are wildly different (they should be: A0≪std≪A3) yet accuracy")
    print("    is flat, the null is REAL, not 'the conditions are secretly identical'.")

    # ── 4. CORRECTNESS STRUCTURE (why is accuracy flat?) ─────────────────────────
    print("\n-- 4. CROSS-CONDITION CORRECTNESS (are the SAME items right/wrong?) --")
    ks = list(D)
    # per-item: how many of the 5 conditions get it right
    n_right = {tid: sum(1 for k in ks if g(D[k][tid])) for tid in common}
    hist = defaultdict(int)
    for tid in common:
        hist[n_right[tid]] += 1
    print(f"  of {len(common)} items, # conditions (out of {len(ks)}) that get it right:")
    for nr in range(len(ks) + 1):
        bar = "#" * round(hist[nr] / max(1, len(common)) * 50)
        print(f"    {nr}/{len(ks)} right: {hist[nr]:>4}  {bar}")
    always_right = hist[len(ks)]
    always_wrong = hist[0]
    item_intrinsic = (always_right + always_wrong) / len(common) * 100
    print(f"  ALWAYS right: {always_right}   ALWAYS wrong: {always_wrong}   "
          f"→ {item_intrinsic:.1f}% of items are condition-invariant")
    print("  (high % ⇒ correctness is item-intrinsic; condition barely matters ⇒ flat acc is the mechanism)")

    # pairwise correctness agreement
    print("\n  pairwise correctness agreement (% items both conditions agree right-or-wrong):")
    print("       " + "".join(f"{k:>6}" for k in ks))
    for k1 in ks:
        row = ""
        for k2 in ks:
            ag = sum(1 for t in common if g(D[k1][t]) == g(D[k2][t])) / len(common) * 100
            row += f"{ag:>6.0f}"
        print(f"  {k1:<5}{row}")

    # ── 5. ANSWERS + GRADE sanity ────────────────────────────────────────────────
    print("\n-- 5. ANSWER variation + GRADE sanity --")
    # do conditions give different extracted answers for the same item?
    diff_ans = sum(1 for t in common
                   if len({str(D[k][t].get("extracted_answer")) for k in ks}) > 1)
    print(f"  items where conditions give ≠ extracted answers: {diff_ans}/{len(common)} "
          f"({diff_ans/len(common)*100:.0f}%) — if ~0, model ignores the manipulation entirely")
    for k in D:
        recs = list(D[k].values())
        tt = sum(1 for r in recs if r.get("grade") is True)
        ff = sum(1 for r in recs if r.get("grade") is False)
        nn = sum(1 for r in recs if r.get("grade") is None)
        print(f"  {k:<4} grade  True={tt:<4} False={ff:<4} None={nn:<3} "
              f"{'✗ None present!' if nn else ''}{'✗ constant!' if (tt==0 or ff==0) else ''}")

    # spot-check: choice items where deterministic letter and judge disagree
    import re
    def letter(s):
        if s is None: return None
        m = re.findall(r"\(([A-Ha-h])\)", str(s)) or re.findall(r"\b([A-Ha-h])\b", str(s))
        return m[-1].upper() if m else None
    print("\n  choice judge-vs-deterministic disagreements (up to 3/cond — inspect for judge error):")
    for k in D:
        ch = [r for r in D[k].values() if r.get("ansType") == "choice"]
        shown = 0
        for r in ch:
            det = letter(r.get("extracted_answer"))
            gl = letter(r.get("gt_answer"))
            det_ok = det is not None and det == gl
            if det_ok != g(r):
                print(f"    {k} id={r['taskId']} gt={gl} extracted={det} "
                      f"det_correct={det_ok} judge_grade={g(r)} "
                      f"raw={str(r.get('grade_raw'))[:14]!r}")
                shown += 1
                if shown >= 3:
                    break

    print("\n" + "=" * 84)
    print("VERDICT GUIDE: alignment ✓ + signatures distinct + lengths wildly different +")
    print("high condition-invariance ⇒ flat accuracy is a REAL item-intrinsic null, not a")
    print("data bug. Any ✗ above (gold mismatch, constant grade, None grades) = stop & fix.")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()
