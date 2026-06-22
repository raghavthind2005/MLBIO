#!/usr/bin/env python3
"""
BabyVision three-way readout: standard vs A0 (no-think) vs A3 (forced-long).

Reasoning-length axis:  A0 (~0 reasoning) → standard (natural) → A3 (forced-long).
Tests the "think-longer-see-less" hypothesis on vision-primitive tasks.

Reads *_graded.jsonl from each condition dir (stdlib only — runs on the login node).
Correctness comes from the `grade` field written by grade.py (the once-and-for-all,
format-agnostic grader), NOT the old `judge_result`:
  standard:  results_run1_graded.jsonl, run2, run3   (3-pass mean±std)
  a0:        results_run1_graded.jsonl                (1 pass)
  a3:        results_run1_graded.jsonl                (1 pass)

Outputs: console tables + a machine-readable summary.json + a markdown readout.md.

Usage:
  python analyze_conditions.py \
    --base /iopsstor/scratch/cscs/raghavthind/babyvision \
    --out  /iopsstor/scratch/cscs/raghavthind/babyvision/analysis
"""

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

CONDITIONS = [  # (key, dir, passes, label)
    ("a0",       "results_a0_nothink",      [1],         "A0 no-think"),
    ("standard", "results_standard",        [1, 2, 3],   "standard"),
    ("a3",       "results_a3_forced_long",  [1],         "A3 forced-long"),
    ("b1",       "results_b1_reinject",     [1],         "B1 reinject"),
    ("b2",       "results_b2_noreinject",   [1],         "B2 no-reinject"),
    ("b1cot",    "results_b1cot_reinject",  [1],         "B1' reinj+CoT"),
    ("b2cot",    "results_b2cot_noreinject",[1],         "B2' noreinj+CoT"),
]
ORDER = ["a0", "standard", "a3", "b2", "b1", "b1cot"]   # reasoning-length / re-grounding axis
# (b2cot kept out of ORDER so the wide tables don't carry an empty column until B2' runs;
#  it is still loaded and used in the dedicated b1cot_report paired contrast when present.)

# A-priori perception-vs-reasoning grouping for the B1' prediction test. NOT the same as
# the serial/holistic split: our traces show 3D Cube Unfold / Paper Folding flip on
# REASONING (same figure read, different mental fold) while counting flips on PERCEPTION
# (the counts themselves change). Debatable at the margins — read the per-subtype Δ too.
PERCEPTION_TASKS = {"Count 3D blocks", "Count Same Patterns", "Count Clusters", "Maze",
                    "Connect the lines", "Metro map", "Lines Observation", "Find the same",
                    "Find the different", "Find the shadow"}
REASONING_TASKS  = {"3D Cube Unfold", "Paper Folding", "3D Views", "Rotation Patterns",
                    "Mirroring Patterns", "2D Pattern Completion", "3D Pattern Completion",
                    "Logic Patterns", "Overlay Patterns", "Reconstruction",
                    "Recognize numbers and letters", "Pattern and Color Completion"}


# ── Loading ──────────────────────────────────────────────────────────────────

def load_pass(cond_dir: Path, pass_idx: int) -> list:
    p = cond_dir / f"results_run{pass_idx}_graded.jsonl"
    if not p.exists():
        print(f"  WARN: missing {p}")
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def valid(recs: list) -> list:
    """Records that actually ran (drop inference errors)."""
    return [r for r in recs if "error" not in r]


# ── Metrics ──────────────────────────────────────────────────────────────────

def acc(recs: list):
    """Accuracy over judged records. Returns (acc, n_correct, n_total)."""
    v = valid(recs)
    if not v:
        return None, 0, 0
    nc = sum(1 for r in v if r.get("grade") is True)
    return nc / len(v), nc, len(v)


def acc_by_key(recs: list, key: str) -> dict:
    g = defaultdict(list)
    for r in valid(recs):
        g[r.get(key)].append(r)
    return {k: acc(v) for k, v in g.items()}


def length_metric(r: dict):
    """Primary comparable length axis across all conditions: total generated tokens."""
    return r.get("completion_tokens")


def fmt_pct(a):
    return "  n/a " if a is None else f"{a*100:5.1f}"


# ── Condition-level summary ────────────────────────────────────────────────────

def summarize_condition(key, cond_dir, passes, label):
    per_pass = {pi: valid(load_pass(cond_dir, pi)) for pi in passes}
    # pool all passes for breakdowns; per-pass accuracy for mean±std
    pooled = [r for pi in passes for r in per_pass[pi]]
    pass_accs = [acc(per_pass[pi])[0] for pi in passes if per_pass[pi]]
    pass_accs = [a for a in pass_accs if a is not None]

    overall = st.mean(pass_accs) if pass_accs else None
    overall_std = st.pstdev(pass_accs) if len(pass_accs) > 1 else 0.0

    # length / reasoning stats
    lens = [length_metric(r) for r in pooled if length_metric(r) is not None]
    no_answer = sum(1 for r in pooled if r.get("extracted_answer") is None)
    truncated = sum(1 for r in pooled if r.get("finish_reason") == "length")

    # A3-specific
    forces = [r.get("n_forces") for r in pooled if r.get("n_forces") is not None]

    return {
        "key": key, "label": label, "passes": passes,
        "n_pooled": len(pooled),
        "overall_acc": overall, "overall_std": overall_std,
        "pass_accs": pass_accs,
        "len_mean": st.mean(lens) if lens else None,
        "len_median": st.median(lens) if lens else None,
        "no_answer": no_answer, "truncated": truncated,
        "forces_mean": st.mean(forces) if forces else None,
        "forces_nonzero": sum(1 for f in forces if f and f > 0) if forces else None,
        "by_type":    acc_by_key(pooled, "type"),
        "by_subtype": acc_by_key(pooled, "subtype"),
        "by_anstype": acc_by_key(pooled, "ansType"),
        "pooled": pooled,
    }


# ── Within-condition accuracy vs reasoning length (the see-less curve) ─────────

def acc_vs_length(pooled: list, n_bins: int = 4):
    rows = [(length_metric(r), r.get("grade") is True)
            for r in pooled if length_metric(r) is not None]
    if len(rows) < n_bins:
        return []
    rows.sort(key=lambda x: x[0])
    out, n = [], len(rows)
    for b in range(n_bins):
        chunk = rows[b * n // n_bins:(b + 1) * n // n_bins]
        if not chunk:
            continue
        lo, hi = chunk[0][0], chunk[-1][0]
        a = sum(1 for _, c in chunk if c) / len(chunk)
        out.append((b + 1, lo, hi, a, len(chunk)))
    return out


# ── Cross-pass instability (standard only) ──────────────────────────────────────

def flip_stats(cond_dir, passes):
    if len(passes) < 2:
        return None
    by_task = defaultdict(list)
    for pi in passes:
        for r in valid(load_pass(cond_dir, pi)):
            by_task[r["taskId"]].append(r.get("grade") is True)
    full = [v for v in by_task.values() if len(v) == len(passes)]
    if not full:
        return None
    unstable = sum(1 for v in full if len(set(v)) > 1)   # disagreement across passes
    allright = sum(1 for v in full if all(v))
    allwrong = sum(1 for v in full if not any(v))
    return {"n": len(full), "unstable": unstable,
            "unstable_frac": unstable / len(full),
            "all_correct": allright, "all_wrong": allwrong}


# ── B-family: two-turn integrity & re-grounding contrast ───────────────────────

def _t2_unconcluded(r: dict) -> bool:
    """Turn-2 ran out of budget while still thinking (channel never closed) → no
    real 2-turn answer; grade.py graded this record on its standing turn-1 answer."""
    return (r.get("finish_reason") == "length"
            and not (r.get("thinking_trace") or "").strip())


def b_family_report(summ: dict):
    """Three honest views of the B1/B2 result (see babyvision_b_truncation_decision):
      1. runaway-loop truncation rates  (behavioral finding; the asymmetry)
      2. paired concluded-only contrast (clean 2-turn, both conditions real)
      3. full-388 T1-fallback vs concluded-only  (sensitivity — verdict must agree)
    """
    b1p, b2p = summ.get("b1", {}).get("pooled"), summ.get("b2", {}).get("pooled")
    if not b1p or not b2p:
        return
    b1 = {r["taskId"]: r for r in b1p}
    b2 = {r["taskId"]: r for r in b2p}

    print("\n" + "-" * 78)
    print("B-FAMILY · two-turn integrity & re-grounding contrast (B1 reinject vs B2)")
    print("-" * 78)

    # 1. truncation / runaway-loop rates (behavioral finding)
    print("turn-2 runaway-loop truncation (hit budget mid-thinking → graded on standing T1):")
    rates = {}
    for key, d in (("b1", b1), ("b2", b2)):
        u = sum(1 for r in d.values() if _t2_unconcluded(r))
        rates[key] = (u, len(d))
        print(f"  {summ[key]['label']:<16}{u:>4}/{len(d):<4} = {u/len(d)*100:4.1f}%")
    if rates["b1"][1] and rates["b2"][1]:
        d = (rates["b1"][0]/rates["b1"][1] - rates["b2"][0]/rates["b2"][1]) * 100
        print(f"  → re-grounding induces {d:+.1f} pts more runaway loops "
              f"(behavioral finding — do NOT fold into accuracy)")

    # 2. paired concluded-only contrast (clean 2-turn on identical items)
    common = set(b1) & set(b2)
    concl = [i for i in common if not _t2_unconcluded(b1[i]) and not _t2_unconcluded(b2[i])]
    print(f"\npaired concluded-only contrast (both B1 & B2 produced a real 2-turn answer):")
    if concl:
        b1a = sum(1 for i in concl if b1[i].get("grade") is True) / len(concl)
        b2a = sum(1 for i in concl if b2[i].get("grade") is True) / len(concl)
        # McNemar discordant pairs
        b1r_b2w = sum(1 for i in concl if b1[i].get("grade") is True and b2[i].get("grade") is not True)
        b1w_b2r = sum(1 for i in concl if b1[i].get("grade") is not True and b2[i].get("grade") is True)
        print(f"  N = {len(concl)}   (dropped {len(common)-len(concl)} of {len(common)} as loop-truncated)")
        print(f"  B1 = {b1a*100:5.1f}%   B2 = {b2a*100:5.1f}%   Δ(B1−B2) = {(b1a-b2a)*100:+.1f} pts")
        print(f"  discordant pairs: B1-right/B2-wrong = {b1r_b2w}   B1-wrong/B2-right = {b1w_b2r}")
        print(f"  NOTE: the {len(common)-len(concl)} dropped items are where re-grounding most "
              f"changes behavior; read with the loop-rate finding above.")
    else:
        print("  (no paired concluded items)")

    # 3. sensitivity: full-388 T1-fallback (main table) vs concluded-only
    print(f"\nsensitivity — full set (T1-fallback) vs concluded-only:")
    for key, d in (("b1", b1), ("b2", b2)):
        full = summ[key]["overall_acc"]
        ids = [i for i in d if not _t2_unconcluded(d[i])]
        co = (sum(1 for i in ids if d[i].get("grade") is True) / len(ids)) if ids else None
        print(f"  {summ[key]['label']:<16} full={fmt_pct(full)}%   "
              f"concluded-only={fmt_pct(co)}%   (concluded n={len(ids)})")
    print("  → B1-vs-B2 verdict is robust iff the sign of Δ agrees across both views.")


# ── B1' corrected (reconsider WITH folded CoT) ──────────────────────────────────

def _grade_map(pooled):
    return {r["taskId"]: (r.get("grade") is True) for r in pooled}


def _std_majority(pooled):
    """standard per-item label = majority correct across its passes (≥2 of 3)."""
    by = defaultdict(list)
    for r in pooled:
        by[r["taskId"]].append(r.get("grade") is True)
    return {t: (sum(v) > len(v) / 2) for t, v in by.items()}


def _paired(name_new, gnew, name_ref, gref):
    ids = set(gnew) & set(gref)
    if not ids:
        print("  (no common items)"); return None
    an = sum(gnew[i] for i in ids) / len(ids)
    ar = sum(gref[i] for i in ids) / len(ids)
    nr_rw = sum(1 for i in ids if gnew[i] and not gref[i])
    nw_rr = sum(1 for i in ids if not gnew[i] and gref[i])
    print(f"  {name_new} {an*100:5.1f}%   vs {name_ref} {ar*100:5.1f}%   "
          f"Δ={(an-ar)*100:+.1f} pts   (n={len(ids)})")
    print(f"    discordant: {name_new}-right/{name_ref}-wrong={nr_rw}   "
          f"{name_new}-wrong/{name_ref}-right={nw_rr}")
    return ids


def b1cot_report(summ):
    """B1' = corrected re-grounding that actually carries the turn-1 CoT into turn 2.
    Tests the prediction from the trajectory read: re-grounding + CoT should help the
    REASONING-type flips (spatial transform) more than the PERCEPTION-type ones
    (counting/search), where re-grounding only re-rolls an unstable percept."""
    b1cp = summ.get("b1cot", {}).get("pooled")
    if not b1cp:
        return
    print("\n" + "-" * 78)
    print("B1' · CORRECTED reconsider-with-CoT (turn-1 reasoning folded into turn-2 user msg)")
    print("-" * 78)
    g_b1c = _grade_map(b1cp)

    std_pool = summ.get("standard", {}).get("pooled") or []
    g_std = _std_majority(std_pool) if std_pool else {}
    if g_std:
        print("paired vs standard (majority of 3 passes):")
        _paired("B1'", g_b1c, "std", g_std)

    b1_pool = summ.get("b1", {}).get("pooled") or []
    if b1_pool:
        print("paired vs B1 (original reinject — CoT was stripped by the template):")
        _paired("B1'", g_b1c, "B1", _grade_map(b1_pool))

    # prediction test: where does B1' help?  Δ vs standard, by perception/reasoning family
    if g_std:
        sub_of = {r["taskId"]: r.get("subtype") for r in b1cp}
        ids = set(g_b1c) & set(g_std)
        print("\nwhere B1' helps — Δ vs standard by a-priori family "
              "(prediction: reasoning > perception):")
        for name, grp in (("perception (count / search / trace)", PERCEPTION_TASKS),
                          ("reasoning  (spatial transform / completion)", REASONING_TASKS)):
            gi = [i for i in ids if sub_of.get(i) in grp]
            if gi:
                an = sum(g_b1c[i] for i in gi) / len(gi)
                ar = sum(g_std[i] for i in gi) / len(gi)
                print(f"  {name:<44} B1'={an*100:5.1f}%  std={ar*100:5.1f}%  "
                      f"Δ={(an-ar)*100:+.1f} pts  (n={len(gi)})")
        # data-driven per-subtype Δ (don't trust the grouping alone)
        print("\nper-subtype Δ(B1' − standard), largest gain first:")
        b1c_sub, std_sub = summ["b1cot"]["by_subtype"], summ["standard"]["by_subtype"]
        rows = []
        for sub, (a, _, n) in b1c_sub.items():
            sa = std_sub.get(sub, (None,))[0]
            if a is not None and sa is not None:
                rows.append((a - sa, sub, a, sa, n))
        for d, sub, a, sa, n in sorted(rows, reverse=True):
            fam = "P" if sub in PERCEPTION_TASKS else "R" if sub in REASONING_TASKS else "?"
            print(f"  [{fam}] {str(sub)[:26]:<26} Δ={d*100:+6.1f}  "
                  f"(B1'={a*100:4.0f}%  std={sa*100:4.0f}%  n={n})")

    # clean image-reshow contrast (identical turn-1) — only once B2' exists
    b2cp = summ.get("b2cot", {}).get("pooled")
    if b2cp:
        print("\npaired vs B2' (identical turn-1 reused; image NOT reshown — isolates re-show):")
        _paired("B1'", g_b1c, "B2'", _grade_map(b2cp))


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="dir containing the results_* condition dirs")
    ap.add_argument("--out",  default=None, help="dir for summary.json + readout.md")
    ap.add_argument("--bins", type=int, default=4)
    args = ap.parse_args()

    base = Path(args.base)
    summ = {}
    print("\n" + "=" * 78)
    print("BabyVision — three-way readout (reasoning-length axis: A0 → standard → A3)")
    print("=" * 78)

    for key, d, passes, label in CONDITIONS:
        print(f"\nLoading {label} ({d}) …")
        summ[key] = summarize_condition(key, base / d, passes, label)

    # ── 1. Overall + reasoning length ──────────────────────────────────────────
    print("\n" + "-" * 78)
    print("OVERALL ACCURACY  &  REASONING LENGTH")
    print("-" * 78)
    print(f"{'condition':<16}{'acc %':>8}{'±std':>7}{'len(med tok)':>14}{'no-ans':>8}{'trunc':>7}")
    for key in ORDER:
        s = summ[key]
        std = f"±{s['overall_std']*100:.1f}" if len(s["pass_accs"]) > 1 else "   -"
        lm = f"{s['len_median']:.0f}" if s["len_median"] is not None else "n/a"
        print(f"{s['label']:<16}{fmt_pct(s['overall_acc']):>8}{std:>7}{lm:>14}"
              f"{s['no_answer']:>8}{s['truncated']:>7}")
    a3 = summ["a3"]
    if a3["forces_mean"] is not None:
        print(f"\nA3 forcing: mean Waits/sample = {a3['forces_mean']:.2f}, "
              f"samples with ≥1 Wait = {a3['forces_nonzero']}/{a3['n_pooled']}")

    # ── 2. The contrast (Δ vs standard) ─────────────────────────────────────────
    base_acc = summ["standard"]["overall_acc"]
    print("\n" + "-" * 78)
    print("THINK-LONGER-SEE-LESS CONTRAST  (Δ accuracy vs standard, pts)")
    print("-" * 78)
    for key in ORDER:
        s = summ[key]
        if s["overall_acc"] is None or base_acc is None:
            continue
        d = (s["overall_acc"] - base_acc) * 100
        arrow = "→ baseline" if key == "standard" else (f"{d:+.1f} pts")
        print(f"  {s['label']:<16}{fmt_pct(s['overall_acc'])}%   {arrow}")

    # ── 3. By type ──────────────────────────────────────────────────────────────
    types = sorted({t for key in ORDER for t in summ[key]["by_type"]})
    print("\n" + "-" * 78)
    print("ACCURACY BY TYPE  (%)")
    print("-" * 78)
    print(f"{'type':<34}" + "".join(f"{summ[k]['label'][:10]:>12}" for k in ORDER))
    for t in types:
        cells = "".join(fmt_pct(summ[k]["by_type"].get(t, (None,))[0]).rjust(12) for k in ORDER)
        print(f"{str(t)[:34]:<34}{cells}")

    # ── 4. By subtype ───────────────────────────────────────────────────────────
    subs = sorted({s for key in ORDER for s in summ[key]["by_subtype"]})
    print("\n" + "-" * 78)
    print("ACCURACY BY SUBTYPE  (%)   [n = pooled count in standard]")
    print("-" * 78)
    print(f"{'subtype':<30}{'n':>5}" + "".join(f"{summ[k]['label'][:10]:>12}" for k in ORDER))
    for sub in subs:
        n = summ["standard"]["by_subtype"].get(sub, (None, 0, 0))[2]
        cells = "".join(fmt_pct(summ[k]["by_subtype"].get(sub, (None,))[0]).rjust(12) for k in ORDER)
        print(f"{str(sub)[:30]:<30}{n:>5}{cells}")

    # ── 5. ansType split ────────────────────────────────────────────────────────
    print("\n" + "-" * 78)
    print("ACCURACY BY ANSWER TYPE  (%)")
    print("-" * 78)
    print(f"{'ansType':<16}" + "".join(f"{summ[k]['label'][:10]:>12}" for k in ORDER))
    for at in ("blank", "choice"):
        cells = "".join(fmt_pct(summ[k]["by_anstype"].get(at, (None,))[0]).rjust(12) for k in ORDER)
        print(f"{at:<16}{cells}")

    # ── 6. Within-condition see-less curve ──────────────────────────────────────
    print("\n" + "-" * 78)
    print(f"ACCURACY vs REASONING LENGTH  ({args.bins} quartiles by completion_tokens)")
    print("  (the see-less curve from natural length variation, within each condition)")
    print("-" * 78)
    for key in ORDER:
        s = summ[key]
        curve = acc_vs_length(s["pooled"], args.bins)
        if not curve:
            continue
        print(f"\n  {s['label']}:")
        for b, lo, hi, a, n in curve:
            print(f"    Q{b}  tok[{lo:>6.0f}–{hi:>6.0f}]  acc={a*100:5.1f}%  (n={n})")

    # ── 7. Cross-pass instability (standard) ────────────────────────────────────
    fs = flip_stats(base / "results_standard", [1, 2, 3])
    if fs:
        print("\n" + "-" * 78)
        print("CROSS-PASS INSTABILITY (standard, 3 passes)")
        print("-" * 78)
        print(f"  questions with disagreement across passes: "
              f"{fs['unstable']}/{fs['n']} = {fs['unstable_frac']*100:.1f}%")
        print(f"  all-3-correct: {fs['all_correct']}   all-3-wrong: {fs['all_wrong']}")

    # ── 8. B-family two-turn integrity & re-grounding contrast ──────────────────
    b_family_report(summ)

    # ── 9. B1' corrected reconsider-with-CoT (+ perception/reasoning prediction) ─
    b1cot_report(summ)

    # ── Persist ──────────────────────────────────────────────────────────────────
    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        ser = {}
        for key in ORDER:
            s = summ[key]
            ser[key] = {
                "label": s["label"], "overall_acc": s["overall_acc"],
                "overall_std": s["overall_std"], "pass_accs": s["pass_accs"],
                "len_mean": s["len_mean"], "len_median": s["len_median"],
                "no_answer": s["no_answer"], "truncated": s["truncated"],
                "forces_mean": s["forces_mean"], "forces_nonzero": s["forces_nonzero"],
                "by_type":    {str(k): v[0] for k, v in s["by_type"].items()},
                "by_subtype": {str(k): v[0] for k, v in s["by_subtype"].items()},
                "by_anstype": {str(k): v[0] for k, v in s["by_anstype"].items()},
                "acc_vs_length": acc_vs_length(s["pooled"], args.bins),
            }
        (out / "summary.json").write_text(json.dumps(ser, indent=2))
        print(f"\nWrote {out/'summary.json'}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
