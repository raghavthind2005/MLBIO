#!/usr/bin/env python3
"""
BabyVision three-way readout: standard vs A0 (no-think) vs A3 (forced-long).

Reasoning-length axis:  A0 (~0 reasoning) → standard (natural) → A3 (forced-long).
Tests the "think-longer-see-less" hypothesis on vision-primitive tasks.

Reads *_judged.jsonl from each condition dir (stdlib only — runs on the login node):
  standard:  results_run1_judged.jsonl, run2, run3   (3-pass mean±std)
  a0:        results_run1_judged.jsonl                (1 pass)
  a3:        results_run1_judged.jsonl                (1 pass)

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
    ("a0",       "results_a0_nothink",     [1],         "A0 no-think"),
    ("standard", "results_standard",       [1, 2, 3],   "standard"),
    ("a3",       "results_a3_forced_long", [1],         "A3 forced-long"),
]
ORDER = ["a0", "standard", "a3"]   # reasoning-length axis order


# ── Loading ──────────────────────────────────────────────────────────────────

def load_pass(cond_dir: Path, pass_idx: int) -> list:
    p = cond_dir / f"results_run{pass_idx}_judged.jsonl"
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
    nc = sum(1 for r in v if r.get("judge_result") is True)
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
    rows = [(length_metric(r), r.get("judge_result") is True)
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
            by_task[r["taskId"]].append(r.get("judge_result") is True)
    full = [v for v in by_task.values() if len(v) == len(passes)]
    if not full:
        return None
    unstable = sum(1 for v in full if len(set(v)) > 1)   # disagreement across passes
    allright = sum(1 for v in full if all(v))
    allwrong = sum(1 for v in full if not any(v))
    return {"n": len(full), "unstable": unstable,
            "unstable_frac": unstable / len(full),
            "all_correct": allright, "all_wrong": allwrong}


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
