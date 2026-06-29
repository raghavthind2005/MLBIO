"""
summarize_deltas.py — read deltas.csv → text readout of the S2 finding + freeze check

Pure stdlib (csv only) — runs on a login node, no GPU/pandas needed.

Tests:
  - FREEZE SANITY: per condition, mean rel_fro by component {vision, llm, embed}.
      full     → both vision and llm > 0
      llm_only → vision ≈ 0 (ViT frozen), llm > 0
      vit_only → llm ≈ 0 (LLM frozen), vision > 0
  - S2 LOCALIZATION: within the LLM, mean rel_fro of mlp vs attn, binned by
      layer band (early/mid/late). H predicts late-MLP > late-attn and rising with depth.
  - S5 DYNAMICS (if multiple steps): mean llm/mlp rel_fro vs step (does it grow?).

Usage:
  python summarize_deltas.py --csv deltas.csv
  python summarize_deltas.py --csv deltas.csv --condition full
  python summarize_deltas.py --csv deltas.csv --per-layer-out per_layer.csv   # for local plotting
"""

import argparse
import csv
from collections import defaultdict


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            # numeric coercions
            for k in ("step", "n_params"):
                r[k] = int(r[k]) if r[k] not in ("", None) else None
            for k in ("base_fro", "abs_fro", "rel_fro", "mean_abs", "cos_sim"):
                r[k] = float(r[k]) if r[k] not in ("", None) else None
            r["layer_idx"] = int(r["layer_idx"]) if r["layer_idx"] not in ("", None) else None
            rows.append(r)
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def banded(layer_idx, n_layers):
    """early / mid / late thirds."""
    third = n_layers / 3.0
    if layer_idx < third:
        return "early"
    elif layer_idx < 2 * third:
        return "mid"
    return "late"


def summarize_condition(rows, cond):
    sub = [r for r in rows if r["condition"] == cond]
    if not sub:
        return
    steps = sorted({r["step"] for r in sub})
    final = steps[-1]
    fin = [r for r in sub if r["step"] == final]

    print(f"\n{'='*70}")
    print(f"CONDITION: {cond}   (steps {steps[0]}..{final}, {len(steps)} checkpoints)")
    print(f"{'='*70}")

    # ── Freeze / component sanity (final step) ───────────────────────────────
    print(f"\n[freeze check]  mean rel_fro by component @ step {final}:")
    by_comp = defaultdict(list)
    for r in fin:
        by_comp[r["component"]].append(r["rel_fro"])
    for comp in ("vision", "llm", "embed", "head", "other"):
        if comp in by_comp:
            vals = by_comp[comp]
            print(f"    {comp:8s}: mean_rel_fro={mean(vals):.3e}   "
                  f"max={max(vals):.3e}   n={len(vals)}")

    # ── S2 localization: LLM mlp vs attn by layer band (final step) ──────────
    llm = [r for r in fin if r["component"] == "llm" and r["layer_idx"] is not None]
    if llm:
        n_layers = max(r["layer_idx"] for r in llm) + 1
        print(f"\n[S2 localization]  LLM mean rel_fro by module × layer-band "
              f"({n_layers} layers) @ step {final}:")
        print(f"    {'band':6s} {'attn':>12s} {'mlp':>12s} {'mlp/attn':>10s}")
        for band in ("early", "mid", "late"):
            attn = mean([r["rel_fro"] for r in llm
                         if r["module"] == "attn" and banded(r["layer_idx"], n_layers) == band])
            mlp = mean([r["rel_fro"] for r in llm
                        if r["module"] == "mlp" and banded(r["layer_idx"], n_layers) == band])
            ratio = mlp / attn if attn and attn == attn else float("nan")
            print(f"    {band:6s} {attn:>12.3e} {mlp:>12.3e} {ratio:>10.2f}")

        # Verdict
        late_mlp = mean([r["rel_fro"] for r in llm
                         if r["module"] == "mlp" and banded(r["layer_idx"], n_layers) == "late"])
        late_attn = mean([r["rel_fro"] for r in llm
                          if r["module"] == "attn" and banded(r["layer_idx"], n_layers) == "late"])
        early_mlp = mean([r["rel_fro"] for r in llm
                          if r["module"] == "mlp" and banded(r["layer_idx"], n_layers) == "early"])
        print(f"\n    S2 signal: late_mlp={late_mlp:.3e}  late_attn={late_attn:.3e}  "
              f"early_mlp={early_mlp:.3e}")
        verdict = []
        if late_mlp > late_attn:
            verdict.append(f"late MLP > late attn ({late_mlp/late_attn:.2f}x) ✓")
        else:
            verdict.append(f"late MLP NOT > late attn ✗")
        if late_mlp > early_mlp:
            verdict.append(f"MLP rises with depth ({late_mlp/early_mlp:.2f}x late/early) ✓")
        else:
            verdict.append(f"MLP does NOT rise with depth ✗")
        print(f"    → {'; '.join(verdict)}")

    # ── S5 dynamics: late-MLP rel_fro vs step ────────────────────────────────
    if len(steps) > 1:
        print(f"\n[S5 dynamics]  mean late-layer LLM/mlp rel_fro across training:")
        for s in steps:
            ss = [r for r in sub if r["step"] == s and r["component"] == "llm"
                  and r["module"] == "mlp" and r["layer_idx"] is not None]
            if ss:
                nl = max(r["layer_idx"] for r in ss) + 1
                late = mean([r["rel_fro"] for r in ss if banded(r["layer_idx"], nl) == "late"])
                bar = "#" * int(late / 1e-3) if late == late else ""
                print(f"    step {s:>3d}: {late:.3e}  {bar}")


def write_per_layer(rows, path):
    """Per (condition, step, layer_idx, module) mean rel_fro — for local matplotlib plotting."""
    agg = defaultdict(list)
    for r in rows:
        if r["component"] == "llm" and r["layer_idx"] is not None and r["module"] in ("attn", "mlp"):
            agg[(r["condition"], r["step"], r["layer_idx"], r["module"])].append(r["rel_fro"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "step", "layer_idx", "module", "mean_rel_fro", "n"])
        for (cond, step, idx, mod), vals in sorted(agg.items()):
            w.writerow([cond, step, idx, mod, f"{mean(vals):.6e}", len(vals)])
    print(f"\nWrote per-layer aggregate to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="deltas.csv")
    ap.add_argument("--condition", default=None,
                    help="Summarize one condition; default = all present.")
    ap.add_argument("--per-layer-out", default=None,
                    help="Also write per-layer aggregate CSV for plotting.")
    args = ap.parse_args()

    rows = load(args.csv)
    conds = ([args.condition] if args.condition
             else sorted({r["condition"] for r in rows}))
    for cond in conds:
        summarize_condition(rows, cond)

    if args.per_layer_out:
        write_per_layer(rows, args.per_layer_out)


if __name__ == "__main__":
    main()
