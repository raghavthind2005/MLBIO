#!/usr/bin/env python3
"""
Mechanistic "think longer, see less" analysis from attention_results.jsonl.

Tests, at the attention level, the behavioral findings from proof_of_concept.py:

  A. Visual attention DECAYS across reasoning positions (early → late).
  B. Lower late-stage visual attention is associated with WRONG answers.
  C. Longer reasoning chains have LOWER mean visual attention (the mechanism
     behind the behavioral length→error collapse).
  D. FORCED re-examination: the model attends LESS to the re-injected image
     (visual_turn1) than to the original (visual_turn0) — it does not re-perceive.

Outputs plots to plots_attention/ and prints a numeric summary.

  python attention_analysis.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
PLOTS = SCRIPT_DIR / "plots_attention"

CONDITIONS = {
    "normal": SCRIPT_DIR / "results_normal" / "attention_results.jsonl",
    "tool":   SCRIPT_DIR / "results_tool"   / "attention_results.jsonl",
    "forced": SCRIPT_DIR / "results_forced" / "attention_results.jsonl",
}
COLORS = {"normal": "steelblue", "tool": "darkorange", "forced": "seagreen"}


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def resample(trace: list[float], n: int = 100) -> np.ndarray:
    a = np.asarray(trace, dtype=float)
    if len(a) < 2:
        return np.full(n, a.mean() if len(a) else 0.0)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a)), a)


def mean_trace(recs, key, n=100):
    traces = [resample(r[key], n) for r in recs if r.get(key) and len(r[key]) >= 5]
    return (np.mean(traces, axis=0), np.std(traces, axis=0), len(traces)) if traces else (None, None, 0)


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    xm, ym = x - x.mean(), y - y.mean()
    d = np.sqrt((xm**2).sum() * (ym**2).sum())
    return float((xm * ym).sum() / d) if d else float("nan")


# ─── A. Visual attention decay over reasoning position ────────────────────────

def plot_decay_over_position(data, n=100):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 100, n)
    for cond, recs in data.items():
        if not recs:
            continue
        avg, std, k = mean_trace(recs, "attn_visual_per_pos", n)
        if avg is None:
            continue
        ax.plot(x, avg, color=COLORS[cond], lw=2, label=f"{cond} visual (n={k})")
        ax.fill_between(x, avg - std/np.sqrt(k), avg + std/np.sqrt(k),
                        color=COLORS[cond], alpha=0.15)
    # instruction/system reference from normal
    if data.get("normal"):
        for key, c, lab in [("attn_instruction_per_pos", "gray", "normal instruction"),
                            ("attn_system_per_pos", "lightcoral", "normal system")]:
            avg, _, k = mean_trace(data["normal"], key, n)
            if avg is not None:
                ax.plot(x, avg, color=c, lw=1.5, ls="--", label=lab)
    ax.set_xlabel("Reasoning progress (%)")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Attention over reasoning: does the model 'see less' as it thinks longer?")
    ax.legend()
    out = PLOTS / "A_visual_decay_over_position.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


# ─── B. Visual attention thirds, correct vs wrong ─────────────────────────────

def plot_thirds_correct_wrong(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    stages = ["early", "mid", "late"]
    for ax, (cond, recs) in zip(axes, data.items()):
        if not recs:
            continue
        cor = {s: [] for s in stages}
        wro = {s: [] for s in stages}
        for r in recs:
            t = r.get("attn_visual_by_thirds")
            if not t:
                continue
            (cor if r.get("is_correct") == 1 else wro)
            tgt = cor if r.get("is_correct") == 1 else wro
            for s in stages:
                if s in t:
                    tgt[s].append(t[s])
        xs = np.arange(3); w = 0.38
        cm = [np.mean(cor[s]) if cor[s] else 0 for s in stages]
        wm = [np.mean(wro[s]) if wro[s] else 0 for s in stages]
        ax.bar(xs - w/2, cm, w, label=f"correct (n={len(cor['early'])})", color="steelblue")
        ax.bar(xs + w/2, wm, w, label=f"wrong (n={len(wro['early'])})", color="tomato")
        ax.set_xticks(xs); ax.set_xticklabels(["early", "mid", "late"])
        ax.set_title(cond); ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean visual attention")
    fig.suptitle("Visual attention by reasoning stage: correct vs wrong")
    out = PLOTS / "B_visual_thirds_correct_wrong.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")


# ─── C. Visual attention vs reasoning length (the mechanism) ──────────────────

def plot_attention_vs_length(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    summary = {}
    for cond, recs in data.items():
        pts = [(r["n_output_tokens"], r["attn_visual_mean"]) for r in recs
               if r.get("n_output_tokens") and r.get("attn_visual_mean") is not None]
        if len(pts) < 8:
            continue
        pts.sort()
        lens = np.array([p[0] for p in pts]); att = np.array([p[1] for p in pts])
        r_corr = pearson(lens, att)
        summary[cond] = r_corr
        # quartile means
        n = len(pts); qx, qy = [], []
        for q in range(4):
            chunk = att[q*n//4:(q+1)*n//4]
            qx.append(q); qy.append(chunk.mean())
        ax.plot(qx, qy, "o-", color=COLORS[cond], lw=2,
                label=f"{cond} (r={r_corr:+.2f}, n={n})")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Q1\n(short)", "Q2", "Q3", "Q4\n(long)"])
    ax.set_xlabel("Reasoning length quartile (output tokens)")
    ax.set_ylabel("Mean visual attention")
    ax.set_title("Think longer, see less: visual attention vs reasoning length")
    ax.legend()
    out = PLOTS / "C_visual_attention_vs_length.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")
    return summary


# ─── D. Forced: original vs re-injected image attention ───────────────────────

def plot_forced_turn0_vs_turn1(forced, n=100):
    if not forced:
        return None
    t0 = [r["attn_visual_turn0_mean"] for r in forced if r.get("attn_visual_turn0_mean") is not None]
    t1 = [r["attn_visual_turn1_mean"] for r in forced if r.get("attn_visual_turn1_mean") is not None]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # bar: mean turn0 vs turn1
    ax1.bar([0, 1], [np.mean(t0), np.mean(t1)],
            color=["steelblue", "tomato"], width=0.6)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["turn0\n(original image)", "turn1\n(re-injected image)"])
    ax1.set_ylabel("Mean visual attention")
    ax1.set_title(f"Forced re-examination: attention to original vs re-injected\n"
                  f"turn0={np.mean(t0):.4f}  turn1={np.mean(t1):.4f}  "
                  f"({(np.mean(t1)/np.mean(t0)-1)*100:+.0f}%)")
    for i, v in enumerate([np.mean(t0), np.mean(t1)]):
        ax1.text(i, v + 0.002, f"{v:.4f}", ha="center")

    # per-position traces of turn0 and turn1
    x = np.linspace(0, 100, n)
    a0, _, k0 = mean_trace(forced, "attn_visual_turn0_per_pos", n)
    a1, _, k1 = mean_trace(forced, "attn_visual_turn1_per_pos", n)
    if a0 is not None: ax2.plot(x, a0, color="steelblue", lw=2, label="turn0 (original)")
    if a1 is not None: ax2.plot(x, a1, color="tomato", lw=2, label="turn1 (re-injected)")
    ax2.set_xlabel("Reasoning progress in final turn (%)")
    ax2.set_ylabel("Mean visual attention")
    ax2.set_title("Per-position attention to each image")
    ax2.legend()
    out = PLOTS / "D_forced_turn0_vs_turn1.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {out}")
    return np.mean(t0), np.mean(t1)


def main():
    PLOTS.mkdir(exist_ok=True)
    data = {c: load(p) for c, p in CONDITIONS.items()}
    print("Loaded:", {c: len(r) for c, r in data.items()}, "\n")

    print("Generating plots...")
    plot_decay_over_position(data)
    plot_thirds_correct_wrong(data)
    length_corr = plot_attention_vs_length(data)
    forced_means = plot_forced_turn0_vs_turn1(data.get("forced", []))

    # ── Numeric summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("MECHANISTIC SUMMARY")
    print("=" * 64)

    for cond, recs in data.items():
        if not recs:
            continue
        thirds = {s: [] for s in ["early", "mid", "late"]}
        for r in recs:
            t = r.get("attn_visual_by_thirds") or {}
            for s in thirds:
                if s in t: thirds[s].append(t[s])
        e, m, l = (np.mean(thirds[s]) for s in ["early", "mid", "late"])
        print(f"\n[{cond}] visual attention by stage: "
              f"early={e:.4f}  mid={m:.4f}  late={l:.4f}  "
              f"(late/early {(l/e-1)*100:+.0f}%)")
        # correct vs wrong late attention
        cl = [r['attn_visual_by_thirds']['late'] for r in recs
              if r.get('is_correct') == 1 and r.get('attn_visual_by_thirds')]
        wl = [r['attn_visual_by_thirds']['late'] for r in recs
              if r.get('is_correct') == 0 and r.get('attn_visual_by_thirds')]
        if cl and wl:
            print(f"         late visual attn: correct={np.mean(cl):.4f}  "
                  f"wrong={np.mean(wl):.4f}")

    print(f"\n[length→attention correlation] (negative = think longer see less):")
    for cond, r in (length_corr or {}).items():
        print(f"  {cond}: r = {r:+.3f}")

    if forced_means:
        t0, t1 = forced_means
        print(f"\n[forced re-examination] attention to original={t0:.4f} vs "
              f"re-injected={t1:.4f}  ({(t1/t0-1)*100:+.0f}%)")
        print("  → model attends LESS to the re-injected image: it re-reasons, "
              "it does not re-perceive." if t1 < t0 else
              "  → model attends more to the re-injected image.")

    print(f"\nAll plots in {PLOTS}/")


if __name__ == "__main__":
    main()
