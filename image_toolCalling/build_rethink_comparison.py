#!/usr/bin/env python3
"""
Rethink vs Forced re-examination comparison.

Reads results_forced/forced_results.jsonl, results_rethink/rethink_results.jsonl,
and optionally results_normal/raw_results.jsonl and produces a self-contained HTML
with all plots embedded as base64 PNGs.

Usage:
  python build_rethink_comparison.py
  python build_rethink_comparison.py --out my_comparison.html
"""

import argparse
import base64
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR = Path(__file__).parent

FORCED_PATH  = SCRIPT_DIR / "results_forced"  / "forced_results.jsonl"
RETHINK_PATH = SCRIPT_DIR / "results_rethink" / "rethink_results.jsonl"
NORMAL_PATH  = SCRIPT_DIR / "results_normal"  / "raw_results.jsonl"

# ── Palette ──────────────────────────────────────────────────────────────────
C_NORMAL  = "#5b8db8"   # blue   — normal baseline
C_FORCED  = "#e07b39"   # orange — forced (image re-injected)
C_RETHINK = "#5aaa6f"   # green  — rethink (no image)
C_T0      = "#aaaaaa"   # grey   — turn 0 (shared baseline)

ALPHA_LIGHT = 0.35


# ── Data loading ─────────────────────────────────────────────────────────────

def load(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r and r.get("is_correct") is not None:
                records.append(r)
    return records


# ── Figure → base64 PNG ──────────────────────────────────────────────────────

def fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Plot helpers ─────────────────────────────────────────────────────────────

def styled_fig(w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", zorder=0)
    return fig, ax


# ── Plot 1: Headline accuracy — turn0 vs turn1 per condition ─────────────────

def plot_headline(forced, rethink, normal):
    fig, ax = styled_fig(10, 5)

    # compute accuracies
    def acc(recs, key):
        vals = [r[key] for r in recs if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0

    norm_acc = acc(normal, "is_correct")

    conds = ["Normal\n(1-turn baseline)", "Forced\n(image re-injected)", "Rethink\n(no image)"]
    t0s   = [norm_acc,           acc(forced, "is_correct_turn0"),  acc(rethink, "is_correct_turn0")]
    t1s   = [None,               acc(forced, "is_correct"),        acc(rethink, "is_correct")]
    ns    = [len(normal),        len(forced),                      len(rethink)]
    colors_t0 = [C_NORMAL, C_FORCED, C_RETHINK]
    colors_t1 = [None,     C_FORCED, C_RETHINK]

    x = np.arange(len(conds))
    w = 0.35

    bars0 = ax.bar(x - w/2, t0s, width=w, color=colors_t0, alpha=ALPHA_LIGHT + 0.15,
                   label="Turn 0", zorder=3)
    # turn 1 bars (not for normal)
    for i, (t1, c) in enumerate(zip(t1s, colors_t1)):
        if t1 is not None:
            ax.bar(x[i] + w/2, t1, width=w, color=c, alpha=0.85, zorder=3)

    # delta annotations
    for i, (t0, t1) in enumerate(zip(t0s, t1s)):
        if t1 is not None:
            delta = t1 - t0
            color = "#2a7a2a" if delta >= 0 else "#c0392b"
            sign  = "+" if delta >= 0 else ""
            ax.annotate(f"Δ {sign}{delta*100:.2f}pp",
                        xy=(x[i] + w/2, t1 + 0.003),
                        ha="center", va="bottom", fontsize=9.5,
                        color=color, fontweight="bold")

    # value labels on turn-0 bars
    for bar, t0 in zip(bars0, t0s):
        ax.text(bar.get_x() + bar.get_width()/2, t0 + 0.003,
                f"{t0*100:.1f}%", ha="center", va="bottom", fontsize=8.5, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=11)
    ax.set_ylabel("Question Accuracy", fontsize=11)
    ax.set_ylim(0.65, 0.88)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_title("Turn 0 vs Turn 1 Accuracy by Condition", fontsize=13, fontweight="bold", pad=12)

    # legend
    p_t0 = mpatches.Patch(color="#999999", alpha=0.55, label="Turn 0 (initial)")
    p_t1 = mpatches.Patch(color="#555555", alpha=0.85, label="Turn 1 (final)")
    ax.legend(handles=[p_t0, p_t1], fontsize=9, loc="upper right")

    # sample sizes
    for i, n in enumerate(ns):
        ax.text(x[i], 0.655, f"n={n}", ha="center", fontsize=8, color="#888")

    return fig_to_b64(fig)


# ── Plot 2: Change-type breakdown ────────────────────────────────────────────

def plot_change_type(forced, rethink):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, recs, label, color in [
        (axes[0], forced,  "Forced (image re-injected)", C_FORCED),
        (axes[1], rethink, "Rethink (no image)",         C_RETHINK),
    ]:
        counts = defaultdict(int)
        for r in recs:
            ct = r.get("change_type")
            if ct:
                counts[ct] += 1
        total = sum(counts.values())

        cats   = ["right_right", "wrong_wrong", "wrong_right", "right_wrong"]
        labels = ["✓→✓\nStayed right", "✗→✗\nStayed wrong", "✗→✓\nHelped ↑", "✓→✗\nHurt ↓"]
        bar_colors = ["#5aaa6f", "#e07b7b", "#2ecc71", "#e74c3c"]
        vals = [counts.get(c, 0) for c in cats]

        bars = ax.bar(labels, vals, color=bar_colors, alpha=0.82, zorder=3, width=0.55)
        for bar, v in zip(bars, vals):
            pct = v / total * 100
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                    f"{v}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

        ax.set_title(label, fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel("Sample Count", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e0e0e0", zorder=0)
        ax.set_ylim(0, max(vals) * 1.25)

    fig.suptitle("Turn 0 → Turn 1 Answer Transitions", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── Plot 3: Accuracy by visual_input (vi=1 original vs vi=2 edited) ──────────

def plot_by_visual_input(forced, rethink, normal):
    fig, ax = styled_fig(11, 5.5)

    vi_labels = {"1": "Original (vi=1)", "2": "Edited/Illusion (vi=2)"}
    vis = ["1", "2"]

    def group_acc(recs, vi, key):
        vals = [r[key] for r in recs if r.get("visual_input") == vi and r.get(key) is not None]
        return (sum(vals) / len(vals) if vals else 0), len(vals)

    x      = np.arange(len(vis))
    w      = 0.18
    offset = [-1.5, -0.5, 0.5, 1.5]
    series = [
        ("Normal (1-turn)",         normal,  "is_correct",       C_NORMAL,  0.70, "-"),
        ("Forced – Turn 0",         forced,  "is_correct_turn0", C_FORCED,  0.50, "--"),
        ("Forced – Turn 1",         forced,  "is_correct",       C_FORCED,  0.85, "-"),
        ("Rethink – Turn 0",        rethink, "is_correct_turn0", C_RETHINK, 0.50, "--"),
        ("Rethink – Turn 1",        rethink, "is_correct",       C_RETHINK, 0.85, "-"),
    ]

    # Use 5 bars per vi group
    offsets5 = [-2, -1, 0, 1, 2]
    w5 = 0.16
    for off, (lbl, recs, key, col, alpha, ls) in zip(offsets5, series):
        vals = []
        for vi in vis:
            a, _ = group_acc(recs, vi, key)
            vals.append(a)
        bars = ax.bar(x + off * w5, vals, width=w5, color=col, alpha=alpha,
                      label=lbl, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([vi_labels[v] for v in vis], fontsize=12)
    ax.set_ylabel("Question Accuracy", fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_title("Accuracy by Image Type — Normal vs Forced vs Rethink", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0.55, 1.0)
    ax.legend(fontsize=8.5, loc="upper right", ncol=2)

    # annotate n per group
    for vi_i, vi in enumerate(vis):
        for off, (lbl, recs, key, col, alpha, ls) in zip(offsets5, series):
            _, n = group_acc(recs, vi, key)
            ax.text(vi_i + off * w5, 0.56, f"n={n}", ha="center", fontsize=6.5, color="#888", rotation=90)

    return fig_to_b64(fig)


# ── Plot 4: Δ accuracy by subcategory (rethink vs forced) ───────────────────

def plot_delta_by_subcategory(forced, rethink):
    subs_f = defaultdict(lambda: {"t0": [], "t1": []})
    subs_r = defaultdict(lambda: {"t0": [], "t1": []})

    for r in forced:
        sub = r.get("subcategory", "?")
        if r.get("is_correct_turn0") is not None: subs_f[sub]["t0"].append(r["is_correct_turn0"])
        if r.get("is_correct") is not None:        subs_f[sub]["t1"].append(r["is_correct"])
    for r in rethink:
        sub = r.get("subcategory", "?")
        if r.get("is_correct_turn0") is not None: subs_r[sub]["t0"].append(r["is_correct_turn0"])
        if r.get("is_correct") is not None:        subs_r[sub]["t1"].append(r["is_correct"])

    all_subs = sorted(set(list(subs_f.keys()) + list(subs_r.keys())))

    deltas_f, deltas_r = [], []
    for sub in all_subs:
        def delta(d):
            if not d["t0"] or not d["t1"]: return 0
            return sum(d["t1"]) / len(d["t1"]) - sum(d["t0"]) / len(d["t0"])
        deltas_f.append(delta(subs_f[sub]))
        deltas_r.append(delta(subs_r[sub]))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x  = np.arange(len(all_subs))
    w  = 0.35

    bars_f = ax.bar(x - w/2, [d*100 for d in deltas_f], width=w,
                    color=[C_FORCED if d >= 0 else "#e74c3c" for d in deltas_f],
                    alpha=0.82, label="Forced (image re-injected)", zorder=3)
    bars_r = ax.bar(x + w/2, [d*100 for d in deltas_r], width=w,
                    color=[C_RETHINK if d >= 0 else "#c0392b" for d in deltas_r],
                    alpha=0.82, label="Rethink (no image)", zorder=3)

    ax.axhline(0, color="#333", linewidth=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(all_subs, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Δ Accuracy (Turn1 − Turn0), pp", fontsize=11)
    ax.set_title("Per-Subcategory Accuracy Delta: Forced vs Rethink", fontsize=13, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", zorder=0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── Plot 5: Thinking length — turn0 vs turn1 for each condition ──────────────

def plot_thinking_length(forced, rethink):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, recs, label, color in [
        (axes[0], forced,  "Forced (image re-injected)", C_FORCED),
        (axes[1], rethink, "Rethink (no image)",         C_RETHINK),
    ]:
        t0 = [r["thinking_turn0_chars"] for r in recs if "thinking_turn0_chars" in r]
        t1 = [r["thinking_turn1_chars"] for r in recs if "thinking_turn1_chars" in r]

        bins = np.linspace(0, max(max(t0, default=1), max(t1, default=1)), 35)
        ax.hist(t0, bins=bins, alpha=0.55, color=C_T0,    label=f"Turn 0 (μ={np.mean(t0):.0f})")
        ax.hist(t1, bins=bins, alpha=0.65, color=color,   label=f"Turn 1 (μ={np.mean(t1):.0f})")

        ax.axvline(np.mean(t0), color=C_T0,  linestyle="--", linewidth=1.5)
        ax.axvline(np.mean(t1), color=color, linestyle="--", linewidth=1.5)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Thinking Length (chars)", fontsize=10)
        ax.set_ylabel("Sample Count", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9)

    fig.suptitle("Thinking Length Distribution: Turn 0 vs Turn 1", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig_to_b64(fig)


# ── Plot 6: Edited-image accuracy — the key PoC ──────────────────────────────

def plot_vi2_comparison(forced, rethink, normal):
    """Focus on vi=2 (edited/illusion) samples — where the effect is strongest."""
    fig, ax = styled_fig(9, 5)

    def vi2_acc(recs, key):
        vals = [r[key] for r in recs if r.get("visual_input") == "2" and r.get(key) is not None]
        return (sum(vals) / len(vals) if vals else 0), len(vals)

    norm_a, norm_n   = vi2_acc(normal,  "is_correct")
    f_t0, fn         = vi2_acc(forced,  "is_correct_turn0")
    f_t1, _          = vi2_acc(forced,  "is_correct")
    r_t0, rn         = vi2_acc(rethink, "is_correct_turn0")
    r_t1, _          = vi2_acc(rethink, "is_correct")

    labels = ["Normal\n(1-turn)", "Forced\nTurn 0", "Forced\nTurn 1", "Rethink\nTurn 0", "Rethink\nTurn 1"]
    vals   = [norm_a, f_t0, f_t1, r_t0, r_t1]
    colors = [C_NORMAL,
              matplotlib.colors.to_rgba(C_FORCED, 0.5), C_FORCED,
              matplotlib.colors.to_rgba(C_RETHINK, 0.5), C_RETHINK]

    bars = ax.bar(labels, vals, color=colors, zorder=3, width=0.55)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.004,
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # delta arrows
    for (xi, xj, t0, t1, col) in [
        (1, 2, f_t0, f_t1, C_FORCED),
        (3, 4, r_t0, r_t1, C_RETHINK),
    ]:
        delta = t1 - t0
        sign  = "+" if delta >= 0 else ""
        color = "#2a7a2a" if delta >= 0 else "#c0392b"
        mid_x = (xi + xj) / 2
        ax.annotate("", xy=(xj, t1 + 0.015), xytext=(xi, t0 + 0.015),
                    arrowprops=dict(arrowstyle="->" if delta >= 0 else "<-",
                                   color=color, lw=1.8))
        ax.text(mid_x, max(t0, t1) + 0.025, f"{sign}{delta*100:.2f}pp",
                ha="center", fontsize=9.5, color=color, fontweight="bold")

    ax.set_ylabel("Question Accuracy", fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0.45, 0.85)
    ax.set_title("Edited / Illusion Images (vi=2) — Where Effects Are Strongest",
                 fontsize=12, fontweight="bold", pad=12)
    ax.text(0.98, 0.02, f"Forced n={fn}  |  Rethink n={rn}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#888")

    return fig_to_b64(fig)


# ── HTML assembly ────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rethink vs Forced Re-examination — HallusionBench</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f8f9fa; color: #222; margin: 0; padding: 0; }}
  .hero {{ background: #1a1a2e; color: #fff; padding: 40px 60px 32px; }}
  .hero h1 {{ margin: 0 0 8px; font-size: 1.9rem; }}
  .hero .subtitle {{ color: #aab; font-size: 1rem; margin: 0; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}
  .section {{ background: #fff; border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,.08);
              padding: 28px 32px; margin-bottom: 32px; }}
  .section h2 {{ margin-top: 0; font-size: 1.2rem; color: #1a1a2e; border-bottom: 2px solid #e8e8f0;
                 padding-bottom: 10px; margin-bottom: 16px; }}
  .section p  {{ color: #444; line-height: 1.65; font-size: 0.95rem; }}
  .figure     {{ text-align: center; margin: 16px 0 4px; }}
  .figure img {{ max-width: 100%; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
  .stat-row   {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }}
  .stat-card  {{ flex: 1; min-width: 150px; background: #f0f2ff; border-radius: 8px;
                 padding: 16px 20px; text-align: center; }}
  .stat-card .val {{ font-size: 1.8rem; font-weight: 700; }}
  .stat-card .lbl {{ font-size: 0.8rem; color: #666; margin-top: 2px; }}
  .pos {{ color: #2a7a2a; }} .neg {{ color: #c0392b; }} .neu {{ color: #444; }}
  .pill {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
           font-size: 0.78rem; font-weight: 600; margin-right: 6px; }}
  .pill.forced  {{ background: #fde8d8; color: #9a4010; }}
  .pill.rethink {{ background: #d8f0e0; color: #1a5c2a; }}
  .pill.normal  {{ background: #d8e8f8; color: #1a3a5c; }}
  .takeaway {{ background: #1a1a2e; color: #e8e8ff; border-radius: 8px; padding: 18px 24px;
               margin-top: 8px; font-size: 0.95rem; line-height: 1.7; }}
  .takeaway strong {{ color: #ffd; }}
  footer {{ text-align: center; color: #aaa; font-size: 0.8rem; padding: 24px; }}
</style>
</head>
<body>
<div class="hero">
  <h1>Rethink vs Forced Re-examination</h1>
  <p class="subtitle">HallusionBench · Gemma-4-31B-it · 30% stratified subset (seed=42) · {n_forced} forced / {n_rethink} rethink samples</p>
</div>
<div class="container">

  <!-- Key numbers -->
  <div class="section">
    <h2>Headline Numbers</h2>
    <div class="stat-row">
      <div class="stat-card">
        <div class="val neu">{norm_acc}</div>
        <div class="lbl"><span class="pill normal">Normal</span>Turn 0 only (baseline)</div>
      </div>
      <div class="stat-card">
        <div class="val neu">{f_t0_acc}</div>
        <div class="lbl"><span class="pill forced">Forced</span>Turn 0</div>
      </div>
      <div class="stat-card">
        <div class="val pos">{f_t1_acc}</div>
        <div class="lbl"><span class="pill forced">Forced</span>Turn 1 &nbsp;<strong class="pos">{f_delta}</strong></div>
      </div>
      <div class="stat-card">
        <div class="val neu">{r_t0_acc}</div>
        <div class="lbl"><span class="pill rethink">Rethink</span>Turn 0</div>
      </div>
      <div class="stat-card">
        <div class="val neg">{r_t1_acc}</div>
        <div class="lbl"><span class="pill rethink">Rethink</span>Turn 1 &nbsp;<strong class="neg">{r_delta}</strong></div>
      </div>
    </div>
    <div class="takeaway">
      <strong>Key finding:</strong> Asking the model to "re-examine carefully" <em>with</em> the image again yields a small positive nudge
      (<strong class="pos">{f_delta}</strong>). Asking it to "re-examine carefully" <em>without</em> the image hurts accuracy
      (<strong class="neg">{r_delta}</strong>). A second reasoning pass alone is not enough — and actively
      causes more correct answers to flip wrong ({r_hurt} hurt) than wrong answers to flip right ({r_helped} helped).
    </div>
  </div>

  <!-- Plot 1: headline -->
  <div class="section">
    <h2>1 · Turn 0 vs Turn 1 Accuracy by Condition</h2>
    <p>Light bars = Turn 0 (initial answer); dark bars = Turn 1 (final answer after re-prompt).
       The Δ annotation shows the change induced by the second turn.</p>
    <div class="figure"><img src="data:image/png;base64,{p1}" alt="Headline accuracy"></div>
  </div>

  <!-- Plot 2: change type -->
  <div class="section">
    <h2>2 · Answer Transition Breakdown</h2>
    <p>Every sample is classified by whether it was correct or wrong before and after the second turn.
       The asymmetry between Forced and Rethink in the Helped ↑ / Hurt ↓ columns is the core result.</p>
    <div class="figure"><img src="data:image/png;base64,{p2}" alt="Change type breakdown"></div>
  </div>

  <!-- Plot 6: vi=2 focus -->
  <div class="section">
    <h2>3 · Edited / Illusion Images (vi=2) — Effect Is Strongest Here</h2>
    <p>On <em>original</em> images the model is already near ceiling; the manipulation effects are
       most visible on <em>edited</em> images where the image contradicts a strong visual prior.
       Re-examining the image helps slightly; re-thinking without it hurts more.</p>
    <div class="figure"><img src="data:image/png;base64,{p6}" alt="vi=2 comparison"></div>
  </div>

  <!-- Plot 3: by visual input -->
  <div class="section">
    <h2>4 · Accuracy by Image Type</h2>
    <p>Original (vi=1) images are easier across the board; the gap between Forced and Rethink
       is most pronounced on the harder edited (vi=2) images.</p>
    <div class="figure"><img src="data:image/png;base64,{p3}" alt="By visual input"></div>
  </div>

  <!-- Plot 4: delta by subcategory -->
  <div class="section">
    <h2>5 · Per-Subcategory Accuracy Delta (Turn1 − Turn0)</h2>
    <p>Bars above zero = the second turn helped; below zero = it hurt.
       Forced (orange) and Rethink (green) differ most on illusion and figure subcategories
       where visual perception — not just reasoning — is required.</p>
    <div class="figure"><img src="data:image/png;base64,{p4}" alt="Delta by subcategory"></div>
  </div>

  <!-- Plot 5: thinking length -->
  <div class="section">
    <h2>6 · Thinking Length Distribution</h2>
    <p>Distribution of thinking characters in Turn 0 (grey) and Turn 1 (coloured).
       Rethink Turn 1 thinking is generally shorter than Forced Turn 1 thinking — consistent
       with the model having less to work with (no new visual tokens to process).</p>
    <div class="figure"><img src="data:image/png;base64,{p5}" alt="Thinking length"></div>
  </div>

</div>
<footer>Generated from forced_results.jsonl &amp; rethink_results.jsonl · Gemma-4-31B-it · HallusionBench 30% subset</footer>
</body>
</html>
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SCRIPT_DIR / "rethink_comparison.html"))
    args = ap.parse_args()

    print("Loading data…")
    forced  = load(FORCED_PATH)
    rethink = load(RETHINK_PATH)
    normal  = load(NORMAL_PATH) if NORMAL_PATH.exists() else []

    def acc(recs, key):
        vals = [r[key] for r in recs if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0

    norm_acc = acc(normal,  "is_correct")
    f_t0     = acc(forced,  "is_correct_turn0")
    f_t1     = acc(forced,  "is_correct")
    r_t0     = acc(rethink, "is_correct_turn0")
    r_t1     = acc(rethink, "is_correct")

    from collections import Counter
    f_ct = Counter(r.get("change_type") for r in forced)
    r_ct = Counter(r.get("change_type") for r in rethink)

    fmt = lambda v: f"{v*100:.2f}%"
    dlt = lambda a, b: (f"+{(b-a)*100:.2f}pp" if b >= a else f"{(b-a)*100:.2f}pp")

    print("Rendering plots…")
    p1 = plot_headline(forced, rethink, normal)
    p2 = plot_change_type(forced, rethink)
    p3 = plot_by_visual_input(forced, rethink, normal)
    p4 = plot_delta_by_subcategory(forced, rethink)
    p5 = plot_thinking_length(forced, rethink)
    p6 = plot_vi2_comparison(forced, rethink, normal)

    html = HTML_TEMPLATE.format(
        n_forced  = len(forced),
        n_rethink = len(rethink),
        norm_acc  = fmt(norm_acc),
        f_t0_acc  = fmt(f_t0),
        f_t1_acc  = fmt(f_t1),
        f_delta   = dlt(f_t0, f_t1),
        r_t0_acc  = fmt(r_t0),
        r_t1_acc  = fmt(r_t1),
        r_delta   = dlt(r_t0, r_t1),
        f_helped  = f_ct.get("wrong_right", 0),
        f_hurt    = f_ct.get("right_wrong", 0),
        r_helped  = r_ct.get("wrong_right", 0),
        r_hurt    = r_ct.get("right_wrong", 0),
        p1=p1, p2=p2, p3=p3, p4=p4, p5=p5, p6=p6,
    )

    out = Path(args.out)
    out.write_text(html)
    print(f"Written → {out}")


if __name__ == "__main__":
    main()
