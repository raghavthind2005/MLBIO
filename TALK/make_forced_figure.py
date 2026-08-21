#!/usr/bin/env python3
"""
Corrected F1 — forced re-injection re-engages vision.

WHY THIS EXISTS
  The original image_toolCalling/plots_forced/F1_attention_reengage.png plots three separate bars
  (0.097, 0.065, 0.120) and puts "turn-0 0.097 -> turn-1 0.184 (+91%)" in the title. The 0.184 is
  never drawn: it is i1_t1 + i2_t1, the SUM of the two turn-1 bars, because during turn-1 reasoning
  the model can attend to BOTH copies of the image. A reader sees the tallest bar at 0.120 against
  0.097 and reads +24%, not +91%. This version plots the total as a stacked bar so the headline
  number is what the eye measures.

Segment definitions (identical to forced_reexam_analysis.py:segments):
  i1_t0 = turn-0 reasoning positions -> original image
  i1_t1 = turn-1 reasoning positions -> original image
  i2_t1 = turn-1 reasoning positions -> re-injected image
  tot0  = i1_t0            (only one image exists during turn 0)
  tot1  = i1_t1 + i2_t1    (both copies visible during turn 1)
Measured WITHIN each reasoning segment: turn-0 tokens precede the re-injected image, so by causal
masking they attend exactly 0 to it — a naive turn-0 vs turn-1 mean is misleading.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ITC = HERE.parent / "image_toolCalling"
OUT = HERE / "figures"; OUT.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def segments(r):
    t0 = np.array(r.get("attn_visual_turn0_per_pos") or [])
    t1 = np.array(r.get("attn_visual_turn1_per_pos") or [])
    if len(t0) != len(t1) or len(t0) < 6:
        return None
    m1 = t1 > 1e-9
    if m1.sum() < 3 or (~m1).sum() < 3:
        return None
    return t0[~m1].mean(), t0[m1].mean(), t1[m1].mean()


att = [json.loads(l) for l in open(ITC / "results_forced" / "attention_results.jsonl") if l.strip()]
S = np.array([s for s in (segments(a) for a in att) if s is not None])
n = len(S)
i1_t0, i1_t1, i2_t1 = S.mean(axis=0)
tot0 = i1_t0
tot1_per_sample = S[:, 1] + S[:, 2]
tot1 = tot1_per_sample.mean()


def ci95(x):
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})

# ---- LEFT: the honest headline — total visual attention, turn-1 stacked
axL.bar([0], [tot0], width=0.55, color="#9ecae1", label="original image")
axL.bar([1], [i1_t1], width=0.55, color="#9ecae1")
axL.bar([1], [i2_t1], width=0.55, bottom=[i1_t1], color="#e6550d", label="re-injected copy")
axL.errorbar([0, 1], [tot0, tot1], yerr=[ci95(S[:, 0]), ci95(tot1_per_sample)],
             fmt="none", ecolor="#2c3e50", capsize=5, lw=1.5)

axL.text(0, tot0 + 0.012, f"{tot0:.3f}", ha="center", fontweight="bold", fontsize=11)
axL.text(1, tot1 + 0.012, f"{tot1:.3f}", ha="center", fontweight="bold", fontsize=11)
axL.text(1, i1_t1 / 2, f"{i1_t1:.3f}", ha="center", va="center", fontsize=9.5, color="#2c3e50")
axL.text(1, i1_t1 + i2_t1 / 2, f"{i2_t1:.3f}", ha="center", va="center", fontsize=9.5, color="white")

axL.annotate("", xy=(1.36, tot1), xytext=(1.36, tot0),
             arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=2))
axL.text(1.42, (tot0 + tot1) / 2, f"+{(tot1/tot0-1)*100:.0f}%",
         color="#c0392b", fontweight="bold", fontsize=13, va="center")

axL.set_xticks([0, 1])
axL.set_xticklabels(["turn-0 reasoning\n(1 image in context)",
                     "turn-1 reasoning\n(2 copies in context)"], fontsize=10)
axL.set_ylabel("mean attention weight to image tokens")
axL.set_ylim(0, 0.235)
axL.set_xlim(-0.55, 1.85)
axL.legend(fontsize=9, loc="upper left")
axL.set_title("Total visual attention nearly doubles", fontsize=11.5)

# ---- RIGHT: the decomposition, kept but labelled so it cannot be misread as the headline
vals = [i1_t0, i1_t1, i2_t1]
errs = [ci95(S[:, 0]), ci95(S[:, 1]), ci95(S[:, 2])]
cols = ["#9ecae1", "#9ecae1", "#e6550d"]
bars = axR.bar([0, 1, 2], vals, yerr=errs, capsize=4, color=cols, width=0.6,
               error_kw=dict(ecolor="#2c3e50", lw=1.4))
for b, v in zip(bars, vals):
    axR.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=10)
axR.set_xticks([0, 1, 2])
axR.set_xticklabels(["turn-0 reasoning\n→ original", "turn-1 reasoning\n→ original",
                     "turn-1 reasoning\n→ RE-INJECTED"], fontsize=9.5)
axR.set_ylabel("mean attention weight")
axR.set_ylim(0, 0.16)
axR.set_title("Component view: the fresh copy (0.120) outdraws\nthe original at its own peak (0.097)",
              fontsize=10.5)

fig.suptitle(f"Forced re-injection genuinely re-engages vision  "
             f"(within-segment, n={n})", fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_forced_reengage.png", bbox_inches="tight")
plt.close(fig)

print(f"n = {n}")
print(f"i1_t0 (turn-0 -> original)     = {i1_t0:.4f}  ±{ci95(S[:,0]):.4f}")
print(f"i1_t1 (turn-1 -> original)     = {i1_t1:.4f}  ±{ci95(S[:,1]):.4f}")
print(f"i2_t1 (turn-1 -> re-injected)  = {i2_t1:.4f}  ±{ci95(S[:,2]):.4f}")
print(f"tot0                            = {tot0:.4f}")
print(f"tot1 = i1_t1 + i2_t1            = {tot1:.4f}  ±{ci95(tot1_per_sample):.4f}")
print(f"change                          = {(tot1/tot0-1)*100:+.1f}%")
print(f"\nwrote {OUT/'fig_forced_reengage.png'}")
