#!/usr/bin/env python3
"""
ACT 3 figures — "attention is not the bottleneck".

PROVENANCE
  - Set 2 probe / causal controls: RL_SeeingToThinking/runs/set2_perception_drift/
    SET2_EXPERIMENT_RECORD.md (L52, L100-103, L111) and NEGATIVE_RESULTS.md (N1).
    Raw `out/` is gitignored and lives on the cluster, so values are transcribed from the record.
  - babyVision: babyVision/RESULTS.md Finding 5.
    NOTE: standard and B1' are stated directly (perception 15.7 -> 22.5; reasoning 39.6 -> 33.0).
    B2' is stated only as a DELTA ("perception +5.2, reasoning -7.6"), so its absolute values
    below are derived: 15.7+5.2 = 20.9 and 39.6-7.6 = 32.0. Flagged on the figure.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

# ===================================================== FIG 3.4a — no stored percept
fig, ax = plt.subplots(figsize=(7.4, 4.4))
labels = ["image-token\npositions\n(positive control)",
          "text / reasoning\npositions\nPCA-16", "PCA-32", "PCA-64"]
vals = [0.918, 0.504, 0.511, 0.517]
cols = ["#2c3e50", "#c0392b", "#c0392b", "#c0392b"]
bars = ax.bar(range(4), vals, color=cols, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.3f}",
            ha="center", fontweight="bold", fontsize=11)
ax.axhline(0.5, color="grey", ls="--", lw=1.6)
ax.text(3.45, 0.512, "chance", fontsize=9.5, color="grey", ha="right")
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("linear-probe balanced accuracy")
ax.set_ylim(0.4, 1.0)
ax.set_title("No stored scene-percept in the reasoning stream\n"
             "The probe works on image tokens and fails everywhere else", fontsize=11.5)
fig.text(0.5, -0.06, "CLEVR, 15 attribute-marginals, ridge on teacher-forced hidden states, n=152 correct items.\n"
                     "Chance result holds across all PCA-k, all 6 sampled layers, all positions.",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout()
fig.savefig(OUT / "fig_set2_probe.png", bbox_inches="tight")
plt.close(fig)

# ===================================================== FIG 3.4b — the instrument works
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})

names = ["normal\n(no injection)", "inline splice\n(mid-assistant)", "user turn\n(real image)",
         "user turn\n(scrambled)", "no image\nat all"]
vals = [1.00, 1.00, 0.80, 0.00, 0.00]
cols = ["#7f8c8d", "#27ae60", "#27ae60", "#c0392b", "#c0392b"]
bars = axL.bar(range(5), vals, color=cols, width=0.62)
for b, v in zip(bars, vals):
    axL.text(b.get_x() + b.get_width()/2, v + 0.03, f"{v:.2f}", ha="center", fontweight="bold")
axL.set_xticks(range(5)); axL.set_xticklabels(names, fontsize=9)
axL.set_ylabel("accuracy")
axL.set_ylim(0, 1.15)
axL.set_title("The injection channel is real and content-bearing\n(A0 mechanics, 5 easy items, greedy)",
              fontsize=10.5)

axR.bar([0, 1], [0.90, 0.40], color=["#7f8c8d", "#c0392b"], width=0.55)
for x, v in zip([0, 1], [0.90, 0.40]):
    axR.text(x, v + 0.03, f"{v:.2f}", ha="center", fontweight="bold", fontsize=11)
axR.annotate("", xy=(1, 0.40), xytext=(0, 0.90),
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2, ls="--"))
axR.set_xticks([0, 1])
axR.set_xticklabels(["re-inject the\nSAME image", "re-inject a\nCONFLICTING image"], fontsize=10)
axR.set_ylabel("accuracy"); axR.set_ylim(0, 1.1)
axR.set_title("Same image = attended but INERT.\nDifferent image MOVES answers. (n=10)", fontsize=10.5)

fig.suptitle("The pixel nulls are real results, not a broken instrument",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_set2_causal_control.png", bbox_inches="tight")
plt.close(fig)

# ===================================================== FIG 3.3 — babyVision, with the B2' control
fig, ax = plt.subplots(figsize=(8.2, 4.6))
groups = ["perception tasks\n(counting, search, tracing)", "reasoning tasks\n(rotation, folding, overlay)"]
standard = [15.7, 39.6]
b1 = [22.5, 33.0]          # image RE-SHOWN
b2 = [20.9, 32.0]          # image NOT re-shown  (derived: 15.7+5.2, 39.6-7.6)
x = np.arange(2); w = 0.26
r1 = ax.bar(x - w, standard, w, label="standard (one look)", color="#95a5a6")
r2 = ax.bar(x,     b1,       w, label="B1′  re-grounding + own reasoning (image re-shown)", color="#2980b9")
r3 = ax.bar(x + w, b2,       w, label="B2′  same, image NOT re-shown", color="#8e44ad")
for rr in (r1, r2, r3):
    for b in rr:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.7,
                f"{b.get_height():.1f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=10)
ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 50)
ax.legend(fontsize=8.5, loc="upper left")
ax.annotate("+6.8  (p=0.015)", xy=(0 - 0.02, 26.8), fontsize=9.5, color="#1e6091", fontweight="bold", ha="center")
ax.annotate("−6.6  (p=0.12, trend)", xy=(1 - 0.02, 44.0), fontsize=9.5, color="#c0392b", fontweight="bold", ha="center")
ax.set_title("Re-examining with your own reasoning helps perception, hurts reasoning —\n"
             "and re-showing the image is NOT what does it", fontsize=11.5)
fig.text(0.5, -0.05,
         "B1′ vs B2′ differ ONLY in whether the image is re-shown: just 9 of 388 items change. "
         "The driver is the reasoning, not the image.\n"
         "B2′ absolute values derived from the deltas stated in babyVision/RESULTS.md (+5.2 / −7.6).",
         ha="center", fontsize=8.5, color="#555")
fig.tight_layout()
fig.savefig(OUT / "fig_babyvision_dissociation.png", bbox_inches="tight")
plt.close(fig)

print("wrote:")
for f in ["fig_set2_probe.png", "fig_set2_causal_control.png", "fig_babyvision_dissociation.png"]:
    print(f"  {OUT/f}")
