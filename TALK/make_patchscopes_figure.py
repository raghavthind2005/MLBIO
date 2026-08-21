#!/usr/bin/env python3
"""
Patchscopes belief-readout figure — companion to fig_set2_probe.png (the linear-probe figure).

PROVENANCE — RL_SeeingToThinking/runs/set2_perception_drift/SET2_EXPERIMENT_RECORD.md
  §3 (Phase 0, L60-62): Patchscopes = patch a hidden state into the readout prompt "Answer: \\boxed{"
      and read the distribution over CLEVR answer tokens; logit-lens = final-norm + unembed.
      At the answer position: LL@ans = 0.92, PS@ans = 1.0 (best at layer 30).
      At a single mid-<think> position with hard argmax: "= chance" (NO numeric value is recorded).
  §4 (Phase 1, L70-85): Patchscopes @ layer 30, margin(t) = P(gt) - P(wrong) at ~12 positions.
      Hard pool (39+39): RIPE FLIP 2/39, correct FLIP 4/39.
      Offline subtype breakdown (pooled 56 RIPE + 56 correct) as plotted in panel C.

⚠ Two numbers disagree between the record and NEGATIVE_RESULTS.md. The RECORD is primary and is used:
    - original-pool flips: record "FLIP 5/17 (strict </think> criterion)"  vs  ledger "~8/17"
    - flipAns column:      record "18 for RIPE vs 0"                        vs  ledger "11 vs 0"
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.6, 4.5),
                                    gridspec_kw={"width_ratios": [1, 0.78, 1.18]})

# ---------------------------------------------------------------- A: instrument validation
labels = ["Patchscopes\n@ answer pos\n(layer 30)", "logit-lens\n@ answer pos",
          "Patchscopes\n@ mid-<think>\nhard argmax"]
vals = [1.00, 0.92, 0.50]
bars = axA.bar(range(3), vals, width=0.6,
               color=["#2c3e50", "#34495e", "#c0392b"])
bars[2].set_hatch("///"); bars[2].set_edgecolor("white"); bars[2].set_linewidth(0)
axA.text(0, 1.00 + 0.025, "1.00", ha="center", fontweight="bold", fontsize=11)
axA.text(1, 0.92 + 0.025, "0.92", ha="center", fontweight="bold", fontsize=11)
axA.text(2, 0.50 + 0.025, "≈ chance", ha="center", fontweight="bold", fontsize=10.5, color="#c0392b")
axA.axhline(0.5, color="grey", ls="--", lw=1.6)
axA.text(-0.42, 0.515, "chance", fontsize=9, color="grey", ha="left")
axA.set_xticks(range(3)); axA.set_xticklabels(labels, fontsize=9)
axA.set_ylabel("belief-readout accuracy")
axA.set_ylim(0.35, 1.12)
axA.set_title("A · The instrument is valid at the endpoint,\nunderpowered at a single mid-chain position",
              fontsize=10.5)

# ---------------------------------------------------------------- B: flips do not replicate
axB.bar([0, 1], [2/39*100, 4/39*100], width=0.55, color=["#c0392b", "#7f8c8d"])
for x, (k, n) in zip([0, 1], [(2, 39), (4, 39)]):
    axB.text(x, k/n*100 + 0.4, f"{k}/{n}", ha="center", fontweight="bold", fontsize=11)
axB.set_xticks([0, 1])
axB.set_xticklabels(["RIPE items\n(should flip)", "CORRECT controls\n(should not)"], fontsize=9.5)
axB.set_ylabel("belief flips at `</think>`  (%)")
axB.set_ylim(0, 14)
axB.set_title("B · Controls flip MORE than the\nitems the drift story predicts", fontsize=10.5)

# ---------------------------------------------------------------- C: it is a count-0 prior
x = np.arange(2); w = 0.34
ripe = [0.476, 0.008]      # count0 (n=7), non-count0 (n=49)
corr = [0.319, 0.011]      # count0 (n=4), non-count0 (n=52)
r1 = axC.bar(x - w/2, ripe, w, label="RIPE (model was wrong)", color="#c0392b")
r2 = axC.bar(x + w/2, corr, w, label="CORRECT controls", color="#7f8c8d")
for rr in (r1, r2):
    for b in rr:
        axC.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
                 f"{b.get_height():.3f}", ha="center", fontsize=9.5)
axC.text(-0.17, 0.545, "7/7 flip", ha="center", fontsize=9, color="#c0392b", style="italic")
axC.text(0.17, 0.39, "4/4 flip", ha="center", fontsize=9, color="#555", style="italic")
axC.text(1, 0.075, "0/49", ha="center", fontsize=9, color="#c0392b", style="italic")
axC.text(1.34, 0.075, "0/52", ha="center", fontsize=9, color="#555", style="italic")
axC.set_xticks(x)
axC.set_xticklabels(["count-0 items\n(“how many … ?” → 0)", "everything else"], fontsize=9.5)
axC.set_ylabel("early belief margin   P(gt) − P(wrong)")
axC.set_ylim(0, 0.62)
axC.legend(fontsize=9, loc="upper right")
axC.set_title("C · The early “belief” is a count-0 PRIOR —\npresent just as strongly in the controls",
              fontsize=10.5)

fig.suptitle("Patchscopes belief trajectory: no maintained-then-corrupted percept — "
             "the answer is produced at emission",
             fontsize=12.5, fontweight="bold", y=1.03)
fig.tight_layout()
fig.text(0.5, -0.055,
         "Patchscopes @ layer 30, margin = P(gt) − P(wrong), ~12 positions across the chain (CLEVR, "
         "Qwen3-VL-Thinking). Panels B–C: hard pool 39+39 / pooled 56+56.\n"
         "Panel A's mid-<think> bar is drawn at the chance line: the record states “≈ chance” and "
         "gives no numeric value.",
         ha="center", fontsize=8.5, color="#555")
fig.savefig(OUT / "fig_set2_patchscopes.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT/'fig_set2_patchscopes.png'}")
