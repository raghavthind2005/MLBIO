#!/usr/bin/env python3
"""
ACT 2 figures — "we replicate the phenomenon".

PROVENANCE
  - Attention decay by thirds: computed live from
    image_toolCalling/results_{normal,tool,forced}/attention_results.jsonl
    (fields attn_{visual,system,instruction}_by_thirds). Gemma-4-31B-it, HallusionBench 30% subset.
  - Quartile accuracy: image_toolCalling/clarifications.md (verbatim tables).
  - RH-Bench: RH-Bench/Qwen3-VL-4B-Thinking_Results.md (verbatim).
"""
import json, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ITC  = HERE.parent / "image_toolCalling"
OUT  = HERE / "figures"; OUT.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

# ============================================ FIG 2.1b — the decay is NOT vision-specific
conds = {"standard": "results_normal", "tool": "results_tool", "forced": "results_forced"}
data = {}
for cname, d in conds.items():
    rows = [json.loads(l) for l in open(ITC / d / "attention_results.jsonl")]
    data[cname] = {ch: [st.mean([r[f"attn_{ch}_by_thirds"][k] for r in rows])
                        for k in ("early", "mid", "late")]
                   for ch in ("visual", "system", "instruction")}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})

ax = axes[0]
x = [0, 1, 2]
styles = {"visual": ("#2980b9", "-", "o"), "system": ("#7f8c8d", "--", "s"),
          "instruction": ("#c0392b", "--", "^")}
for ch, (c, ls, mk) in styles.items():
    y = data["standard"][ch]
    pct = 100 * (y[2] - y[0]) / y[0]
    ax.plot(x, y, color=c, ls=ls, marker=mk, lw=2, ms=6,
            label=f"{ch}  ({pct:+.0f}%)")
ax.set_xticks(x); ax.set_xticklabels(["early", "mid", "late"])
ax.set_xlabel("position within the reasoning chain")
ax.set_ylabel("mean attention weight")
ax.set_yscale("log")
ax.legend(fontsize=9.5)
ax.set_title("Standard condition: every prompt channel decays", fontsize=11)

ax = axes[1]
chans = ["visual", "system", "instruction"]
pcts = [100 * (data["standard"][ch][2] - data["standard"][ch][0]) / data["standard"][ch][0]
        for ch in chans]
cols = ["#2980b9", "#7f8c8d", "#c0392b"]
bars = ax.barh(chans, pcts, color=cols, height=0.55)
for b, p in zip(bars, pcts):
    ax.text(p + 1.5, b.get_y() + b.get_height()/2, f"{p:.1f}%",
            va="center", ha="left", color="white", fontweight="bold", fontsize=10)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("change in attention, early → late (%)")
ax.set_xlim(-58, 2)
ax.tick_params(axis="y", labelsize=10)
ax.invert_yaxis()
ax.set_title("Visual decay is not the largest — instruction decays more", fontsize=11)

fig.suptitle("Attention decay replicates — but it is NOT vision-specific",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_attention_decay_control.png", bbox_inches="tight")
plt.close(fig)

# ============================================ FIG 2.3 — accuracy collapse by reasoning length
# from image_toolCalling/clarifications.md
Q = ["Q1\n(shortest)", "Q2", "Q3", "Q4\n(longest)"]
edited = {                       # vi=2, prior-contradicting images
    "standard":       [93.9, 79.4, 72.7, 73.5],
    "voluntary tool": [87.5, 81.2, 68.8, 62.5],
    "forced (turn0)": [94.4, 75.0, 69.4, 52.8],
}
allimg = {
    "standard":       [93.8, 83.1, 87.5, 69.2],
    "voluntary tool": [93.5, 82.5, 80.6, 69.8],
    "forced (turn0)": [94.1, 83.8, 79.4, 60.3],
}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
for ax, (title, dct) in zip(axes, [("Edited images (vi=2) — prior-contradicting", edited),
                                   ("All images", allimg)]):
    for (name, y), c in zip(dct.items(), ["#c0392b", "#e67e22", "#2c3e50"]):
        ax.plot(Q, y, marker="o", lw=2, ms=6, color=c, label=name)
    ax.axhline(50, color="grey", ls=":", lw=1.4)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(45, 100)
axes[0].text(0.05, 51.5, "chance (binary questions)", fontsize=8.5, color="grey")
axes[0].set_ylabel("accuracy (%)")
axes[0].legend(fontsize=9.5, loc="lower left")
axes[0].annotate("94.4 → 52.8", xy=(3, 52.8), xytext=(-88, 20), textcoords="offset points",
                 fontsize=10, fontweight="bold", color="#c0392b",
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4))
for ax in axes:
    ax.set_xlabel("reasoning-length quartile")
fig.suptitle("Accuracy falls as the chain lengthens — sharpest where the image contradicts the prior",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_length_collapse.png", bbox_inches="tight")
plt.close(fig)

# ============================================ FIG 2.4 — RH-Bench
# from RH-Bench/Qwen3-VL-4B-Thinking_Results.md
fig, ax = plt.subplots(figsize=(6.6, 4.2))
groups = ["multi-choice", "free-form", "OVERALL"]
reason = [78.2, 60.9, 69.3]
percep = [73.7, 55.0, 64.4]
xx = np.arange(len(groups)); w = 0.36
b1 = ax.bar(xx - w/2, reason, w, label="reasoning subset (n=450)", color="#2c3e50")
b2 = ax.bar(xx + w/2, percep, w, label="perception subset (n=450)", color="#c0392b")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f"{b.get_height():.1f}", ha="center", fontsize=9.5, fontweight="bold")
ax.set_xticks(xx); ax.set_xticklabels(groups)
ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 90)
ax.legend(fontsize=9.5)
ax.set_title("RH-Bench, Qwen3-VL-4B-Thinking:\nreasoning 69.3% > perception 64.4% (gap 4.9pp)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig_rhbench.png", bbox_inches="tight")
plt.close(fig)

print("decay, standard condition (early -> late):")
for ch in chans:
    y = data["standard"][ch]
    print(f"  {ch:12} {y[0]:.4f} -> {y[2]:.4f}  ({100*(y[2]-y[0])/y[0]:+.1f}%)")
print("\nwrote:")
for f in ["fig_attention_decay_control.png", "fig_length_collapse.png", "fig_rhbench.png"]:
    print(f"  {OUT/f}")
