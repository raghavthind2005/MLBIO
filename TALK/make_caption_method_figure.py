#!/usr/bin/env python3
"""
Simple explainer for the caption-distortion method.
Deliberately plain: three steps, one comparison, one formula.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

BLUE, RED, GREY, DARK = "#2471a3", "#c0392b", "#7f8c8d", "#2c3e50"

W, H = 11.5, 5.8
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, fc, ec, txt, fs=11, tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=fc, ec=ec, lw=1.8, zorder=2))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3)


def arrow(p1, p2, c, lw=2.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                                 color=c, lw=lw, zorder=4))


ax.text(W/2, 5.45, "Describe the image well enough to answer without it",
        ha="center", fontsize=14, fontweight="bold", color=DARK)

# ── top row: the two paths
box(0.35, 3.55, 2.05, 0.80, "#eaf3fa", BLUE, "image  $I$\n+ question $x$", fs=11, tc=BLUE, bold=True)
arrow((2.45, 3.95), (3.25, 3.95), BLUE)
box(3.30, 3.55, 2.35, 0.80, "#ffffff", BLUE, "answer\n$\\pi(\\cdot \\mid I, x)$", fs=11.5, tc=BLUE)

# ── the caption path
box(0.35, 1.55, 2.05, 0.80, "#fdf3e2", "#d68910", "caption  $c$\n+ question $x$", fs=11,
    tc="#a5680a", bold=True)
arrow((2.45, 1.95), (3.25, 1.95), "#d68910")
box(3.30, 1.55, 2.35, 0.80, "#ffffff", "#d68910", "answer\n$\\pi(\\cdot \\mid c, x)$", fs=11.5,
    tc="#a5680a")

# writing the caption
arrow((1.375, 3.50), (1.375, 2.40), GREY)
ax.text(1.52, 2.95, "the model looks\nand writes $c$", fontsize=9.5, color=GREY, va="center")
ax.text(1.375, 1.35, "image taken away", fontsize=9.5, color="#a5680a",
        ha="center", va="top", style="italic")

# ── the comparison
ax.add_patch(FancyArrowPatch((5.85, 3.95), (5.85, 2.35), arrowstyle="<->",
                             mutation_scale=18, color=DARK, lw=2.4, zorder=4))
ax.text(6.05, 3.15, "same?", fontsize=13, color=DARK, fontweight="bold", va="center")

# ── the objective
box(6.95, 2.55, 4.20, 1.45, "#f7f9fa", DARK,
    "$D(c)=D_{KL}\\!\\left(\\pi(\\cdot \\mid c,x)\\;\\|\\;\\pi(\\cdot \\mid I,x)\\right)$",
    fs=13, tc=DARK)
ax.text(9.05, 4.15, "how much do the answers change\nwhen the caption replaces the picture?",
        ha="center", va="bottom", fontsize=10, color=GREY, style="italic")
ax.text(9.05, 2.30, "train the caption to make this SMALL",
        ha="center", va="top", fontsize=11.5, color=DARK, fontweight="bold")

# ── bottom line
ax.add_patch(FancyBboxPatch((0.35, 0.20), W - 0.70, 0.72, boxstyle="round,pad=0.05",
                            fc="#eafaf1", ec="#1e8449", lw=1.6, zorder=1))
ax.text(W/2, 0.56,
        "No human captions. The target is the model's own behaviour when it can see.",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#145a32")

fig.savefig(OUT / "fig_caption_method.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT/'fig_caption_method.png'}")
