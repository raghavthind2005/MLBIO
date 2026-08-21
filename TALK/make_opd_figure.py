#!/usr/bin/env python3
"""
On-policy distillation schematic for the Act 4 -> Act 5 bridge slide.

The point the figure must make: teacher and student are THE SAME WEIGHTS, differing only in whether
the perceptual content has been serialised into the context. So the per-token KL between them IS the
extraction deficit, measured at every position of the model's own trajectory.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

BLUE, RED, GREEN = "#2980b9", "#c0392b", "#1e8449"
W, H = 13.0, 6.4

fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, fc, ec, txt, fs=9.5, tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                fc=fc, ec=ec, lw=1.6, zorder=2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3)


def arrow(p1, p2, c, lw=2.0, style="-|>", conn=None):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=15,
                                 color=c, lw=lw, zorder=4,
                                 connectionstyle=conn or "arc3,rad=0"))


LX, LW = 0.30, 4.15                       # left column
CX = LX + LW / 2

# ══════════════════ LEFT COLUMN ══════════════════
ax.text(CX, 6.15, "one set of weights $\\pi_\\theta$ — only the CONTEXT differs",
        ha="center", fontsize=10.5, style="italic", color="#333")

box(LX, 5.20, LW, 0.62, "#eaf3fa", BLUE,
    "STUDENT  ·  [image $I$] + question $x$", fs=10, tc=BLUE, bold=True)

box(LX, 3.95, LW, 0.72, "#eaf3fa", BLUE,
    "samples its OWN rollout\n$y \\sim \\pi_\\theta(\\cdot \\mid I, x)$", fs=9.8, tc=BLUE)
arrow((CX, 5.18), (CX, 4.72), BLUE)
ax.text(CX + 0.14, 4.95, "on-policy", fontsize=8.6, color=BLUE, va="center")

box(LX, 2.55, LW, 0.62, "#fdeeec", RED,
    "TEACHER  ·  [image $I$] + $\\bf{caption\\ c}$ + $x$", fs=10, tc=RED, bold=True)
ax.text(CX, 2.38, "$c \\sim \\pi_\\theta^{\\rm Stage\\,1}(\\cdot \\mid I, x)$ — the model's own serialisation,\n"
                  "so the teacher's advantage is always derivable  (imitation gap = 0)",
        ha="center", va="top", fontsize=8.5, color="#555")

# ══════════════════ RIGHT: tokens, KL bars, loss ══════════════════
toks = ["the", "left", "sphere", "is", "behind", "the", "cube"]
x0, tw, gap = 5.15, 0.92, 0.07
TOK_Y, TOK_H = 3.30, 0.60
BAR_BASE = 4.15

for i, t in enumerate(toks):
    box(x0 + i * (tw + gap), TOK_Y, tw, TOK_H, "#f4f6f7", "#95a5a6", t, fs=8.4)

mid = x0 + 3.5 * (tw + gap)
ax.text(mid, TOK_Y - 0.24,
        "the student's own trajectory — BOTH distributions are scored on THESE tokens",
        ha="center", fontsize=8.8, color="#555", style="italic")

kl = np.array([0.05, 0.42, 0.88, 0.07, 0.71, 0.04, 0.31])
for i, k in enumerate(kl):
    cx = x0 + i * (tw + gap) + tw / 2
    ax.add_patch(plt.Rectangle((cx - tw * 0.32, BAR_BASE), tw * 0.64, k * 1.05,
                               fc=RED if k > 0.3 else "#e3b4af", ec="none", zorder=3))
ax.plot([x0 - 0.16, x0 - 0.16], [BAR_BASE, BAR_BASE + 1.02], color="#bbb", lw=1)
ax.text(x0 - 0.28, BAR_BASE + 0.5, "per-token\n$D_{KL}$", ha="right", va="center",
        fontsize=9, color=RED, fontweight="bold")

# arrows into the token strip
arrow((LX + LW, 4.30), (x0 - 0.05, 3.75), BLUE, conn="arc3,rad=-0.18")
arrow((LX + LW, 2.86), (x0 - 0.05, 3.35), RED, conn="arc3,rad=0.18")

# loss
ax.text(mid, 5.72,
        "$\\mathcal{L}_{\\rm OPD}(\\theta)=\\mathbb{E}_{y\\sim\\pi_\\theta(\\cdot|I,x)}"
        "\\left[\\ \\sum_t D_{KL}\\left(\\pi_\\theta(\\cdot|y_{<t},I,x)\\ \\|\\ "
        "\\pi_\\theta(\\cdot|y_{<t},I,c,x)\\right)\\right]$",
        ha="center", fontsize=13)
ax.text(mid, 5.36, "MINIMISED — an attractive KL toward a strictly better-informed teacher",
        ha="center", fontsize=9.2, color="#555", style="italic")

ax.annotate("large KL = a token where failing to\nextract the percept cost the model",
            xy=(x0 + 2 * (tw + gap) + tw / 2, BAR_BASE + 0.95),
            xytext=(x0 + 5.45 * (tw + gap), 5.02),
            fontsize=8.8, color=RED, ha="center",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))

# ══════════════════ gradient ══════════════════
arrow((mid + 1.5, 2.92), (4.70, 2.92), GREEN, lw=2.2)
ax.text((mid + 1.5 + 4.70) / 2, 2.68,
        "gradient reaches the STUDENT branch only\n(teacher branch: stop-gradient)",
        ha="center", va="top", fontsize=9, color=GREEN, fontweight="bold")

# ══════════════════ banner ══════════════════
ax.add_patch(FancyBboxPatch((0.30, 0.42), W - 0.60, 1.10, boxstyle="round,pad=0.04",
                            fc="#f7f9fa", ec="#bdc3c7", lw=1.4, zorder=1))
ax.text(W / 2, 1.20,
        "Teacher and student differ ONLY by whether the perception is already serialised.",
        ha="center", va="center", fontsize=11.4, fontweight="bold", color="#2c3e50")
ax.text(W / 2, 0.78,
        "→  the per-token KL IS the extraction deficit — dense, positional, and re-derived from each "
        "input, exactly the shape Act 4 requires.",
        ha="center", va="center", fontsize=10.4, color="#2c3e50")

fig.savefig(OUT / "fig_opd_schematic.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT/'fig_opd_schematic.png'}")
