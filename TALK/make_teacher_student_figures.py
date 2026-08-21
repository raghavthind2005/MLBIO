#!/usr/bin/env python3
"""
Two slide diagrams for "How do we formulate the teacher?"

  fig_teacher_student_conditioning.png — one policy, two contexts (S = privileged information)
  fig_teacher_student_kl.png           — the loss, and the two directions it can be driven
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

BLUE, RED, GOLD, GREEN = "#2471a3", "#c0392b", "#d68910", "#1e8449"


def box(ax, x, y, w, h, fc, ec, txt, fs=10, tc="black", bold=False, lw=1.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.045",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, p1, p2, c, lw=2.0, conn=None, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                                 color=c, lw=lw, zorder=5,
                                 connectionstyle=conn or "arc3,rad=0"))


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1 — one policy, two contexts
# ═══════════════════════════════════════════════════════════════════════
W, H = 12.2, 4.8
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

ax.text(W/2, 4.52, "The teacher is the SAME model — it just knows one more thing",
        ha="center", fontsize=13.5, fontweight="bold", color="#2c3e50")

# shared policy
box(ax, 0.30, 1.72, 1.85, 1.30, "#f4f6f7", "#2c3e50", "$\\pi_\\theta$", fs=26, lw=2.2)
ax.text(1.225, 1.54, "one set of weights", ha="center", va="top",
        fontsize=9.5, color="#2c3e50", fontweight="bold")

CW, CH = 1.28, 0.66

# ---------- STUDENT
ys = 3.14
arrow(ax, (2.20, 2.62), (2.85, ys + CH/2), BLUE, conn="arc3,rad=-0.18")
box(ax, 2.95, ys, CW, CH, "#eaf3fa", BLUE, "image $I$", fs=10.5, tc=BLUE, bold=True)
box(ax, 4.38, ys, CW, CH, "#eaf3fa", BLUE, "question $Q$", fs=10.5, tc=BLUE, bold=True)
ax.text(5.98, ys + CH/2, "$\\longrightarrow$", fontsize=17, va="center", color=BLUE)
box(ax, 6.60, ys - 0.09, 3.00, 0.84, "#ffffff", BLUE,
    "$\\pi_\\theta(\\,\\cdot \\mid I, Q\\,)$", fs=16, tc=BLUE)
ax.text(10.95, ys + CH/2 + 0.09, "STUDENT", fontsize=13, color=BLUE, fontweight="bold",
        va="center", ha="center")
ax.text(10.95, ys + CH/2 - 0.24, "what we deploy", fontsize=9, color=BLUE,
        va="center", ha="center", style="italic")

# ---------- TEACHER
yt = 1.02
arrow(ax, (2.20, 2.12), (2.85, yt + CH/2), RED, conn="arc3,rad=0.18")
box(ax, 2.95, yt, CW, CH, "#fdeeec", RED, "image $I$", fs=10.5, tc=RED, bold=True)
box(ax, 4.38, yt, CW, CH, "#fdeeec", RED, "question $Q$", fs=10.5, tc=RED, bold=True)
box(ax, 5.81, yt, CW, CH, "#fdf3e2", GOLD, "$\\mathbf{S}$", fs=15, tc=GOLD, bold=True, lw=2.4)
ax.text(6.45, yt - 0.16, "privileged information", ha="center", va="top",
        fontsize=9.5, color=GOLD, fontweight="bold")

ax.text(7.42, yt + CH/2, "$\\longrightarrow$", fontsize=17, va="center", color=RED)
box(ax, 8.02, yt - 0.09, 2.62, 0.84, "#ffffff", RED,
    "$\\pi_\\theta(\\,\\cdot \\mid I, Q, S\\,)$", fs=15, tc=RED)
ax.text(11.40, yt + CH/2 + 0.09, "TEACHER", fontsize=13, color=RED, fontweight="bold",
        va="center", ha="center")
ax.text(11.40, yt + CH/2 - 0.24, "train-time only", fontsize=9, color=RED,
        va="center", ha="center", style="italic")

fig.savefig(OUT / "fig_teacher_student_conditioning.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT/'fig_teacher_student_conditioning.png'}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — the loss, and the two directions
# ═══════════════════════════════════════════════════════════════════════
W2, H2 = 11.0, 5.6
fig = plt.figure(figsize=(W2, H2))
ax = fig.add_axes([0.09, 0.30, 0.86, 0.42])

vocab = ["cube", "sphere", "left", "behind", "red", "two", "none"]
student = np.array([0.10, 0.09, 0.26, 0.08, 0.22, 0.07, 0.18])
teacher = np.array([0.05, 0.06, 0.52, 0.05, 0.09, 0.05, 0.18])

xx = np.arange(len(vocab)); w = 0.38
ax.bar(xx - w/2, student, w, color=BLUE, label="student  $\\pi_\\theta(\\cdot \\mid I,Q)$")
ax.bar(xx + w/2, teacher, w, color=RED, alpha=0.9,
       label="teacher  $\\pi_\\theta(\\cdot \\mid I,Q,S)$")
ax.set_xticks(xx); ax.set_xticklabels(vocab, fontsize=10)
ax.set_ylim(0, 0.62); ax.set_ylabel("P(next token)", fontsize=10.5)
ax.grid(alpha=0.25, axis="y"); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=10, loc="upper left", framealpha=0.95)

# the loss
fig.text(0.5, 0.885,
         "$\\mathcal{L}(\\theta)\\;=\\;\\sum_t D_{KL}\\!\\left("
         "\\pi_\\theta(\\cdot \\mid y_{<t}, I, Q)\\;\\;\\|\\;\\;"
         "\\pi_\\theta(\\cdot \\mid y_{<t}, I, Q, S)\\right)$",
         ha="center", fontsize=17)
fig.text(0.5, 0.795, "a per-token distance between the two distributions",
         ha="center", fontsize=10.5, color="#555", style="italic")

# the two directions
fig.patches.append(FancyBboxPatch((0.09, 0.045), 0.40, 0.175,
                                  boxstyle="round,pad=0.012", transform=fig.transFigure,
                                  fc="#eafaf1", ec=GREEN, lw=1.8, zorder=1))
fig.text(0.29, 0.163, "minimise  $\\mathcal{L}$", ha="center", fontsize=13.5,
         color=GREEN, fontweight="bold")
fig.text(0.29, 0.082, "the two distributions CONVERGE", ha="center",
         fontsize=11, color=GREEN)

fig.patches.append(FancyBboxPatch((0.545, 0.045), 0.40, 0.175,
                                  boxstyle="round,pad=0.012", transform=fig.transFigure,
                                  fc="#fdeeec", ec=RED, lw=1.8, zorder=1))
fig.text(0.745, 0.163, "maximise  $\\mathcal{L}$", ha="center", fontsize=13.5,
         color=RED, fontweight="bold")
fig.text(0.745, 0.082, "the two distributions DIVERGE", ha="center",
         fontsize=11, color=RED)

fig.savefig(OUT / "fig_teacher_student_kl.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT/'fig_teacher_student_kl.png'}")
