#!/usr/bin/env python3
"""
Explainer figure for the "what is on-policy distillation" slide.

Panel A — off-policy (sequence-level KD) vs on-policy distillation: who generates, who scores.
Panel B — the privileged-information ladder: the teacher's advantage IS what gets taught,
          and it only transfers if the student can eventually derive it.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

BLUE, RED, GREEN, GREY = "#2980b9", "#c0392b", "#1e8449", "#7f8c8d"
W, H = 13.8, 7.1
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, fc, ec, txt, fs=9.5, tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                fc=fc, ec=ec, lw=1.5, zorder=2))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3)


def arrow(p1, p2, c, lw=1.9, conn=None):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                 color=c, lw=lw, zorder=5,
                                 connectionstyle=conn or "arc3,rad=0"))


def tokens(x0, y, n=6, tw=0.62, gap=0.05, fc="#f4f6f7", ec="#95a5a6"):
    for i in range(n):
        box(x0 + i*(tw+gap), y, tw, 0.42, fc, ec, "", fs=8)
    return x0 + n*(tw+gap) - gap


# ═══════════════════════════ PANEL A ═══════════════════════════
ax.text(3.75, 6.82, "A  ·  Who generates, and who grades?", ha="center",
        fontsize=12.5, fontweight="bold", color="#2c3e50")

# ---------- off-policy
ax.text(0.25, 6.42, "OFF-POLICY   (sequence-level KD)", fontsize=9.4, color="#555",
        fontweight="bold", va="top")
box(0.25, 5.62, 1.75, 0.52, "#fdeeec", RED, "TEACHER", fs=10, tc=RED, bold=True)
end = tokens(2.35, 5.67, n=5)
ax.text((2.35+end)/2, 6.22, "teacher's OWN text", fontsize=8.4, color=RED, ha="center")
arrow((2.00, 5.88), (2.30, 5.88), RED)
box(5.90, 5.62, 1.55, 0.52, "#eaf3fa", BLUE, "student\ncopies", fs=9, tc=BLUE)
arrow((end+0.06, 5.88), (5.85, 5.88), GREY)
ax.text(3.75, 5.36,
        "✗  trained on the TEACHER's distribution, then operates on its own\n"
        "     →  exposure bias, compounding error",
        fontsize=8.9, color=RED, ha="center", va="top")

# ---------- on-policy
ax.text(0.25, 4.62, "ON-POLICY   DISTILLATION", fontsize=9.4, color="#555",
        fontweight="bold", va="top")

BAR_BASE = 3.62
for i in range(6):
    cx = 2.35 + i*(0.67) + 0.31
    h = [0.10, 0.34, 0.62, 0.12, 0.48, 0.22][i]
    ax.add_patch(plt.Rectangle((cx-0.17, BAR_BASE), 0.34, h,
                               fc=RED if h > 0.3 else "#e3b4af", ec="none", zorder=3))
ax.text(6.45, BAR_BASE + 0.34, "per-token  $D_{KL}$", fontsize=9, color=RED,
        va="center", ha="left", fontweight="bold")

box(0.25, 2.92, 1.75, 0.52, "#eaf3fa", BLUE, "STUDENT", fs=10, tc=BLUE, bold=True)
end2 = tokens(2.35, 2.97, fc="#eaf3fa", ec=BLUE)
arrow((2.00, 3.18), (2.30, 3.18), BLUE)
ax.text((2.35+end2)/2, 2.82, "the student's OWN rollout", fontsize=8.4, color=BLUE,
        ha="center", va="top")

box(0.25, 1.72, 1.75, 0.52, "#fdeeec", RED, "TEACHER", fs=10, tc=RED, bold=True)
ax.text(1.125, 1.56, "scores, never generates", fontsize=8.2, color=RED,
        ha="center", va="top", style="italic")
arrow((2.00, 1.98), (2.55, 2.90), RED, conn="arc3,rad=-0.25")

ax.text(4.85, 2.06,
        "✓  supervised exactly where it operates\n"
        "✓  dense signal at EVERY token, not one scalar at the end\n"
        "✓  the training distribution follows the student's current failures",
        fontsize=8.9, color=GREEN, ha="center", va="center")

ax.plot([7.62, 7.62], [0.95, 6.95], color="#d5d8dc", lw=1.6)

# ═══════════════════════════ PANEL B ═══════════════════════════
BX, BW = 7.95, 2.05
C_TRANS, C_DERIV = 11.15, 12.85

ax.text(10.7, 6.82, "B  ·  The teacher's advantage IS the lesson", ha="center",
        fontsize=12.5, fontweight="bold", color="#2c3e50")
ax.text(10.7, 6.50, "the KL transfers whatever the teacher has and the student lacks",
        ha="center", fontsize=9, color="#555", style="italic")

ax.text(BX + BW/2, 6.10, "privileged info  $z$", fontsize=9, fontweight="bold",
        color="#2c3e50", ha="center")
ax.text(C_TRANS, 6.10, "what transfers", fontsize=9, fontweight="bold",
        color="#2c3e50", ha="center")
ax.text(C_DERIV, 6.10, "derivable by\nthe student?", fontsize=9, fontweight="bold",
        color="#2c3e50", ha="center", va="center")

rows = [
    ("a LARGER model", "general capability", "n/a", GREY, "#f4f6f7"),
    ("ground-truth answer\n(oracle facts)", "confident assertion\nwithout the evidence",
     "✗   no", RED, "#fdeeec"),
    ("the percept, already\nEXTRACTED into text", "the ability\nto extract",
     "✓   yes\n(Stage 1)", GREEN, "#eafaf1"),
]
y = 5.42
for z_txt, transfer, deriv, col, fcol in rows:
    box(BX, y - 0.42, BW, 0.84, fcol, col, z_txt, fs=8.3, tc=col, bold=True)
    ax.text(C_TRANS, y, transfer, fontsize=8.7, va="center", ha="center", color="#2c3e50")
    ax.text(C_DERIV, y, deriv, fontsize=9, va="center", ha="center",
            color=col, fontweight="bold")
    y -= 1.22

ax.text(10.7, 2.20,
        "Oracle privilege breaks the transfer: the student cannot\n"
        "reproduce it at test time (the imitation gap), so it learns the\n"
        "FORM of confident perception rather than the substance.",
        ha="center", va="top", fontsize=8.8, color=RED)

# ═══════════════════════════ banner ═══════════════════════════
ax.add_patch(FancyBboxPatch((0.25, 0.15), W - 0.50, 0.66, boxstyle="round,pad=0.04",
                            fc="#f7f9fa", ec="#bdc3c7", lw=1.4, zorder=1))
ax.text(W/2, 0.48,
        "Give the teacher the percept the student failed to extract — and the KL between them "
        "teaches extraction.",
        ha="center", va="center", fontsize=11.8, fontweight="bold", color="#2c3e50")

fig.savefig(OUT / "fig_opd_explainer.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT/'fig_opd_explainer.png'}")
