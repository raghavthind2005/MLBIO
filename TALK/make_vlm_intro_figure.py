#!/usr/bin/env python3
"""
Intro slide figure: what a VLM is (ViT + connector + LLM), and the puzzle.
Deliberately plain.

Numbers:
  Qwen3-VL-4B split 0.42B vision / 4.0B language — RL_SeeingToThinking/runs/analysis/SLIDES.md
  babyVision 31.6% (Gemma-4-31B-it, 388 items)   — babyVision/RESULTS.md Finding 1
  BlindTest 58.57% avg over 4 VLMs               — Rahmanzadehgervi et al., ACCV 2024
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

BLUE, GREY, DARK, RED = "#2471a3", "#7f8c8d", "#2c3e50", "#c0392b"

W, H = 11.5, 5.4
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, fc, ec, txt, fs=11, tc="black", bold=False, lw=1.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3)


def arrow(p1, p2, c, lw=2.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15,
                                 color=c, lw=lw, zorder=4))


# ───────── architecture row
ax.text(0.35, 5.05, "A VLM is an LLM that has been given eyes",
        fontsize=13.5, fontweight="bold", color=DARK, va="top")

y, h = 3.55, 0.95
box(0.35, y, 1.55, h, "#f4f6f7", GREY, "image", fs=11.5, tc=DARK)
arrow((1.98, y + h/2), (2.38, y + h/2), GREY)
box(2.45, y, 1.75, h, "#eaf3fa", BLUE, "ViT", fs=14, tc=BLUE, bold=True)
arrow((4.28, y + h/2), (4.68, y + h/2), GREY)
box(4.75, y, 1.55, h, "#eaf3fa", BLUE, "connector", fs=10.5, tc=BLUE)
arrow((6.38, y + h/2), (6.78, y + h/2), GREY)
box(6.85, y, 3.05, h, "#e8eaed", DARK, "LLM", fs=17, tc=DARK, bold=True, lw=2.2)
arrow((9.98, y + h/2), (10.38, y + h/2), GREY)
ax.text(10.95, y + h/2, "answer", fontsize=11.5, color=DARK, va="center", ha="center")

ax.text(3.32, y - 0.22, "0.42 B", fontsize=10.5, color=BLUE, ha="center", va="top",
        fontweight="bold")
ax.text(8.37, y - 0.22, "4.0 B", fontsize=10.5, color=DARK, ha="center", va="top",
        fontweight="bold")
ax.text(6.0, y - 0.62, "Qwen3-VL-4B  —  about nine-tenths of the model is the language model",
        fontsize=9.5, color=GREY, ha="center", va="top", style="italic")

# ───────── the puzzle
ax.plot([0.35, 11.15], [2.42, 2.42], color="#d5d8dc", lw=1.4)
ax.text(0.35, 2.20, "The same weights:", fontsize=12, fontweight="bold",
        color=DARK, va="top")

box(0.35, 0.95, 4.95, 1.00, "#eef7f0", "#1e8449",
    "given TEXT\n\nmulti-step maths · code · graduate-level questions",
    fs=11, tc="#145a32")

box(5.85, 0.95, 5.30, 1.00, "#fdeeec", RED,
    "given a PICTURE\n\n31.6% on puzzles young children pass\n"
    "58.6% at telling if two circles overlap",
    fs=11, tc=RED)

ax.text(W/2, 0.62, "counting · tracing a line · do these shapes intersect",
        ha="center", va="top", fontsize=10, color=GREY, style="italic")
ax.text(W/2, 0.24, "Not the hard part of any task — and exactly where it breaks.",
        ha="center", va="top", fontsize=12, fontweight="bold", color=DARK)

fig.savefig(OUT / "fig_vlm_intro.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT/'fig_vlm_intro.png'}")
