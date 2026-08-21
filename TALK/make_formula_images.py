#!/usr/bin/env python3
"""
Render the OPD formulas as transparent PNGs for direct drop-in to slides.
Each is cropped tight, 300 dpi, transparent background — scales cleanly in PowerPoint/Keynote.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "figures" / "formulas"
OUT.mkdir(parents=True, exist_ok=True)

FORMULAS = {
    "f1_opd_objective": (
        r"$\mathcal{L}_{\rm OPD}(\theta)\;=\;\mathbb{E}_{x}\;\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}"
        r"\left[\;\sum_{t}\,D_{KL}\left(\pi_\theta(\cdot\mid y_{<t},x)\;\;\|\;\;"
        r"\pi_T(\cdot\mid y_{<t},x)\right)\right]$", 22),

    "f2_instantiation": (
        r"$\pi_T(\cdot\mid y_{<t},x)\;=\;\pi_\theta(\cdot\mid y_{<t},\,I,\,c,\,x)$"
        "\n\n"
        r"$\pi_S(\cdot\mid y_{<t},x)\;=\;\pi_\theta(\cdot\mid y_{<t},\,I,\,x)$"
        "\n\n"
        r"$c\;\sim\;\pi_\theta^{\rm Stage\,1}(\cdot\mid I,x)$", 20),

    "f3_full": (
        r"$\mathcal{L}_{\rm OPD}(\theta)=\mathbb{E}_{y\sim\pi_\theta(\cdot\mid I,x)}"
        r"\left[\;\sum_{t} D_{KL}\left(\pi_\theta(\cdot\mid y_{<t},I,x)\;\;\|\;\;"
        r"\pi_\theta(\cdot\mid y_{<t},I,c,x)\right)\right]$", 21),

    "f4_stage1_caption": (
        r"$D(c)\;=\;D_{KL}\left(\pi_\theta(\cdot\mid c,x)\;\;\|\;\;"
        r"\pi_\theta(\cdot\mid I,x)\right)$", 22),

    "f5_papo_contrast": (
        r"$\mathrm{PAPO:}\;\;\max_\theta\;\;D_{KL}\left(\pi_\theta(\cdot\mid I)\;\|\;"
        r"\pi_\theta(\cdot\mid \tilde{I})\right)$"
        "\n\n"
        r"$\mathrm{Ours:}\;\;\;\;\min_\theta\;\;D_{KL}\left(\pi_\theta(\cdot\mid I)\;\|\;"
        r"\pi_\theta(\cdot\mid I,c)\right)$", 20),
}

for name, (tex, size) in FORMULAS.items():
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, tex, fontsize=size, color="#1a1a1a")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight",
                pad_inches=0.12, transparent=True)
    plt.close(fig)
    print(f"wrote {OUT / (name + '.png')}")

# dark-slide variants
for name, (tex, size) in FORMULAS.items():
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, tex, fontsize=size, color="white")
    fig.savefig(OUT / f"{name}_dark.png", dpi=300, bbox_inches="tight",
                pad_inches=0.12, transparent=True)
    plt.close(fig)
print(f"\n(+ *_dark.png variants for dark slides)")
