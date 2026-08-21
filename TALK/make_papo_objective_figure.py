#!/usr/bin/env python3
"""
PAPO objective figure: the masked-image branch, and the KL that is MAXIMISED.

The masked image is produced by PAPO's OWN function, copied verbatim from
PAPO_clone/PAPO/verl/trainer/papo_utils.py:17-30 (patch_size=14, black_prob=0.6).
Nothing about the corruption is invented for the slide.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).parent
OUT = HERE / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

BLUE, RED, GREEN = "#2471a3", "#c0392b", "#1e8449"


# ─── verbatim from papo_utils.py ────────────────────────────────────────
def random_patch_blackening(pil_img, patch_size=14, black_prob=0.6):
    """Randomly blacken square patches in a PIL image."""
    img = np.array(pil_img).astype(np.float32)
    h, w = img.shape[:2]
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            if np.random.rand() < black_prob:
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                if img.ndim == 3:
                    img[y:y_end, x:x_end, :] = 0
                else:
                    img[y:y_end, x:x_end] = 0
    return Image.fromarray(img.astype(np.uint8))
# ────────────────────────────────────────────────────────────────────────


# ─── verbatim resize logic from verl/utils/dataset.py:process_image ────
#     PAPO config: max_pixels 1003520 (1280*28*28), min_pixels 200704 (256*28*28)
#     Masking is applied AFTER this resize (ray_trainer.py:616-617 reads
#     multi_modal_data["images"], which process_image already produced), so the
#     patches below are at the SAME scale the vision tower actually sees.
import math
MAX_PIXELS, MIN_PIXELS = 1003520, 200704

def process_image(image, min_pixels, max_pixels):
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        f = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((int(image.width * f), int(image.height * f)))
    if min_pixels is not None and (image.width * image.height) < min_pixels:
        f = math.sqrt(min_pixels / (image.width * image.height))
        image = image.resize((int(image.width * f), int(image.height * f)))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

SRC = HERE.parent / "image_toolCalling" / "examples_illustration" / "Q1_cheesecake.png"
orig = process_image(Image.open(SRC).convert("RGB"), MIN_PIXELS, MAX_PIXELS)
np.random.seed(0)
masked = random_patch_blackening(orig, patch_size=14, black_prob=0.6)
print(f"pipeline-faithful size: {orig.size}  ({orig.size[0]*orig.size[1]:,} px)")

W, H = 12.6, 6.6
fig = plt.figure(figsize=(W, H))

# ── images
axL = fig.add_axes([0.075, 0.28, 0.245, 0.44]); axL.imshow(orig); axL.axis("off")
axL.set_title("original image  $I$", fontsize=12, color=BLUE, fontweight="bold", pad=8)
for sp in axL.spines.values():
    sp.set_visible(False)

axR = fig.add_axes([0.385, 0.28, 0.245, 0.44]); axR.imshow(masked); axR.axis("off")
axR.set_title("masked image  $\\tilde{I}$", fontsize=12, color=RED, fontweight="bold", pad=8)

fig.text(0.5075, 0.245,
         "$\\tilde{I}$ = 14×14-pixel patches, each blackened with probability 0.6",
         ha="center", fontsize=10, color=RED)
fig.text(0.5075, 0.207,
         "black_prob 0.6 = the paper's stated setting; patch_size 14 from the released config",
         ha="center", fontsize=8.3, color="#777", style="italic")
fig.text(0.5075, 0.175,
         "generated with PAPO's own random_patch_blackening(), at the pipeline's own resolution",
         ha="center", fontsize=8.3, color="#777", style="italic")

# ── conditionals under each image
fig.text(0.1975, 0.125, "$\\pi_\\theta(\\,\\cdot \\mid y_{<t},\\, I,\\, Q\\,)$",
         ha="center", fontsize=14, color=BLUE)
fig.text(0.5075, 0.125, "$\\pi_\\theta(\\,\\cdot \\mid y_{<t},\\, \\tilde{I},\\, Q\\,)$",
         ha="center", fontsize=14, color=RED)

# ── the objective, right column
fig.text(0.815, 0.685, "PAPO's perception term", ha="center",
         fontsize=12.5, fontweight="bold", color="#2c3e50")

fig.text(0.815, 0.575,
         "$\\max_\\theta\\;\\; \\sum_t D_{KL}\\!\\left("
         "\\pi_\\theta(\\cdot \\mid y_{<t}, I, Q)\\;\\|\\;"
         "\\pi_\\theta(\\cdot \\mid y_{<t}, \\tilde{I}, Q)\\right)$",
         ha="center", fontsize=13.5)

fig.text(0.815, 0.475, "as implemented — note the sign:", ha="center",
         fontsize=9.5, color="#555", style="italic")
fig.text(0.815, 0.405,
         "$\\mathcal{L}\\;=\\;\\mathcal{L}_{\\rm PG}\\;\\mathbf{-}\\;"
         "\\gamma\\,\\mathcal{L}_{\\rm prcp}$",
         ha="center", fontsize=14)
fig.text(0.815, 0.345, "dp_actor.py:412   ·   $\\gamma = 0.01$",
         ha="center", fontsize=8.5, color="#777", style="italic")

# ── the direction box (mirrors the OPD figure's red box)
fig.patches.append(FancyBboxPatch((0.685, 0.145), 0.26, 0.155,
                                  boxstyle="round,pad=0.012", transform=fig.transFigure,
                                  fc="#fdeeec", ec=RED, lw=1.9, zorder=1))
fig.text(0.815, 0.253, "maximise  $\\mathcal{L}_{\\rm prcp}$", ha="center",
         fontsize=13, color=RED, fontweight="bold")
fig.text(0.815, 0.183, "the two distributions DIVERGE", ha="center",
         fontsize=10.5, color=RED)

# ── header + closing line
fig.text(0.5, 0.945,
         "PAPO: the same machinery, with the teacher blinded and the KL pushed APART",
         ha="center", fontsize=14, fontweight="bold", color="#2c3e50")
fig.text(0.5, 0.895,
         "the second branch is the SAME weights scoring the SAME tokens — only the pixels are destroyed",
         ha="center", fontsize=10.5, color="#555", style="italic")

fig.text(0.5, 0.055,
         "Rewards a correlate of grounding — sensitivity to pixels — not grounding itself. "
         "A repulsive KL has no optimum and no ceiling:\nany change that increases the "
         "discrepancy satisfies it, including changes orthogonal to accuracy.",
         ha="center", fontsize=10.8, color="#2c3e50")

fig.savefig(OUT / "fig_papo_objective.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT/'fig_papo_objective.png'}")

# also save the standalone image pair, in case you want it on its own slide
fig2, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 4.4))
a1.imshow(orig); a1.axis("off")
a1.set_title("original  $I$", fontsize=13, color=BLUE, fontweight="bold")
a2.imshow(masked); a2.axis("off")
a2.set_title("masked  $\\tilde{I}$   (14×14 patches, $p=0.6$)", fontsize=13,
             color=RED, fontweight="bold")
fig2.tight_layout()
fig2.savefig(OUT / "fig_papo_mask_example.png", bbox_inches="tight")
plt.close(fig2)
print(f"wrote {OUT/'fig_papo_mask_example.png'}")
