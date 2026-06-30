"""
make_figures.py — render the talk figures from the VERIFIED result numbers.

Numbers are hardcoded from the actual runs (see FINDINGS.md / the logs) so this renders
anywhere with matplotlib, no CSV needed. Outputs PNGs to runs/analysis/figures/.

  python make_figures.py        # writes figures/fig_*.png

Provenance of each number is in FINDINGS.md (Parts 8/9/11/12) and METHODS.md appendix.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "font.size": 12, "axes.titlesize": 12, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

C_BASE, C_TRAIN, C_MLP, C_ATTN = "#888888", "#1f77b4", "#d62728", "#2ca02c"


# ── Fig 1 (Slide 8): depth-probe — decodability vs layer, base vs trained ─────
# argmax-accuracy by LLM layer (logit-lens). Layers 7-36 (0-6 are near-chance, omitted).
depth_layers = list(range(7, 37))
depth_base = [0.250,0.250,0.260,0.250,0.410,0.250,0.250,0.250,0.250,0.250,0.293,0.340,
              0.040,0.040,0.040,0.043,0.043,0.173,0.350,0.370,0.390,0.370,0.373,0.387,
              0.390,0.377,0.380,0.397,0.373,0.377]
depth_train = [0.250,0.250,0.260,0.250,0.447,0.250,0.250,0.250,0.250,0.250,0.303,0.293,
               0.040,0.040,0.040,0.043,0.053,0.273,0.623,0.613,0.617,0.613,0.593,0.627,
               0.617,0.607,0.597,0.620,0.647,0.657]

def fig_depth():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(depth_layers, depth_base, "-o", color=C_BASE, ms=4, label="base (untrained)")
    ax.plot(depth_layers, depth_train, "-o", color=C_TRAIN, ms=4, label="trained (full, step 96)")
    ax.axvline(24, color="k", ls="--", lw=1, alpha=0.6)
    ax.annotate("diverge at L24", xy=(24, 0.45), xytext=(26.5, 0.30),
                arrowprops=dict(arrowstyle="->", color="k"), fontsize=11)
    ax.axhline(0.25, color="gray", ls=":", lw=1, alpha=0.7)
    ax.text(7.2, 0.265, "chance (¼)", fontsize=9, color="gray")
    ax.set_xlabel("LLM layer"); ax.set_ylabel("answer decodable here? (argmax-acc)")
    ax.set_title("Depth probe: the answer becomes readable only at the LATE layers (L24+)")
    ax.set_ylim(0, 0.72); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_depth.png"); plt.close(fig)


# ── Fig 2 (Slide 9): graft — % of the gain recovered by each subset ──────────
graft_labels = ["base", "early_mlp\n(L0-11)", "attn\n(all)", "mlp\n(all)", "full"]
graft_recov  = [0.0, 19.0, 30.0, 63.0, 100.0]   # late_mlp shown separately as a callout
graft_colors = [C_BASE, "#9ecae1", C_ATTN, C_MLP, "#333333"]

def fig_graft():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(graft_labels, graft_recov, color=graft_colors)
    for b, v in zip(bars, graft_recov):
        ax.text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.0f}%", ha="center", fontsize=11)
    # late_mlp callout
    ax.bar(["late_mlp\n(L24-35)"], [3.6], color="#fc9272")
    ax.text(5, 5.5, "3.6%", ha="center", fontsize=11)
    ax.set_ylabel("% of the +0.28 perception gain recovered")
    ax.set_title("Module graft (causal): MLP-dominant, but DISTRIBUTED across depth")
    ax.set_ylim(0, 108)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_graft.png"); plt.close(fig)


# ── Fig 3 (Slide 11): activation patch — per-item patch vs fixed steering, by layer ─
patch_layers = [8, 12, 16, 20, 24, 28, 32, 35]
patch_recov  = [0, 0, 1, 11, 82, 93, 99, 100]      # per-item patch (oracle residual)
steer_recov  = [-2.4, 0, 1.2, 7.1, 26.2, 29.8, 34.5, 39.3]  # fixed steering vector, best alpha (=4)
def fig_patch():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(patch_layers, patch_recov, "-o", color=C_TRAIN, ms=6,
            label="per-item patch (inject trained residual)")
    ax.plot(patch_layers, steer_recov, "-s", color=C_ATTN, ms=5,
            label="fixed steering vector (best α=4)")
    ax.axvline(24, color="k", ls="--", lw=1, alpha=0.5)
    ax.annotate("82% at L24", xy=(24, 82), xytext=(14, 86),
                arrowprops=dict(arrowstyle="->", color="k"), fontsize=11)
    ax.annotate("steering tops out ~40%", xy=(35, 39), xytext=(24, 62),
                arrowprops=dict(arrowstyle="->", color=C_ATTN), fontsize=10, color=C_ATTN)
    ax.set_xlabel("inject at LLM layer L")
    ax.set_ylabel("% of gain recovered (base→trained)")
    ax.set_title("Activation patch: per-item works (≈full), a fixed vector does NOT (input-specific)")
    ax.set_ylim(-6, 108); ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.62))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_patch.png"); plt.close(fig)


# ── Fig 4 (Slide 7): H ablation — accuracy by condition (reward + probe) ──────
cond = ["base", "full", "llm_only", "vit_only"]
reward = [0.365, 0.746, 0.749, 0.443]
probe  = [0.377, 0.657, 0.593, 0.423]
def fig_ablation():
    import numpy as np
    x = np.arange(len(cond)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(x-w/2, reward, w, color="#1f77b4", label="training reward (with reasoning)")
    b2 = ax.bar(x+w/2, probe,  w, color="#ff7f0e", label="direct probe (no reasoning)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.012, f"{b.get_height():.3f}",
                    ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(cond)
    ax.set_ylabel("perception accuracy")
    ax.set_title("Freeze ablation (H): llm_only ≈ full ≫ vit_only  (the fix is LLM-internal)")
    ax.set_ylim(0, 0.86); ax.legend(loc="upper center", ncol=2, fontsize=9, framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ablation.png"); plt.close(fig)


if __name__ == "__main__":
    fig_depth(); fig_graft(); fig_patch(); fig_ablation()
    print("wrote:", ", ".join(sorted(os.listdir(OUT))))
