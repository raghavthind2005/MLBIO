#!/usr/bin/env python3
"""
PAPO figures for the talk.

DATA PROVENANCE (read before trusting any number here):
  - C-pure per-step series: TALK/data/cpure_experiment_log.jsonl, pulled 2026-08-19 from
    clariden:/iopsstor/scratch/cscs/raghavthind/runs/papo_2b_8k_cpure_run/checkpoints/experiment_log.jsonl
    This is the verl "file" logger, i.e. the same dict that went to wandb. 60 training steps + 2 val points.
  - Arm A (GRPO baseline) and Arm B (C+DE) final val: taken from PAPO_fixed/PAPO_THREE_RUNS_EXPLAINED.md.
    Their wandb .wandb data files no longer exist on the cluster (verified 2026-08-19), so NO per-step
    series can be plotted for A or B. Only their documented final val_reward_score is used.

NOTE on sign: dp_actor.py logs actor/kl_prcp_loss as the NEGATED value
(`append_to_dict(metrics, {"actor/kl_prcp_loss": -kl_prcp_loss...})`), so we plot |value| and call it
the perception-KL magnitude.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- load C-pure
rows = [json.loads(l) for l in open(HERE / "data" / "cpure_experiment_log.jsonl")]
train, val = {}, {}
for r in rows:
    if "actor" in r and "kl_prcp_loss" in r["actor"]:
        train[r["step"]] = (abs(r["actor"]["kl_prcp_loss"]), r["reward"]["accuracy"])
    if "val" in r:
        val.setdefault(r["step"], []).append(r["val"]["reward_score"])
# step 60 has TWO final-val passes (the run's own + one re-fired by the resumed job).
# Both measure the same checkpoint under val sampling n=8 @ temp 1.0, so we report the mean
# and keep both values visible.
CPURE_VAL_RUNS = val[60]                                   # [0.53958, 0.53602]
val = {k: sum(v) / len(v) for k, v in val.items()}

steps = sorted(train)
kl = [train[s][0] for s in steps]
acc = [train[s][1] for s in steps]

BASE_VAL   = val[0]              # 0.2579
CPURE_VAL  = val[60]              # mean of the two final-val passes
GRPO_VAL   = 0.540               # Arm A,  PAPO_THREE_RUNS_EXPLAINED.md
CDE_VAL    = 0.466               # Arm B,  PAPO_THREE_RUNS_EXPLAINED.md

# ============================================================ FIG 1: the dissociation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3),
                               gridspec_kw={"width_ratios": [1.45, 1]})

# -- panel A: the objective is achieved
ax1.plot(steps, kl, color="#c0392b", lw=2, marker="o", ms=3, label="perception-KL  $|D_{KL}[\\pi_\\theta \\| \\pi_\\theta^{mask}]|$")
ax1.set_xlabel("training step")
ax1.set_ylabel("perception-KL magnitude", color="#c0392b")
ax1.tick_params(axis="y", labelcolor="#c0392b")
ax1.annotate(f"{kl[0]:.3f}", xy=(steps[0], kl[0]), xytext=(6, -16),
             textcoords="offset points", color="#c0392b", fontsize=10)
ax1.annotate(f"{kl[-1]:.3f}   ({kl[-1]/kl[0]:.1f}×)", xy=(steps[-1], kl[-1]), xytext=(-78, 8),
             textcoords="offset points", color="#c0392b", fontsize=10, fontweight="bold")

axr = ax1.twinx()
axr.plot(steps, acc, color="#7f8c8d", lw=1.6, ls="--", alpha=0.85, label="training accuracy reward")
axr.set_ylabel("training accuracy reward", color="#7f8c8d")
axr.tick_params(axis="y", labelcolor="#7f8c8d")
axr.grid(False)
axr.spines["top"].set_visible(False)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = axr.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, framealpha=0.9)
ax1.set_title("The objective is achieved: PAPO's perception-KL more than doubles", fontsize=11)

# -- panel B: the accuracy is not
names  = ["base\n(untrained)", "GRPO\n(Arm A)", "PAPO C-pure\n(Arm C)", "PAPO C+DE\n(Arm B)"]
vals   = [BASE_VAL, GRPO_VAL, CPURE_VAL, CDE_VAL]
colors = ["#bdc3c7", "#34495e", "#c0392b", "#e67e22"]
bars = ax2.bar(names, vals, color=colors, width=0.62)
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.3f}",
             ha="center", fontsize=10, fontweight="bold")
ax2.axhline(GRPO_VAL, color="#34495e", ls=":", lw=1.4, zorder=0)
ax2.set_ylabel("validation reward score")
ax2.set_ylim(0, 0.68)
ax2.set_title("…and buys nothing: C-pure lands on the GRPO baseline", fontsize=11)
ax2.tick_params(axis="x", labelsize=9)

fig.suptitle("PAPO = on-policy distillation from a deliberately blinded teacher, with the KL maximized",
             fontsize=12.5, fontweight="bold", y=1.00)
fig.tight_layout()
fig.savefig(OUT / "fig_papo_dissociation.png", bbox_inches="tight")
plt.close(fig)

# ============================================================ FIG 2: KL trajectory alone (backup slide)
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.plot(steps, kl, color="#c0392b", lw=2, marker="o", ms=3.5)
# 10-step block means, to show it is a trend not noise
blocks = [(a, a + 9) for a in range(1, 61, 10)]
for a, b in blocks:
    seg = [train[s][0] for s in steps if a <= s <= b]
    ax.hlines(sum(seg)/len(seg), a, b, color="#2c3e50", lw=2.4, alpha=0.8, zorder=5)
ax.hlines([], [], [], color="#2c3e50", lw=2.4, label="10-step block mean")
ax.set_xlabel("training step")
ax.set_ylabel("perception-KL magnitude")
ax.set_title("PAPO C-pure: the maximized objective rises monotonically over 60 steps", fontsize=11)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig_papo_kl_trajectory.png", bbox_inches="tight")
plt.close(fig)

print(f"base val   : {BASE_VAL:.4f}")
print(f"C-pure val : {CPURE_VAL:.4f}   (two final-val passes: {CPURE_VAL_RUNS[0]:.4f}, {CPURE_VAL_RUNS[1]:.4f})")
print(f"GRPO  (A)  : {GRPO_VAL:.4f}   [from PAPO_THREE_RUNS_EXPLAINED.md]")
print(f"C+DE  (B)  : {CDE_VAL:.4f}   [from PAPO_THREE_RUNS_EXPLAINED.md]")
print(f"KL {kl[0]:.4f} -> {kl[-1]:.4f}  ({kl[-1]/kl[0]:.2f}x)")
print(f"train acc {acc[0]:.4f} -> {acc[-1]:.4f}")
print(f"\nwrote {OUT}/fig_papo_dissociation.png")
print(f"wrote {OUT}/fig_papo_kl_trajectory.png")
