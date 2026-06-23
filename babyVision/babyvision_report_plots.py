#!/usr/bin/env python3
"""
Generate the shareable figures for RESULTS.md from the real data (no hand-entered
numbers — everything is recomputed from the graded results + attention files).

Figures written to <base>/plots/:
  1. dissociation.png   — accuracy by task family: standard vs re-grounding+CoT (B1'),
                          showing the +perception / -reasoning split.
  2. attention_decay.png — image attention early->mid->late across the turn-2 reasoning
                          (the "see less" curve), b1cot and b2cot.
  3. attn_correct_wrong.png — image attention for correct vs wrong turn-2 answers.

Prints the underlying numbers so they can be checked against the figures.

  python babyvision_report_plots.py --base /iopsstor/.../babyvision
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PERCEPTION = {"Count 3D blocks", "Count Same Patterns", "Count Clusters", "Maze",
              "Connect the lines", "Metro map", "Lines Observation", "Find the same",
              "Find the different", "Find the shadow"}
REASONING  = {"3D Cube Unfold", "Paper Folding", "3D Views", "Rotation Patterns",
              "Mirroring Patterns", "2D Pattern Completion", "3D Pattern Completion",
              "Logic Patterns", "Overlay Patterns", "Reconstruction",
              "Recognize numbers and letters", "Pattern and Color Completion"}


def fam(s):
    return "perception" if s in PERCEPTION else "reasoning" if s in REASONING else "other"


def load(path):
    return [json.loads(l) for l in open(path)] if path.exists() else []


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def std_majority(base):
    """taskId -> (majority-of-3 grade, subtype)."""
    votes, sub = {}, {}
    for pi in (1, 2, 3):
        p = base / "results_standard" / f"results_run{pi}_graded.jsonl"
        for r in load(p):
            if "error" in r:
                continue
            votes.setdefault(r["taskId"], []).append(1 if r.get("grade") else 0)
            sub[r["taskId"]] = r.get("subtype")
    return {t: (sum(v) >= len(v) / 2.0, sub[t]) for t, v in votes.items()}


def graded_map(path):
    out = {}
    for r in load(path):
        if "error" not in r:
            out[r["taskId"]] = (bool(r.get("grade")), r.get("subtype"))
    return out


def fam_acc(gmap, family):
    gs = [g for g, s in gmap.values() if fam(s) == family]
    return 100.0 * sum(gs) / len(gs) if gs else float("nan"), len(gs)


def fig_dissociation(base, outdir):
    std = std_majority(base)
    b1 = graded_map(base / "results_b1cot_reinject" / "results_run1_graded.jsonl")
    rows = []
    for family in ("perception", "reasoning"):
        sa, sn = fam_acc(std, family)
        ba, bn = fam_acc(b1, family)
        rows.append((family, sa, ba, bn))
        print(f"  {family}: standard={sa:.1f}%  B1'(regrounding+CoT)={ba:.1f}%  "
              f"Δ={ba-sa:+.1f}  (n={bn})")

    labels = [f"{f}\n(n={n})" for f, _, _, n in rows]
    std_v  = [r[1] for r in rows]
    b1_v   = [r[2] for r in rows]
    x = range(len(rows)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar([i - w/2 for i in x], std_v, w, label="standard (one look)", color="#9aa0a6")
    ax.bar([i + w/2 for i in x], b1_v, w, label="re-grounding + own reasoning (B1')",
           color="#1a73e8")
    for i, r in enumerate(rows):
        ax.annotate(f"{r[2]-r[1]:+.1f} pts", (i, max(r[1], r[2]) + 1.5),
                    ha="center", fontsize=10, fontweight="bold",
                    color="#137333" if r[2] > r[1] else "#c5221f")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Re-examining with your own reasoning:\nhelps perception, hurts reasoning")
    ax.legend(fontsize=9, loc="upper left"); ax.set_ylim(0, 50)
    fig.tight_layout(); fig.savefig(outdir / "dissociation.png", dpi=150)
    print(f"  -> {outdir/'dissociation.png'}")


def fig_decay(base, outdir):
    def thirds(recs, key="attn_visual_all_by_thirds"):
        e = mean([r[key]["early"] for r in recs if r.get(key)])
        m = mean([r[key]["mid"]   for r in recs if r.get(key)])
        l = mean([r[key]["late"]  for r in recs if r.get(key)])
        return [e, m, l]
    b1 = load(base / "results_b1cot_reinject"   / "attention_b.jsonl")
    b2 = load(base / "results_b2cot_noreinject" / "attention_b.jsonl")
    c1, c2 = thirds(b1), thirds(b2)
    print(f"  b1cot decay early/mid/late = {['%.4f'%v for v in c1]} (n={len(b1)})")
    print(f"  b2cot decay early/mid/late = {['%.4f'%v for v in c2]} (n={len(b2)})")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    xs = ["early", "middle", "late"]
    ax.plot(xs, c1, "o-", color="#1a73e8", lw=2, label=f"image re-shown (B1', n={len(b1)})")
    ax.plot(xs, c2, "s--", color="#e8710a", lw=2, label=f"image not re-shown (B2', n={len(b2)})")
    ax.set_ylabel("attention to the image (mean)")
    ax.set_xlabel("position within the second-turn reasoning")
    ax.set_title("The model looks at the image less as it reasons longer\n(“see less”)")
    ax.legend(fontsize=9); ax.set_ylim(0, max(c1 + c2) * 1.2)
    fig.tight_layout(); fig.savefig(outdir / "attention_decay.png", dpi=150)
    print(f"  -> {outdir/'attention_decay.png'}")


def fig_correct_wrong(base, outdir):
    def cw(recs, key="attn_visual_all_mean"):
        c = mean([r.get(key) for r in recs if r.get("grade") is True])
        w = mean([r.get(key) for r in recs if r.get("grade") is False])
        nc = sum(1 for r in recs if r.get("grade") is True)
        nw = sum(1 for r in recs if r.get("grade") is False)
        return c, w, nc, nw
    b1 = load(base / "results_b1cot_reinject"   / "attention_b.jsonl")
    b2 = load(base / "results_b2cot_noreinject" / "attention_b.jsonl")
    c1, w1, nc1, nw1 = cw(b1); c2, w2, nc2, nw2 = cw(b2)
    print(f"  b1cot: correct={c1:.4f}(n={nc1})  wrong={w1:.4f}(n={nw1})")
    print(f"  b2cot: correct={c2:.4f}(n={nc2})  wrong={w2:.4f}(n={nw2})")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = range(2); w = 0.36
    ax.bar([i - w/2 for i in x], [c1, c2], w, label="correct answers", color="#137333")
    ax.bar([i + w/2 for i in x], [w1, w2], w, label="wrong answers", color="#c5221f")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"B1' (image re-shown)\nn={nc1}+{nw1}",
                        f"B2' (not re-shown)\nn={nc2}+{nw2}"])
    ax.set_ylabel("attention to the image (mean)")
    ax.set_title("When the model looks at the image, it gets it right")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(outdir / "attn_correct_wrong.png", dpi=150)
    print(f"  -> {outdir/'attn_correct_wrong.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    args = ap.parse_args()
    base = Path(args.base)
    outdir = base / "plots"; outdir.mkdir(exist_ok=True)
    print("FIG 1 — perception/reasoning dissociation:")
    fig_dissociation(base, outdir)
    print("FIG 2 — attention decay across turn-2 reasoning:")
    fig_decay(base, outdir)
    print("FIG 3 — image attention, correct vs wrong:")
    fig_correct_wrong(base, outdir)
    print("\nDone. Copy the PNGs in", outdir, "into babyVision/plots/ to embed in RESULTS.md.")


if __name__ == "__main__":
    main()
