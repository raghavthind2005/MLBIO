#!/usr/bin/env python3
"""
Analyze and plot HallusionBench results.

Expects (in results/ by default):
  - raw_results.jsonl      from run_eval.py
  - attention_results.jsonl from extract_attention.py  (optional)

Plots produced (saved to results/plots/):
  1. acc_by_category.png         — qAcc by category / subcategory / visual_input
  2. thinking_length_dist.png    — thinking chars distribution, correct vs wrong
  3. thinking_vs_accuracy.png    — scatter: thinking length vs is_correct + trend
  4. attn_over_reasoning.png     — mean attention to visual/instruction/system
                                   tokens as a function of output position (% through reasoning)
  5. attn_per_layer.png          — per-layer mean attention to each token group
  6. attn_visual_decay_thirds.png — early/mid/late attention to visual tokens,
                                   broken down by correct / wrong
  7. attn_layer_heatmap.png      — heatmap: layer x position, colored by attention to visual tokens
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


# ─── Accuracy metrics ─────────────────────────────────────────────────────────

def compute_accuracy(records: list[dict]) -> dict:
    scored = [r for r in records if r.get("is_correct") is not None]
    qacc   = sum(r["is_correct"] for r in scored) / len(scored) if scored else 0.0

    # figure-level
    from collections import defaultdict
    fig_results = defaultdict(list)
    for r in scored:
        key = (r.get("category"), r.get("set_id"), r.get("figure_id"))
        fig_results[key].append(r["is_correct"])
    facc = (sum(1 for v in fig_results.values() if all(v)) / len(fig_results)
            if fig_results else 0.0)

    def group_acc(field):
        buckets = defaultdict(lambda: [0, 0])
        for r in scored:
            k = r.get(field, "?")
            buckets[k][0] += r["is_correct"]
            buckets[k][1] += 1
        return {k: v[0] / v[1] for k, v in buckets.items()}

    return {
        "qAcc":          round(qacc, 4),
        "fAcc":          round(facc, 4),
        "n":             len(scored),
        "by_category":   group_acc("category"),
        "by_subcategory":group_acc("subcategory"),
        "by_visual_input": group_acc("visual_input"),
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_accuracy(records: list[dict], plots_dir: Path) -> None:
    scored = [r for r in records if r.get("is_correct") is not None]
    if not scored:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("HallusionBench Accuracy (Gemma-4)", fontsize=13)

    def bar_plot(ax, field, title, labels_map=None):
        buckets = defaultdict(lambda: [0, 0])
        for r in scored:
            k = r.get(field, "?")
            buckets[k][0] += r["is_correct"]
            buckets[k][1] += 1
        keys = sorted(buckets)
        accs = [buckets[k][0] / buckets[k][1] for k in keys]
        ns   = [buckets[k][1] for k in keys]
        xlabels = [labels_map.get(k, k) if labels_map else k for k in keys]
        bars = ax.bar(xlabels, accs, color=plt.cm.Set2.colors[:len(keys)])
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_ylabel("qAcc")
        ax.tick_params(axis="x", rotation=30)
        for bar, acc, n in zip(bars, accs, ns):
            ax.text(bar.get_x() + bar.get_width()/2, acc + 0.01,
                    f"{acc:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8)

    bar_plot(axes[0], "category",     "By Category")
    bar_plot(axes[1], "subcategory",  "By Subcategory")
    bar_plot(axes[2], "visual_input", "By Visual Input",
             {"0": "text-only", "1": "original", "2": "edited"})

    plt.tight_layout()
    out = plots_dir / "acc_by_category.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_thinking_distribution(records: list[dict], plots_dir: Path) -> None:
    correct = [r["thinking_chars"] for r in records
               if r.get("is_correct") == 1 and r.get("thinking_chars") is not None]
    wrong   = [r["thinking_chars"] for r in records
               if r.get("is_correct") == 0 and r.get("thinking_chars") is not None]
    if not correct and not wrong:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, max(max(correct, default=0), max(wrong, default=0)) * 1.05, 40)
    if correct:
        ax.hist(correct, bins=bins, alpha=0.6, label=f"Correct (n={len(correct)})", color="steelblue")
    if wrong:
        ax.hist(wrong,   bins=bins, alpha=0.6, label=f"Wrong (n={len(wrong)})",     color="tomato")
    ax.set_xlabel("Thinking length (characters)")
    ax.set_ylabel("Count")
    ax.set_title("Thinking length distribution by correctness")
    ax.legend()

    out = plots_dir / "thinking_length_dist.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_thinking_vs_accuracy(records: list[dict], plots_dir: Path) -> None:
    data = [(r["thinking_chars"], r["is_correct"]) for r in records
            if r.get("thinking_chars") is not None and r.get("is_correct") is not None]
    if not data:
        return

    chars, correct = zip(*data)
    chars   = np.array(chars,   dtype=float)
    correct = np.array(correct, dtype=float)

    # Bin by thinking length and compute accuracy per bin
    n_bins   = 12
    bin_edges = np.percentile(chars, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    bin_acc, bin_mid, bin_n = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (chars >= lo) & (chars < hi)
        if mask.sum() >= 3:
            bin_acc.append(correct[mask].mean())
            bin_mid.append((lo + hi) / 2)
            bin_n.append(mask.sum())

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(chars, correct + np.random.uniform(-0.02, 0.02, len(chars)),
                    alpha=0.25, s=20, color="slategray", label="Samples")
    if bin_mid:
        ax.plot(bin_mid, bin_acc, "o-", color="crimson", lw=2, label="Binned mean accuracy")
    ax.set_xlabel("Thinking length (characters)")
    ax.set_ylabel("Correct (1) / Wrong (0)")
    ax.set_title("Does thinking longer hurt visual accuracy? (Think-longer-see-less)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.legend()
    ax.axhline(correct.mean(), color="steelblue", linestyle="--", alpha=0.5, label=f"Overall acc={correct.mean():.2f}")

    out = plots_dir / "thinking_vs_accuracy.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─── Attention plots ──────────────────────────────────────────────────────────

def align_and_average(pos_lists: list[list[float]], n_bins: int = 100) -> np.ndarray:
    """Resample each per-position attention trace to n_bins bins and average."""
    resampled = []
    for arr in pos_lists:
        if len(arr) < 2:
            continue
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, n_bins)
        resampled.append(np.interp(x_new, x_old, arr))
    return np.array(resampled).mean(axis=0) if resampled else np.zeros(n_bins)


def plot_attention_over_reasoning(attn_records: list[dict], plots_dir: Path) -> None:
    """Mean attention to each token group as a function of normalised output position."""
    N_BINS = 100
    groups = {
        "visual":      {"key": "attn_visual_per_pos",      "color": "steelblue",  "label": "Visual tokens"},
        "instruction": {"key": "attn_instruction_per_pos", "color": "darkorange", "label": "Instruction tokens"},
        "system":      {"key": "attn_system_per_pos",      "color": "seagreen",   "label": "System tokens"},
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(0, 100, N_BINS)

    for gname, cfg in groups.items():
        traces = [r[cfg["key"]] for r in attn_records if cfg["key"] in r and len(r[cfg["key"]]) >= 5]
        if not traces:
            continue
        avg = align_and_average(traces, N_BINS)
        ax.plot(x, avg, color=cfg["color"], lw=2, label=cfg["label"])
        # std band
        std = np.array([np.interp(np.linspace(0, 1, N_BINS),
                                  np.linspace(0, 1, len(t)), t) for t in traces]).std(axis=0)
        ax.fill_between(x, avg - std, avg + std, alpha=0.15, color=cfg["color"])

    ax.set_xlabel("Reasoning progress (%)")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Attention to token groups over reasoning steps\n(evidence for visual forgetting)")
    ax.legend()

    out = plots_dir / "attn_over_reasoning.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_attention_per_layer(attn_records: list[dict], plots_dir: Path) -> None:
    groups = {
        "visual":      ("attn_visual_per_layer",      "steelblue",  "Visual"),
        "instruction": ("attn_instruction_per_layer", "darkorange", "Instruction"),
        "system":      ("attn_system_per_layer",      "seagreen",   "System"),
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    for gname, (key, color, label) in groups.items():
        layers_list = [r[key] for r in attn_records if key in r]
        if not layers_list:
            continue
        n_layers = max(len(l) for l in layers_list)
        # Pad shorter ones (shouldn't happen, but defensive)
        padded = [l + [float("nan")] * (n_layers - len(l)) for l in layers_list]
        mean = np.nanmean(padded, axis=0)
        ax.plot(range(n_layers), mean, color=color, lw=2, label=label)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Per-layer attention to token groups (averaged over samples and output positions)")
    ax.legend()

    out = plots_dir / "attn_per_layer.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_attn_visual_decay_thirds(attn_records: list[dict], plots_dir: Path) -> None:
    """Early/mid/late visual attention for correct vs wrong predictions."""
    stages = ["early", "mid", "late"]
    correct_means, wrong_means = {s: [] for s in stages}, {s: [] for s in stages}
    for r in attn_records:
        if "attn_visual_by_thirds" not in r:
            continue
        thirds = r["attn_visual_by_thirds"]
        target = correct_means if r.get("is_correct") == 1 else wrong_means
        for s in stages:
            if s in thirds:
                target[s].append(thirds[s])

    x = np.arange(len(stages))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    c_means = [np.mean(correct_means[s]) if correct_means[s] else 0 for s in stages]
    w_means = [np.mean(wrong_means[s])   if wrong_means[s]   else 0 for s in stages]
    ax.bar(x - w/2, c_means, w, label="Correct", color="steelblue")
    ax.bar(x + w/2, w_means, w, label="Wrong",   color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(["Early\n(first ⅓)", "Mid\n(second ⅓)", "Late\n(final ⅓)"])
    ax.set_ylabel("Mean attention to visual tokens")
    ax.set_title("Visual attention at reasoning stages: correct vs wrong")
    ax.legend()

    out = plots_dir / "attn_visual_decay_thirds.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_layer_position_heatmap(attn_records: list[dict], plots_dir: Path) -> None:
    """Heatmap: layer (y) × normalised output position (x) → mean visual attention."""
    N_BINS   = 50
    n_layers = max((len(r.get("attn_visual_per_layer", [])) for r in attn_records), default=0)
    if n_layers == 0:
        return

    # Build [n_layers, N_BINS] array by averaging over samples
    heat = np.zeros((n_layers, N_BINS))
    counts = np.zeros(n_layers, dtype=int)

    for r in attn_records:
        per_layer = r.get("attn_visual_per_layer")
        per_pos   = r.get("attn_visual_per_pos")
        if not per_layer or not per_pos or len(per_pos) < 5:
            continue
        # Resample per_pos to N_BINS
        resampled = np.interp(np.linspace(0, 1, N_BINS),
                              np.linspace(0, 1, len(per_pos)), per_pos)
        # Build outer product: layer weight × position weight (as heuristic)
        per_layer_np = np.array(per_layer[:n_layers])
        # Normalise layer weights
        lw = per_layer_np / (per_layer_np.sum() + 1e-9)
        heat += np.outer(lw, resampled)
        counts += 1

    if counts.max() == 0:
        return
    heat /= counts.max()

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(heat, aspect="auto", origin="lower",
                   cmap="viridis", interpolation="nearest",
                   extent=[0, 100, 0, n_layers])
    plt.colorbar(im, ax=ax, label="Mean visual attention weight")
    ax.set_xlabel("Reasoning progress (%)")
    ax.set_ylabel("Layer index")
    ax.set_title("Visual attention: layer × reasoning position")

    out = plots_dir / "attn_layer_heatmap.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    raw_path    = results_dir / "raw_results.jsonl"
    attn_path   = results_dir / "attention_results.jsonl"
    plots_dir   = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"No raw_results.jsonl found at {raw_path}")
        return

    print(f"Loading raw results from {raw_path}...")
    records = load_jsonl(raw_path)
    records = [r for r in records if "error" not in r]
    print(f"  {len(records)} valid samples\n")

    # Summary
    acc = compute_accuracy(records)
    print(f"qAcc: {acc['qAcc']}  fAcc: {acc['fAcc']}  n={acc['n']}")
    print(f"By category:    {acc['by_category']}")
    print(f"By visual_input:{acc['by_visual_input']}")
    print()

    print("Generating plots...")
    plot_accuracy(records, plots_dir)
    plot_thinking_distribution(records, plots_dir)
    plot_thinking_vs_accuracy(records, plots_dir)

    if attn_path.exists():
        print(f"\nLoading attention results from {attn_path}...")
        attn_records = load_jsonl(attn_path)
        # Merge is_correct into attn records from raw results
        raw_by_id = {r["sample_id"]: r for r in records}
        for ar in attn_records:
            raw = raw_by_id.get(ar["sample_id"], {})
            ar.setdefault("is_correct",     raw.get("is_correct"))
            ar.setdefault("thinking_chars", raw.get("thinking_chars"))
        print(f"  {len(attn_records)} attention records\n")

        plot_attention_over_reasoning(attn_records, plots_dir)
        plot_attention_per_layer(attn_records, plots_dir)
        plot_attn_visual_decay_thirds(attn_records, plots_dir)
        plot_layer_position_heatmap(attn_records, plots_dir)
    else:
        print(f"\nNo attention_results.jsonl found — skipping attention plots.")

    print(f"\nAll plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
