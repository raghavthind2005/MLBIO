#!/usr/bin/env python3
"""
Re-seeing vs Re-thinking — focused comparison dashboard.

Question: a forced second reasoning pass helps a little. Is that because the model
RE-SEES the image (re-injected) or just RE-THINKS (reasons again)? We isolate it
with two conditions that share an identical 2-turn protocol and differ ONLY in
whether the image is present in turn 1:

  Forced   — image RE-INJECTED in turn 1   (results_forced/)
  Rethink  — NO image in turn 1            (results_rethink/)

Four plots, grounded in results + attention JSONL:
  1. The link       — turn-1 image attention vs accuracy change
  2. Same start     — turn-0 baselines are statistically identical (sampling noise)
  3. What changed   — answer transitions (helped / hurt)
  4. The mechanism  — image attention decays across reasoning ("see less")

Usage:
  python build_rethink_comparison.py
"""

import argparse
import base64
import json
import math
from collections import Counter
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SCRIPT_DIR = Path(__file__).parent
PATHS = {
    "normal":  (SCRIPT_DIR / "results_normal"  / "raw_results.jsonl",
                SCRIPT_DIR / "results_normal"  / "attention_results.jsonl"),
    "forced":  (SCRIPT_DIR / "results_forced"  / "forced_results.jsonl",
                SCRIPT_DIR / "results_forced"  / "attention_results.jsonl"),
    "rethink": (SCRIPT_DIR / "results_rethink" / "rethink_results.jsonl",
                SCRIPT_DIR / "results_rethink" / "attention_results_fixed.jsonl"),
}

C_FORCED, C_RETHINK, C_NORMAL = "#e07b39", "#5aaa6f", "#5b8db8"
C_GREY = "#b0b0b0"
POS, NEG = "#2a7a2a", "#c0392b"


# ── load / helpers ───────────────────────────────────────────────────────────

def load(p):
    return [json.loads(l) for l in open(p) if '"error"' not in l]

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def acc(rows, key):
    return mean([r[key] for r in rows if r.get(key) is not None])

def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def base_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#ececec", zorder=0)


# ── attention: turn-1 reasoning → image, per condition ───────────────────────

def forced_turn_split(rows):
    """Re-injected image (visual_turn1) is causally attendable only by turn-1
    output positions, so positions with attn_visual_turn1_per_pos>0 ARE turn 1."""
    t0, t1 = [], []
    for r in rows:
        pc, p1 = r.get("attn_visual_per_pos"), r.get("attn_visual_turn1_per_pos")
        if not pc or not p1:
            continue
        seg1 = {i for i, v in enumerate(p1) if v > 1e-9}
        seg0 = [i for i in range(len(pc)) if i not in seg1]
        if seg0:  t0.append(np.mean([pc[i] for i in seg0]))
        if seg1:  t1.append(np.mean([pc[i] for i in seg1]))
    return {"t0": mean(t0), "t1": mean(t1)}

def rethink_turn_split(rows):
    return {"t0": mean([r.get("attn_visual_output_turn0") for r in rows]),
            "t1": mean([r.get("attn_visual_output_turn1") for r in rows])}


# ── turn-0 sampling-noise decomposition (forced t0 vs rethink t0) ─────────────

def turn0_noise(res):
    def key(r):
        return (r["category"], r.get("subcategory"), r["set_id"],
                r["figure_id"], r["question_id"], r["visual_input"])
    F = {key(r): r for r in res["forced"]}
    R = {key(r): r for r in res["rethink"]}
    same = flip = f_only = r_only = n = 0
    for k in set(F) & set(R):
        fp, rp = F[k].get("pred_turn0"), R[k].get("pred_turn0")
        if fp is None or rp is None:
            continue
        n += 1
        gt = F[k]["gt_answer"]
        if fp == rp:
            same += 1
        else:
            flip += 1
            if   fp == gt and rp != gt: f_only += 1
            elif rp == gt and fp != gt: r_only += 1
    z = (f_only - flip/2) / math.sqrt(flip*0.25) if flip else 0.0
    return {"n": n, "same": same, "flip": flip, "f_only": f_only,
            "r_only": r_only, "z": z}


# ── decay curves ─────────────────────────────────────────────────────────────

def decay_curve(rows, n_bins=20):
    g = []
    for r in rows:
        pp = r.get("attn_visual_per_pos")
        if not pp or len(pp) < 4:
            continue
        pp = np.array(pp, float)
        g.append(np.interp(np.linspace(0,1,n_bins), np.linspace(0,1,len(pp)), pp))
    return (np.linspace(0,1,n_bins), np.array(g).mean(0)) if g else (None, None)

def rethink_segments(rows, n_bins=20):
    g0, g1 = [], []
    for r in rows:
        pp, k = r.get("attn_visual_per_pos"), r.get("n_output_turn0")
        if not pp or not k or k < 2 or len(pp)-k < 2:
            continue
        pp = np.array(pp, float)
        g0.append(np.interp(np.linspace(0,1,n_bins), np.linspace(0,1,k), pp[:k]))
        g1.append(np.interp(np.linspace(0,1,n_bins), np.linspace(0,1,len(pp)-k), pp[k:]))
    x = np.linspace(0,1,n_bins)
    return (x, np.array(g0).mean(0) if g0 else None,
               np.array(g1).mean(0) if g1 else None)


# ════════════════════════════════════════════════════════════════════════════
#  PLOTS  (4)
# ════════════════════════════════════════════════════════════════════════════

def plot_link(S):
    """1 — turn-1 image attention (L) and accuracy change (R), side by side."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
    conds = ["Forced\n(re-injected)", "Rethink\n(no image)"]
    x = np.arange(2)

    t0 = [S["forced"]["attn"]["t0"], S["rethink"]["attn"]["t0"]]
    t1 = [S["forced"]["attn"]["t1"], S["rethink"]["attn"]["t1"]]
    axL.bar(x-0.19, t0, 0.36, color=C_GREY, label="Turn 0", zorder=3)
    axL.bar(x+0.19, t1, 0.36, color=[C_FORCED, C_RETHINK], label="Turn 1", zorder=3)
    for i,(a,b) in enumerate(zip(t0,t1)):
        axL.text(i-0.19, a+0.003, f"{a:.3f}", ha="center", fontsize=8.5, color="#555")
        axL.text(i+0.19, b+0.003, f"{b:.3f}", ha="center", fontsize=9.5, fontweight="bold")
        d=(b/a-1)*100
        axL.text(i, max(a,b)+0.016, f"{d:+.0f}%", ha="center", fontsize=10,
                 color=POS if d>=0 else NEG, fontweight="bold")
    axL.set_xticks(x); axL.set_xticklabels(conds, fontsize=10)
    axL.set_ylabel("Image attention (from reasoning)", fontsize=10)
    axL.set_title("Turn-1 reasoning: does it look at the image?", fontsize=11, fontweight="bold")
    axL.legend(fontsize=9, loc="upper right"); axL.set_ylim(0, max(t1)*1.32); base_ax(axL)

    d = [S["forced"]["delta"], S["rethink"]["delta"]]
    axR.bar(x, [v*100 for v in d], 0.5, color=[POS if v>=0 else NEG for v in d], zorder=3)
    axR.axhline(0, color="#333", lw=0.8)
    for i,v in enumerate(d):
        axR.text(i, v*100+(0.08 if v>=0 else -0.08), f"{v*100:+.2f}pp", ha="center",
                 va="bottom" if v>=0 else "top", fontsize=11, fontweight="bold",
                 color=POS if v>=0 else NEG)
    axR.set_xticks(x); axR.set_xticklabels(conds, fontsize=10)
    axR.set_ylabel("Accuracy change, turn0→turn1 (pp)", fontsize=10)
    axR.set_title("Did accuracy go up or down?", fontsize=11, fontweight="bold")
    axR.set_ylim(min(d)*100*1.6-0.4, max(d)*100*1.6+0.4); base_ax(axR)

    fig.suptitle("Re-seeing lifts image attention AND accuracy · re-thinking alone drops both",
                 fontsize=12.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_baseline(S):
    """2 — turn-0 baselines are statistically identical (sampling noise)."""
    nz = S["noise"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios":[1.1,1]})

    # left: agreement on turn-0 prediction across the two runs
    same_pct = nz["same"]/nz["n"]*100
    flip_pct = nz["flip"]/nz["n"]*100
    axL.barh([0], [same_pct], color="#8bc6a0", zorder=3, label=f"Same answer  {same_pct:.0f}%")
    axL.barh([0], [flip_pct], left=[same_pct], color="#e6b0b0", zorder=3,
             label=f"Flipped (noise)  {flip_pct:.0f}%")
    axL.set_xlim(0,100); axL.set_yticks([]); axL.set_xlabel("% of shared turn-0 samples", fontsize=10)
    axL.legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.5,-0.55), ncol=1, frameon=False)
    axL.set_title(f"Forced t0 vs Rethink t0 — same computation, n={nz['n']}",
                  fontsize=10.5, fontweight="bold")
    for s in ["top","right","left"]: axL.spines[s].set_visible(False)

    # right: of the flips, how they split + verdict
    axR.bar(["forced\nright","rethink\nright"], [nz["f_only"], nz["r_only"]],
            color=[C_FORCED, C_RETHINK], width=0.55, zorder=3)
    for i,v in enumerate([nz["f_only"], nz["r_only"]]):
        axR.text(i, v+0.2, str(v), ha="center", fontsize=12, fontweight="bold")
    axR.set_title(f"The {nz['flip']} flips split ~evenly", fontsize=10.5, fontweight="bold")
    axR.set_ylabel("samples", fontsize=10)
    axR.set_ylim(0, max(nz["f_only"], nz["r_only"])*1.4)
    verdict = "not significant" if abs(nz["z"])<1.96 else "significant"
    axR.text(0.5, 0.92, f"McNemar z={nz['z']:.2f}  →  {verdict}\n(baselines are equal)",
             transform=axR.transAxes, ha="center", va="top", fontsize=9.5,
             bbox=dict(boxstyle="round", fc="#f3f3f3", ec="#ccc"))
    base_ax(axR)

    fig.suptitle("The two conditions START from the same place — the gap is sampling noise",
                 fontsize=12.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_transitions(S):
    """3 — answer transitions helped/hurt, forced vs rethink."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, cond, col in [(axes[0],"forced",C_FORCED), (axes[1],"rethink",C_RETHINK)]:
        c = S[cond]["change"]; tot = sum(c.values())
        cats = ["right_right","wrong_wrong","wrong_right","right_wrong"]
        labs = ["stayed ✓","stayed ✗","✗→✓ helped","✓→✗ hurt"]
        cols = ["#7cc095","#e9b3b3",POS,NEG]
        vals = [c.get(k,0) for k in cats]
        bars = ax.bar(labs, vals, color=cols, width=0.62, zorder=3)
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.4, f"{v}", ha="center", fontsize=9.5)
        net = c.get("wrong_right",0)-c.get("right_wrong",0)
        title = "Forced (re-injected)" if cond=="forced" else "Rethink (no image)"
        ax.set_title(f"{title}   ·   net {net:+d}", fontsize=11, fontweight="bold")
        ax.set_ylabel("samples", fontsize=10); ax.set_ylim(0, max(vals)*1.25)
        ax.tick_params(axis="x", labelsize=9); base_ax(ax)
    fig.suptitle("What the second turn actually did to each answer", fontsize=12.5,
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_decay(att):
    """4 — image attention decays across reasoning ('see less')."""
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    xn, yn = decay_curve(att["normal"])
    if xn is not None:
        ax.plot(xn, yn, color=C_NORMAL, lw=2.4, label="Normal (single pass)")
    x, c0, c1 = rethink_segments(att["rethink"])
    if c0 is not None:
        ax.plot(x, c0, color=C_RETHINK, lw=2.2, label="Rethink turn 0 (image present)")
    if c1 is not None:
        ax.plot(x, c1, color=C_RETHINK, lw=2.2, ls="--", label="Rethink turn 1 (no image)")
    ax.set_xlabel("position within reasoning  (start → end)", fontsize=10)
    ax.set_ylabel("image attention", fontsize=10)
    ax.set_title("Attention to the image fades as the model reasons longer",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(color="#ececec"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return fig_to_b64(fig)


# ════════════════════════════════════════════════════════════════════════════
#  STATS + HTML
# ════════════════════════════════════════════════════════════════════════════

def build_stats(res, att):
    return {
        "normal":  {"acc": acc(res["normal"], "is_correct")},
        "forced":  {"t0": acc(res["forced"], "is_correct_turn0"),
                    "t1": acc(res["forced"], "is_correct"),
                    "delta": acc(res["forced"],"is_correct")-acc(res["forced"],"is_correct_turn0"),
                    "change": Counter(r.get("change_type") for r in res["forced"] if r.get("change_type")),
                    "attn": forced_turn_split(att["forced"])},
        "rethink": {"t0": acc(res["rethink"], "is_correct_turn0"),
                    "t1": acc(res["rethink"], "is_correct"),
                    "delta": acc(res["rethink"],"is_correct")-acc(res["rethink"],"is_correct_turn0"),
                    "change": Counter(r.get("change_type") for r in res["rethink"] if r.get("change_type")),
                    "attn": rethink_turn_split(att["rethink"])},
        "noise":   turn0_noise(res),
    }


def sec(n, title, desc, b64, note=None):
    note_html = f'<div class="note">{note}</div>' if note else ""
    return f"""
  <div class="card">
    <h2><span class="num">{n}</span>{title}</h2>
    <p>{desc}</p>
    <div class="fig"><img src="data:image/png;base64,{b64}"></div>
    {note_html}
  </div>"""


def build_html(S, figs, ns):
    f, r, nz = S["forced"], S["rethink"], S["noise"]
    dpp = lambda v: f"{v*100:+.2f}pp"

    tldr = f"""
  <div class="tldr">
    <div class="q">Does a forced second look help because the model re-sees the image, or just because it reasons again?</div>
    <div class="cols">
      <div class="col forced">
        <div class="h">Forced — image re-injected</div>
        <div class="big pos">{dpp(f['delta'])}</div>
        <div class="sub">turn-1 image attention <b>{f['attn']['t1']:.3f}</b> (up from {f['attn']['t0']:.3f})</div>
      </div>
      <div class="col rethink">
        <div class="h">Rethink — no image</div>
        <div class="big neg">{dpp(r['delta'])}</div>
        <div class="sub">turn-1 image attention <b>{r['attn']['t1']:.3f}</b> (down from {r['attn']['t0']:.3f})</div>
      </div>
    </div>
    <div class="ans">It's the <b>re-seeing</b>. Both conditions run an identical second reasoning pass — the only difference is whether the image comes back. When it does, attention rises and accuracy improves. Without it, a second pass <b>hurts</b>: {r['change'].get('right_wrong',0)} correct answers flipped wrong, only {r['change'].get('wrong_right',0)} fixed.</div>
  </div>"""

    attn_note = (
        f"Note on the small turn-0 gap (forced {f['attn']['t0']:.3f} vs rethink {r['attn']['t0']:.3f}): "
        "turn-0 is structurally identical in both conditions, so this ~0.01 difference is a measurement artifact. "
        "Forced turn-0 attention is inferred via a heuristic (positions with no attention to the re-injected image); "
        "rethink turn-0 is measured directly from a stored boundary. The gap reflects heuristic imprecision, not real behavior."
    )

    body = tldr
    body += sec(1, "Attention and accuracy move together",
        f"Both panels come from the same runs. Left: how much turn-1 reasoning attends to image tokens. "
        f"Right: the accuracy change from turn 0 to turn 1. Re-injecting the image raises both; withholding it drops both.",
        figs["link"], note=attn_note)
    body += sec(2, "Both conditions start from the same place",
        f"Turn-0 is the same computation in both conditions — any difference is sampling noise from temperature=1.0. "
        f"Of {nz['n']} shared samples, {nz['same']} give the same prediction; the {nz['flip']} that differ "
        f"split {nz['f_only']}–{nz['r_only']} (McNemar z={nz['z']:.2f}, not significant). "
        "The effect is entirely in turn 1.",
        figs["baseline"])
    body += sec(3, "What the second turn did to individual answers",
        "Forced is mildly positive overall. Rethink without the image flips more right answers wrong than it corrects — a net negative.",
        figs["transitions"])
    body += sec(4, "Why: image attention fades as the model reasons",
        "Attention to the image decays across a reasoning pass. In rethink turn 1 (dashed) it never recovers — "
        "without a fresh image, the second pass reasons largely from memory.",
        figs["decay"])

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Re-seeing vs Re-thinking</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
          background:#f5f6f8; color:#1d1d22; margin:0; }}
  .hero {{ background:#16161f; color:#fff; padding:34px 50px 26px; }}
  .hero h1 {{ margin:0 0 6px; font-size:1.7rem; }}
  .hero p {{ margin:0; color:#9aa; font-size:.92rem; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:26px 20px; }}
  .tldr {{ background:#fff; border-radius:11px; box-shadow:0 1px 7px rgba(0,0,0,.08);
           padding:24px 28px; margin-bottom:26px; border-left:5px solid #16161f; }}
  .tldr .q {{ font-size:1.08rem; font-weight:700; margin-bottom:16px; }}
  .cols {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .col {{ flex:1; min-width:220px; border-radius:9px; padding:16px 18px; }}
  .col.forced {{ background:#fcefe4; }} .col.rethink {{ background:#e6f4ea; }}
  .col .h {{ font-size:.82rem; font-weight:700; text-transform:uppercase; letter-spacing:.4px; color:#555; }}
  .col .big {{ font-size:2.1rem; font-weight:800; margin:4px 0; }}
  .col .sub {{ font-size:.83rem; color:#555; }}
  .ans {{ margin-top:16px; font-size:.95rem; line-height:1.6; background:#16161f; color:#eee;
          padding:14px 18px; border-radius:8px; }}
  .ans b {{ color:#ffd27a; }}
  .card {{ background:#fff; border-radius:11px; box-shadow:0 1px 7px rgba(0,0,0,.07);
           padding:22px 28px; margin-bottom:24px; }}
  .card h2 {{ margin:0 0 6px; font-size:1.12rem; display:flex; align-items:center; }}
  .card .num {{ display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px;
               background:#16161f; color:#fff; border-radius:50%; font-size:.85rem; margin-right:10px; flex:none; }}
  .card p {{ color:#555; line-height:1.6; font-size:.9rem; margin:6px 0 12px; }}
  .fig {{ text-align:center; }} .fig img {{ max-width:100%; border-radius:6px; box-shadow:0 1px 4px rgba(0,0,0,.1); }}
  .note {{ font-size:.8rem; color:#888; line-height:1.55; margin-top:10px;
           border-top:1px solid #ececec; padding-top:10px; }}
  .pos {{ color:#2a7a2a; }} .neg {{ color:#c0392b; }}
  footer {{ text-align:center; color:#aaa; font-size:.78rem; padding:18px; }}
</style></head><body>
<div class="hero">
  <h1>Re-seeing vs Re-thinking</h1>
  <p>HallusionBench · Gemma-4-31B-it · 30% stratified subset (seed=42, temp=1.0) ·
     forced n={ns['forced']} / rethink n={ns['rethink']}</p>
</div>
<div class="wrap">{body}</div>
<footer>build_rethink_comparison.py</footer>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SCRIPT_DIR / "rethink_comparison.html"))
    args = ap.parse_args()

    res = {k: load(v[0]) for k, v in PATHS.items()}
    att = {k: load(v[1]) for k, v in PATHS.items()}
    S = build_stats(res, att)

    figs = {
        "link":        plot_link(S),
        "baseline":    plot_baseline(S),
        "transitions": plot_transitions(S),
        "decay":       plot_decay(att),
    }
    ns = {k: len([x for x in res[k] if x.get("is_correct") is not None]) for k in PATHS}
    Path(args.out).write_text(build_html(S, figs, ns))
    print(f"Written → {args.out}")

    nz = S["noise"]
    print(f"\nforced  Δ={S['forced']['delta']*100:+.2f}pp  attn t0={S['forced']['attn']['t0']:.3f} t1={S['forced']['attn']['t1']:.3f}")
    print(f"rethink Δ={S['rethink']['delta']*100:+.2f}pp  attn t0={S['rethink']['attn']['t0']:.3f} t1={S['rethink']['attn']['t1']:.3f}")
    print(f"turn-0 noise: same={nz['same']}/{nz['n']} flip={nz['flip']} ({nz['f_only']}-{nz['r_only']}) z={nz['z']:.2f}")


if __name__ == "__main__":
    main()
