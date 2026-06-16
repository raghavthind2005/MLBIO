#!/usr/bin/env python3
"""
Forced re-examination: does putting the image back in front of the model help?

Segment-aware analysis of the FORCED condition (image re-injected for a 2nd turn).
Avoids the causal-masking dilution that made a naive turn0-vs-turn1 mean misleading:
turn-0 reasoning tokens precede the re-injected image, so they attend exactly 0 to it.
We therefore measure attention WITHIN each reasoning segment.

Outputs to plots_forced/:
  F1_attention_reengage.png   — visual attention by reasoning segment (re-engagement)
  F2_gain_by_subcategory.png  — turn0->turn1 accuracy gain per subcategory
  F3_answer_stickiness.png    — turn0 x turn1 correctness contingency

Prints all statistics used in the summary.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SD = Path(__file__).parent
OUT = SD / "plots_forced"; OUT.mkdir(exist_ok=True)

att = [json.loads(l) for l in open(SD/"results_forced"/"attention_results.jsonl") if l.strip()]
forced = [json.loads(l) for l in open(SD/"results_forced"/"forced_results.jsonl")
          if l.strip() and "error" not in json.loads(l)]
# unique join key (sample_id alone collides; +subcategory+visual_input is unique)
fk = {(r["sample_id"], r["subcategory"], r["visual_input"]): r for r in forced}


def segments(r):
    """Return (img1_turn0, img1_turn1, img2_turn1) attention means, or None."""
    t0 = np.array(r.get("attn_visual_turn0_per_pos") or [])
    t1 = np.array(r.get("attn_visual_turn1_per_pos") or [])
    if len(t0) != len(t1) or len(t0) < 6:
        return None
    m1 = t1 > 1e-9                      # turn-1 output positions (can see re-injected img)
    if m1.sum() < 3 or (~m1).sum() < 3:
        return None
    return t0[~m1].mean(), t0[m1].mean(), t1[m1].mean()


# ── Gather ────────────────────────────────────────────────────────────────────
seg = {"i1_t0": [], "i1_t1": [], "i2_t1": []}
joined = []   # per-sample: img2 attn + turn correctness
for a in att:
    s = segments(a)
    if not s:
        continue
    seg["i1_t0"].append(s[0]); seg["i1_t1"].append(s[1]); seg["i2_t1"].append(s[2])
    f = fk.get((a["sample_id"], a.get("subcategory"), a.get("visual_input")))
    if f and f.get("is_correct_turn0") is not None and f.get("is_correct") is not None:
        joined.append({"img2": s[2], "c0": f["is_correct_turn0"], "c1": f["is_correct"],
                       "sub": a.get("subcategory"), "change": f.get("change_type")})

i1_t0, i1_t1, i2_t1 = (np.mean(seg[k]) for k in ("i1_t0", "i1_t1", "i2_t1"))
tot0, tot1 = i1_t0, i1_t1 + i2_t1

print("="*60)
print("SEGMENT-AWARE VISUAL ATTENTION (n=%d)" % len(seg["i1_t0"]))
print("="*60)
print(f"  turn0 reasoning -> original image     : {i1_t0:.4f}")
print(f"  turn1 reasoning -> original image     : {i1_t1:.4f}")
print(f"  turn1 reasoning -> RE-INJECTED image  : {i2_t1:.4f}")
print(f"  TOTAL visual: turn0 {tot0:.4f} -> turn1 {tot1:.4f}  ({(tot1/tot0-1)*100:+.0f}%)")

# correlation img2 vs correctness
im2 = np.array([j["img2"] for j in joined]); c1 = np.array([j["c1"] for j in joined])
xm, ym = im2-im2.mean(), c1-c1.mean()
rcorr = (xm*ym).sum()/np.sqrt((xm**2).sum()*(ym**2).sum())
print(f"\n  re-injected-image attn: correct={im2[c1==1].mean():.4f} wrong={im2[c1==0].mean():.4f}"
      f"  (r={rcorr:+.3f})")

# ── Stats: stickiness + McNemar ───────────────────────────────────────────────
cont = {(0,0):0,(0,1):0,(1,0):0,(1,1):0}
for j in joined:
    cont[(j["c0"], j["c1"])] += 1
n = len(joined)
wr, rw = cont[(0,1)], cont[(1,0)]
unchanged = cont[(0,0)]+cont[(1,1)]
chi = (abs(wr-rw)-1)**2/(wr+rw) if (wr+rw) else 0
acc0 = (cont[(1,0)]+cont[(1,1)])/n; acc1 = (cont[(0,1)]+cont[(1,1)])/n
print("\n"+"="*60); print("ANSWER STICKINESS / SIGNIFICANCE (n=%d)" % n); print("="*60)
print(f"  unchanged correctness: {unchanged}/{n} = {unchanged/n*100:.0f}%")
print(f"  wrong->right={wr}  right->wrong={rw}")
print(f"  acc turn0={acc0*100:.1f}%  turn1={acc1*100:.1f}%  (Δ {(acc1-acc0)*100:+.1f}pp)")
print(f"  McNemar χ²(cc)={chi:.2f}  -> {'significant' if chi>3.84 else 'NOT significant (p>0.05)'}")

# ── Stats: gain by subcategory ────────────────────────────────────────────────
bysub = defaultdict(lambda:[0,0,0])
for j in joined:
    bysub[j["sub"]][0]+=j["c0"]; bysub[j["sub"]][1]+=j["c1"]; bysub[j["sub"]][2]+=1
gains = {s:((c1-c0)/nn*100, nn, c0/nn*100, c1/nn*100) for s,(c0,c1,nn) in bysub.items()}
print("\n"+"="*60); print("turn0->turn1 ACCURACY GAIN BY SUBCATEGORY"); print("="*60)
for s,(g,nn,a0,a1) in sorted(gains.items(), key=lambda x:-x[1][0]):
    print(f"  {s:10s} n={nn:2d}  {a0:5.1f}% -> {a1:5.1f}%  ({g:+.1f}pp)")

# ── Plot F1: attention re-engagement ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar([0,1,2],[i1_t0,i1_t1,i2_t1],
              color=["#9ecae1","#9ecae1","#e6550d"], width=0.6)
ax.set_xticks([0,1,2])
ax.set_xticklabels(["turn-0 reasoning\n→ original img","turn-1 reasoning\n→ original img",
                    "turn-1 reasoning\n→ RE-INJECTED img"])
ax.set_ylabel("Mean attention weight")
ax.set_title(f"Forced re-injection re-engages vision\n"
             f"total visual attention: turn-0 {tot0:.3f} → turn-1 {tot1:.3f} (+{(tot1/tot0-1)*100:.0f}%)")
for b,v in zip(bars,[i1_t0,i1_t1,i2_t1]):
    ax.text(b.get_x()+b.get_width()/2, v+0.003, f"{v:.3f}", ha="center")
plt.tight_layout(); plt.savefig(OUT/"F1_attention_reengage.png", dpi=150); plt.close()
print(f"\nsaved {OUT/'F1_attention_reengage.png'}")

# ── Plot F2: gain by subcategory ──────────────────────────────────────────────
order = sorted(gains.items(), key=lambda x:x[1][0])
labels = [f"{s} (n={gains[s][1]})" for s,_ in order]
vals = [gains[s][0] for s,_ in order]
def col(v): return "#31a354" if v>1 else ("#de2d26" if v<-1 else "#969696")
fig, ax = plt.subplots(figsize=(8,5))
ax.barh(range(len(vals)), vals, color=[col(v) for v in vals])
ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels)
ax.axvline(0, color="k", lw=.8)
ax.set_xlabel("turn-0 → turn-1 accuracy gain (pp)")
ax.set_title("Where forced re-examination helps\n(perception-limited tasks gain; prior-dominated / saturated don't)")
for i,v in enumerate(vals):
    ax.text(v+(0.2 if v>=0 else -0.2), i, f"{v:+.1f}", va="center",
            ha="left" if v>=0 else "right", fontsize=9)
plt.tight_layout(); plt.savefig(OUT/"F2_gain_by_subcategory.png", dpi=150); plt.close()
print(f"saved {OUT/'F2_gain_by_subcategory.png'}")

# ── Plot F3: stickiness contingency ───────────────────────────────────────────
M = np.array([[cont[(1,1)], cont[(1,0)]],
              [cont[(0,1)], cont[(0,0)]]])
fig, ax = plt.subplots(figsize=(5.6,5))
im = ax.imshow(M, cmap="Blues")
ax.set_xticks([0,1]); ax.set_xticklabels(["turn-1\ncorrect","turn-1\nwrong"])
ax.set_yticks([0,1]); ax.set_yticklabels(["turn-0\ncorrect","turn-0\nwrong"])
for i in range(2):
    for j in range(2):
        ax.text(j,i,M[i,j], ha="center", va="center", fontsize=16,
                color="white" if M[i,j]>M.max()/2 else "black")
ax.set_title(f"Answers barely move: {unchanged}/{n} = {unchanged/n*100:.0f}% unchanged\n"
             f"wrong→right={wr}, right→wrong={rw} (McNemar p>0.05)")
plt.tight_layout(); plt.savefig(OUT/"F3_answer_stickiness.png", dpi=150); plt.close()
print(f"saved {OUT/'F3_answer_stickiness.png'}")
