# Think Longer, See Less — Gemma-4-31B on HallusionBench

**Setup.** Gemma-4-31B-it (thinking mode) on a stratified 30% subset of HallusionBench
(~273 visual yes/no questions; `vi=1` original images, `vi=2` edited images designed to
contradict linguistic priors). Three conditions: **(1) standard**, **(2) voluntary tool**
(model may emit `LOOK_AGAIN` to re-see the image), **(3) forced** re-examination (image
re-injected every sample). Attention extracted via HuggingFace eager attention, tracking
visual / instruction / system tokens per reasoning position and per layer.

## Behavioral findings

| Metric | Result |
|---|---|
| Standard accuracy | 83.3% |
| Prior reliance (vi=1 vs vi=2) | 87.1% vs 79.9% |
| Voluntary re-examination rate | **2.4%** (model rarely chooses to look again, bight be due to Gemma4's training) |

**Accuracy collapses as reasoning lengthens** (edited images, by reasoning-length quartile):
Q1 (shortest) **94.4%** → Q2 75.0% → Q3 69.4% → Q4 (longest) **52.8%** (≈ random chance, since these are binary questions).

## Mechanistic findings (attention)

**A — Visual attention decays over reasoning** (`A_visual_decay_over_position.png`):
mean attention to image tokens vs % through reasoning. Visual attention falls
**early 0.104 → late 0.065 (−38%)** in the standard run (−34% tool).

**B — Visual attention by stage, correct vs wrong** (`B_visual_thirds_correct_wrong.png`):
early/mid/late visual attention split by correctness. Correct answers hold slightly higher
late-stage visual attention (tool 0.073 vs 0.063; forced 0.184 vs 0.148; weak in standard).

**C — Visual attention vs reasoning length** (`C_visual_attention_vs_length.png`):
mean visual attention by reasoning-length quartile. Strong negative correlation —
**r = −0.52 (standard), −0.58 (tool), −0.60 (forced)**: longer chains attend less to the
image. This is the mechanism behind the behavioral length→error collapse.

**D — Forced: original vs re-injected image** (`D_forced_turn0_vs_turn1.png`):
attention to the original image (turn0) vs the re-injected copy (turn1). The model attends
**−28% less** to the re-injected image (0.081 → 0.058) — it re-reasons but does not
re-perceive, explaining why forced re-examination barely moves accuracy.

## Takeaway

As Gemma-4 thinks longer, it attends less to the
image (r ≈ −0.5 to −0.6) and drifts toward linguistic priors, collapsing to chance on
prior-contradicting images. Re-showing the image — voluntarily or by force — does not
restore visual grounding.