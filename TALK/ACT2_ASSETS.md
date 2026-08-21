# ACT 2 — asset map and tables

Every slide below lists the **exact file** to put on it. `[HAVE]` = pre-existing; `[NEW]` = generated
2026-08-19 by `make_act2_figures.py`. All numbers traced to the stated source file.

**Substrate for 2.1–2.3:** Gemma-4-31B-it (thinking mode), HallusionBench stratified 30% subset
(~273 visual yes/no questions; `vi=1` original, `vi=2` edited to contradict linguistic priors).
Attention via HuggingFace eager attention. Source: `image_toolCalling/SUMMARY.md`.

---

## 2.1 — Attention to the image decays over the chain

**Primary figure — [HAVE]** `image_toolCalling/plots_attention/A_visual_decay_over_position.png`
*Mean attention weight vs % through reasoning, all three conditions + system/instruction channels.
n = 271 / 269 / 265. The forced condition (green) visibly re-engages after the ~50% mark — that is the
re-injection, and it sets up Act 3.*

**Supporting figure — [HAVE]** `image_toolCalling/plots_attention/C_visual_attention_vs_length.png`
*Mean visual attention by reasoning-length quartile, with r in the legend.*
> ⚠️ The figure legend reads **r = −0.51** for `normal`; `SUMMARY.md` states **−0.52**. Use the figure's
> value on the slide, or re-derive. Do not quote both.

**Table 2.1a — visual attention decay** *(source: `SUMMARY.md`; recomputed and confirmed)*

| Condition | early | late | change |
|---|---|---|---|
| standard | 0.1043 | 0.0649 | **−37.8%** |
| voluntary tool | 0.1029 | 0.0677 | −34.2% |
| forced | 0.1073 | 0.1774 | **+65.4%** ← re-injection |

**Table 2.1b — correlation with chain length** *(source: `SUMMARY.md` / figure C)*

| Condition | r (visual attention vs reasoning length) | n |
|---|---|---|
| standard | −0.52 *(figure: −0.51)* | 271 |
| voluntary tool | −0.58 | 269 |
| forced | −0.60 | 265 |

---

## 2.1b — ★ NEW: the decay is *not* vision-specific

**Figure — [NEW]** `TALK/figures/fig_attention_decay_control.png`

Computed live from `results_{normal,tool,forced}/attention_results.jsonl`
(`attn_{visual,system,instruction}_by_thirds`). **This control did not exist before today.**

**Table 2.1c — all three prompt channels, standard condition**

| Channel | early | mid | late | change |
|---|---|---|---|---|
| visual | 0.1043 | 0.0873 | 0.0649 | −37.8% |
| system | 0.4562 | 0.3782 | 0.3370 | −26.1% |
| **instruction** | 0.0908 | 0.0465 | 0.0455 | **−49.9%** |

**Two things this buys you, and you should say both:**

1. **It pre-empts the obvious objection.** Someone will ask whether "visual attention decays" is just
   attention mass diluting across a growing generated sequence. Answer: partly yes — *every* prompt-side
   channel decays, and visual is not even the fastest. So decay alone cannot carry a causal story. This
   makes Act 3's falsification land as the natural next question rather than a surprise.
2. **★ It is a first-party divergence from the published account.** *More Thinking, Less Seeing?*
   reports that attention to visual tokens falls **while attention to instruction tokens intensifies**.
   In our replication instruction attention **falls by 49.9%** — more than visual. The direction of their
   proposed mechanism does not reproduce here.
   > Scope it honestly: different model (Gemma-4-31B-it), different subset, our own attention pipeline.
   > State it as "does not reproduce in our setting," not as a refutation.

---

## 2.2 — Attention correlates with being right

**Primary figure — [HAVE]** `babyVision/plots/attn_correct_wrong.png`
*"When the model looks at the image, it gets it right."*

**Supporting — [HAVE]** `image_toolCalling/plots_attention/B_visual_thirds_correct_wrong.png`
*(early/mid/late visual attention split by correctness, per condition)*
**Supporting — [HAVE]** `babyVision/plots/attention_decay.png`

**Table 2.2 — visual attention, correct vs wrong answers**

| Study | Condition | correct | wrong | ratio | source |
|---|---|---|---|---|---|
| babyVision | B1′ | 0.021 | 0.011 | **~2×** | `babyVision/RESULTS.md` F6 |
| babyVision | B2′ | 0.012 | 0.007 | ~1.7× | `babyVision/RESULTS.md` F6 |
| HallusionBench | tool (late) | 0.073 | 0.063 | 1.16× | `SUMMARY.md` B |
| HallusionBench | forced (late) | 0.184 | 0.148 | 1.24× | `SUMMARY.md` B |
| HallusionBench | standard | — | — | weak | `SUMMARY.md` B |

⚠️ **Caveats that must be on the slide.** (a) Correlational — easier items may both attract more looking
*and* be answered correctly. (b) babyVision attention covers only the **shorter items, ~16% of the set**
(61 and 67 questions), because holding the full attention matrix is memory-bound. (c) The HallusionBench
effect is weak in the standard condition.

---

## 2.3 — Accuracy collapses as the chain lengthens

**Figure — [NEW]** `TALK/figures/fig_length_collapse.png`
*(two panels: edited images vs all images, three conditions, chance line marked)*

**Table 2.3a — edited images (`vi=2`), by reasoning-length quartile** *(source: `clarifications.md`)*

| Condition | Q1 (shortest) | Q2 | Q3 | Q4 (longest) |
|---|---|---|---|---|
| standard | 93.9 | 79.4 | 72.7 | **73.5** |
| voluntary tool | 87.5 | 81.2 | 68.8 | 62.5 |
| **forced (turn0)** | **94.4** | 75.0 | 69.4 | **52.8** ← ≈ chance |

**Table 2.3b — all images**

| Condition | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| standard | 93.8 | 83.1 | 87.5 | 69.2 |
| voluntary tool | 93.5 | 82.5 | 80.6 | 69.8 |
| forced (turn0) | 94.1 | 83.8 | 79.4 | 60.3 |
| forced (turn1, final) | 92.6 | 85.3 | 80.9 | 63.8 |

> ⚠️ **PRECISION FIX — this matters.** The headline **"94.4 → 52.8"** is the **forced (turn0)** row on
> edited images, **not** the standard condition. On standard-edited the fall is **93.9 → 73.5**, which is
> a real decline but **not** a collapse to chance. `SUMMARY.md` and the earlier deck plan both presented
> 94.4 → 52.8 without naming the condition. Label the condition on the slide, or show the full table.
> An audience member who checks will find the discrepancy.

**Third replication, different model and benchmark** *(source: `text_privilege/PROBE_A_FULL_RUN_REPORT.md` §4.1)*

| Arm | mean think_tok (correct) | mean think_tok (wrong) | median (correct) | median (wrong) | point-biserial r |
|---|---|---|---|---|---|
| T0 (no payload) | 1504.4 | 2365.9 | 287 | 603 | **−0.115** |
| T3 (targeted caption) | 1654.7 | 2636.3 | 318 | 551 | **−0.114** |

Qwen3-VL-4B-Thinking on MMStar, n=1500 × K=3. Longer chains are disproportionately the wrong ones —
the same direction, in a different model, on a different benchmark. **[MAKE]** optional small figure.

---

## 2.4 — We ran the field's own benchmark

**Figure — [NEW]** `TALK/figures/fig_rhbench.png`

**Table 2.4 — RH-Bench, Qwen3-VL-4B-Thinking, 900 items**
*(source: `RH-Bench/Qwen3-VL-4B-Thinking_Results.md`)*

| Subset | multi-choice | free-form | **overall** |
|---|---|---|---|
| Reasoning (MathVision, MathVista, MMMU, ScienceQA) | 78.2% (172/220) | 60.9% (140/230) | **69.3%** (312/450) |
| Perception (MMhalu, MMVP, HallusionBench, VMCBench) | 73.7% (168/228) | 55.0% (122/222) | **64.4%** (290/450) |
| | | | **gap 4.9pp** |

**Table 2.4b — declared deviations from the official protocol** *(put this on the slide, not in backup)*

| Item | Official | Ours |
|---|---|---|
| Judge | GPT-4o (Azure) | **Qwen3-32B** (restricted symlink on the intended model) |
| Dataset size | 1000 | 900 (version difference) |
| max_tokens | unspecified | 16384 |
| Unanswered | — | **57/900 (6.3%)** exhausted the budget mid-`<think>`, scored **incorrect**; 54 of these are in the reasoning subset (12%) |

> ⚠️ **Two hard limits.** (a) Absolute scores are **not comparable** to published RH-Bench numbers —
> different judge, different dataset size. (b) **RH-AUC cannot be reported**: it requires multiple
> (reasoning, perception) points at different thinking budgets; we have **one**. Present this as
> "we engaged the field's instrument and reproduced its qualitative direction," nothing stronger.
> The 6.3% unanswered rate penalizes the reasoning subset ~17× more than perception, which
> *works against* the observed gap — so the gap is conservative. Worth one sentence.

---

## Generated files

| File | Slide |
|---|---|
| `TALK/figures/fig_attention_decay_control.png` | 2.1b |
| `TALK/figures/fig_length_collapse.png` | 2.3 |
| `TALK/figures/fig_rhbench.png` | 2.4 |
| `TALK/make_act2_figures.py` | regenerates all three |
