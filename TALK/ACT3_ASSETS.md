# ACT 3 — asset map and tables
### "Attention decay is correlational, not causal" — the first real result

`[HAVE]` = pre-existing · `[NEW]` = generated 2026-08-19 · `[FIXED]` = rebuilt because the original
mis-drew its own headline. Every number below re-derived from raw data or traced to a frozen record.

---

## 3.1 — We restored the attention. The answers did not move. ★ the strongest slide in the deck

Show **two figures side by side**. The whole act turns on the contrast.

**Left — [FIXED]** `TALK/figures/fig_forced_reengage.png`
> ⛔ **Do not use** `image_toolCalling/plots_forced/F1_attention_reengage.png`. It draws three separate
> bars (0.097 / 0.065 / 0.120) under a title claiming "0.097 → 0.184 (+91%)" — but **0.184 is never
> drawn**. It is the *sum* of the two turn-1 bars, because during turn-1 reasoning the model attends to
> **both** copies. A reader measures 0.120 against 0.097 and sees +24%. Bug is at source in
> [`forced_reexam_analysis.py:96-107`](../image_toolCalling/forced_reexam_analysis.py#L96).

**Right — [HAVE]** `image_toolCalling/plots_forced/F3_answer_stickiness.png` *(verified against raw data — correct)*

**Table 3.1a — the intervention works** *(recomputed from `results_forced/attention_results.jsonl`, n=265, 95% CI)*

| Reasoning segment | → original | → re-injected | **total visual** |
|---|---|---|---|
| turn-0 | 0.0967 ±0.0028 | — | **0.0967** |
| turn-1 | 0.0648 ±0.0053 | 0.1196 ±0.0034 | **0.1844 ±0.0070** |
| | | | **+90.7%** |

The fresh copy (0.120) draws **more** attention than the original ever did at its own peak (0.097).
CIs do not overlap — the re-engagement is solid.

**Table 3.1b — the answers don't** *(recomputed; n=265)*

| | turn-1 correct | turn-1 wrong |
|---|---|---|
| **turn-0 correct** | 210 | 3 *(right→wrong)* |
| **turn-0 wrong** | 8 *(wrong→right)* | 44 |

- unchanged **254/265 = 95.8%**
- accuracy **80.4% → 82.3%**, Δ **+1.9pp**, McNemar χ²(cc) = **1.45** → **not significant**
- re-injected-image attention barely tracks correctness: 0.122 (correct) vs 0.111 (wrong), r = **+0.14**

> ⚠️ **`forced_reexamination.md` §2 has a transcription error.** Its contingency table prints
> **205 / 3 / 8 / 49**; the true values are **210 / 3 / 8 / 44**. The doc's own quoted accuracies
> (80.4% → 82.3%) only reconcile with 210/44 — so the prose is right and the table is wrong.
> **Fix the doc before anyone reads it.** The figure F3 is correct.

**The line to say:** *"We doubled the model's attention to the image. Ninety-six percent of its answers
did not change."*

---

## 3.2 — Where it helps, it is perception-limited items only

**Figure — [HAVE]** `image_toolCalling/plots_forced/F2_gain_by_subcategory.png` *(verified — correct)*

**Table 3.2 — turn-0 → turn-1 accuracy by subcategory**

| Subcategory | turn-0 → turn-1 | Δ | n | reading |
|---|---|---|---|---|
| math | 83.3 → 90.0 | **+6.7** | 30 | a discrete element that can be **re-read** |
| ocr | 88.1 → 92.9 | **+4.8** | 42 | ″ |
| figure | 83.3 → 87.5 | **+4.2** | 24 | ″ |
| illusion | 65.9 → 68.2 | +2.3 | 44 | |
| chart | 91.7 → 91.7 | 0.0 | 24 | already at ceiling |
| table | 93.8 → 93.8 | 0.0 | 32 | already at ceiling |
| video | 69.4 → 69.4 | 0.0 | 49 | failure is **interpretation**, not a missed detail |
| map | 80.0 → 75.0 | −5.0 | 20 | n=20, within noise |

⚠️ Per-subcategory n is small throughout; treat the ordering as suggestive, not established.

**Why this slide matters:** it is the first appearance of the **information-gain condition** — the idea
that re-perception pays only where a second look supplies something new. Plant it here; Act 5.4 cashes
it with DeepEyes' +1.6 / −0.3 / **+11.8**.

**Framing line:** *"Re-examination repairs **perception**, not **commitment**."*

---

## 3.3 — Isolating the two things re-examination does

**Figure — [NEW]** `TALK/figures/fig_babyvision_dissociation.png`
*(three bars: standard / B1′ / B2′ — supersedes `babyVision/plots/dissociation.png`, which shows only
standard vs B1′ and therefore omits the control that carries the argument)*

**Table 3.3 — babyVision, Gemma-4-31B-it, 388 items**

| Task family | standard | B1′ *(image re-shown)* | B2′ *(image NOT re-shown)* | n |
|---|---|---|---|---|
| perception (counting, search, tracing) | 15.7% | **22.5%** (+6.8, **p=0.015**) | 20.9% (+5.2) | 191 |
| reasoning (rotation, folding, overlay) | 39.6% | 33.0% (−6.6, p=0.12) | 32.0% (−7.6) | 197 |

★ **The control is the point.** B1′ and B2′ differ *only* in whether the image is re-shown, and
**just 9 of 388 items change** between them.
> **"The driver is the reasoning, not the image."** — `babyVision/RESULTS.md` Finding 5

⚠️ **Honesty riders for the slide.** (a) Only the perception gain is significant; the reasoning drop is a
consistent *direction*, not a proven effect. (b) B1′ also altered turn-2 wording vs the old B1, so it is
not a perfectly clean one-variable swap. (c) B2′ absolutes are **derived** from the stated deltas
(+5.2 / −7.6); only the deltas appear in the record. (d) Overall it is a wash: B1′ 27.8% vs the fair
baseline 27.8%, exactly 42 items flipping each way.

**Supporting, if you want a fourth panel — [HAVE]** `babyVision/plots/attn_correct_wrong.png`:
in ~**8 of 10** items the model attends *more* to the **original** copy than the fresh one.
> ⚠️ This is the **opposite** of the HallusionBench-forced result (fresh copy 0.120 > original 0.065).
> Different injection protocol, different benchmark, same model family. **Report the discrepancy** —
> do not present either as the general case.

---

## 3.4 — There is no stored percept to decay

**Figure A — [NEW]** `TALK/figures/fig_set2_probe.png`

**Table 3.4a — linear probe for a stored scene-percept** *(CLEVR, 15 attribute-marginals, ridge on
teacher-forced hidden states, n=152 correct items)*

| Position | PCA-16 | PCA-32 | PCA-64 |
|---|---|---|---|
| **image tokens** (positive control) | — | **0.918** | — |
| text / reasoning tokens | 0.504 | 0.511 | 0.517 |

Chance across **all** PCA-k, **all 6** sampled layers, **all** positions. The probe works where the
information lives and fails everywhere else.

**Figure B — [NEW]** `TALK/figures/fig_set2_causal_control.png` ★ **never show a pixel null without this**

**Table 3.4b — the injection channel is real** *(A0 mechanics, 5 easy items, greedy)*

| Condition | accuracy |
|---|---|
| normal (no injection) | 1.00 |
| inline splice, mid-assistant | 1.00 |
| user turn, real image | 0.80 |
| user turn, **scrambled** image | 0.00 |
| **no image at all** | 0.00 |

**Table 3.4c — same vs conflicting image** *(vLLM two-image config, n=10)*

| Re-injected content | accuracy |
|---|---|
| the **same** image | 0.90 — attended but **inert** |
| a **conflicting** image | 0.40 — **answers move** |

> **This is the control that makes every pixel null in the talk legitimate.** A skeptic's first move is
> "your injection was broken." Answer: a *different* image moves answers; the *same* image is attended
> and does nothing. The instrument works; the redundancy is the finding.

**Table 3.4d — the "drift" did not replicate** *(E2b Phase 1)*

| | RIPE items | controls |
|---|---|---|
| `</think>` belief flips | **2/39** | **4/39** — controls flip *more* |
| early margin | +0.026 | +0.052 |

`count0` items carry **every** flip (RIPE 7/7, correct 4/4); non-`count0` = 0/0 in both groups.
The early "belief" is a **prior** ("nothing there → 0"), present equally in correct items.

★ **The cautionary slide — worth 60 seconds.** The `flipAns` column reads **11 vs 0**, which looks like
a clean RIPE-only signal. It is a **construction confound**: RIPE sets `wrong_target = model_ans`,
forcing the answer-margin negative, while controls use a distractor, forcing it positive — controls
*cannot* score it. Showing a metric that was true by construction, and how you caught it, buys more
credibility with this audience than any positive result.

**Verdict for the act:** *the answer is determined at emission, not drifted.*

---

## Act-closing verdict slide

- Attention decay is **real** (Act 2), **not vision-specific** (Act 2.1b), and **not causal** (3.1).
- `[PAPER]` corroboration: *MLLMs Know Where to Look* (ICLR 2025) — attention ratio > 1 in most layers
  **even when the answer is wrong**. The model looks in the right place and still fails.
- ⛔ **Do NOT claim "text potent, pixels inert."** Set 3 overturned D6 (`CORRECTIONS.md` C1). The correct
  statement is *the asymmetry does not replicate*.
- ⛔ Do **not** cite 2510.23482 here as replicating the dissociation — that finding was withdrawn.
  It belongs in Act 6.2, for the articulation claim only.

---

## Files generated for Act 3

| File | Slide | Regenerate with |
|---|---|---|
| `TALK/figures/fig_forced_reengage.png` | 3.1 left | `make_forced_figure.py` |
| `TALK/figures/fig_babyvision_dissociation.png` | 3.3 | `make_act3_figures.py` |
| `TALK/figures/fig_set2_probe.png` | 3.4 A | `make_act3_figures.py` |
| `TALK/figures/fig_set2_causal_control.png` | 3.4 B | `make_act3_figures.py` |

## Source-document fixes this pass surfaced

1. **`forced_reexamination.md` §2** — contingency table prints 205/49, true values 210/44. *(3.1)*
2. **`forced_reexam_analysis.py:96-107`** — F1 never plots the 0.184 total it claims in its title. *(3.1)*
3. **`SUMMARY.md` finding D** — "−28% to the re-injected image (0.081 → 0.058)" is the naive turn-0-vs-turn-1
   mean that `forced_reexam_analysis.py` explicitly rejects as causally-masked. Within-segment value is
   **0.120**. Do not put this on a slide.
