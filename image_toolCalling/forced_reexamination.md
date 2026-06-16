# Forced Re-examination — Perception Recovers, but the Answer Is Committed

*Forced condition: every sample is answered, then the **same image is re-injected** for a second turn and the model answers again. n = 265 (samples with valid attention segments).*

A naive turn-0-vs-turn-1 attention mean is misleading: turn-0 reasoning tokens come **before** the re-injected image, so by causal masking they attend exactly 0 to it. All numbers below are measured **within** each reasoning segment.

## 1. Re-injection genuinely re-engages vision

![attention re-engagement](plots_forced/F1_attention_reengage.png)

| Reasoning segment | → original image | → re-injected image | total visual |
|---|---|---|---|
| turn-0 | 0.097 | — | **0.097** |
| turn-1 | 0.065 | **0.120** | **0.184 (+91%)** |

During turn-1 the model attends to the fresh image at **0.120 — higher than it ever attended to the original during turn-0 (0.097)** — and total visual attention nearly **doubles (+91%)**.

## 2. …but the answer barely moves

![answer stickiness](plots_forced/F3_answer_stickiness.png)

| | turn-1 correct | turn-1 wrong |
|---|---|---|
| **turn-0 correct** | 205 | 3 (right→wrong) |
| **turn-0 wrong** | 8 (wrong→right) | 49 |

- **Correctness unchanged for 254/265 = 96%** of samples.
- Accuracy 80.4% → 82.3% (**Δ +1.9pp**), **not significant** (McNemar χ² = 1.45, p > 0.05).
- Re-injected-image attention is only weakly tied to being correct: 0.122 (correct) vs 0.111 (wrong), **r = +0.14**.

## 3. The gain depends on whether the edited detail can be *re-read*

![gain by subcategory](plots_forced/F2_gain_by_subcategory.png)

| Subcategory | turn-0 → turn-1 | n |
|---|---|---|
| math | 83.3% → 90.0% (**+6.7**) | 30 |
| ocr | 88.1% → 92.9% (**+4.8**) | 42 |
| figure | 83.3% → 87.5% (**+4.2**) | 24 |
| illusion | 65.9% → 68.2% (+2.3) | 44 |
| chart | 91.7% → 91.7% (0.0) | 24 |
| table | 93.8% → 93.8% (0.0) | 32 |
| video | 69.4% → 69.4% (0.0) | 49 |
| map | 80.0% → 75.0% (**−5.0**) | 20 |

My analysis:

- **math (+6.7), ocr (+4.8), figure (+4.2)** — the edit might changes a *discrete element the model can simply re-read*: an angle/length in a diagram, a word on a label, etc. A second look lets it re-read that element and overturn the turn-0 incorrect read.
- **table (0.0), chart (0.0)** — already correct in turn-0 (94% / 92%); there is almost no headroom left to gain from perception maybe.
- **video (0.0)** — the failure is not a missed detail but the model's *interpretation* (a temporal sequence, or a perceptual illusion). Re-reading reproduces the same interpretation, so the answer does not move.
- **map (−5.0, n=20)** — re-examination *hurts* accuraccy; with only 20 samples this is within noise.
*(Per-subcategory n is small for all instances...)*

## 4. All turn-0 vs turn-1 traces (browse by category)

Complete turn-0 vs turn-1 reasoning and answers for **every forced sample**, grouped by category with flips colour-coded (✗→✓ corrected, ✓→✗ broke), are in a separate browsable page: **`forced_traces.html`** (open in any browser; cards are collapsed, click to expand).

## What is going on

Forcing the image back **restores perception** (visual attention +91%). Where the failure was a missed *detail*, the second look re-reads it and flips the answer (math/ocr/figure, +4–7pp). Where the failure was a committed *interpretation*, the model re-perceives the same thing and re-derives the same answer (96% unchanged). Re-examination repairs **perception**, not **commitment**.
