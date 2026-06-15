# Clarifications — Experiment Design & Full Result Tables

## What the "two images" mean (two unrelated things)

1. **Dataset image type** (`vi=1` vs `vi=2`) — a property of HallusionBench, not our setup.
   Each question has **one** image, either an **original** (`vi=1`) or an **edited**
   image designed to contradict common-sense priors (`vi=2`). These are different
   questions in the benchmark; we report accuracy split by type to show prior reliance.

2. **The 3 conditions** — how we *run* the model, on the **same** samples. They do not
   create new images.

So it is **not** a 2-images × 3-conditions grid. It is: one image per question (original
or edited, fixed by the dataset), evaluated under 3 conditions.

## How the conditions are implemented

All conditions are **turn-based** — we never splice an image into the middle of a
reasoning chain (Gemma-4's reasoning parser breaks if a `<think>` block is interrupted).
A full think→answer always completes, then a new turn may begin.

- **Standard** — show image + question once; model thinks and answers Yes/No.
- **Voluntary tool** — system prompt offers a `LOOK_AGAIN: <region>` tool. In turn 0 the
  model may answer with `LOOK_AGAIN` *instead of* Yes/No; if so, we crop that region and
  append it as a **new user turn**, and the model reasons again (≤3 times). Only **2.4%**
  of samples ever requested a look.
- **Forced** — every sample gets a second turn **unconditionally**: turn 0 (image +
  question → think + answer, recorded as `turn0`), then we append a new user turn with the
  **same image re-injected** + "re-examine, give final answer" (→ `turn1`). turn0 and
  turn1 are the **identical image shown twice**, letting us track attention to each
  presentation separately (`visual_turn0` vs `visual_turn1`).

## Accuracy by image type (%)

| Condition | Overall | vi=1 (original) | vi=2 (edited) |
|---|---|---|---|
| Standard | 83.3 | 87.1 | 79.9 |
| Voluntary tool | 81.6 | 88.5 | 75.0 |
| Forced (turn0) | 79.4 | 86.7 | 72.9 |
| Forced (turn1, final) | 80.6 | 87.6 | 74.3 |

## Accuracy by reasoning-length quartile (%)

Q1 = shortest reasoning chain, Q4 = longest.

| Condition | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| Standard | 93.8 | 83.1 | 87.5 | 69.2 |
| Voluntary tool | 93.5 | 82.5 | 80.6 | 69.8 |
| Forced (turn0) | 94.1 | 83.8 | 79.4 | 60.3 |
| Forced (turn1, final) | 92.6 | 85.3 | 80.9 | 63.8 |

### Quartiles, edited (vi=2) images only — sharpest cut

| Condition | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| Standard | 93.9 | 79.4 | 72.7 | 73.5 |
| Voluntary tool | 87.5 | 81.2 | 68.8 | 62.5 |
| Forced (turn0) | 94.4 | 75.0 | 69.4 | **52.8** |

## Reading notes

- The Q1→Q4 collapse appears in **every** condition — robust, not an artifact of one run.
- Steepest on **edited images, forced turn0** (94.4 → 52.8 ≈ chance for binary questions).
- The standard vi=2 quartiles are noisier (≈33 samples per quartile after splitting).
- **Forced turn1 ≥ turn0** at every image type, but only ~1pp — a noise-level lift,
  consistent with the −28% attention to the re-injected image.
- **Caveat on the 2.4% voluntary rate:** Gemma-4 was not trained for "thinking with
  images" (no native image re-examination tool during reasoning — confirmed from its model
  card), so this rate is partly a capability gap, not purely low felt-uncertainty. The
  forced and attention results do not depend on the model choosing a tool, so they are
  unaffected.
