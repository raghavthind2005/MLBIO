# Clarifications — Experiment Design & Full Result Tables

## What the "two images" mean 

1. **Dataset image type** (`vi=1` vs `vi=2`) — a property of HallusionBench, not our setup.
   Each question has **one** image, either an **original** (`vi=1`) or an **edited**
   image designed to contradict common-sense priors (`vi=2`). 

2. **The 3 conditions** — how we run the model, on the same samples. They do not
   create new images.

## How the conditions are implemented

All conditions are turn-based — we never splice an image into the middle of a
reasoning chain (Gemma-4's reasoning parser breaks if a `<think>` block is interrupted).
A full think→answer always completes, then a new turn may begin.

- **Standard** — show image + question once; model thinks and answers Yes/No.
- **Voluntary tool** — system prompt offers a `LOOK_AGAIN: <region>` tool. In turn 0 the
  model may answer with `LOOK_AGAIN` instead of Yes/No; if so, we crop that region and
  append it as a **new user turn**, and the model reasons again (≤3 times). Only **2.4%**
  of samples ever requested a look.
- **Forced** — every sample gets a second turn unconditionally: turn 0 (image +
  question → think + answer, recorded as `turn0`), then we append a new user turn with the
  same image re-injected + "re-examine, give final answer" (→ `turn1`). turn0 and
  turn1 are the identical image shown twice.

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

### Quartiles, edited (vi=2) images only (The ones where inferring visual cues are more important)

| Condition | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| Standard | 93.9 | 79.4 | 72.7 | 73.5 |
| Voluntary tool | 87.5 | 81.2 | 68.8 | 62.5 |
| Forced (turn0) | 94.4 | 75.0 | 69.4 | **52.8** |

## Example questions per quartile (Standard run, by reasoning length)

One representative example per quartile, showing how questions that elicit longer
reasoning are the ones the model talks itself out of. All examples are *wrong*
predictions except where noted, to illustrate the failure mode.

**Q1 — shortest reasoning (204–615 ch, 93.8% acc).** Simple perceptual checks; the
model answers quickly and usually correctly.
> *"Are two circles in the image different color? Yes or No"* (illusion, gt=No)
> Short chains; the few misses are classic illusions where the prior fires instantly.

**Q2 — (617–904 ch, 83.1% acc).** Prior starts to override perception.
> *"According to the image, is Key West the southernmost point of Florida?"* (map, gt=No
> — the image shows otherwise, but "Key West = southernmost" is a strong prior) → predicted Yes.

**Q3 — (906–1434 ch, 87.5% acc).** Longer deliberation on visual details, still drifting.
> *"Is the vertical line in the middle actually curved?"* (illusion, gt=Yes) → predicted No.
> The model reasons about geometry but trusts the expected/"straight" prior over the figure.

**Q4 — longest reasoning (1445–13562 ch, 69.2% acc).** The model often *reads the image
correctly, then argues itself back to the prior.* Clearest case:
> *"According to the text in this image, is this poster for the TV series Reply 1988?"*
> (OCR, gt=**No**) → predicted **Yes**, after 6,573 characters of reasoning.
>
> The model correctly transcribes the Korean text and even verifies it character-by-character:
> *"The text in the image: 보-내-주-세-요 (5 characters)… '응답하라' (4 characters). It is
> definitely 보내주세요 [‘please send’], not 응답하라 [‘Reply’]."* — i.e. the poster text does
> **not** say Reply 1988. Then it overrides its own correct perception:
> *"Regardless, the image is definitively from the series Reply 1988… Most AI benchmarks
> want the answer that aligns with the most prominent subject… I'll go with Yes."*

This is "think longer, see less" in a single trace: the longer the chain, the more room the
model has to reason *away* from what the pixels say and back toward the linguistic prior —
consistent with the −38% decay in visual attention from early to late reasoning.

## Notes

- **Forced turn1 ≥ turn0** at every image type,
  consistent with the −28% attention to the re-injected image.
