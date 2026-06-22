# BabyVision results

Model: Gemma-4-31B-it. Benchmark: BabyVision, 388 vision-primitive questions
(135 multiple-choice, 253 free-form), 4 task types, 22 subtypes. Scoring: a Qwen3-32B
judge reads the model's full final answer and decides if it is correct in substance
(no rigid format matching).

## What we did

We ran the same 388 questions five ways. Only the reasoning changed; the prompt,
sampling settings, and scoring were identical across all of them.

| Condition | What changes | Median reasoning tokens |
|-----------|--------------|-------------------------|
| A0 | answer directly, no thinking | 536 |
| standard | normal thinking (run 3 times) | 7,334 |
| A3 | forced to think much longer | 12,599 |
| B2 | reconsider, image NOT shown again | 9,308 |
| B1 | reconsider, image shown again | 10,105 |


## Finding 1: No major trends in accuraccy

| Condition | Accuracy | vs standard | Significant? |
|-----------|----------|-------------|--------------|
| A0 | 29.4% | −1.0 | no (p=0.76) |
| standard | 31.6% (±1.4) | — | — |
| A3 | 30.7% | +0.3 | no (p=1.00) |
| B2 | 28.1% | −2.3 | no (p=0.41) |
| B1 | 33.2% | +2.8 | no (p=0.29) |

Accuracy barely moved: A0 29.4%, standard 31.6%, A3 30.7%, B2 28.1%, B1 33.2%. None
of the differences are statistically real. Standard run three times already
varied by 3.1 points on its own (30.4 / 33.5 / 30.9), which is bigger than most of the
differences between conditions. 

The one result that looked promising — re-showing the image (B1) beating no-reshow (B2)
by 5.2 points — disappeared when we cleaned it up. About 1 in 5 of B1's reconsideration
attempts ran out of room while still thinking (they fell into repetitive loops) and
never produced a real second answer; for those we fairly fell back to the model's first
answer. On the clean subset where both conditions actually produced a second answer, the
gap shrank to +2.1 points with p = 0.55. So the apparent win was anbartifact, not re-grounding helping.

checked the data is valid, not buggy

## Finding 2: whether the model is right is decided by the question, not the reasoning

We treated the 7 runs per item (A0, standard x3, A3, B1, B2) as 7 attempts at the same
question and counted how often each item was answered correctly.

| Across all 7 attempts | Items | Share |
|-----------------------|-------|-------|
| always right | 27 | 7% |
| always wrong | 135 | 35% |
| sometimes right, sometimes wrong | 226 | 58% |

Take any two of the five conditions and go item by item: do they land the same way — both
right, or both wrong? They match on 72-77% of items. That might sound automatic, but it
isn't. Two conditions that had nothing to do with each other would still match about 58%
of the time purely by luck — mostly by both being wrong, since the model fails most items.
So the item largely decides the outcome;
switching the reasoning barely changes which items are answered correctly.

The next check makes the same point another way. A "flip" is when one item comes out right
on one attempt and wrong on another. How often do flips happen?

| Flip rate (an item changes right↔wrong) | Value |
|------------------------------------------|-------|
| changing the reasoning regime (e.g. A0 vs A3) | 25.5% |
| just re-running standard with a new random seed | 25.3% |

Changing the reasoning regime flips an item 25.5% of the time. But keeping everything the
same and only re-running standard with a new random seed flips it 25.3% of the time — the
same rate. So switching how the model reasons disturbs the answers no more than simply
re-rolling the dice does. Reasoning re-samples the answer around a fixed perception; it
does not steer it toward the right one.

## Finding 3: the hard tasks are the ones needing step-by-step looking

The 135 always-wrong items are not spread evenly. Sorting subtypes by how often the model
gets them right shows a clear pattern: the tasks that almost never get solved are the ones
that need serial, step-by-step inspection (tracing a path, counting, matching elements one
by one), while the ones solved more often are single-glance recognitions. We split all 22
benchmark subtypes into these two groups:

| Group | n subtypes | Mean % correct | Ever solved |
|-------|------------|----------------|-------------|
| Serial / step-by-step | 11 | 18.8% | only 12% of these items |
| Single-glance | 11 | 43.4% | — |

The official benchmark subtypes in each group (named exactly as they appear in the dataset):

- **Serial / step-by-step:** Maze, Connect the lines, Metro map, Lines Observation, Find
  the same, Find the different, Find the shadow, Count 3D blocks, Count Same Patterns,
  Paper Folding, 3D Cube Unfold.
- **Single-glance:** Rotation Patterns, Recognize numbers and letters, Overlay Patterns,
  3D Views, 2D Pattern Completion, 3D Pattern Completion, Mirroring Patterns, Logic
  Patterns, Reconstruction, Count Clusters, Pattern and Color Completion.

Grouping all subtypes into "serial" vs "single-glance": serial 18.8% correct vs
single-glance 43.4% (p ≈ 0). This is real but not the whole story — the separation is
moderate (a random single-glance item is harder-to-beat only ~73% of the time), and about
a third of single-glance items also fail. The honest version: needing step-by-step looking
almost guarantees failure (only 12% of serial items are ever solved), but not needing it
does not guarantee success. There is a second source of failure we did not pin down.

## Finding 4: on the unsure items, the model is an uncalibrated coin-flip

The 226 "sometimes right" items were the interesting case: what tips a single attempt
from wrong to right? We checked, holding the item fixed, whether a right attempt differs
from a wrong attempt of the same item.

| Signal (compared between right and wrong attempts of the *same* item) | If it predicted the flip, we'd see... | What we actually found | Predicts the flip? |
|----------------------------------------------------------------------|----------------------------------------|------------------------|--------------------|
| **Confidence** — answer entropy and average token log-probability | wrong attempts noticeably less confident than right ones | right vs wrong separated only ~50% of the time (entropy 54%, log-prob 47%); the model is just as confident when it's wrong as when it's right | No |
| **Length** — total tokens the model spent on the attempt | a consistent direction, e.g. longer attempts more often wrong | only a weak, non-significant hint that longer attempts go wrong on easy items (54% of items, n=48); no effect overall | No |
| **Chosen wrong answer** (multiple-choice items) | wrong attempts converging on one specific distractor (a systematic confusion) | wrong answers scattered across the options (~2.15 distinct wrong answers per item, about what random picking over the 3 distractors gives) | No |

So on an unsure item, whether a given attempt lands right or wrong is not predictable from
the attempt's length, its confidence, or its answer. Each attempt is a coin-flip at the
item's own success rate, and the model has no internal sense of which way it went.

## What this tells us

The model's perception of these puzzles is set when it looks at the image. If it can see
the cue, it answers right; if it can't, no amount of extra thinking, reconsidering, or
looking again recovers it — those just re-sample around the same fixed perception. The
clearest deficit is in tasks that need attention deployed step-by-step across the image
(tracing, counting, one-by-one comparison), which the model essentially cannot do. 

In short: BabyVision measures perception, and on this model reasoning cannot substitute for
a cue that perception failed to extract.... which largely determines right from wrong

Maybe one good idea would be to fixate on these hard, largely unsolvable problems, 
and think how to increase their accuracy.

## What we could not do, and why

We could not explain what makes a single attempt flip from wrong to right. We showed that
flip is just sampling noise here, and nothing observable predicts it. 

A direct mechanism check is still open: extracting the model's visual attention to test
*why* the serial tasks fail (does attention fail to move across the image?). That is the
natural next step for the "why is it hard" question — not for the "what flips it" question,
which this data cannot answer.

## Side note: re-showing the image makes the model loop more

Conditions B1 and B2 both work in two turns — the model gives a first answer, then takes a
second turn to reconsider it. The only difference between them is that B1 shows the image
again in that second turn while B2 does not.

In that second turn the model sometimes gets stuck repeating itself — restating the same
observation again and again (for example, listing "(C): yellow background... (D): yellow
background..." on and on) — and burns through its token budget without ever reaching a new
answer. This happened more often when the image was re-shown (B1: 19.3% of second turns)
than when it was not (B2: 12.6%). So, counterintuitively, giving the model a fresh look at
the image made it *more* likely to spiral into repetition, not less.

This is a genuine behavioral difference between the two conditions, but it did not change
accuracy. (These unfinished second turns were scored fairly: when a second turn never
produced a real answer, we fell back to the model's first-turn answer — see Finding 1.)


Significance: paired McNemar tests and bootstrap confidence intervals (babyvision_validity.py).
Data integrity: alignment, manipulation, and grading checks (babyvision_integrity.py).
Item structure and churn: babyvision_flips.py, babyvision_churn.py. Difficulty grouping:
babyvision_serial.py. Flip structure: babyvision_transitions.py. Scoring: grade.py.

## Appendix: per-subtype solvability

Mean % correct over all 7 attempts, sorted from hardest to easiest. "Group" marks our
serial (S) vs single-glance (H) labelling. Note it is a smooth gradient, not two clean
clusters — a few single-glance subtypes sit low and one serial subtype sits mid-pack.

| Subtype | Group | n | Mean % correct |
|---------|-------|---|----------------|
| Find the same | S | 17 | 7.6 |
| Maze | S | 20 | 8.6 |
| Count 3D blocks | S | 22 | 11.0 |
| Lines Observation | S | 9 | 11.1 |
| Connect the lines | S | 19 | 11.3 |
| Metro map | S | 12 | 17.9 |
| Paper Folding | S | 12 | 19.0 |
| Find the different | S | 16 | 22.3 |
| Find the shadow | S | 23 | 23.0 |
| 3D Cube Unfold | S | 12 | 25.0 |
| Mirroring Patterns | H | 10 | 27.1 |
| Pattern and Color Completion | H | 20 | 30.7 |
| Count Same Patterns | S | 35 | 34.7 |
| Logic Patterns | H | 14 | 41.8 |
| 3D Pattern Completion | H | 18 | 42.1 |
| 2D Pattern Completion | H | 20 | 42.1 |
| 3D Views | H | 27 | 44.4 |
| Reconstruction | H | 14 | 44.9 |
| Count Clusters | H | 18 | 45.2 |
| Overlay Patterns | H | 17 | 48.7 |
| Recognize numbers and letters | H | 23 | 51.6 |
| Rotation Patterns | H | 10 | 55.7 |
