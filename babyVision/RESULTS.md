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

That is a 23x range in reasoning length, plus two ways of "looking again."

## Finding 1: more reasoning does not help

| Condition | Accuracy | vs standard | Significant? |
|-----------|----------|-------------|--------------|
| A0 | 29.4% | −1.0 | no (p=0.76) |
| standard | 31.6% (±1.4) | — | — |
| A3 | 30.7% | +0.3 | no (p=1.00) |
| B2 | 28.1% | −2.3 | no (p=0.41) |
| B1 | 33.2% | +2.8 | no (p=0.29) |

Accuracy barely moved: A0 29.4%, standard 31.6%, A3 30.7%, B2 28.1%, B1 33.2%. None
of the differences are statistically real. We tested every pair with a paired test and
confidence intervals; every interval includes zero. Standard run three times already
varied by 3.1 points on its own (30.4 / 33.5 / 30.9), which is bigger than most of the
differences between conditions. Forcing the model to think 1.7x longer than it normally
does (A3) changed accuracy by +0.3 points (p = 1.00).

The one result that looked promising — re-showing the image (B1) beating no-reshow (B2)
by 5.2 points — disappeared when we cleaned it up. About 1 in 5 of B1's reconsideration
attempts ran out of room while still thinking (they fell into repetitive loops) and
never produced a real second answer; for those we fairly fell back to the model's first
answer. On the clean subset where both conditions actually produced a second answer, the
gap shrank to +2.1 points with p = 0.55 — i.e. nothing. So the apparent win was an
artifact, not re-grounding helping.

## We checked the data is real, not buggy

A flat result across very different conditions is suspicious, so we verified it.

- All five conditions cover the exact same 388 questions, with identical correct answers
  and identical question text. No misalignment.
- The conditions really are different: 23x spread in reasoning length, A3 was forced on
  328 of 388 items, B1 carries two image passes and B2 zero. The manipulations took.
- Grading is sound. On multiple-choice (where the right letter is known) the judge agrees
  with the known answer 84-96% of the time, and nearly all disagreements are the judge
  correctly crediting an answer written in prose that a strict format check would miss.

So the flat result is real.

## Finding 2: whether the model is right is decided by the question, not the reasoning

We treated the 7 runs per item (A0, standard x3, A3, B1, B2) as 7 attempts at the same
question and counted how often each item was answered correctly.

| Across all 7 attempts | Items | Share |
|-----------------------|-------|-------|
| always right | 27 | 7% |
| always wrong | 135 | 35% |
| sometimes right, sometimes wrong | 226 | 58% |

Any two conditions agree on which items are right or wrong 72-77% of the time. If the
conditions were independent, chance agreement would be about 58%. So which items are
right or wrong is mostly a property of the item, not the condition.

| Flip rate (an item changes right↔wrong) | Value |
|------------------------------------------|-------|
| changing the reasoning regime (e.g. A0 vs A3) | 25.5% |
| just re-running standard with a new random seed | 25.3% |

The key check: changing the reasoning regime flips an item's outcome 25.5% of the time —
but simply re-running standard with a different random seed flips it 25.3% of the time.
They are the same. So changing how the model reasons does no more than re-rolling the
dice. Reasoning re-samples the answer around a fixed perception; it does not steer it.
(92% of items get a differently-worded answer across conditions, so this is not the model
just repeating itself — the wording changes, the right/wrong outcome does not.)

## Finding 3: the hard tasks are the ones needing step-by-step looking

The 135 always-wrong items are not spread evenly. Sorting subtypes by how often the model
gets them right shows a clear pattern. The bottom (almost never solved) are tasks that
need serial, step-by-step inspection — tracing a path (maze, connect-the-lines, metro
map), counting (3D blocks), or matching elements one by one (find-the-same, find-the-
different, find-the-shadow). The top (more often solved) are single-glance recognitions
(rotation, letters, overlay, a single 3D view).

| Group | Mean % correct | Ever solved |
|-------|----------------|-------------|
| Serial / step-by-step (tracing, counting, one-by-one matching) | 18.8% | only 12% of these items |
| Single-glance (rotation, letters, overlay, single 3D view) | 43.4% | — |

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

| Signal (right vs wrong attempt of the same item) | Result | Predicts the flip? |
|---------------------------------------------------|--------|--------------------|
| Confidence (entropy, log-probability) | separates them only ~50% of the time | no |
| Length | "longer goes wrong" hint on easy items, 54% (n=48), not significant | no |
| Which wrong answer it picks | scattered across options, no consistent mistake | no |

So on an unsure item, whether a given attempt lands right or wrong is not predictable from
the attempt's length, its confidence, or its answer. Each attempt is a coin-flip at the
item's own success rate, and the model has no internal sense of which way it went.

## What this tells us

The model's perception of these puzzles is set when it looks at the image. If it can see
the cue, it answers right; if it can't, no amount of extra thinking, reconsidering, or
looking again recovers it — those just re-sample around the same fixed perception. The
clearest deficit is in tasks that need attention deployed step-by-step across the image
(tracing, counting, one-by-one comparison), which the model essentially cannot do. And on
the puzzles it is unsure about, it is unsure in an uncalibrated way: it cannot tell its own
right answers from its wrong ones.

In short: BabyVision measures perception, and on this model reasoning cannot substitute for
a cue that perception failed to extract.

## What we could not do, and why

We could not explain what makes a single attempt flip from wrong to right. We showed that
flip is just sampling noise here, and nothing observable predicts it. That question is real
but needs a different setup: controlled images where one perceptual variable is dialed up
and down with many repeats per setting, so a flip threshold can actually be measured.
BabyVision is too varied and has too few attempts per item for that.

A direct mechanism check is still open: extracting the model's visual attention to test
*why* the serial tasks fail (does attention fail to move across the image?). That is the
natural next step for the "why is it hard" question — not for the "what flips it" question,
which this data cannot answer.

## Side note

Re-showing the image (B1) made the model fall into repetitive reasoning loops more often
than not re-showing it (B2): 19.3% vs 12.6% of second attempts. This is a real behavioral
difference, but it did not change accuracy.

## How to trust these numbers

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
