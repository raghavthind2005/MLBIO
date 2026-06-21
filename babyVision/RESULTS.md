# BabyVision — Results: perception-bound, not reasoning-bound

**Model:** Gemma-4-31B-it (Clariden, sglang). **Benchmark:** BabyVision, 388 vision-primitive
VQA items (135 choice / 253 free-form blank), 4 types / 22 subtypes. **Scoring:** `grade.py`
— a Qwen3-32B judge reads the model's full final answer and decides correctness in substance
(format-agnostic; no `\boxed{}` regex). Standard baseline = 3 passes; all other arms = 1 pass.

## Thesis

On these vision primitives, **accuracy is bound by perception, not by reasoning.** Whether the
model gets an item right is determined by whether it can *perceive* the relevant visual cue
from the image — not by how long it reasons, whether it reconsiders, or whether it re-sees the
image. We manipulate reasoning along two axes while holding everything else byte-identical
(prompt, temperature, top-k/p, gold construction, judge), and accuracy does not move.

## Conditions (only reasoning changes; everything else fixed)

| ID | Arm | Manipulation | Reasoning length (median tok) |
|----|-----|--------------|-------------------------------|
| A0 | no-think | `enable_thinking=False`, answer directly | **536** |
| — | standard | natural thinking, 3 passes | **7 334** |
| A3 | forced-long | s1-style "Wait"-injection to a 12k floor | **12 599** |
| B2 | no-reinject | 2-turn: reconsider, **no** image in turn 2 | 9 308 |
| B1 | reinject | 2-turn: reconsider **with image re-shown** | 10 105 |

The length axis (A0→std→A3) spans a **23× range** in reasoning tokens. The freshness axis
(B2 vs B1) isolates the effect of re-grounding the image at answer time.

## Headline: every reasoning manipulation is a null

| Arm | Accuracy | Δ vs standard | paired bootstrap 95% CI | McNemar p |
|-----|----------|---------------|--------------------------|-----------|
| A0 no-think | 29.4% | −1.0 | [−5.9, +3.9] | 0.76 |
| **standard** | **31.6% ±1.4** | — (passes 30.4 / 33.5 / 30.9) | — | — |
| A3 forced-long | 30.7% | +0.3 | [−5.2, +5.7] | 1.00 |
| B2 no-reinject | 28.1% | −2.3 | [−7.2, +2.6] | 0.41 |
| B1 reinject | 33.2% | +2.8 | [−1.8, +7.7] | 0.29 |

Standard's own 3-pass range is **3.1 pts** — larger than most of the cross-condition deltas.
**No delta is statistically significant**; every confidence interval spans 0. Forcing ~1.7× more
reasoning than the model naturally produces (A3) changes accuracy by +0.3 pts (p=1.00).

The re-grounding contrast looks tempting at first (B1−B2 full set = +5.2, CI [+0.5,+10.1]),
but it is a **fallback artifact**, not re-grounding: on the clean paired subset where both arms
produced a genuine two-turn answer, it collapses to **+2.1 pts, CI [−3.8,+7.5], p=0.55** (see
*Two-turn integrity*). Re-showing the image does not help.

## Why it's flat: correctness is item-locked

The null is not "the conditions are secretly identical" — the manipulations demonstrably took
effect (23× length spread; A3 forced on 328/388 items; B1 carries 2 image passes, B2 zero). It
is that **the same items are right or wrong regardless of how the model reasons.**

- **48.2% of items are condition-invariant** across all 5 arms: **154 always wrong** (40%),
  **33 always right** (8.5%).
- **Every pair of conditions agrees on right/wrong 72–77% of the time** — including A0-vs-A3
  (73%), the most extreme length contrast. Chance agreement at this accuracy is ~58%, so
  conditions are coupled **~15 pts above chance.** Correctness is dominantly a property of the
  *item*, not the condition.
- **The ~52% that do flip, flip symmetrically.** Every McNemar contrast has balanced discordant
  pairs — A3-vs-std 55/54, A0-vs-std 46/50, B1-vs-std 51/40, B2-vs-std 43/52. As many items are
  gained as lost when reasoning changes. The flips are **stochastic noise, not a reasoning arm
  systematically winning.**

This rules out the trivial explanation that the model just repeats itself: **92% of items receive
a *different* extracted answer across conditions.** The wording and the answer vary; the
right/wrong *outcome* stays locked to the item. The model reasons differently every time and
still lands on the same perceptual verdict.

**Interpretation:** the model forms its perception of the visual primitive at encoding, and
reasoning operates on that fixed (often wrong) percept without re-perceiving. More thinking,
forced reconsideration, and even re-showing the image cannot recover a cue the model did not
extract in the first place. This matches the public leaderboard, where reasoning-heavy models
also score low on BabyVision.

## Within-condition "see-less" curve (correlational — not causal)

Inside *every* condition, accuracy falls as the trace lengthens (concluded-only, fallback-removed):

| | Q1 (short) | Q2 | Q3 | Q4 (long) |
|---|---|---|---|---|
| standard | 46.7% | 29.6% | 26.8% | 23.4% |
| A3 | 49.5% | 22.7% | 29.9% | 20.6% |
| B1 | 48.7% | 34.6% | 23.1% | 29.1% |
| B2 | 46.4% | 30.6% | 21.2% | 22.4% |

This is striking but **correlational**: trace length proxies item difficulty (hard items induce
longer reasoning *and* are more often wrong). It is **not** evidence that reasoning causes errors —
the clean causal test is the between-condition contrast (A3 forces the *same* items to reason
longer), and that is the null above. We report the curve, we do not claim it is causal.

## Behavioral co-finding: re-grounding induces more runaway loops

Re-showing the image in turn 2 (B1) makes the model spiral into degenerate enumeration loops more
often than text-only reconsideration (B2): **B1 19.3% vs B2 12.6%** of turn-2 traces run to the
budget ceiling mid-thinking (compression ratio 0.21 vs 0.40 for concluded traces — i.e. genuine
loops, not long reasoning). This is a real behavioral effect of re-grounding — but it **does not
convert into accuracy**, consistent with the perception-bound thesis.

## Rigor / threats addressed

- **Alignment:** all 5 conditions cover the identical 388 taskIds with 0 gold-answer and 0
  question mismatches (`babyvision_integrity.py`).
- **Judge faithfulness:** on choice items (where the gold letter is deterministic), the judge
  agrees with the gold letter 84–96% of the time (A0 84.4, std 93.3, A3 95.6, B1 92.6, B2 96.3).
  Nearly all disagreements are the judge correctly crediting a right answer written in prose that
  a `\boxed{}` regex missed — i.e. the judge is *more* faithful than extraction, not less. One
  reverse case in 675 choice gradings.
- **Two-turn integrity:** when turn 2 ran out of budget mid-thinking (no conclusion), we grade the
  model's standing turn-1 answer (the protocol's pre-defined fallback), report the fallback rate
  per arm, and confirm the B1-vs-B2 verdict is identical under full-set / concluded-only /
  raw-grade views. Truncation is reported as a behavioral finding, never folded silently into
  accuracy.
- **Significance:** all deltas tested with paired McNemar (exact) + paired bootstrap CIs
  (`babyvision_validity.py`); none survive.

## Limitations

- Single model (Gemma-4-31B-it) and single judge (Qwen3-32B). Cross-model replication open.
- Non-baseline arms are 1-pass; the baseline's 3-pass band (±1.4) is our noise yardstick.
- The see-less curve is correlational (difficulty confound), explicitly not claimed causal.
- **Mechanism not yet shown directly.** The item-locking result predicts that always-wrong items
  reflect a perceptual failure — the model never attends to / extracts the relevant cue. The
  planned Phase-2 attention pass tests this: does image-token attention on always-wrong items stay
  low / decay with trace length, while always-right items hold it, and does B1's re-grounding fail
  to restore it? That is the direct mechanistic test of the perception-bound claim.

## One-line summary

Across a 23× range of reasoning length, plus forced reconsideration and image re-grounding,
BabyVision accuracy does not move; correctness is locked to the item (48% invariant, all pairs
coupled ~15 pts above chance, remaining flips symmetric/noise). **BabyVision measures perception,
and reasoning cannot substitute for a cue the model failed to perceive.**
