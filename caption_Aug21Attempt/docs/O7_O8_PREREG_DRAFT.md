# O7 / O8 — Evaluation and Success Criterion

> **STATUS: DRAFT FOR REVIEW. NOT FROZEN. NO HASH.**
> Nothing here is binding until the user signs it off and it is committed with a SHA-256
> recorded in `DECISION_LOG.md`. Until then it must not be cited as a pre-registration.
>
> **This is the rule the previous attempt broke** — five GPU jobs ran before anyone wrote
> down what winning looked like. Drafted *before* the trainer exists, deliberately, so no
> number has been seen that could shape it.

---

## 0. What is actually being claimed

> `J(θ) = J_success + λ·J_cap` produces a model that answers perception-dependent VQA
> better than `J_success` alone, **because** the caption term makes the model's internal
> serialisation of the image more faithful.

Two clauses, and **the second is not optional**. A win on accuracy with leaking captions is
not this mechanism — it is a different result wearing its clothes. So the design tests both,
and §6 makes the mechanism evidence capable of *disqualifying* a positive headline.

---

## 1. Arms (O8)

| arm | objective | everything else |
|---|---|---|
| **A — control** | `J_success` only (`λ = 0`) | identical |
| **B — treatment** | `J_success + λ·J_cap` | identical |

Identical means: same data and order, same seed, same steps, same batch shapes, same LR
schedule, same backbone and revision, same container, same kernels (S10). **The only
difference between A and B is `λ`.** Arm A is exactly the accuracy-only GRPO configuration
Vision-SR1 reports (§4.7), which is what makes the external anchor meaningful.

**Seeds: 3 per arm, minimum.** Reasons in §4 — this is the part most likely to be cut for
cost and least safe to cut.

---

## 2. Evaluation sets (O7)

Three sets, disjoint by **image** (S5.3), all drawn from the untouched remainder of the
eligible pool — 25,498 images remain after the current 6,300 (§4.3), so no existing split
moves and the trial pool is undisturbed.

| set | size | role | may be looked at during training? |
|---|---|---|---|
| `dev` | 300 (existing) | debugging, gates, sanity | yes, freely |
| `eval_monitor` | 1,000 (existing `eval`) | training curves | yes — **never a confirmatory measure** |
| `eval_final` | **12,000 (new)** | the confirmatory comparison | **NO. Once, at the end, per run.** |

`eval_final` is disjoint from `eval_monitor` so that monitoring cannot contaminate the
confirmatory set — the usual justification ("we only looked at the curve") does not survive
the sets being the same items.

**External generalisation set:** MMStar (`zli12321/mmstar@test`), which is Vision-SR1's own
`val_files`. Secondary, and the only measure that speaks to generalisation beyond our
substrate. In-distribution improvement on Vision-SR1-47K is a weaker claim and will be
labelled as such.

---

## 3. Power — why `eval_final` is 12,000 and not 1,000

The comparison is paired (same items, two models), so McNemar applies:
`x = z·√(d/n)` for minimum detectable difference `x` at discordant rate `d`,
`z = 2.8` (80% power, α = 0.05 two-sided).

| n | `d = 0.25` | `d = 0.40` |
|---|---|---|
| 1,000 | 4.4 pp | 5.6 pp |
| 4,000 | 2.2 pp | 2.8 pp |
| 8,000 | 1.6 pp | 2.0 pp |
| **12,000** | **1.3 pp** | **1.6 pp** |

**Vision-SR1's effect is +1.7 pp** (47.1 → 48.8, 3B). At n = 1,000 we could not have
detected it — a +2 pp result would have looked like a win and been indistinguishable from
noise. §4.8's 71.7% live-group rate implies high within-item variance, so `d = 0.40` is the
prudent column.

**Evaluation is cheap; training is not.** 12,000 items × 2 arms × 3 seeds ≈ 72,000
generations at ~200 tokens ≈ **under 20 minutes total** on one GH200. There is no good
reason to be underpowered on the axis that costs almost nothing.

---

## 4. Seed variance — the part that is easy to get wrong

McNemar quantifies *item-sampling* noise for **one** pair of runs. It says nothing about
**run-to-run** variance, which in RL is frequently comparable to the effect being chased.
A single-seed A-vs-B difference confounds treatment with seed and cannot be repaired
afterwards.

**Therefore the unit of analysis is the RUN, not the item.**

- 3 seeds per arm, paired by seed (A₁ vs B₁, A₂ vs B₂, A₃ vs B₃).
- **Report every per-seed number, never only the mean.**
- Report the within-arm seed spread alongside the between-arm difference.

**The disqualifying rule, fixed now:**

> If the between-arm difference does not exceed the within-arm seed spread, the result is
> reported as **INCONCLUSIVE** — not as positive with a caveat, and not as a trend.

3 seeds gives a genuinely weak run-level test. That is a real limitation and is stated
rather than hidden: with 3 seeds we can detect effects that are large relative to seed
noise, and we **cannot** precisely estimate small ones. If the honest answer turns out to
need 5+ seeds, that is a cost decision to take with eyes open, not a corner to cut quietly.

---

## 5. Endpoints

### 5.1 Primary — ONE number, fixed in advance

**Accuracy on `eval_final`, Arm B minus Arm A, averaged over the 3 seed-pairs.**

`R(y) = grade_answer(extract_boxed_content(y))` — identical to training, identical to
Vision-SR1's. Sampling for evaluation is fixed here and does not vary by arm.

### 5.2 Secondary — the mechanism family (Holm-corrected)

Pre-specified, corrected as a family, reported whatever they show:

| # | measure | direction predicted |
|---|---|---|
| M-1 | `D̂` on held-out items | **down** (captions track sighted behaviour better) |
| M-2 | blind accuracy from `c` alone | **up** (captions carry more usable evidence) |
| M-3 | entropy gap `H(blind) − H(sighted)` | **bounded** (no over-dispersion, failure mode 2) |
| M-4 | MMStar accuracy | **up** (generalisation) |

### 5.3 Exploratory — reported, never used to support the claim

Per-category effects, caption length, boxed-rate trajectory, dead-group fraction, timing.
Anything not listed in 5.1/5.2 is exploratory **by definition**, including anything we think
of after seeing results.

---

## 6. Validity checks — these run BEFORE the primary endpoint is looked at

If any fails, the primary comparison is **not interpreted** until it is understood.

| # | check | why it can disqualify a positive result |
|---|---|---|
| V-1 | **Arm A anchor.** Arm A lands within a pre-set band of Vision-SR1's accuracy-only GRPO on comparable evaluation. | If our control does not reproduce the published control, the setup is wrong and B-vs-A measures our bug, not our method. This is the free correctness check S4 was chosen for. |
| V-2 | **No leak (L1).** Gold-string containment, verdict-assertion phrasing, and the strong one — **answer from `c` with `x` removed**. Arm B's leak rate must not exceed Arm A's beyond a pre-set margin. | A caption that states the answer improves accuracy through a mechanism we are not claiming. **This is the single most likely way to get a real-looking positive for the wrong reason.** |
| V-3 | **KL oracle never fired**, and exact vs one-sample agreed in expectation. | A silently wrong estimator makes every number meaningless. |
| V-4 | **G-PARITY / G-BLIND held** for every scored pair, all steps. | If the blind pass ever saw the image, `D ≈ 0` and the caption term measured nothing while looking excellent. |
| V-5 | **Boxed-rate did not fall** below its §4.5 baseline of 89.5%. | `J_success` would be partly scoring formatting rather than perception. |

**V-2 is the one to be most suspicious of, and it is why L1 must exist before the first
training run rather than after a surprising result.**

---

## 7. Two tiers, and what each may claim

### Tier 1 — TRIAL RUN. Mechanism health only.

Pre-committed as **incapable of establishing the claim**, so it can never be quoted as if it
had. Its questions are: does the machinery work, and is there anything alive here?

- `D̂` decreases over steps
- V-2 … V-5 hold
- dead-group fraction stays near the measured 28.3% (§4.8)
- boxed-rate rises from 89.5%
- captions stay on-task (length, on-topic, no degenerate collapse)

**A Tier-1 accuracy number is not evidence of the effect and will not be reported as one.**
Its only accuracy-related role is to catch catastrophe (e.g. Arm B collapsing).

### Tier 2 — CONFIRMATORY RUN. Everything in §5 and §6.

Runs only after Tier 1 passes, at the pre-registered sizes and seeds, against this frozen
document.

---

## 8. Anti-contamination rules

1. **λ is never selected on `eval_final` or `eval_monitor`.** Fixed a priori or chosen on
   `dev`/trial. If λ is tuned at all, the tuning budget is stated and the confirmatory run
   uses the single chosen value.
2. **`eval_final` is touched once per run, at the end.** No interim peeks, no "just checking".
3. **No post-hoc endpoint substitution.** If the primary is null, the secondaries do not get
   promoted. A null primary is reported as a null primary.
4. **Every run is reported**, including crashed, abandoned, and mis-configured ones, with
   the reason. No quiet re-runs.
5. **This document is hashed before the first Tier-2 step.** Any later change is a new
   version with both hashes recorded, and the reason.

---

## 9. What counts as each outcome

| outcome | condition |
|---|---|
| **Positive** | Primary > 0, exceeds seed spread (§4), **and** V-1…V-5 all pass, **and** at least one of M-1/M-2 moves as predicted |
| **Positive but unexplained** | Primary > 0 and validity passes, but no mechanism measure moves. Reported as an accuracy effect **without** the mechanism claim — the second clause of §0 fails. |
| **Confounded** | Primary > 0 but V-2 fails. Reported as leak-driven. **Not our mechanism.** |
| **Inconclusive** | Difference within seed spread |
| **Null** | Primary ≤ 0 with validity passing |

**A null is a real result here and is pre-committed to be reported as one.** S1 already
names the load-bearing assumption — that training the model to serialise what it sees
reshapes representations the sighted pathway reuses — as *plausible, motivated by Track T,
and unproven*. A clean null is evidence about that assumption and is worth having. This
programme has published nulls before (Set 2 closed on validated nulls; Set 3's H1 failed at
Δ = +0.007); the standard does not change because we would prefer a different answer.

---

## 10. Open dependencies

This document cannot be frozen until:

- **O6** — steps, batch shapes, LR, seeds. §4's "3 seeds" and §7's tier split assume a step
  budget that does not exist yet. **A 5,000-item trial pool at `rollout_batch_size` 512 is
  ~10 steps per epoch, which is almost certainly too few** — either the trial pool grows or
  we run multiple epochs, and that choice belongs to O6.
- **O2 / λ** — §8.1 needs the selection procedure named.
- **L1** — V-2 needs its instruments specified before it can be a gate.
- **V-1's band** — the acceptable distance from Vision-SR1's anchor must be a number, and it
  is not one yet.

**Nothing here is frozen until those are closed and the user signs off.**
