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

Three sets, disjoint by **image** (S5.3), drawn in the same stratified pass as `trial`
(DECISION_LOG S5 v2, manifest `9a109667…`).

| set | size | role | may be looked at during training? |
|---|---|---|---|
| `dev` | 300 | debugging, gates, sanity | yes, freely |
| `eval_monitor` | 1,000 | training curves | yes — **never a confirmatory measure** |
| `eval_final` | **8,000 as built — see §3 and O9** | the confirmatory comparison | **NO. Once, at the end, per run.** |

> **Correction to an earlier draft of this section.** It said these sets came "from the
> untouched remainder of the eligible pool — 25,498 images remain … so no existing split
> moves and the trial pool is undisturbed." **That is no longer true and this document must
> not be frozen while it says so.** S5 v2 allocates 31,300 of 31,798 eligible images; 498
> remain. `eval_final` and `trial` now come out of the same fixed budget, so every image
> given to one is taken from the other. That is O9, and §3 is where it bites.

`eval_final` is disjoint from `eval_monitor` so that monitoring cannot contaminate the
confirmatory set — the usual justification ("we only looked at the curve") does not survive
the sets being the same items.

**External generalisation set:** MMStar (`zli12321/mmstar@test`), which is Vision-SR1's own
`val_files`. Secondary, and the only measure that speaks to generalisation beyond our
substrate. In-distribution improvement on Vision-SR1-47K is a weaker claim and will be
labelled as such.

---

## 3. Power — and the one number in this document that does not yet clear its own bar

The comparison is paired (same items, two models), so McNemar applies:
`x = z·√(d/n)` for minimum detectable difference `x` at discordant rate `d`,
`z = 2.8` (80% power, α = 0.05 two-sided).

| n | `d = 0.25` | `d = 0.40` | trial images left over |
|---|---|---|---|
| 1,000 | 4.4 pp | 5.6 pp | 29,000 |
| 4,000 | 2.2 pp | 2.8 pp | 26,000 |
| **8,000 (as built)** | **1.57 pp** | **1.98 pp** | **22,000** |
| 10,000 | 1.40 pp | 1.77 pp | 20,000 |
| 12,000 | 1.28 pp | 1.62 pp | 18,000 |

**Vision-SR1's effect is +1.7 pp** (47.1 → 48.8, 3B). §4.8's 71.7% live-group rate implies
high within-item variance, so `d = 0.40` is the prudent column — and **in that column the
built `eval_final` of 8,000 has an MDE of 1.98 pp, which is larger than the effect we are
chasing.** At `d = 0.25` it clears (1.57 pp), so the honest statement is: *8,000 is powered
for a Vision-SR1-sized effect only under the optimistic discordance assumption.*

This is stated plainly rather than resolved quietly, because resolving it quietly is exactly
how a pre-registration stops meaning anything. **The trade is now zero-sum** (§2): 12,000
buys 1.62 pp but costs 4,000 trial images.

**What the trial side of that trade actually costs.**

> ⚠️ **CORRECTION [CC, 2026-08-24].** An earlier version of this section compared our step
> count at `rollout_batch_size` **128** against Vision-SR1's ~93 steps at **512**, and
> concluded "every cell exceeds Vision-SR1's ~93 steps." **That comparison was invalid** — a
> step over 128 prompts is a quarter of a step over 512. Read from source,
> `vision_sr1/config.yaml:12` sets `data.rollout_batch_size: 512` (prompts per step) while
> `:36` sets `worker.actor.global_batch_size: 128` (the *optimizer* minibatch). I had
> conflated the two. The comparable unit is **prompts seen**, not steps.

Vision-SR1: 47,628 rows ÷ 512 = **93 steps at `total_epochs: 1`** (`config.yaml:12, 94`), so
47,628 prompts × `rollout.n: 8` ≈ 381k rollouts. Against that:

| trial | prompts / epoch | epochs to match Vision-SR1's exposure | if online filtering were on (25–28% dead) |
|---|---|---|---|
| 22,000 | 22,000 | 2.2 | ≈ 2.9 |
| 20,000 | 20,000 | 2.4 | ≈ 3.2 |
| **18,000** | 18,000 | **2.6** | **≈ 3.5** |

**Neither split reaches Vision-SR1's single-epoch exposure without repeating data, so the
trial side is not where the decision lives.** The ceiling is structural: the *entire* eligible
pool of 31,798 images gives only **62** fresh steps at 512, two-thirds of their one epoch.
That is a consequence of **S5.3** — one row per image collapses 42,288 eligible rows to 31,798
images — and it would bind identically at any O9 split.

So the real trade is **3.5 epochs vs 2.9 epochs of repetition** on the training side, against
**1.62 pp vs 1.98 pp MDE** on the eval side. The first is a difference in degree; the second
is the difference between a study that can detect its target effect and one that cannot.

**O9 closed [U, 2026-08-24]: trial 22,000 / `eval_final` 8,000** — 18,000/12,000 was proposed
and reversed before any rebuild ran. See §3.1 for why, and DECISION_LOG S5 v3/v4.

⚠️ **The ~2.9-epoch repetition is a real deviation from the reference run and belongs to O6.**
It carries overfitting risk Vision-SR1's 1-epoch schedule does not, and V-1 (the Arm A anchor)
is the check most exposed to it: if Arm A fails to reproduce their control, this is the first
thing to suspect.

### 3.1 The table above is the wrong table — read §4 first

⚠️ **[CC] Everything above prices *item* noise, and §4 of this document says the unit of
analysis is the RUN.** Those two sections were inconsistent, and an earlier draft used the item
table alone to argue for `eval_final` = 12,000. Seed variance does not shrink with `n`, so the
primary endpoint's SE is `√(seed² + item²)`:

| `eval_final` | item SE | total SE at σ_seed = 1.0 pp | gain over previous |
|---|---|---|---|
| 4,000 | 1.00 pp | 1.29 pp | — |
| **8,000 (as built)** | 0.71 pp | **1.08 pp** | −16% |
| 12,000 | 0.58 pp | 1.00 pp | −7% |

8,000 is at the knee, and that holds at any σ: the 8,000 → 12,000 gain is 7% at σ = 1.0 pp,
16% at σ = 0.3 pp, **18% even at σ = 0.** No assumption makes 12,000 worth 4,000 training images.

**And `eval_final` was never the checkpoint-time cost.** It is read **once per run, at the end**
(§8.2); `eval_monitor` (1,000) is what runs at every checkpoint. **[V]** One GH200 produced
2,400 generations in **104 s** (job 3169217), so 8,000 ≈ **6 min per run**, ≈ 35 min for the
whole 6-run study.

**The number that actually binds — O11.** At 3 seeds and σ_seed = 1.0 pp the **run-level MDE is
≈ 3.0 pp**, whatever `eval_final` is:

| seeds per arm | seed term | total SE @ 8,000 | run-level MDE |
|---|---|---|---|
| 3 | 0.82 pp | 1.08 pp | **≈ 3.0 pp** |
| 5 | 0.63 pp | 0.95 pp | ≈ 2.7 pp |
| 8 | 0.50 pp | 0.87 pp | ≈ 2.4 pp |

**Detecting +1.7 pp at run level would take ~16 seeds.** So the honest position is: this design
can detect a *large* effect, and cannot resolve a Vision-SR1-sized one — and **no eval size
changes that.** σ_seed is currently an assumption, not a measurement. **It is the most
load-bearing unknown in this document, Tier 1 is where to measure it, and §5.1 may have to be
restated as an effect size with a confidence interval rather than a detect/no-detect verdict.**
This document must not be frozen until that is faced.

**Evaluation is cheap; training is not.** 12,000 items × 2 arms × 3 seeds ≈ 72,000
generations at ~200 tokens ≈ **under 20 minutes total** on one GH200. The cost of more power
is trial images, not GPU time.

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
| V-5 | **Boxed-rate did not fall** below its baseline of **89.1%** — pooled over two disjoint 300-item dev samples, 4,800 generations, Wilson95 [88.2, 90.0] (§4.11). Judged on the **aggregate only**: per-category boxed rates did not replicate across those samples and are not a usable instrument at this n. | `J_success` would be partly scoring formatting rather than perception. |

**V-2 is the one to be most suspicious of, and it is why L1 must exist before the first
training run rather than after a surprising result.**

---

## 7. Two tiers, and what each may claim

### Tier 1 — TRIAL RUN. Mechanism health only.

Pre-committed as **incapable of establishing the claim**, so it can never be quoted as if it
had. Its questions are: does the machinery work, and is there anything alive here?

- `D̂` decreases over steps
- V-2 … V-5 hold
- dead-group fraction stays inside the measured 25.0–28.3% band (§4.11, two disjoint samples)
- boxed-rate rises from 89.1% (pooled baseline, §4.11)
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

- ~~**O9** — the `trial` / `eval_final` split.~~ **CLOSED [U, 2026-08-24] at 22,000 / 8,000.**
- **O11 — run-level power (§3.1). The blocker that matters.** σ_seed is assumed, not measured;
  at 3 seeds the run-level MDE is ≈ 3.0 pp regardless of eval size. Freezing §5.1 as a
  detect/no-detect verdict without measuring σ_seed would pre-commit six training runs to a
  test that may be unable to resolve the effect it targets.
- **O6** — steps, batch shapes, LR, seeds. §4's "3 seeds" and §7's tier split assume a step
  budget that does not exist yet. (The earlier note here — "a 5,000-item trial pool at
  `rollout_batch_size` 512 is ~10 steps per epoch" — is superseded: the pool is 22,000 and
  the batch is 128, giving ≈ 124 fresh steps. The point that the *step budget itself* is
  still unwritten stands.)
- **O2 / λ** — §8.1 needs the selection procedure named.
- **L1** — V-2 needs its instruments specified before it can be a gate.
- **V-1's band** — the acceptable distance from Vision-SR1's anchor must be a number, and it
  is not one yet.

**Nothing here is frozen until those are closed and the user signs off.**
