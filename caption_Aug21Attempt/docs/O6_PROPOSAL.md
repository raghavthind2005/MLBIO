# O6 — RL configuration for run 1

> **STATUS: PROPOSAL. Awaiting user sign-off.** Nothing here is settled until approved and
> recorded in `DECISION_LOG.md`. Every value is either **[P]** parity with Vision-SR1 (cited to
> its config line) or **[D]** a deviation with a stated reason. There are no unexamined defaults.

---

## 0. What run 1 is for

Run 1 is **not** a small version of the confirmatory study. Its job is to make the machinery
observable and the next decision cheap. That sets three rules:

1. **Simplified, but not simplified where the simplification is the thing under test.** The
   caption term, the blind pass, and the estimator run at full fidelity from step 1. Cutting
   those would produce a run that proves nothing about this project.
2. **Every knob either matches Vision-SR1 or is a recorded deviation.** V-1 (the Arm A anchor)
   is the only free correctness check we have; each unforced difference weakens it.
3. **The run must be readable afterwards.** Curve resolution and per-component logging are
   design requirements, not nice-to-haves — the stated purpose is to analyse and iterate.

---

## 1. Parity — taken from `vision_sr1/config.yaml`, not from memory

| knob | value | source |
|---|---|---|
| `data.rollout_batch_size` | **512** | `:12` — deviation proposed then **withdrawn**, see §2 |
| `worker.rollout.n` (`G`) | **8** | `:67` |
| `worker.actor.global_batch_size` | **128** | `:36` |
| `algorithm.online_filtering` | **false** | `:29` — same value, **our own reason**, see §2 |
| `optim.lr` | **1.0e-6** | `:54` |
| `optim.weight_decay` | **1.0e-2** | `:55` |
| `optim.lr_warmup_ratio` | **0.0** | `:57` |
| `max_grad_norm` | **1.0** | `:39` |
| `algorithm.adv_estimator` | **grpo** | `:24` |
| `use_kl_loss` / `kl_penalty` / `kl_coef` | **true / low_var_kl / 1.0e-2** | `:26-28` |
| `rollout.temperature` / `top_p` | **1.0 / 0.99** | `:68-69` |
| `val_override_config` | **temperature 0.0, n 1** | `:78-80` |
| `max_prompt_length` / `max_response_length` | **12800 / 4096** | `:10-11` |
| `freeze_vision_tower` | **false** | `:47` |
| `ppo_epochs` | **1** | `verl/workers/actor/config.py:118` (default; Vision-SR1 does not override) |
| `trainer.val_freq` / `val_before_train` | **5 / true** | `:102-103` |

**`G` = 8 is load-bearing, not cosmetic.** §4.11's live-group rate (75.0%) and mean-correct
(2.62/8) were measured *at 8 draws*. Any other `G` silently invalidates the one measurement that
says a GRPO gradient exists here.

**Greedy validation (`temperature 0.0, n 1`) is parity and it also helps O11.** It removes
per-item sampling noise from every validation number, so the endpoint's variance is item +
seed only. Worth stating because it is the one free reduction in the variance O11 is about.

---

## 2. Deviations — three, each with a reason

### ~~D1 — `rollout_batch_size` 256~~ **WITHDRAWN [U, 2026-08-24]. `rollout_batch_size` = 512 `:12` [P]**

Proposed as a deviation, then withdrawn on review. Kept in the record because the reason it was
withdrawn is worth more than the proposal was.

⚠️ **[CC] One of the three arguments for 256 was simply false.** I wrote that a 512-shaped 40-step
run could not stay inside one epoch. **40 × 512 = 20,480 < 22,000** — 512 is *also* entirely fresh
data at 40 steps. That was the argument which made 256 look necessary rather than merely tidier,
and it does not survive arithmetic.

Compared honestly at **equal compute** (512 × 40 steps ≡ 256 × 80 steps — the same 20,480 prompts
and the same 1,280 optimizer updates, since updates = `rollout_batch_size × n / global_batch_size`):

| | 512 | 256 |
|---|---|---|
| θ_old staleness | 32 updates | 16 |
| logged curve points | 40 | 80 |
| V-1 anchor | **intact** | weakened |
| demonstrated to work on *this* dataset + reward | **yes** | no |

They are near-equivalent on substance; 256's remaining edge is diagnostic convenience. That does
not outweigh the standard practice for a first run — **reproduce the reference configuration, then
deviate on evidence.** At 256, an unstable R1 would leave "our batch-size deviation" as a live
explanation costing a second run to eliminate.

**And the deviation is testable for free.** The PPO **clip fraction** is a direct readout of θ_old
staleness. If it runs high at 512, that is the measured evidence that justifies 256 for run 2 —
which is the right order: measure, then deviate. Logged per §6.

### D2 — `algorithm.online_filtering` **false** **[D in reasoning, P in value]**

Same value as Vision-SR1 (`:29` — **false**, see the retraction in DECISION_LOG §S5v3), but it
must be decided on our own objective, because **our situation has a reason theirs does not**:

> `filter_key: overall` filters on the **accuracy** reward. A group with zero accuracy variance
> — all 8 wrong, 20.3% of items — can still carry **non-zero caption-distortion variance**.
> Filtering on accuracy alone would silently discard caption-term signal, which is the one term
> this project exists to measure.

So filtering is off, and if it is ever turned on it must filter on the *composite*, not on
`overall`. Cost of leaving it off: the measured 25.0% dead groups (§4.11) contribute no
accuracy gradient. That is Vision-SR1's cost too, and it is the honest baseline.

### D3 — `λ` = **1.0**, fixed, **confirmed by T0 before R1 commits** **[D — no Vision-SR1 analogue]**

`J = J_success + λ·J_cap`. Both advantage terms are **group-standardised**: GRPO's is
`(score − mean)/std` per uid (`core_algos.py:176-202`), and S12 applies the same normalisation to
the caption score. Both therefore sit at roughly ±1.3, so λ = 1.0 is *literally* equal weight, not
an aggressive one — and it is why λ here is **not** comparable to PAPO's γ = 0.01, which
multiplies an unnormalised KL. The field norm of starting an auxiliary term at 0.01–0.1 is sound
but does not transfer: those coefficients are absorbing a scale mismatch that standardisation has
already removed.

**Why err high rather than low, given R1 gets one shot.** The two failure directions are not
symmetric:

- **λ too small → `D̂` never moves → the run teaches nothing.** This is the expensive failure,
  because Tier-1's entire job is mechanism health.
- **λ too large → accuracy degrades.** This is the cheap failure: §8's "not destructive" exit
  condition catches it, and it is still informative about the scale.

**But it will not be left as an assumption.** T0 already runs both terms, so five steps yield the
*realised* magnitudes of the two advantage components at no extra cost. If both come out ~unit
scale, λ = 1.0 is confirmed empirically before R1 commits; if the caption advantage comes out much
larger, that is learned for free rather than by burning the run. **T0's component magnitudes are a
gate on R1's λ, not just a log line.**

**No warmup.** A 0 → 1.0 ramp is standard for auxiliary losses, but over 40 steps it would consume
a quarter of the run and blur the trajectory being read. It is the **first remedy if instability
appears**, not a run-1 complication.

**λ is not tuned in R1** (O7/O8 §8.1 forbids selecting it on eval sets). R1 logs both components so
λ for run 2 is chosen from data — the "analyse post-run, then iterate" purpose.

---

## 3. O5 resolved — θ_old is not a free parameter

**[V]** `ray_trainer.py:609-612` recomputes `old_log_probs` from the *current* policy once per
step, after generation and before the actor update. `dp_actor.py:229-236` then splits the batch
into `global_batch_size_per_device` minibatches and loops `ppo_epochs` (= 1) times.

So there is no cadence knob. Staleness is fully determined:
`updates per θ_old refresh = (rollout_batch_size × n) / global_batch_size = (512 × 8) / 128 = **32**`.
**O5 closes as a derived quantity, not a decision** — and the **clip fraction** is its diagnostic,
which is what would justify moving to 256 in run 2.

---

## 4. One correctness requirement that is easy to get wrong

**[V] `ray_trainer.py:597-598` warns in-source:** *"this breaks the order of data inside the
batch. Please take care when you implement group based adv computation such as GRPO and rloo."*

`_balance_batch` reorders rows across DP ranks. GRPO survives it because it groups by
`non_tensor_batch["uid"]` — a uuid4 assigned per prompt at `:486-488` and repeated across the `n`
rollouts at `:513`.

> **Therefore S12's `group_normalise` MUST group by the same `uid`, never by position or by
> assuming rollouts of a prompt are contiguous.** A positional implementation would look correct,
> run without error, and compute the caption advantage against the wrong group — §4.6's pattern
> exactly. This is a gate for the smoke: plant a reordering and show the grouping survives it.

---

## 5. Staging — **ONE run, then decide** **[U, 2026-08-24]**

**User decision: no multi-seed, no paired arms, on the first pass.** One run at one setting,
analysed to completion, and the next run chosen from what it shows — same setting with a new
seed, or a different setting.

| stage | data | steps | what | status |
|---|---|---|---|---|
| **T0 smoke** | `trial_smoke` 2,000 | ~5 | correctness only — gates fire on planted violations, `assert_forward_matches_verl` passes, `uid` grouping survives a planted reorder | **not a training run**; see below |
| **R1** | `trial` 22,000 | 40 × 512 = 20,480 prompts = **0.93 epoch, all fresh** | **Arm B only, one seed** | the one run |
| — | | | *decide from R1* | |

**T0 is a correctness check, not a run, and I am counting it as outside "once."** It executes 5
steps on the throwaway subset purely to prove the machinery does what the unit tests claim.
Flagging it explicitly rather than assuming: `ca21_estimator.py`, `ca21_packing.py`, and
`ca21_worker.py` have **never executed against a real model** — they are tested only against
fixtures, and §4.6 records five separate occasions in this project where a fixture-passing
component failed on the real artifact shape. If you want T0 folded into R1 instead, say so.

### Why the one run is Arm B, not Arm A

| | Arm A (λ=0) | **Arm B (λ=1)** |
|---|---|---|
| exercises the caption pass, blind pass, S11/S12/S13 | **no** | **yes** |
| produces a `D̂` trajectory (the Tier-1 mechanism evidence) | no | **yes** |
| if it succeeds, what we learn | published work reproduces | substrate **and** our machinery both work |

Arm A alone would consume the run reproducing someone else's result while testing none of our
code. Arm B strictly dominates on information per GPU-hour.

⚠️ **The honest cost, stated rather than discovered later:** without a paired Arm A, a *flat or
unstable* R1 is ambiguous — substrate, or the caption term? Two things mitigate it, and neither
is an afterthought: §6 logs the **two advantage components separately**, so the accuracy term can
be read as if it were Arm A; and `val_before_train: true` gives a step-0 validation baseline, so
"did anything improve at all" is answerable within the single run. That is weaker than a real
control and is not claimed to be otherwise.

### What deferring the seeds costs

**O11 stays open and T2 stays unsized.** σ_seed remains assumed, not measured, so the number of
confirmatory seeds cannot be chosen yet. That is a deferral, not a resolution — it must be
settled before T2, and R1 cannot settle it, because one run has no run-to-run variance to
observe. Recorded here so it is not quietly forgotten at freezing time.

---

## 6. What run 1 must log to be analysable

Non-negotiable, because the run is otherwise unreadable afterwards:

- `D̂` per step: mean, and the `H(p,q)` / `H(p)` split (failure-mode-2 instrument, S11)
- **both advantage components separately**, pre-λ — sets λ for run 2
- dead-group fraction per step (against §4.11's 25.0% band)
- boxed-rate per step (against §4.11's 89.1% pooled aggregate — **aggregate only**, per §4.11)
- caption length distribution + degenerate-collapse check
- KL-oracle firings (must be zero, V-3)
- G-PARITY / G-BLIND assertion count per step (V-4)
- grad-norm, and the clip fraction

---

## 7. Open

**Closed by the user, 2026-08-24:**
- ~~D1 (256 vs 512)~~ → **512, full parity.** Deviation withdrawn; clip fraction will decide run 2.
- ~~λ~~ → **1.0, gated on T0's measured component magnitudes.** No warmup.
- ~~T1b / multi-seed~~ → deferred, one run first (§5).

**Still open:**

1. **Is T0 outside "once"?** — 5 steps on `trial_smoke`, correctness only. I am treating it as a
   check rather than a run; say if you want it folded into R1.
2. **Hardware shape** (`n_gpus_per_node`, `tensor_parallel_size`, `gpu_memory_utilization`) —
   deliberately unproposed. Vision-SR1's values are for 7B on 8×A100; ours is 3B on GH200. These
   should be settled **by the T0 smoke**, not guessed in a document.
3. **Wall-clock is unestimated on purpose.** The only measured throughput we have is 23 gen/s for
   inference-only generation on one GH200 (job 3169217). Training adds a caption generation pass,
   two scored forward passes, and backward — extrapolating from that number would be the kind of
   guess this project keeps getting caught by. T0 measures it.

---

## 8. R1's exit conditions — written before the run, not after

R1 has no control arm, so what counts as "worth continuing" must be fixed now or it will be
decided by whatever the curves happen to look like. Tier 1 rules (O7/O8 §7) apply: **an accuracy
number from R1 is not evidence of the effect and will not be reported as one.**

| | condition | reading |
|---|---|---|
| **Machinery sound** | V-3 (KL oracle never fired), V-4 (G-PARITY/G-BLIND held every step), `uid` grouping intact | if any fails, R1 measured nothing regardless of its curves |
| **Mechanism alive** | `D̂` decreases over 40 steps | the caption term is doing what it is for |
| **Not degenerate** | caption length stable, no collapse; boxed-rate ≥ 89.1% aggregate; dead-group fraction in the 25–28% band | rules out reward hacking through formatting or caption collapse |
| **Not destructive** | accuracy advantage component behaves like ordinary GRPO; step-0 → step-40 validation not *falling* | λ = 1.0 has not overwhelmed `J_success` |

**A flat `D̂` with everything else healthy is an informative negative**, and is the outcome that
would send us to λ before anything else. **A rising `D̂` is a bug hypothesis first**, not a
finding — forward KL should not increase under a term that minimises it.
