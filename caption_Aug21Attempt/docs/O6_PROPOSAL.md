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
| `worker.rollout.n` (`G`) | **8** | `:67` |
| `worker.actor.global_batch_size` | **128** | `:36` |
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

### D1 — `rollout_batch_size` **256**, not 512 `:12` **[D]**

**This does not change total compute.** For a fixed data exposure the total number of optimizer
updates is identical, because updates per step = `(rollout_batch_size × n) / global_batch_size`:

| | prompts/step | updates/step | steps for 47,628-prompt exposure | total updates |
|---|---|---|---|---|
| Vision-SR1 | 512 | 32 | 93 | 2,976 |
| **proposed** | **256** | **16** | 186 | **2,976** |

What it changes is two things that both favour 256 here:

- **θ_old goes stale half as fast.** `ray_trainer.py:609-612` recomputes `old_log_probs` once per
  generation phase, so the policy drifts 32 updates from θ_old at Vision-SR1's shape and **16** at
  ours. Less off-policy drift is more stable training — which is the stated goal for run 1.
- **Twice the curve resolution.** 186 logged steps instead of 93, for the same GPU-hours. Run 1
  exists to be analysed; 2× the points is 2× the post-run signal at zero cost.

Third, at 22,000 it makes **1 epoch = 85 steps**, so a 40-step Tier-1 run sits at **0.47 epoch —
entirely fresh data**. That removes O10's repetition confound from run 1 completely, which a
512-shaped run of the same duration could not do.

*Cost to V-1:* real but small. We already break parity on model size (3B vs their 7B config) and
cannot match their 1-epoch exposure at any split (S5 v3). Batch shape is the least of those.

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

### D3 — `λ` = **1.0**, fixed **[D — no Vision-SR1 analogue]**

`J = J_success + λ·J_cap`. Both advantage terms are **group-standardised**: GRPO's is
`(score − mean)/std` per uid (`core_algos.py:176-202`), and S12 applies the same normalisation to
the caption score. Two standardised quantities make λ = 1.0 the principled "equal weight" default
rather than an arbitrary pick — and it is why λ here is **not** comparable to PAPO's γ = 0.01,
which multiplies an unnormalised KL.

**λ is not tuned in run 1** (O7/O8 §8.1 forbids selecting it on eval sets). Instead run 1 logs the
two advantage components separately so their realised magnitudes are visible, and λ for run 2 is
chosen from that — which is exactly the "analyse post-run, then iterate" purpose.

---

## 3. O5 resolved — θ_old is not a free parameter

**[V]** `ray_trainer.py:609-612` recomputes `old_log_probs` from the *current* policy once per
step, after generation and before the actor update. `dp_actor.py:229-236` then splits the batch
into `global_batch_size_per_device` minibatches and loops `ppo_epochs` (= 1) times.

So there is no cadence knob. Staleness is fully determined:
`updates per θ_old refresh = (rollout_batch_size × n) / global_batch_size = 16` at D1.
**O5 closes as a derived quantity, not a decision.**

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

## 5. Proposed staging

The ordering exists so that **the cheapest run answers the question that decides the expensive
ones.**

| stage | data | steps | arms × seeds | purpose |
|---|---|---|---|---|
| **T0 smoke** | `trial_smoke` 2,000 | ~5 | A + B, 1 seed | machinery runs; gates fire on planted violations; `assert_forward_matches_verl` passes |
| **T1 mechanism** | `trial` 22,000 | 40 (0.47 epoch, all fresh) | A + B, 1 seed | Tier-1 health: `D̂` trend, reward trend, V-2…V-5 |
| **T1b σ_seed** | `trial` 22,000 | 40 | **A only, 3 seeds** (reuses T1's A as seed 1 → 2 extra runs) | **measures σ_seed and closes O11** |
| **T2 confirmatory** | `trial` 22,000 | TBD | A + B, `k` seeds set by T1b | the frozen pre-registration |

**T1b is the part I would argue hardest for.** O11 says the run-level MDE is ≈3.0 pp at 3 seeds
under an *assumed* σ_seed = 1.0 pp, and that no eval size fixes it. Two extra short Arm-A runs
convert that assumption into a measurement, and it is the difference between choosing `k` and
guessing it. Spending six confirmatory runs against a guessed σ_seed is the expensive mistake
available here.

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

## 7. Open — needs the user

1. **D1 (256 vs 512)** — the only deviation with a real V-1 cost.
2. **T1b** — 2 extra Arm-A runs to measure σ_seed before committing to T2. Recommended.
3. **λ = 1.0** — confirmed as the run-1 constant, tuned only after run 1.
4. **Hardware shape** (`n_gpus_per_node`, `tensor_parallel_size`, `gpu_memory_utilization`) —
   deliberately unproposed. Vision-SR1's values are for 7B on 8×A100; ours is 3B on GH200. These
   should be settled **by the T0 smoke**, not guessed in a document.
5. **Wall-clock is unestimated on purpose.** The only measured throughput we have is 23 gen/s for
   inference-only generation on one GH200 (job 3169217). Training adds a caption generation pass,
   two scored forward passes, and backward — extrapolating from that number would be the kind of
   guess this project keeps getting caught by. T0 measures it.
