# Offline Perception-KL Probe — Experiment Design (pre-registration draft)

**Status:** DESIGN, not yet built. Nothing here runs until the frozen parameters (§4) and hypotheses
(§8) are signed off. Written to a pre-registration standard: a reader should be able to predict every
number the probe will emit and every decision rule we will apply *before* seeing the data.

**One-sentence purpose.** Measure, on a fixed held-out set and under identical controlled corruption,
how much each trained policy's output distribution depends on the image — the *offline perception-KL* —
for every checkpoint of all three arms, so we can compare **visual grounding** across arms and across
the training trajectory, cleanly and fairly.

---

## 1. Why this probe exists (what WandB cannot tell us)

During training, PAPO logs a scalar `actor/kl_prcp_loss` per step: the mean KL between the model's
outputs on the real vs. masked image, **on the tokens that particular policy sampled that step, under a
fresh random mask, with a coefficient applied.** That number is useful but scientifically *contaminated*
for cross-arm comparison, for four reasons:

1. It is **on-policy on a moving distribution** — each step's tokens come from a different policy on
   different (shuffled) prompts, so step-to-step and arm-to-arm changes mix "grounding" with "what got
   sampled."
2. The **mask is fresh and unseeded** each step → mask noise rides on the signal.
3. Arm A's value is a **bug artifact** (the term was never trained), so its curve is not comparable to
   B/C's in kind.
4. It is measured on the **training set**, not held-out data → it speaks to fit, not generalization.

The probe removes all four contaminants by measuring on a **fixed held-out set**, with a **fixed seeded
mask bank**, using the **same code path for every checkpoint**, and (optionally) on **fixed tokens**. It
converts an incidental training log into a controlled measurement.

---

## 2. Operational definition of "grounding" (no hand-waving)

We define **offline perception-KL** for one (policy θ, prompt q with image I, token sequence o) as the
mean, over the response tokens of o, of the per-token divergence between the policy's next-token
distribution conditioned on the real image vs. the masked image, evaluated on the *same* tokens o:

```
G(θ; q, o, mask) = mean over response tokens t of  D_KL_lowvar( logπ_θ(o_t | q, I_real) , logπ_θ(o_t | q, I_mask) )
```

This is **exactly** the training quantity (§7 gives the byte-level estimator). It is a **distributional**
grounding measure: high G means "when I blacken the image, the probabilities the model assigns to these
very tokens change a lot" → the tokens depended on the image. G ≥ 0; G = 0 means the image was irrelevant
to producing o.

**Two evaluation modes**, because "which tokens o" is itself a scientific choice:
- **On-policy (primary):** o = a response the *evaluated checkpoint itself* sampled under the real image.
  This is what PAPO optimizes; it answers *"does the trained model, on its own outputs, remain grounded?"*
  Confound: different checkpoints emit different o (length, content) → G differs partly because the tokens
  differ, not only because grounding differs.
- **Fixed-token (control):** o = a single frozen reference response, identical for every checkpoint/arm,
  teacher-forced. Removes the token-choice confound and isolates *"for the same words, how much does this
  policy lean on the image?"* Confound: the reference is off-policy for trained checkpoints (that is the
  intended trade — we buy comparability).

Reporting both, and requiring the on-policy conclusion to **survive** the fixed-token control, is the
core of the design. If a cross-arm difference appears on-policy but vanishes under fixed tokens, we will
conclude the difference is token-choice, not grounding.

---

## 3. Design at a glance

- **Subjects:** 13 model checkpoints — one shared base (Qwen3-VL-2B-Thinking, step 0) + steps
  {10,20,30,40,50,60} × {Arm A, Arm B, Arm C}. (Base is shared because all three arms start from identical
  weights — verified: base val 0.253/0.254/0.254.)
- **Stimuli:** a fixed set of N held-out prompts (each with its image) + a fixed seeded bank of K masks per
  image + a fixed reference response per prompt.
- **Design:** within-stimulus, fully crossed — every checkpoint is measured on the *same* prompts, *same*
  masks, and (for the control) *same* tokens. The only thing that varies between measurements is the
  checkpoint's weights. This is a **paired/repeated-measures** design → we compare arms *per prompt* and can
  bootstrap over prompts.
- **Two-pass execution** (memory-safe): Pass-1 generation (vLLM) → save responses; Pass-2 scoring (HF
  forward) → log-probs & KL. vLLM and the HF scorer never co-reside (avoids the sleep/wake OOM class).

---

## 4. Frozen parameters (LOCKED 2026-07-28 — user-approved)

| Parameter | Value | Rationale |
|---|---|---|
| Held-out set | fixed seeded subset of `PAPO_MMK12_test` | disjoint from ViRL39K train; **also gives an acc-sanity anchor** (probe acc(real) should ≈ final val) |
| N prompts | **100** (smoke: 16) | paired design; adequate CIs for the large arm gaps (~0.02–0.03); ~half the generation cost of 200 |
| n on-policy samples/prompt | **4** (smoke: 2) | averages sampling noise in G; also yields acc@4 for the anchor |
| K masks/image | **4** (smoke: 2) | averages the mask nuisance; K masks also give a per-response mask-variance estimate |
| Mask op | `random_patch_blackening(patch_size=14, black_prob=0.6)` | **byte-identical to training** ([papo_utils.py:17](../PAPO_clone/PAPO/verl/trainer/papo_utils.py#L17)) |
| Mask seeding | fixed global seed; deterministic per (prompt_idx, mask_idx); **reused for all 13 checkpoints** | removes mask noise from cross-arm/step comparison |
| KL estimator | `low_var_kl` (Schulman k3), clamps as in code | **byte-identical to training** ([core_algos.py:626](../PAPO_clone/PAPO/verl/trainer/core_algos.py#L626)) |
| Aggregation | per-token → token-mean over response; then mean over K masks, n samples, prompts | matches training's token averaging ([average_loss, core_algos.py:444](../PAPO_clone/PAPO/verl/trainer/core_algos.py#L444)) |
| Generation | temp 1.0, top_p 0.99, top_k −1, max_new_tokens 8192, fixed vLLM seed | **matches training/eval sampling** ([config:85](../PAPO_clone/PAPO/examples/configs/config_grpo_papo.yaml#L85)) |
| Temperature in scoring | 1.0 (logits unscaled, as `logits.div_(1.0)`) | matches the training log-prob convention ([dp_actor.py:159](../PAPO_clone/PAPO/verl/workers/actor/dp_actor.py#L159)) |
| Fixed-token reference | base-model **greedy** (temp 0) response per prompt, frozen once | deterministic; common ancestor of all arms |
| Checkpoint merge | their `scripts/model_merger.py` → HF format | the only supported merge; fidelity checked by C1 (§9) |

---

## 5. Measurement protocol (exact, reproducible)

**Pass 0 — freeze the stimuli (once, before anything):**
1. Draw N prompt indices from `MMK12_test` with a fixed seed → the prompt set. Save indices + a content
   hash.
2. For each prompt image: generate K masks via `random_patch_blackening` under deterministic seeds
   `seed0 + prompt_idx*K + k`. Save the K masked images (or, better, save the seeds and regenerate — but
   store a hash of each realized mask to guarantee reuse).
3. Generate the fixed-token reference response per prompt from the **base** model (greedy). Save token ids +
   text + a hash.
   → These three artifacts (prompt set, mask bank, reference responses) are **frozen** and reused for every
   checkpoint. Any change to them invalidates all comparisons.

**Pass 1 — on-policy generation (per checkpoint):**
4. Merge the checkpoint to HF (if not already).
5. With vLLM: for each of the N prompts, sample **n** responses under the **real** image (temp 1.0, seed
   fixed). Save token ids, text, and grade each with the training reward fn (`accuracy`, `format`,
   `overall`) → gives acc(real)@n and the per-response tokens for scoring.
6. Release vLLM.

**Pass 2 — scoring (per checkpoint, HF forward, no grad):**
7. Load the HF model + processor.
8. **On-policy G:** for each (prompt, sampled response o):
   a. Build I_real and the K masked images I_mask_k (from the frozen bank), each through the **same**
      processor (min/max pixels as training).
   b. `real_lp` = per-token logπ_θ(o | I_real) [1 forward]; `masked_lp_k` = per-token logπ_θ(o | I_mask_k)
      [K forwards].
   c. per-token `kl_k = lowvar(masked_lp_k − real_lp)` (§7); average over K → per-token kl; **save the full
      per-token vector** (needed for Analysis #2, intra-chain) and the token-mean → G_onpolicy(prompt,
      sample).
9. **Fixed-token G (control):** repeat 8 but with o = the frozen base-greedy reference response for that
   prompt (no generation needed) → G_fixed(prompt).
10. Save a tidy long-format table: one row per (arm, step, prompt, sample) with columns {G_onpolicy,
    G_fixed, per-token-kl vector ref, acc, format, response_length, mask_variance}.

**Pass 3 — aggregation & inference:**
11. Per checkpoint: mean G_onpolicy, mean G_fixed, acc(real), each with a **95% bootstrap CI resampling
    prompts** (the paired unit). Plot vs. step, arms overlaid.

---

## 6. Primary and secondary outputs

- **Primary:** G_onpolicy vs. training step, three arms overlaid, with CIs. Endpoint contrast at step 60.
- **Secondary:** G_fixed (control) same plot; acc(real) trajectory (grounding-vs-accuracy joint); the
  (accuracy, grounding) scatter per arm/step.
- **Diagnostics saved for downstream:** per-token kl vectors (→ Analysis #2), per-response mask variance
  (→ noise floor), response lengths (→ length-confound check).

---

## 7. The estimator, byte-exact (so there is zero ambiguity)

Matching [`compute_kl`, core_algos.py:626-630](../PAPO_clone/PAPO/verl/trainer/core_algos.py#L626) with
`log_probs = real_lp`, `ref_log_probs = masked_lp`:
```
s   = clamp(masked_lp − real_lp, −20, +20)     # per token
kl  = clamp(exp(s) − s − 1, −10, +10)          # per token, ≥ 0 in expectation
G   = masked_mean(kl over response tokens)     # per (response, mask)
```
We report the **magnitude** (this equals the positive `kl_prcp` the model maximizes; training logs its
negative). **Fidelity requirement F1 (verify at build):** confirm the probe's masked-image construction
matches training's pipeline (mask applied to the same PIL the model sees, then the identical processor /
min-max pixels), so that I_real and I_mask differ *only* by the blackening. This will be verified by
byte-comparing a probe-masked image to a training-masked image for one example before any full run.

---

## 8. Pre-registered hypotheses & decision rules

Let G_X(s) = mean on-policy grounding of arm X at step s; A_X(s) = mean accuracy(real).

- **H1 (perception raises grounding vs GRPO):** G_C(60) > G_A(60). Directional; decision = the 95%
  bootstrap CI of the paired difference (C−A over prompts) excludes 0.
- **H2 (double entropy suppresses grounding):** G_C(60) > G_B(60). Same test. (Motivated by the on-policy
  training curves: C ~0.055 vs B ~0.035.)
- **H3 (grounding–accuracy trade):** characterize the joint (A_X, G_X). We do **not** assume a sign for
  accuracy; we report A_C vs A_A and G_C vs G_A jointly. "Trade" = G_C>G_A while A_C≤A_A.
- **Control gate (must pass to claim grounding):** any H1/H2 grounding difference must **also** hold, in
  direction, for G_fixed. If it holds on-policy but not fixed-token, we downgrade the claim to
  "token-choice difference, not grounding."
- **Null (H0):** CIs of the paired differences include 0 → no measurable held-out grounding difference at
  this scale/budget. This is a publishable, honest outcome.
- **Trajectory (exploratory, not confirmatory):** shape of G_X(s) over s (monotone rise? early plateau?
  decay?). Reported descriptively; not used for a pass/fail decision (avoids over-reading a single seed).

**Multiplicity:** the confirmatory tests are exactly H1 and H2 at step 60 (two comparisons) + their two
control gates. We will report raw CIs and note the two-test family; we will not mine the 7×3 grid for
significance.

---

## 9. Validity — positive/negative controls & sanity checks (run BEFORE trusting any number)

- **C1 — merge fidelity + acc anchor.** Probe acc(real)@8 at step 60 must ≈ the run's final val
  `accuracy_reward` (within sampling CI). At base, ≈ 0.254. Fail ⇒ checkpoint merge or eval path is broken;
  stop.
- **C2 — estimator floor (negative control).** With `black_prob = 0` (no masking), G must be ≈ 0 for every
  checkpoint (real vs. real). Non-zero ⇒ a bug in the masked-vs-real plumbing.
- **C3 — estimator ceiling (positive control).** With `black_prob = 1` (fully black image), G must be
  clearly large and ordered sensibly. Confirms the estimator responds to corruption magnitude.
- **C4 — mask-noise floor.** Report the across-K variance of G per response; the cross-arm effect must be
  large relative to this nuisance. If comparable, raise K.
- **C5 — determinism.** Pass-2 scoring is deterministic given fixed masks/tokens; re-running must reproduce
  G bit-for-bit. Pass-1 generation is seeded; re-running reproduces the same responses.
- **C6 — base sanity.** Base checkpoint on-policy G should be in the neighborhood of the training step-1
  `kl_prcp` (~0.046) — a coarse cross-check that offline ≈ online at step 0.

Controls C1, C2, C3, C6 are **gates**: the study does not proceed to interpretation until they pass.

---

## 10. Threats to validity (stated, not hidden)

- **T1 — single training seed.** We have one run per arm; the probe's bootstrap CIs quantify *measurement*
  uncertainty (over prompts/masks/samples), **not** training-seed uncertainty. We therefore cannot fully
  separate "arm effect" from "training-seed luck." Mitigation: report the limitation explicitly; if a
  future budget allows, repeat one arm with a second seed to bound it. **Do not** state arm differences as
  seed-robust.
- **T2 — short horizon.** 60 steps (~0.3 epoch) vs. the paper's ~200. Grounding may not have saturated;
  trajectory conclusions are about *this* horizon.
- **T3 — corruption specificity.** G is defined w.r.t. patch-blackening (14/0.6). Grounding under other
  corruptions (blur, crop, color) is untested; optional robustness sweep (vary black_prob ∈ {0.3,0.6,0.9})
  can probe sensitivity of the *ranking* (not just the level).
- **T4 — KL is a proxy.** High G could reflect useful grounding *or* mere brittleness (over-sensitivity to
  any pixel change). G alone cannot distinguish them. **This is why Analysis #3 (accuracy-under-masking) is
  a required companion:** grounded ⇒ G up **and** clean accuracy maintained **and** accuracy drops under
  masking. We will interpret G only jointly with accuracy, never alone.
- **T5 — fixed-token reference off-policy.** The base-greedy reference is increasingly unlike later
  checkpoints' outputs; G_fixed measures grounding on possibly-unnatural tokens. That is the intended cost
  of removing the token confound; we report both modes and reconcile.
- **T6 — merge/precision drift.** HF-merged bf16 weights vs. FSDP training weights. C1/C5 bound this;
  differences within sampling CI are acceptable.

---

## 11. Reproducibility & artifacts

- Frozen inputs (prompt indices, K masks, reference responses) written once with content **hashes**; every
  checkpoint's run records the hashes it consumed → a mismatch aborts.
- All seeds fixed and logged. Output = one tidy long-format parquet (row per arm/step/prompt/sample) + the
  per-token kl vectors (compressed) + a run manifest (code commit, checkpoint paths, hashes, params).
- Deterministic Pass-2; seeded Pass-1. Re-running from the manifest reproduces every figure.

---

## 12. Compute budget (so the cost is not a surprise)

Per checkpoint: Pass-1 = N·n generations (≈ 100·4 = 400 autoregressive decodes; the dominant cost, well
under one validation pass). Pass-2 = N·n·(1+K) on-policy forwards + N·(1+K) fixed-token forwards (single forwards, no
decode; batched, cheap). ×13 checkpoints. The fixed-token control needs **no** generation, so it can run
first as a fast, cheap signal while on-policy generation proceeds. Knobs N/n/K trade cost vs. CI width;
smoke values (16/2/2) validate the pipeline in minutes.

---

## 13. What I will build (folder `PAPO_probe/`, scripts as committed files)

1. `freeze_stimuli.py` — Pass-0: sample prompts, build seeded mask bank, generate base-greedy references;
   write artifacts + hashes.
2. `probe_generate.py` — Pass-1: vLLM on-policy generation + grading per checkpoint → `responses/*.parquet`.
3. `probe_score.py` — Pass-2: HF teacher-forced forwards, `low_var_kl` (imported/replicated byte-exact),
   on-policy + fixed-token, per-token vectors → `scores/*.parquet`.
4. `probe_controls.py` — runs C1–C6 and refuses to proceed if a gate fails.
5. `probe_aggregate.py` — bootstrap CIs, primary/secondary plots, the (accuracy, grounding) joint.
6. `merge_ckpts.sh` — wraps `model_merger.py` for the 13 checkpoints.
7. `README_PROBE.md` — this design, frozen, + the realized hashes/manifest.

Reused verbatim from the PAPO repo: `random_patch_blackening`, the `low_var_kl` math, the reward
`compute_score`, the processor/min-max-pixel image pipeline — so the probe measures *their* quantity, not a
re-derivation.

---

## 14. Decisions — RESOLVED (user-approved 2026-07-28)

1. **Frozen parameters (§4):** ✅ **N=100, n=4, K=4**, MMK12 held-out subset.
2. **Fixed-token reference:** ✅ **base-model greedy** response per prompt.
3. **Robustness sweep (black_prob ∈ {0.3,0.6,0.9}):** ✅ **deferred to v2** (v1 uses training value 0.6).
4. **Confirmatory scope:** ✅ **endpoint (step 60) only** — H1/H2 at step 60 + fixed-token gates are
   confirmatory; the G(s) trajectory is exploratory/descriptive.
