# Presentation & Mastery Dossier — RL-Perception Mechanism in a VLM

> **What this document is.** Everything you need to (a) give the talk and (b) survive a deep grilling: each
> result paired with *exactly how the number was computed (code-cited, no hand-waving)*, *what it means in terms
> of the model*, *how to interpret it*, and *how it ties to the research goal* — built from absolute scratch to
> expert mastery. **Every formula and number here is grounded in the actual code**, with `file:line` citations
> for the RL internals and script names for the analyses. Nothing is invented; where a value is a measurement,
> its source command is named.
>
> **Companion files:** `METHODS.md` (condensed how-reference), `FINDINGS.md` (results log, Parts 0–12),
> `RESULTS.md` (per-knob config), `RL_MASTERY.md` (RL from scratch). Code: `runs/analysis/`. Training engine:
> `EasyR1/` @ commit `dd71bbd`; reward/config: `VLM-CapCurriculum/training/`.
>
> **How to read it:** Part I is the talk spine. Parts II–VI are the deep mastery (RL training → foundations →
> the five experiments → systems/memory). Part VII is anticipated supervisor questions with exact answers. Part
> VIII is the honest caveats. Skim Part I to build slides; *know* Parts II–VIII cold.

---

# PART I — THE TALK (the narrative arc, in 7 beats)

1. **The problem.** VLMs make *perception* errors that more reasoning can't fix. We post-trained one with RL on
   perception and it improved a lot (perception reward 0.365→0.746). **Where and how, inside 4.4B numbers, does
   that improvement live?**
2. **The lever.** Three byte-identical training runs, differing only in which part is frozen: full / LLM-only /
   ViT-only.
3. **Finding 1 (weights):** the change is *tiny* (0.05%) and biased toward the MLP.
4. **Finding 2 (component):** it's the **LLM, not the vision encoder** (LLM-only = full; ViT-only ≈ base; frozen
   parts provably move 0).
5. **Finding 3 (depth):** the answer becomes readable only in the **late layers (~24+)**; early/mid are untouched.
6. **Finding 4 (causation):** an **MLP-dominant, depth-distributed** edit causes it (graft).
7. **Capstone (re-usability):** the better **representation** is portable — inject it at layer 24 and base
   recovers 82% — but **input-specific** (a fixed steering vector caps at ~40%). → the empirical case for an
   **on-demand, image-derived re-inspection tool-call.**

**One-sentence mechanism (the thing to be able to defend):** *RL's tiny, MLP-dominated, depth-distributed edits
across layers ~0–24 progressively write the answer into the residual stream; by layer 24 it is present and
readable, and the unchanged late layers read it out.*

---

# PART II — THE RL TRAINING, FROM SCRATCH & CODE-GROUNDED

*(This is where supervisors grill hardest. Every claim below cites the file that implements it.)*

## II.0 What "training the model" means here
The model is a **policy** `π_θ` — a probability distribution over the next token given the context. "Training"
changes `θ` (the 4.4B weights) so that token sequences with **higher reward** become **more probable**. We use
**RLVR** (RL with *Verifiable* Rewards): the reward is a deterministic program (is the answer correct?), not a
learned reward model.

## II.1 The exact data + prompt the model sees
- **Data:** `perception_difficulty_curriculum.jsonl`, 3360 DOCCI perception MCQs. Each item:
  `problem` (a question + 4 lettered options + "Respond using only the letter…"), `answer` (a single letter),
  one `image`.
- **Prompt actually fed to the model:** the problem is wrapped by `format_prompts/math.jinja`, which **appends**:
  *"You FIRST think … within `<think> </think>` tags. The final answer MUST BE put in `\boxed{}`."* So **during
  training the model reasons inside `<think>…</think>` and then emits `\boxed{<letter>}`.** (Keep this in mind —
  it differs from how the *probe* later reads the model; Part IV.D.)

## II.2 The reward — exactly (`VLM-CapCurriculum/.../math.py:74-91`)
For each generated response:
- `accuracy_reward` (`math.py:54`): `extract_boxed_content(response)` pulls the text inside `\boxed{}`;
  `grade_answer(answer, ground_truth)` (from `mathruler`) checks correctness; returns **1.0 if correct else 0.0**.
  (Guarded by `too_complex` and a 1s timeout.)
- `format_reward` (`math.py:63`): returns **1.0** iff the response **fullmatches** the regex
  `<think>.*</think>.*\boxed\{.*\}.*` (DOTALL), else 0.0.
- `compute_score` (`math.py:85`): `overall = (1 − 0.1)·accuracy + 0.1·format = 0.9·accuracy + 0.1·format`
  (`format_weight = 0.1`).

**So the number we report as "accuracy 0.365→0.746" is the `accuracy` component** — the fraction of sampled
completions whose `\boxed{}` letter is correct. `overall` (0.9·acc+0.1·format) is the headline reward the
optimizer maximizes.

## II.3 GRPO — the exact advantage (`EasyR1/.../core_algos.py:176-216`)
GRPO = **Group Relative Policy Optimization**. There is **no value/critic network**; the baseline is the group
mean. Per training step:
1. For each prompt, sample **n = 5** completions (`rollout.n=5`, `temperature=1.0`, `config.yaml:61-63`).
2. Each completion's reward is placed on its **last token** → a token-level reward tensor `(batch,
   response_len)`; `scores = token_level_rewards.sum(dim=-1)` (`core_algos.py:199`) recovers the scalar.
3. Group the 5 scores by prompt `index`; compute group mean and **std** (`core_algos.py:209-210`).
4. **Advantage = group-relative z-score** (`core_algos.py:213`):
   `A_i = (score_i − mean_group) / (std_group + 1e-6)`.
5. Broadcast the scalar advantage onto every response token via `response_mask` (`core_algos.py:215`):
   `returns = A.unsqueeze(-1) * response_mask`. (`assert len(group) > 1` — GRPO needs n>1, `core_algos.py:208`.)

**Interpret:** a completion is rewarded for being *better than its 5-sample peers on the same prompt*, scaled by
how spread-out the group is. The std-division is the (debated) GRPO normalization — see Part VII Q2.

## II.4 The policy loss — exact, with dual-clip (`core_algos.py:464-509`)
- **Importance ratio** `r = exp(log π_θ − log π_θ_old)` (`negative_approx_kl = log_probs − old_log_probs`,
  `core_algos.py:464`; `ratio = exp(clamp(·, −20, 20))`, `:478`). `π_θ_old` = the policy that *generated* the
  rollouts (recomputed log-probs); ratio≈1 at the start of each update.
- **Clipped ratio** uses **asymmetric DAPO clipping**: `exp(clamp(·, log(1−0.2), log(1+0.3)))`
  (`:479-481`), i.e. `clip_ratio_low=0.2`, `clip_ratio_high=0.3` (`config.py:108-110`).
- `pg_loss = −A·r`, `pg_loss2 = −A·clipped_r`, take `max` (the standard PPO pessimistic bound, `:497-501`).
- **Dual-clip** (`clip_ratio_dual=3.0`, `config.py:112`): for negative-advantage tokens, also lower-bound by
  `−A·3.0` (`pg_loss3`, `:499,503-504`) to prevent destabilizing updates when `A<0`.
- Averaged over tokens (`loss_avg_mode="token"`, `config.py:114`).

**Interpret:** standard PPO-clip stops any single update from moving the policy too far; the asymmetric 0.2/0.3
and dual-clip 3.0 are DAPO/Dual-clip-PPO defaults. *These are engine defaults, not paper-set — flag as such.*

## II.5 The KL leash — exact (`core_algos.py:590-594`, applied `dp_actor.py:272-281`)
We use **loss-side KL** (`use_kl_loss=true`, `config.yaml:26`), **not** reward-side. The penalty is the
**k3 (low-variance) estimator** of `KL(π_θ ‖ π_ref)`:
`x = (ref_log_probs − log_probs).clamp(−20,20); kld = exp(x) − x − 1` (`core_algos.py:592-593`). This is always
≥ 0, unbiased, and lower-variance than the naive `log_probs−ref_log_probs`. It's added to the loss:
`loss = pg_loss + kl_loss · kl_coef` with **`kl_coef = 1e-2`** (`dp_actor.py:281`, `config.yaml:28`).
`π_ref` = the frozen base policy.

**Interpret + why it matters for our results:** this term *actively penalizes drifting from the base model*.
Together with **`lr = 1e-6`** (`config.yaml:49`, `weight_decay=1e-2`, warmup 0, optimizer `adamw_bf16`), it is
the reason the weights barely move (Part IV.A: 0.05%). *This is not an accident — it's the designed regime, and
it's exactly the regime in which "tiny surgical edit" is the right description.*

## II.6 The fit loop, per step (conceptually, `ray_trainer.py` fit())
generate (vLLM, n=5) → compute reward (math.py) → recompute old log-probs → compute ref log-probs (for KL) →
GRPO advantage → policy update (FSDP). **Sizes (verified in config):** `rollout_batch_size=512` prompts/step,
`global_batch_size=128` prompts/optimizer-update (so 512/128 = 4 updates/step; divisibility is required),
`micro_batch_size_per_device_for_update=4` / `…_for_experience=8` (gradient-accumulation chunking — the gradient
is identical, only memory differs). **16 epochs over 3360 items ≈ 96 optimizer steps** (matches the 96/96 in the
logs). `seed=1`.

## II.7 The three conditions and the freeze mechanism — exact (`fsdp_workers.py:263-287`)
All three runs are byte-identical except two flags. Freezing is *not* a soft penalty — it removes parameters
from the optimizer:
- `freeze_vision_tower=true` → `model.model.visual.requires_grad_(False)` (`:265`), prints
  *"Vision tower is set to not trainable."* (`:267`).
- `freeze_language_model=true` (our added flag) → `model.model.language_model.requires_grad_(False)` +
  `model.lm_head.requires_grad_(False)` (`:280-282`), prints *"Language model is set to not trainable (ViT-only
  condition)."* (`:285`).
- Both set `fsdp_config.use_orig_params = True` (`:266/283`) so FSDP can flatten a mixed-trainability module.
- The optimizer is built over `filter(lambda p: p.requires_grad, …)` (`:343,351`) → **frozen params receive no
  gradient and no update → their saved weights equal the base bit-for-bit → Δ = exactly 0.** *(This is precisely
  why the weight-level freeze proofs read 0.000 — Part IV.B.)*

---

# PART III — FOUNDATIONS (model, weights, checkpoints — 4 concepts)

## III.1 The model anatomy (verified by our parameter census, Part IV.A)
`Qwen3-VL-4B-Instruct = Qwen3VLForConditionalGeneration`:
- `model.model.visual` — **vision tower**, ~0.42B params, **24 transformer blocks** (verified: 96 attn tensors /
  4 per block = 24); turns the image into image-tokens.
- `model.model.language_model` — the **LLM**, ~4.0B params, **36 decoder layers** (verified: 216 attn tensors / 6
  per layer = 36); reads image-tokens + question, writes the answer.
- `lm_head` — output projection, **tied** to `embed_tokens` (so the *base* model lists it once → **713 named
  tensors**; a checkpoint re-saves an untied copy → 714).
- A decoder layer = **attention** (`q,k,v,o_proj` + `q_norm,k_norm`) + **MLP** (`gate,up,down_proj`) + two norms.
  Attention mixes information *across token positions*; the MLP transforms *each token independently* (and in
  interpretability is associated with feature/knowledge readout — why "is the fix in the MLP?" matters).

## III.2 What a "weight" is, and what "learning" is
Every weight is a number in a matrix; a forward pass multiplies activations by these matrices. **Everything the
model learned is the difference `Δ = W_trained − W_base`.** Studying Δ (Part IV.A) and what it does to activations
(IV.D) and behavior (IV.E, IV.F) *is* studying the learning.

## III.3 Checkpoints + the residual stream
`save_freq=6` → 16 checkpoints (`global_step_6…96`); **base = step 0**. The **residual stream** is a running
vector that each layer *adds to*; "information flows up the residual stream" is literal, and is the object the
depth-probe and activation-patch read.

## III.4 Why a checkpoint is 4 files, and how we rebuild a weight (`ckpt_model.reconstruct_full_state_dict`, mirrors `scripts/model_merger.py`)
We trained on **4 GPUs with FSDP**, which splits each weight matrix into 4 slices (one per GPU) → each checkpoint
is `model_world_size_4_rank_{0,1,2,3}.pt`. Each slice is a **DTensor** carrying its `placement` (`Shard(dim)` = a
slice along `dim`, or `Replicate` = a full copy). To get the full weight:
`full = torch.cat([d._local_tensor for d in shards], dim=placement.dim)` (Replicate → take slice 0). **CPU,
single-process, no GPUs, no process group.** We never write merged checkpoints to disk (memory: ~9 GB for the 4
shards + the rebuilt dict; the node has 450 GB). The tied-`lm_head` duplicate (714 vs 713) is bit-identical and
ignored.

---

# PART IV — THE FIVE EXPERIMENTS (each: exact computation → what the number means → interpret → research tie)

## IV.A — Weight-delta: *where did the weights change?* (`weight_delta.py`, S2)

**Inputs.** Base safetensors + a checkpoint's 4 shards → 713 tensor pairs `(W_base, W_ckpt)`.

**Exact computation** (`weight_delta.compute_metrics`, both cast fp32):
- `Δ = W_ckpt − W_base`.
- `abs_fro = ‖Δ‖_F = sqrt(Σ_ij Δ_ij²)` = `delta.norm()`.
- **`rel_fro = abs_fro / max(‖W_base‖_F, 1e-12)`** ← primary. *In model terms:* the fraction of its own
  length the weight matrix moved in parameter space.
- `mean_abs = mean(|Δ|)`; `cos = F.cosine_similarity(vec(W_base), vec(W_ckpt))`.
- Each tensor is tagged `(component, module, layer_idx)` by `qwen3vl_param_map.classify` (regex on the name).

**What the headline number means.** *(from `summarize_deltas.py --condition full`)* LLM mean `rel_fro ≈ 5e-4` =
**the weight matrices moved 0.05% of their length.** `cos ≈ 1`. MLP tensors moved 1.4–1.6× more than attention
(ratio mlp/attn = 1.59 early / 1.40 mid / 1.37 late). Roughly uniform across depth in *magnitude*.

**Interpret.** A 0.05% nudge (not a rewrite) ≈ doubled perception → "surgical edit." The tininess is the
*expected* fingerprint of `lr=1e-6` + the KL leash (Part II.5) — *defend it that way, not as a surprise.*
**Freeze proof:** `llm_only` → all 315 vision tensors `rel_fro = 0.000` exactly; `vit_only` → all 397 LLM tensors
`0.000` (Part II.7 explains why exactly 0).

**Research tie.** First evidence for *re-access, not re-representation*: the model isn't rebuilt; a small,
MLP-biased adjustment suffices.

## IV.B — Freeze ablation: *which component carries the fix?* (the 3 conditions, H)

**Exact computation.** Two measurements per condition: (1) the **training-reward accuracy** (Part II.2,
`accuracy_reward` averaged over rollouts), read from the run log; (2) the **direct DOCCI probe** (Part IV.D);
(3) the **weight-level freeze proof** (Part IV.A `rel_fro = 0` for the frozen part).

**The numbers (all verified):**
| condition | trains | train-reward acc | direct probe | freeze proof |
|---|---|---|---|---|
| base | — | 0.365 | 0.377 | — |
| full | ViT+LLM | 0.746 | 0.657 | both > 0 |
| llm_only | LLM | 0.749 | 0.593 | vision = 0.000 |
| vit_only | ViT | 0.443 | 0.423 | **llm = 0.000** |

**Interpret.** `llm_only ≈ full ≫ vit_only > base`. LLM-only recovers ~100%; ViT-only ~16–20%
((0.423−0.377)/(0.657−0.377)=16%; reward 20%). The fix is **overwhelmingly LLM-internal.** *Honest:* the ViT is
**not** zero (vit_only > base) — the encoder can help a little — but the LLM dominates.

**Research tie.** Tells the tool-call *where to act*: the LLM's readout, **not** re-encoding the image — which
also *redirects* the original "re-inspect in a different representation" framing toward "help the LLM re-read."

## IV.C — *(reserved — folded into B; the freeze conditions ARE the ablation)*

## IV.D — Depth-probe / logit-lens: *where does the answer become readable?* (`depth_probe.py`, S4/S5)

**The probe mechanism first (also the perception metric reused everywhere — `mc_eval.run_mc_probe`):**
1. Build the prompt: the DOCCI problem (native "respond with the letter") via `apply_chat_template(…,
   add_generation_prompt=True)` — **note: we do *not* append the `<think>/\boxed` wrapper**, so this is a
   *direct, no-reasoning* readout (differs from training — Part VII Q6).
2. One forward → `logits = out.logits[0, -1, :]` (the next-token distribution at the last prompt position).
3. Pre-compute letter token-ids (`A→32,…` verified by decoding back). Restrict `logits` to the present options'
   letter-ids, `argmax` → predicted letter; compare to gold (`chr(65+choiceAns)`). Deterministic.

**The logit-lens, exact (`depth_probe.py`):** one forward with `output_hidden_states=True` → `hs` = tuple of 37
(`hs[0]`=embeddings, `hs[L]`=output of decoder layer L). For each L: take the answer-position vector
`h = hs[L][0,-1,:]`; apply the model's **own** head `logits_L = lm_head(final_norm(h))`
(`final_norm = model.model.language_model.norm`; `lm_head = model.get_output_embeddings()`); restrict to letters;
record `P(correct)` (softmax over option logits) and argmax-accuracy.

**What the per-layer number means.** "If the model were forced to answer using its layer-L representation of the
last token, how often is it right?" **Results:** final layer = 0.377 (base)/0.657 (trained) — *reproduces*
`mc_eval` exactly (lens calibrated). Layers 0–23 **identical** base vs trained; sharp divergence at **L24–25**;
trained sustains ~0.62–0.66. (Sub-chance dip at L19–23 is a logit-lens artifact — the head is tuned for the top
layer — and is identical in both models, so it's *not* signal; Part VII Q8.)

**Interpret.** RL leaves early/mid representations untouched and lifts the **late** readout. This reconciled
weight-delta: change is uniform in *magnitude* but late-specific in *effect*.

**Research tie.** Localizes the "better representation" to ≈layer 24 — the target depth for any re-injection.

## IV.E — Module-graft: *which weights CAUSE it?* (`module_graft.py`, S3)

**Exact computation.** Build a counterfactual model: `W_grafted[k] = W_ckpt[k]` if `k` is in a mask, else
`W_base[k]`; run the probe (IV.D). Mechanism: load base once; `base_snapshot = {k: state_dict()[k].clone()}`; per
mode `load_state_dict(base_snapshot)` then `load_state_dict({masked k: ckpt[k]})` (`module_graft.in_mask` selects
masked keys using the **same** classifier as weight-delta). Modes: `mlp`/`attn` (LLM only), `late_mlp`
(layer≥24)/`early_mlp` (<12), `full`/`base`.

**The numbers:**
| graft | accuracy | recovers % of +0.28 |
|---|---|---|
| base | 0.377 | 0% |
| full | 0.657 | 100% |
| mlp | 0.553 | 63% |
| attn | 0.460 | 30% |
| early_mlp | 0.430 | 19% |
| late_mlp | 0.387 | 3.6% |

**What each number means.** `mlp 0.553` = "*if only the 108 LLM MLP matrices had been trained*, accuracy." It's
**causal** because we *built* the model and measured it (an intervention).

**Interpret.** MLP-dominant (63% vs 30% attn → S3 supported; *attn not zero*). **Distributed, not late-localized:**
`late_mlp` 3.6% ≪ `early_mlp` 19%, and full-mlp 63% ≫ sum of thirds (synergy). **This corrected our own
prediction** (we expected late_mlp to dominate). Replicated in Cond 2 (per-graft accuracies near-identical, ViT
frozen).

**Research tie + the reconciliation (defend this):** `late_mlp` weights alone fail (3.6%) but the late
*representation* works (patch@L24 = 82%, IV.F) → **RL's distributed MLP edits across L0–24 build the answer into
the residual stream; the late layers don't need new weights, they read what's now there.**

## IV.F — Activation-patch: *is the better representation RE-USABLE?* (`activation_patch.py`, capstone)

**Exact computation (forward hooks).** "Residual at layer L" = output of decoder layer L at the **last token**. A
PyTorch forward hook on `model.model.language_model.layers[L]` reads/writes `out[0][:, -1, :]` (tuple-safe).
- **capture**: store `out[0][:,-1,:]` (no change). **patch**: `out[0][:,-1,:] = trained_vec` (return modified).
  **steer**: `out[0][:,-1,:] += α·v_L`. Under `torch.no_grad()`.
- **Phased:** (1) trained model → cache its residual at every layer + trained acc; (2) base model → cache + base
  acc + steering vectors `v_L = mean_items(r^trained_L − r^base_L)`; (3) **sanity** self-patch base→base must
  reproduce base acc; (4) per-item **patch** sweep; (5) **steer** sweep.

**The numbers** *(sanity `self-patch@L24 = 0.3767 = base` exactly → hooks correct)*:
| patch at layer | accuracy | recovered |
|---|---|---|
| 8 / 12 | 0.377 | 0% |
| 16 | 0.380 | 1% |
| 20 | 0.407 | 11% |
| **24** | **0.607** | **82%** |
| 28 / 32 / 35 | 0.637 / 0.653 / 0.657 | 93 / 99 / 100% |

Steering (fixed mean direction): best `L35, α=4 → 0.487 (39%)`; ~40% ceiling.

**What the numbers mean.** `patch@L24 = 0.607`: "*take the trained model's last-token vector at layer 24, drop it
into the base model at layer 24, let base's own (untrained) layers 25–35 finish* — base now answers correctly 61%
of the time (82% of the gain)." `steer ≈ 40%`: a single fixed direction (averaged across items) recovers only
part.

**Interpret.** The improvement **is a portable representation**, carried by L24, readable by base's untrained
late layers — but it is **input-specific** (per-item patch ~100% vs fixed vector ~40%).

**Research tie (the payoff).** (1) The better representation is **extractable + re-injectable** → tool-call
substrate confirmed, localized to ≈L24. (2) It must be **re-derived per image** (static vector caps at ~40%) →
**the tool-call must *re-inspect*, not apply a canned vector.** This is the cleanest empirical case for an
on-demand, mid-stack, image-derived re-inspection mechanism.

---

# PART V — THE SYNTHESIS (one story, five angles)

| angle | method (script) | exact result | role |
|---|---|---|---|
| where weights moved | Frobenius rel-change (`weight_delta`) | 0.05%, MLP-biased, uniform-magnitude | small surgical edit |
| which component | 3-cond ablation + freeze proof | llm_only=full≫vit_only; frozen Δ=0 | it's the LLM |
| where it shows | logit-lens (`depth_probe`) | identical 0–23, diverge at L24 | manifests late |
| which weights cause it | counterfactual graft (`module_graft`) | MLP 63% > attn 30%; distributed | MLP, depth-spread |
| is the rep re-usable | residual patch/steer (`activation_patch`) | patch@L24 82%; steer ≤40% | portable, input-specific |

**Mechanism (causally established):** *RL's tiny distributed MLP edits across layers ~0–24 write the answer into
the residual stream; by L24 it is present and readable, and the unchanged late layers read it.* (Proof of the
"write into the stream" claim: late *weights* transplanted fail (3.6%), late *representation* transplanted works
(82%).)

**For the tool-call:** portable (re-inject@L24 → ~full) but input-specific (fixed vector ≤40%) → **on-demand,
image-derived re-inspection**, localized to ≈L24.

---

# PART VI — IMPLEMENTATION & SYSTEMS DETAILS (memory, parameters, the things they'll probe)

- **Hardware:** 1 node = 4× GH200 (96 GB GPU each), aarch64/ARM. Paper used 8× H200 (141 GB). → our
  result-neutral deviations: 4 GPUs (sharding only; global/rollout batch held → same optimization),
  micro-batch 4/8 vs paper 16/32 (gradient-accumulation chunking; **identical gradient**, only memory differs;
  the paper's 16 OOMs at response=2048 on 96 GB).
- **A vision-forward speed fix (result-neutral):** Qwen3-VL's patch-embed is a kernel==stride `Conv3d`; on
  aarch64/cuDNN it was ~3.4e5× slower than the equivalent matmul. We monkey-patched it to a `matmul`
  (`maxdiff = 0.0`, bit-identical). Applied in training *and* in `ckpt_model` for the analyses. *(Pure speed;
  zero effect on any number.)*
- **Precision:** bf16 weights (`adamw_bf16`). The analyses cast to fp32 for the Frobenius/cosine math.
- **Checkpoint analysis memory:** weight-delta holds the 4 shards (~9 GB) + one rebuilt tensor at a time (CPU,
  no GPU). Graft/activation-patch hold one 4B model (~9 GB) on GPU + a CPU residual cache (~28 M floats). Node
  RAM 450 GB → comfortable.
- **Probe cost:** 300 items × 1 forward per condition; depth-probe adds 37 cheap head applications per item;
  activation-patch ≈ 10k forwards (~40–60 min). All 1-GPU jobs; weight-delta is CPU-only.
- **Determinism:** `seed=1` (identical across the 3 conditions, verified by config diff); the probe/graft/patch
  use greedy/argmax readout → deterministic.
- **Python gotcha (real):** the cluster *login node* is Python 3.6 (no `dataclasses`, no `list[…]` subscripts);
  the *container* is 3.12. The loaders use `namedtuple` + bare annotations so they run on both.

---

# PART VII — GRILLING PREP (anticipated questions, exact answers)

**Q1. Why GRPO instead of PPO?** No value/critic network — the **group mean of n=5 rollouts is the baseline**
(`core_algos.py:209-213`). Cheaper (no critic to train/store) and well-suited to verifiable-reward settings.
Trade-off: the baseline is noisier (n=5 small) and you can get dead groups (all-correct or all-wrong → zero
advantage).

**Q2. The std-normalization in the advantage — isn't that controversial?** Yes. `A = (r − mean)/(std + 1e-6)`
(`core_algos.py:213`). Dividing by group std up-weights *low-variance (easy/consistent) groups* and is argued to
bias optimization; some variants (Dr.GRPO) drop it. We kept the EasyR1/paper default. *Know this is a known
debate.*

**Q3. Is the KL on the reward or the loss?** **Loss-side** (`use_kl_loss=true`): `loss = pg_loss + kl_coef·kl`
(`dp_actor.py:281`), `kl_coef=1e-2`. Not added to the reward. `π_ref` is the frozen base policy.

**Q4. What KL estimator, and why?** The **k3 / low-variance** estimator `exp(x) − x − 1`, `x = ref−logp`
(`core_algos.py:592-593`). It's always ≥0 (a proper divergence), **unbiased**, and **lower-variance** than the
naive `logp−ref` (Schulman's blog). Clamped for numerical stability.

**Q5. Why is the weight change so tiny (0.05%) yet behavior changes a lot?** Two reasons it's *small by design*:
`lr=1e-6` and the KL leash to base. And it changes behavior because the edit is *targeted* — it reorganizes the
late-layer readout (IV.D/F), not random weights. The tininess is the point: re-access, not rebuild.

**Q6. (THE one to nail) Your probe reads a *direct* letter with no reasoning, but training used `<think>` +
`\boxed{}`. Are you measuring the same thing?** **No — different protocols, stated honestly.** Training optimized
the boxed answer *after* reasoning (`math.jinja`, `accuracy_reward`); the probe reads the **immediate next-token
letter** (`out.logits[0,-1]` restricted to letters, no reasoning). They are different measurements. Why it's
still valid: (a) the probe is a *consistent* yardstick applied identically to base and every checkpoint, so the
*difference* (0.377→0.657) is meaningful; (b) base-probe 0.377 ≈ base training-reward-accuracy 0.365, so the
direct readout tracks the trained competence; (c) crucially, the probe shows the perception gain appears **even
without reasoning** — evidence the improvement is in *direct perception*, not just a reasoning skill. We disclose
it as a direct-perception (System-1) probe, deliberately.

**Q7. Contamination — you probe on the training distribution.** Yes — trained-model probe numbers are
*train-accuracy*. Fine for **localization** (we dissect *where* the learned competence lives; base numbers are
uncontaminated). We do **not** claim generalization — and the babyVision miss (base 32.6 ≈ trained 33.3) shows
generalization is in fact limited (itself a finding).

**Q8. Is the logit-lens valid? The mid-layers dip below chance.** The lens applies the *final* head to
intermediate states, which is approximate — mid-layer states aren't in the output basis (hence the L19–23 dip).
But: the dip is *identical* in base and trained (so it carries no base-vs-trained signal), and the final-layer
readout *exactly* matches the real probe (calibration check). We rely on the late layers and on differences. A
*tuned lens* (learned per-layer affine) would clean the middle.

**Q9. Graft non-additivity — mlp(63%)+attn(30%)≠100%, thirds don't sum.** Expected in nets. Grafts test
**sufficiency** of a subset, not a clean decomposition; the strong synergy (full-mlp ≫ Σ thirds) is a real
finding, not an error.

**Q10. Why does `late_mlp` graft fail (3.6%) but `patch@L24` succeed (82%)?** Because they transplant different
things. `late_mlp` graft gives base the trained *late weights* but feeds them base's *own* (unimproved) residual
→ nothing to read → 3.6%. `patch@L24` gives base the trained *late representation* → base's untrained late layers
read it → 82%. **The fix is in the residual the early/mid MLPs build, not in the late weights.** This is the core
mechanistic claim.

**Q11. Steering vector only ~40% — is that a failure?** No — it's informative. A single averaged direction
captures the *shared* component (~40%); the rest is **input-specific** (the answer differs per item). → a static
fix is insufficient; a tool-call must compute the representation *from each image*. This *motivates* the
re-inspection design.

**Q12. How do you know the freeze actually worked?** Three independent ways: (1) the log message fires only when
the freeze code runs; (2) config diff shows only the freeze flag differs; (3) **weight-level**: the frozen
component's `rel_fro = 0.000` (mean *and* max, all tensors) — bit-identical to base, because the optimizer's
`filter(p.requires_grad)` never touches it (`fsdp_workers.py:343`).

**Q13. Sample size / significance?** 300 DOCCI items → ±~3% (≈ ±1 item per 0.3%). Treat sub-3% gaps as noise
(e.g. full 0.657 vs llm_only 0.593 is partly real, partly noise; on the *training reward* they're equal).

**Q14. Single seed / 4B / Stage-1 — is this robust?** It's a **mechanism study**, not a 1:1 paper repro. We
strengthen it by: replicating the graft+depth under the ViT-frozen condition; multiple consistent angles; exact
freeze proofs. Stated as scope, not hidden.

**Q15. FSDP reconstruction — how do you know it's correct?** It mirrors the official `model_merger.py`
(`cat` shards by `placement.dim`); and the activation-patch **self-patch sanity** (base→base reproduces base
accuracy exactly) independently confirms the model we load behaves identically to a normally-loaded model.

**Q16. The tied lm_head / 713 vs 714 keys?** `tie_word_embeddings=true` → input embedding *is* the output
projection. Base stores it once (713); a checkpoint re-saves an untied copy (714); it's bit-identical, so we
ignore the duplicate in weight-delta.

---

# PART VIII — HONEST CAVEATS (say these; they build credibility)
- **Probe ≠ training protocol** (direct letter vs reasoning+boxed) — Q6.
- **Contamination** (train-distribution probe) — Q7; generalization limited (babyVision OOD).
- **Logit-lens mid-layer approximation** — Q8.
- **Graft sufficiency, not decomposition; non-additivity** — Q9.
- **Clip ratios 0.2/0.3/3.0 are engine defaults**, not paper-set (flag for fidelity-minded reviewers).
- **Single 4B / seed 1 / Stage-1-only mechanism study** — Q14.
- **Residual-direction mechanism (IV.E reconciliation)** is the best-supported reading (patch vs graft), not a
  formal proof; the activation-patch is the most direct test we have.
- **What we do NOT yet claim:** the *temporal* "perception decays as reasoning proceeds" axis (needs Stage-3 or
  long-reasoning probes), and a *working tool-call prototype* (next experiment, now de-risked: a module that
  produces an L24-style representation from the image on demand).

---

## Appendix — exact result provenance (so you can re-derive any number live)
| number | produced by |
|---|---|
| training reward 0.365→0.746 | the run logs (`accuracy_reward`, Part II.2) |
| weight-delta 0.05%, mlp/attn ratios, freeze=0 | `weight_delta.py` → `deltas.csv` → `summarize_deltas.py` |
| probe base 0.377 / trained 0.657 / cond accuracies | `mc_eval.py` (DOCCI 300) |
| depth curve (L24 divergence) | `depth_probe.py` → `depth_*.csv` |
| graft 63/30/19/3.6% | `module_graft.py` → `graft_full_96.csv` |
| patch@L24 82%, steer ≤40%, sanity 0.377 | `activation_patch.py` → `actpatch_full_96.csv` |
