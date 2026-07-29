# PAPO — The Three Runs, Explained From Scratch (Master Reference)

**Goal of this document.** Take a reader who knows only *basic* ML (what a neural net, a gradient, and
a probability distribution are) and bring them to a **master-level** understanding of the three
training runs — Arm A (GRPO baseline), Arm B ("C+DE"), Arm C ("C-pure") — such that they can reason
about, extend, and draw their own conclusions from them. **Nothing is assumed that isn't stated here.**

**Provenance.** Every quantitative/behavioral claim was read first-hand from the code in
`PAPO_clone/PAPO` (branch `main_qwen3 @ 1263a29`) on 2026-07-28. Citations are `file:line`. Where a
value comes from a config or run-script, that is stated. If you see a claim without a citation, it is a
plain-English restatement of a cited one.

> Companion docs: [`PAPO_MASTER_RECORD.md`](PAPO_MASTER_RECORD.md) (audit, results, provenance of bug B),
> [`README_FIX.md`](README_FIX.md) (Arm B fixes), [`../PAPO_cpure/README_CPURE.md`](../PAPO_cpure/README_CPURE.md) (Arm C).

---

# PART 0 — The one-paragraph summary

All three runs train **the same model** (`Qwen3-VL-2B-Thinking`) on **the same data** (ViRL39K math+vision
problems) with **the same RL algorithm skeleton** (GRPO) and **identical hyperparameters**, on 4 GPUs, for
60 steps. They differ in **exactly one source file** — `verl/workers/actor/dp_actor.py`, which defines the
**loss function** — plus a couple of run-script switches. That single difference determines **which loss
terms actually receive a gradient**, and therefore what the model learns:
- **Arm A** effectively trains **GRPO + reference-KL** only (the perception terms are computed but a bug
  discards their gradient).
- **Arm B** trains **GRPO + reference-KL + Implicit Perception Loss + Double Entropy** (the full paper Eq. 2,
  with an extra masked forward pass).
- **Arm C** trains **GRPO + reference-KL + Implicit Perception Loss** (no double entropy) — the paper's
  *canonical* PAPO_G-2B recipe.

---

# PART 1 — Concepts from scratch (the vocabulary you need)

- **Policy `π_θ`.** The model itself, viewed as a function that, given a prompt (text + image), assigns a
  probability to every possible next token. `θ` = all trainable weights. "Training the policy" = changing
  `θ` so that whole answers we like become more probable. **There is only ONE policy** in these runs; when
  we say "the masked model `π_θ^mask`" we mean *the same weights* `θ` fed a corrupted image — not a second
  network.
- **Rollout / generation.** Sampling actual answers from `π_θ` for a batch of prompts. Each answer is a
  sequence of tokens `o = (o_1,…,o_T)`.
- **Log-prob `log π_θ(o | q, I)`.** For a given answer `o`, prompt `q`, image `I`, the sum/vector of the
  log-probabilities the model assigns to each token it produced. Computed by one forward pass that
  "teacher-forces" the tokens and reads off `log_softmax` at the realized token ids
  ([`dp_actor.py:151-161`](verl/workers/actor/dp_actor.py#L151)).
- **Reward `R`.** A number saying how good an answer is. Here it is **rule-based** (no reward model): did the
  final boxed answer match the ground truth, and was the format right (§2.3.2).
- **RLVR** (Reinforcement Learning with Verifiable Rewards). RL where the reward is a *verifiable* rule
  (exact-match grading), not a learned model. GRPO/PAPO are RLVR algorithms.
- **GRPO** (Group-Relative Policy Optimization). A simplification of PPO for RLVR: instead of a learned
  "critic" network to estimate how good a state is, it samples **several answers per prompt** and uses the
  **group's mean reward as the baseline**. An answer better than its group's average gets a positive
  "advantage"; worse gets negative. No value network (§2.3.3).
- **PPO clip.** The mechanism that stops each update from moving the policy too far: it clips the
  probability ratio between the new and old policy (§2.4.1).
- **KL divergence `D_KL[P‖Q]`.** A non-negative measure of how different two distributions are (0 iff
  identical). Used two ways here: (a) a *penalty* keeping `π_θ` near a frozen reference (regularization),
  (b) a *reward* — PAPO **maximizes** the KL between the model's outputs on the real vs. masked image
  (§2.4).
- **FSDP** (Fully-Sharded Data Parallel). A way to split one big model's weights/gradients/optimizer state
  across the 4 GPUs so it fits and trains in parallel (§2.5).
- **vLLM.** A fast inference engine used only for the *rollout* (generation) phase (§2.3.1).

---

# PART 2 — The SHARED machinery (identical in all three runs)

Everything in Part 2 is **byte-identical across Arm A/B/C** (same config, same run-script values, same code
except `dp_actor.py`). Only Part 3 differs.

## 2.1 What model trains, and which parts of it

- **Model:** `Qwen/Qwen3-VL-2B-Thinking` — a ~2B-parameter vision-language model that natively emits a
  chain-of-thought inside `<think>…</think>` then a final answer. Run scripts set
  `worker.actor.model.model_path=$SCRATCH/models/Qwen3-VL-2B-Thinking`.
- **A single policy is trained.** The same weights are used for: generating rollouts, scoring real-image
  log-probs, and scoring masked-image log-probs.
- **The vision tower IS trained** (not frozen): `freeze_vision_tower: false`
  ([config_grpo_papo.yaml:68](examples/configs/config_grpo_papo.yaml#L68), default also `False` at
  [actor/config.py:30](verl/workers/actor/config.py#L30)). So gradients update the image encoder too — this
  matters: PAPO's perception pressure can reshape how the model *sees*, not just how it reasons.
- **A frozen reference policy `π_ref`** is also held in memory (a copy of the initial weights) — used only to
  compute the reference-KL penalty; it never trains. It exists because `disable_kl=false`
  ([ray_trainer.py:272-277](verl/trainer/ray_trainer.py#L272)) → `use_reference_policy=True`.
- **No critic/value network.** Because the advantage estimator is GRPO, not GAE:
  `use_critic=False` ([ray_trainer.py:280-283](verl/trainer/ray_trainer.py#L280)).
- **Precision & memory:** bf16 mixed precision (`mp_param_dtype: bf16`,
  [actor/config.py:65](verl/workers/actor/config.py#L65)); gradient checkpointing on
  (`enable_gradient_checkpointing: true`, [config:66](examples/configs/config_grpo_papo.yaml#L66));
  parameters **and** optimizer state offloaded to CPU between phases (`offload_params: true`,
  `offload_optimizer: true`, [config:80-81](examples/configs/config_grpo_papo.yaml#L80)).

## 2.2 The data, and every constraint that shapes it

- **Training set:** `PAPOGalaxy/PAPO_ViRL39K_train` (~39K multimodal math/reasoning problems; ~**38,870**
  after filtering). **Validation set:** `PAPOGalaxy/PAPO_MMK12_test` (2,000 problems) — used only to
  *monitor*, never to pick checkpoints.
- **Columns:** `problem` (prompt text), `answer` (ground truth), `images`
  ([config:4-6](examples/configs/config_grpo_papo.yaml#L4)).
- **Prompt construction:** each example's text is wrapped by the Jinja template
  [`math_perception.jinja`](examples/format_prompt/math_perception.jinja): the problem, then *"You first
  think through the reasoning process as an internal monologue, enclosed within `<think> </think>` tags.
  Then, provide your final answer enclosed within `\boxed{}`."* This is fed through the model's chat
  template ([dataset.py:301](verl/utils/dataset.py#L301)).
- **Constraint 1 — prompt length filter.** `filter_overlong_prompts: true`
  ([config:21](examples/configs/config_grpo_papo.yaml#L21)); any example whose tokenized prompt (text +
  encoded image) exceeds **`max_prompt_length = 4096`** tokens is **dropped from the dataset** before
  training ([dataset.py:154-159](verl/utils/dataset.py#L154), [dataset.py:298-311](verl/utils/dataset.py#L298)).
  This is why ViRL39K → ~38,870.
- **Constraint 2 — image resizing.** Every image is resized so its pixel count lies in
  `[min_pixels = 200704 (=256·28²), max_pixels = 1003520 (=1280·28²)]`
  ([config:19-20](examples/configs/config_grpo_papo.yaml#L19), [dataset.py:57-74](verl/utils/dataset.py#L57)).
  This bounds the number of vision tokens per image (roughly 256–1280), which bounds prompt length and memory.
- **Constraint 3 — response length cap.** Generation stops at **`max_response_length = 8192`** tokens
  (run-script override; the config default is 2048). Answers that would be longer are truncated (they then
  lack a closing `\boxed{}` → format+accuracy reward 0). Observed truncation ≈ 33% of training rollouts.
- **Constraint 4 — deterministic order.** `shuffle: true, seed: 1`
  ([config:17-18](examples/configs/config_grpo_papo.yaml#L17)). **All three arms see the same prompts in the
  same order**, which makes the arms directly comparable (differences are due to the loss, not the data).

## 2.3 One training step, end-to-end (the `fit()` loop)

The driver loops `while global_step < training_steps` ([ray_trainer.py:722](verl/trainer/ray_trainer.py#L722)).
`training_steps = max_steps = 60` (because `max_steps` is set, the epoch count is ignored;
[ray_trainer.py:315-316](verl/trainer/ray_trainer.py#L315)). Each step does the following, in order
([ray_trainer.py:722-851](verl/trainer/ray_trainer.py#L722)):

### 2.3.1 Rollout — generate answers (phase "gen")
- The rollout engine (vLLM) is woken ([ray_trainer.py:729](verl/trainer/ray_trainer.py#L729)); a batch of
  **`rollout_batch_size = 384` prompts** is drawn ([config→run script](examples/configs/config_grpo_papo.yaml#L12)).
- For each prompt, **`n = 5`** answers are sampled ([config:84](examples/configs/config_grpo_papo.yaml#L84))
  with **temperature = 1.0, top_p = 0.99, top_k = −1** (unbounded), seed 1
  ([config:85-86](examples/configs/config_grpo_papo.yaml#L85); rollout defaults
  [rollout/config.py:25-29](verl/workers/rollout/config.py#L25)). → **384 × 5 = 1920 answers per step**.
- **If** `use_kl_prcp` is on (it is), a **masked copy of each image** is created *now*, at rollout time, by
  `random_patch_blackening` ([ray_trainer.py:608-620](verl/trainer/ray_trainer.py#L608),
  [ray_trainer.py:550-561](verl/trainer/ray_trainer.py#L550)): the image is tiled into 14×14-pixel patches and
  each patch is set to black independently with probability 0.6 (§2.4.3). Stored as `aug_multi_modal_data`.
- vLLM is put back to sleep to free GPU memory for the training forward/backward
  ([ray_trainer.py:731](verl/trainer/ray_trainer.py#L731)).

### 2.3.2 Reward — grade every answer (phase "reward")
- The **`BatchFunctionRewardManager`** ([reward/function.py:107-135](verl/workers/reward/function.py#L107))
  decodes each answer to text (`skip_special_tokens=True`, [reward/config.py:28](verl/workers/reward/config.py#L28);
  the `</think>` tag is a *normal* vocab token so it survives, which is why the format check can fire) and
  calls `compute_score` ([qwen3_vl_think.py:33-50](examples/reward_function/qwen3_vl_think.py#L33)):
  - **format_reward** = 1.0 if the text matches the regex `.*</think>.*\boxed\{.*\}.*` (DOTALL), else 0.0
    ([qwen3_vl_think.py:21-25](examples/reward_function/qwen3_vl_think.py#L21)).
  - **accuracy_reward** = 1.0 if `grade_answer(extract_boxed_content(text), ground_truth)` is true, else 0.0
    ([qwen3_vl_think.py:28-30](examples/reward_function/qwen3_vl_think.py#L28)).
  - **overall = 0.9·accuracy + 0.1·format** (`format_weight = 0.1`,
    [qwen3_vl_think.py:44](examples/reward_function/qwen3_vl_think.py#L44)).
- The scalar `overall` is written to the reward tensor at the **last token** of each answer
  ([reward/function.py:131](verl/workers/reward/function.py#L131)); all other positions are 0.

### 2.3.3 Advantage — GRPO group normalization (phase "adv")
- Because `use_kl_loss = true`, the reference-KL is **NOT** subtracted from the reward here
  ([ray_trainer.py:798-803](verl/trainer/ray_trainer.py#L798)); instead `token_level_rewards =
  token_level_scores` (raw reward). (The ref-KL enters later, inside the loss — §2.4.2.)
- `compute_grpo_outcome_advantage` ([core_algos.py:187-229](verl/trainer/core_algos.py#L187)):
  1. Per answer, sum the token rewards → a scalar `R_i` (= the last-token `overall`).
  2. Group answers by their prompt (the 5 rollouts of one prompt form a group; they share a `uid`,
     [ray_trainer.py:646-650](verl/trainer/ray_trainer.py#L646)).
  3. Compute the group's mean and std; set **`A_i = (R_i − mean_group) / (std_group + 1e-6)`**
     ([core_algos.py:226](verl/trainer/core_algos.py#L226)).
  4. Broadcast `A_i` to **every token** of that answer ([core_algos.py:228](verl/trainer/core_algos.py#L228)).
  → The "advantage" is a single number per answer: how much better than its prompt-mates. (GRPO **requires
  `n>1`**; asserted at [core_algos.py:221](verl/trainer/core_algos.py#L221) and
  [ray_trainer.py:309-313](verl/trainer/ray_trainer.py#L309).)

### 2.3.4 The three log-prob passes (before the update)
All three teacher-force the **same** sampled answer tokens and read off per-token log-probs
([dp_actor.py:151-161](verl/workers/actor/dp_actor.py#L151)); they differ in *which weights/image*:
- **`old_log_probs`** = `log π_θ(o | q, I_real)` at the step's starting weights, no grad
  ([ray_trainer.py:766-768](verl/trainer/ray_trainer.py#L766)). Used as the PPO "old policy" baseline.
- **`aug_log_probs`** = `log π_θ(o | q, I_masked)` at the step's starting weights, no grad, computed **only
  if `use_kl_prcp`** ([ray_trainer.py:771-775](verl/trainer/ray_trainer.py#L771),
  [fsdp_workers.py:682-700](verl/workers/fsdp_workers.py#L682)). This is the masked-image scoring used by the
  perception loss. (It is *detached* — a fixed snapshot — unless `RECOMPUTE=True`; see §3.)
- **`ref_log_probs`** = `log π_ref(o | q, I_real)` from the frozen reference, no grad
  ([ray_trainer.py:778-781](verl/trainer/ray_trainer.py#L778)). Used by the reference-KL penalty.

### 2.3.5 The actor update (phase "update_actor") — where learning happens
`update_actor` calls `DataParallelPPOActor.update_policy`
([dp_actor.py:246-446](verl/workers/actor/dp_actor.py#L246)). This is the loop that computes the loss and
calls `.backward()`. **This is the ONLY place the three arms differ** (Part 3). The batching:
- The 1920 answers are split into **mini-batches of `global_batch_size = 128`**
  ([config:57](examples/configs/config_grpo_papo.yaml#L57)) → 15 optimizer updates per step.
- `ppo_epochs = 1` ([actor/config.py:97](verl/workers/actor/config.py#L97)): each rollout batch is used for
  exactly one pass (so `old_log_probs == current log_probs` on the first and only pass; the PPO ratio starts
  at 1).
- Each mini-batch is further split into **micro-batches of
  `micro_batch_size_per_device_for_update = 1`** (run-script) for gradient accumulation — one answer per
  forward/backward, summed, to keep memory bounded at 8192-token responses.
- After each mini-batch: gradient clipping to **`max_grad_norm = 1.0`**
  ([actor/config.py:85](verl/workers/actor/config.py#L85)) and one optimizer step
  ([dp_actor.py:445](verl/workers/actor/dp_actor.py#L445), [fsdp_workers.py:789](verl/workers/fsdp_workers.py#L789)).

### 2.3.6 Save / validate
- **Checkpoint** every `save_freq = 10` steps, keeping all (`save_limit = -1`) with optimizer state
  (`save_model_only = false`) so resume is exact ([ray_trainer.py:840-842](verl/trainer/ray_trainer.py#L840)).
- **Validation:** `val_freq = -1` disables mid-training validation; but a **base validation runs before
  training** (`val_before_train = true`, [ray_trainer.py:714-717](verl/trainer/ray_trainer.py#L714)) and a
  **final validation always runs after the loop** ([ray_trainer.py:853-863](verl/trainer/ray_trainer.py#L853)).
  Validation samples **n = 8** answers per prompt at **temperature 1.0**
  ([config:93-95](examples/configs/config_grpo_papo.yaml#L93)) and reports `val/reward_score` = the mean
  **overall** reward, plus a breakdown `val/accuracy_reward`, `val/format_reward`
  ([ray_trainer.py:526-531](verl/trainer/ray_trainer.py#L526)). **Note:** the "0.540 / 0.466 / …" numbers are
  `reward_score` (overall = 0.9·acc+0.1·format); `accuracy_reward` is the clean accuracy.

## 2.4 The loss building blocks (the math, precisely)

The loss is assembled per micro-batch inside `update_policy`. Below are the components; Part 3 says which are
*backpropagated* in each arm. All per-token quantities are averaged with **`loss_avg_mode = "token"`** =
global masked mean over all response tokens in the micro-batch
([average_loss, core_algos.py:444-468](verl/trainer/core_algos.py#L444)); `mask` = the response-token mask
(prompt tokens and padding excluded).

### 2.4.1 PPO clipped policy-gradient loss (`pg_loss`) — the RL objective
([compute_policy_loss, core_algos.py:471-545](verl/trainer/core_algos.py#L471))
- ratio `r = exp(log π_θ − log π_θ_old)` (=1 on the single ppo epoch, then moves as θ updates within the
  mini-batch loop), clamped for stability.
- `L_pg = − mean_t[ min( r_t·A , clip(r_t, 1−0.2, 1+0.3)·A ) ]`, with a **dual clip** at `3.0·A` when `A<0`
  ([core_algos.py:533-543](verl/trainer/core_algos.py#L533)). Clip ranges: `clip_ratio_low=0.2`,
  `clip_ratio_high=0.3`, `clip_ratio_dual=3.0` ([actor/config.py:87-92](verl/workers/actor/config.py#L87)).
- Intuition: push up the probability of above-average answers, down for below-average, but never by more
  than the clip in one step (stability).

### 2.4.2 Reference-KL penalty (`kl_loss`) — stay near the start
- `kl_loss = mean_t D_KL[π_θ ‖ π_ref]` using the **`low_var_kl`** estimator (§2.4.5), coefficient
  `kl_coef = 0.01` ([config:26-28](examples/configs/config_grpo_papo.yaml#L26)). Added **with a + sign** so
  it is **minimized** (keeps the trained policy from drifting too far from the base model). This is the
  β·D_KL[π_θ‖π_ref] term of paper Eq. 2.

### 2.4.3 Implicit Perception Loss (`kl_prcp_loss`) — PAPO's core idea
- Masked image via `random_patch_blackening(patch_size=14, black_prob=0.6)`
  ([papo_utils.py:17-30](verl/trainer/papo_utils.py#L17)): tile into 14×14 patches, blacken each with prob 0.6.
- `kl_prcp_loss = mean_t D_KL[ π_θ(·|I_real) ‖ π_θ(·|I_masked) ]` on the **same** sampled tokens, using
  `low_var_kl`, coefficient `kl_prcp_coef = 0.01`, uniform per-sample weight 1.0
  (`kl_prcp_apply_mode="all"`, [ray_trainer.py:565-570](verl/trainer/ray_trainer.py#L565)), **fixed**
  schedule (no annealing, [ray_trainer.py:324-326](verl/trainer/ray_trainer.py#L324)).
- Added **with a − sign** (`pg_loss = pg_loss − kl_prcp_loss·coef`, [dp_actor.py:399/412/415]) so it is
  **MAXIMIZED**: the model is rewarded for producing *different* outputs when the image is corrupted →
  forcing it to actually rely on the image. This is the +γ·D_KL[π_θ‖π_θ^mask] term of Eq. 2.

### 2.4.4 Double Entropy Loss (`aug_entropy_loss`, `ori_entropy_loss`) — anti-collapse regularizer
- Entropy estimated as the negative mean log-prob: `H = −mean_t log π_θ`
  ([dp_actor.py:316,357](verl/workers/actor/dp_actor.py#L316)); one term on the real branch
  (`ori_entropy_loss`) and one on the masked branch (`aug_entropy_loss`), each coefficient **0.03**
  ([config:50,53](examples/configs/config_grpo_papo.yaml#L50)). Added **with a + sign** so entropy is
  **penalized** (kept low) — this prevents the model from "hacking" the unbounded perception-KL by emitting
  gibberish (high-entropy nonsense that maximizes KL). These are the −η₁H[π_θ] − η₂H[π_θ^mask] terms of Eq. 2.
  **Enabled only in Arm B** (`use_aug/ori_entropy_loss`, off by default at
  [config:49,52](examples/configs/config_grpo_papo.yaml#L49)).

### 2.4.5 The KL estimator (`low_var_kl`, Schulman k3)
([core_algos.py:626-630](verl/trainer/core_algos.py#L626)) For per-token log-probs of the same tokens under
two distributions, with `s = log π_ref − log π_θ`: `D_KL ≈ exp(s) − s − 1` (always ≥ 0, low variance). Both
the reference-KL and the perception-KL use this estimator (`kl_penalty` and `kl_prcp_penalty` = `low_var_kl`,
[config:27,36](examples/configs/config_grpo_papo.yaml#L27)).

## 2.5 Optimizer, schedule, parallelism, memory

- **Optimizer:** `AnyPrecisionAdamW` (because `strategy: adamw_bf16`), betas (0.9, 0.999), weight_decay 0.01,
  **lr = 1e-6** ([fsdp_workers.py:300-312](verl/workers/fsdp_workers.py#L300);
  [config:70-73](examples/configs/config_grpo_papo.yaml#L70)).
- **LR schedule:** `get_constant_schedule_with_warmup` with **0 warmup steps**
  (`lr_warmup_ratio=0.0`, `warmup_style="constant"`, [fsdp_workers.py:318-323](verl/workers/fsdp_workers.py#L318)).
  → **The learning rate is constant the entire run.** This is why stopping at 60 and resuming is *exact*: no
  schedule depends on the total step count, so `max_steps` only changes how many iterations run.
- **Parallelism:** 4 GPUs, FSDP full-shard ([config:76](examples/configs/config_grpo_papo.yaml#L76)),
  `tensor_parallel_size=1` for rollout (run-script), `ulysses_size=1` (no sequence parallelism).
- **Memory tricks:** param+optimizer CPU offload; gradient checkpointing; vLLM sleep/wake between rollout and
  training (frees KV-cache memory for the backward). `gpu_memory_utilization` sets the vLLM KV-cache fraction
  (the one memory knob that differs per arm — see §3).

## 2.6 The complete hyperparameter table (identical across A/B/C unless noted)

| Group | Parameter | Value | Source |
|---|---|---|---|
| Model | base | Qwen3-VL-2B-Thinking | run script |
| Model | vision tower | **trained** (not frozen) | [config:68](examples/configs/config_grpo_papo.yaml#L68) |
| Data | train / val | ViRL39K (~38,870) / MMK12 (2,000) | [config:2-3](examples/configs/config_grpo_papo.yaml#L2) |
| Data | max_prompt_length | 4096 | run script |
| Data | max_response_length | **8192** (paper-unspecified; code default 2048) | run script |
| Data | max/min pixels | 1,003,520 / 200,704 | [config:19-20](examples/configs/config_grpo_papo.yaml#L19) |
| Data | shuffle / seed | true / 1 | [config:17-18](examples/configs/config_grpo_papo.yaml#L17) |
| Rollout | n (train) | 5 | [config:84](examples/configs/config_grpo_papo.yaml#L84) |
| Rollout | temperature / top_p / top_k | 1.0 / 0.99 / −1 | [config:85-86](examples/configs/config_grpo_papo.yaml#L85) |
| Rollout | n (val) | 8 @ temp 1.0 | [config:93-95](examples/configs/config_grpo_papo.yaml#L93) |
| Batch | rollout_batch_size (prompts/step) | 384 | run script |
| Batch | answers/step | 1920 (=384×5) | derived |
| Batch | global_batch_size (mini-batch) | 128 | [config:57](examples/configs/config_grpo_papo.yaml#L57) |
| Batch | micro update / experience | 1 / 4 | run script |
| Optim | optimizer | AnyPrecisionAdamW (bf16), β(0.9,0.999), wd 0.01 | [fsdp_workers.py:308](verl/workers/fsdp_workers.py#L308) |
| Optim | learning rate | 1e-6, **constant, 0 warmup** | [config:70-73](examples/configs/config_grpo_papo.yaml#L70) |
| Optim | max_grad_norm | 1.0 | [actor/config.py:85](verl/workers/actor/config.py#L85) |
| Optim | ppo_epochs | 1 | [actor/config.py:97](verl/workers/actor/config.py#L97) |
| RL | adv estimator | GRPO (no critic) | [config:24](examples/configs/config_grpo_papo.yaml#L24) |
| RL | clip low/high/dual | 0.2 / 0.3 / 3.0 | [actor/config.py:87-92](verl/workers/actor/config.py#L87) |
| RL | loss averaging | token | [actor/config.py:93](verl/workers/actor/config.py#L93) |
| Ref-KL | use / coef / estimator | true / 0.01 / low_var_kl | [config:26-28](examples/configs/config_grpo_papo.yaml#L26) |
| Perception | use / coef / estimator | true / 0.01 / low_var_kl | [config:34-37](examples/configs/config_grpo_papo.yaml#L34) |
| Perception | mask patch / black_prob | 14 / 0.6 | [config:41-42](examples/configs/config_grpo_papo.yaml#L41) |
| Perception | schedule / apply_mode | fixed / all (weight 1.0) | [config:38,43](examples/configs/config_grpo_papo.yaml#L38) |
| Double entropy | coef (each) | 0.03 | [config:50,53](examples/configs/config_grpo_papo.yaml#L50) |
| Schedule | max_steps → training_steps | 60 | run script |
| Schedule | epochs | 2 (**ignored**, max_steps set) | run script |
| Ckpt | save_freq / limit / model_only | 10 / −1 / false | run script |
| Val | val_freq / val_before_train | −1 / true | run script |
| HW | GPUs / tensor_parallel | 4 / 1 | run script |

## 2.7 Batch-shape constraints (the asserts that must hold)
([ray_trainer.py:288-313](verl/trainer/ray_trainer.py#L288)) — `rollout_batch_size % global_batch_size == 0`
(384 % 128 = 0 ✓); `(rollout_batch_size·n) % micro_experience == 0` (1920 % 4 = 0 ✓); GRPO needs `n>1`
(5 ✓). These are why the batch numbers are what they are.

---

# PART 3 — What the three runs actually DIFFER by

## 3.0 The single point of divergence

The three arms use three copies of the repo that are **identical except `verl/workers/actor/dp_actor.py`**,
plus a few run-script switches. Concretely:

| | Code dir | `RECOMPUTE_AUG_LOG_PROBS` | run-script entropy flags | `gpu_memory_utilization` |
|---|---|---|---|---|
| **A** | `PAPO_clone` (released, buggy) | False | `use_aug/ori_entropy_loss=true`* | 0.60 / resume 0.55 |
| **B** | `PAPO_fixed` (B+C fixes) | **True** | `use_aug/ori_entropy_loss=true` | 0.40 / resume 0.38 |
| **C** | `PAPO_cpure` (B+C fixes) | False | *(not set → false)* | 0.55 / resume 0.52 |

*Arm A's script sets the entropy flags true, but they have no effect — see §3.1.

`RECOMPUTE_AUG_LOG_PROBS` is a module-level constant in `dp_actor.py`
([A:47, B:58, C:61](verl/workers/actor/dp_actor.py#L47)). It controls whether the masked-image log-probs used
by the perception/entropy terms are **recomputed with gradients inside the update loop** (True) or reused as
the **detached snapshot** from rollout (False) ([dp_actor.py:318-324](verl/workers/actor/dp_actor.py#L318)).

## 3.1 Arm A — the GRPO + reference-KL baseline (val `reward_score` 0.540)

- **Intent at launch:** full PAPO (config has `use_kl_prcp=true`; script sets both entropy flags true).
- **What actually trains:** **GRPO + reference-KL only.** The released `main_qwen3` code has **bug B**: in
  `update_policy` the variable `loss` is bound at the reference-KL line
  ([dp_actor.py:347](verl/workers/actor/dp_actor.py#L347) `loss = pg_loss + kl_loss*kl_coef`) **before** the
  perception term (399), entropy terms (403/412) and sft (426) are added to a *different* variable `pg_loss`;
  then `.backward()` runs on `loss` ([dp_actor.py:431-432](verl/workers/actor/dp_actor.py#L431)). So every
  PAPO term is **computed and logged but receives zero gradient**. (Full analysis + git-blame provenance in
  [`PAPO_MASTER_RECORD.md`](PAPO_MASTER_RECORD.md) §2.6; it is a `main_qwen3`-only port regression, commit
  961291f2.) The masked forward still runs at rollout, so `kl_prcp` is *logged* — but it **decays** over
  training (nothing maximizes it), the fingerprint of the bug.
- **Effective loss backpropped:** `L = L_pg + 0.01·kl_loss`.
- **Role in the study:** a clean **GRPO baseline** (= paper's `GRPO-2B`), because with the perception terms
  neutralized it is exactly standard GRPO + ref-KL.

## 3.2 Arm B — "C+DE", full paper Eq. 2 with double entropy (val `reward_score` 0.466)

- **Code:** `PAPO_fixed` — three fixes to `dp_actor.py`: **(B)** fold ref-KL into `pg_loss` and backprop
  `pg_loss` (so *all* terms get gradient, [dp_actor.py:360,444](verl/workers/actor/dp_actor.py#L360)); **(C)**
  log aux metrics as means not last-sample; **RECOMPUTE flipped to True**.
- **RECOMPUTE=True** ([dp_actor.py:58]): inside the update loop, an **extra forward pass on the masked image
  with gradients** recomputes `aug_log_probs` ([dp_actor.py:320-322](verl/workers/actor/dp_actor.py#L320),
  [_forward_micro_batch_aug:166](verl/workers/actor/dp_actor.py#L166)). Consequences: the perception-KL now
  backprops through **both** the real and masked branches (θ in both), and the masked-branch entropy term η₂
  becomes a **real** gradient instead of a no-op.
- **Double entropy ON** (script sets both flags; coef 0.03 each).
- **Effective loss backpropped:**
  `L = L_pg + 0.01·kl_loss − 0.01·kl_prcp_loss + 0.03·H[π_θ^mask] + 0.03·H[π_θ]`
  (perception maximized; both entropies penalized).
- **Cost:** the extra masked forward roughly doubles update-time compute/memory → `gpu_memory_utilization`
  lowered to 0.40 and step time ~26 min.
- **Status vs paper:** this is the paper's *written* Eq. 2 in full, **but it is OFF-SPEC for the 2B**: paper
  Table 3 gives `PAPO_G-2B` with **no** double entropy and RECOMPUTE at the default. So Arm B adds two
  variables (perception **and** double entropy, with full-gradient masked branch) on top of GRPO.

## 3.3 Arm C — "C-pure", the faithful paper PAPO_G-2B (running now)

- **Code:** `PAPO_cpure` — identical to `PAPO_fixed` **except `RECOMPUTE_AUG_LOG_PROBS` reverted to False**.
  Its functional diff vs the released `PAPO_clone` is *exactly* the B (backprop) + C (logging) fixes and
  nothing else (verified by `diff` + `py_compile`).
- **Why the B-fix is still required:** without it, perception is orphaned and Arm C would just be Arm A. With
  it + RECOMPUTE=False, `aug_log_probs` is the **detached** rollout snapshot, so the perception-KL is
  maximized **through the real branch only** (θ in `π_θ(·|I_real)`, the masked branch is a fixed target) — and
  there is **no** extra masked forward. This is precisely the authors' documented default and matches paper
  Table 3 for `PAPO_G-2B`: **γ=0.01, no double entropy, ref-KL β=0.01**.
- **Double entropy OFF** (flags not set → false).
- **Effective loss backpropped:** `L = L_pg + 0.01·kl_loss − 0.01·kl_prcp_loss`.
- **Memory:** no extra forward → profile ≈ Arm A → `gpu_memory_utilization` 0.55.
- **Smoke proof it trains:** `kl_prcp` magnitude **rose** 0.046→0.054 over 2 steps (perception being
  maximized), the opposite of Arm A's decay.

## 3.4 The three objectives, side by side (what `.backward()` sees)

Let `L_pg` = PPO-clip loss, `kl` = D_KL[π_θ‖π_ref], `kp` = D_KL[π_θ(·|real)‖π_θ(·|mask)],
`H`/`H^m` = entropy on real/masked branch. All coefficients as in §2.6.

| Arm | Backpropagated loss | Perception grad path | Double entropy |
|---|---|---|---|
| **A** | `L_pg + 0.01·kl` | — (orphaned) | — (orphaned) |
| **B** | `L_pg + 0.01·kl − 0.01·kp + 0.03·H^m + 0.03·H` | real **and** masked branch (RECOMPUTE) | on |
| **C** | `L_pg + 0.01·kl − 0.01·kp` | real branch only (detached mask) | off |

Everything else — data, order, model init, GRPO advantage, PPO clip, ref-KL, optimizer, LR, 60 steps,
checkpoints — is **identical**. So any difference in outcomes across A/B/C is attributable to these loss
differences alone.

---

# PART 4 — The exact loss-assembly trace (read this to *own* it)

Inside `update_policy`, per micro-batch ([dp_actor.py:295-445](verl/workers/actor/dp_actor.py#L295)), in order:

1. `log_probs = _forward_micro_batch(real image)` — **with gradient** (the current policy). Also
   `entropy_loss = −mean(log_probs)`.
2. `aug_log_probs`: if RECOMPUTE → recompute masked forward **with grad**; else → take the **detached**
   rollout snapshot ([dp_actor.py:318-324](verl/workers/actor/dp_actor.py#L318)).
3. `pg_loss, _ = compute_policy_loss(...)` — the PPO term.
4. If `use_kl_loss`: `kl_loss = mean low_var_kl(log_probs, ref_log_probs)`; **Arm A:** `loss = pg_loss +
   0.01·kl_loss` (binds `loss`, the bug). **Arm B/C:** `pg_loss = pg_loss + 0.01·kl_loss` (folds in).
5. If `aug_log_probs` is not None (perception on): `kl_prcp_loss = mean low_var_kl(log_probs, aug_log_probs)`;
   `pg_loss = pg_loss − 0.01·kl_prcp_loss` (maximize). Then if `use_aug_entropy_loss` (Arm B only):
   `pg_loss += 0.03·aug_entropy_loss`.
6. If `use_ori_entropy_loss` (Arm B only): `pg_loss += 0.03·entropy_loss`.
7. Backprop: **Arm A:** `loss = loss · Σmask/Σtokens; loss.backward()` — backprops steps 3–4 only, orphaning
   5–6. **Arm B/C:** `loss = pg_loss · Σmask/Σtokens; loss.backward()` — backprops everything in `pg_loss`.
8. After the mini-batch: clip grads to 1.0, `optimizer.step()`, `lr_scheduler.step()` (constant LR).

The `Σmask/Σtokens` factor re-weights the micro-batch's contribution by its share of response tokens (token-
level averaging across the accumulated micro-batches).

---

# PART 5 — How to reason about the results (interpretation hooks)

- **Base val is identical (~0.254 accuracy_reward)** for all arms — same model, same val set. Divergence is
  purely what the loss taught.
- **Arm A → 0.540, Arm B → 0.466 `reward_score`.** Adding perception **and** double entropy (Arm B) *lowered*
  clean accuracy vs pure GRPO — consistent with the double entropy over-regularizing a 2B (it is off-spec per
  Table 3), and/or the short 60-step budget (paper trains ~200).
- **`kl_prcp` trajectory is the mechanism signal.** Arm A: decays (orphaned). Arm B (run): plateau ~0.033–
  0.037. Arm C (run, early): plateau ~0.052–0.057 — *higher*, hinting the double entropy in B suppresses
  grounding. **Caveat:** on-policy `kl_prcp` also depends on what tokens each policy chose; the fair,
  cross-arm grounding number comes from the offline perception-KL probe (fixed tokens/masks) — see the
  master record §5.
- **The research question** — *does the perception loss make the model more visually grounded, and at what
  cost to accuracy?* — is answered by comparing, across A/B/C over the 60-step checkpoints: (i) clean
  accuracy, (ii) accuracy under masking (behavioral grounding), (iii) offline perception-KL (distributional
  grounding). Arm C isolates *perception alone* (the clean treatment vs the GRPO control); Arm B adds the
  double-entropy variable.

---

# PART 6 — Glossary (quick reference)

- **π_θ / π_ref / π_θ^mask** — trained policy / frozen reference copy / same trained policy fed a masked image.
- **Rollout** — sampled answers. **n** — answers per prompt (5 train, 8 val).
- **Advantage (GRPO)** — `(answer_reward − group_mean)/(group_std+1e-6)`, group = the n answers of one prompt.
- **PPO clip** — bounds the per-step policy change; ranges 0.2/0.3, dual 3.0.
- **Reference-KL (β=0.01)** — penalty keeping π_θ near π_ref (minimized).
- **Implicit Perception Loss (γ=0.01)** — KL between real- and masked-image outputs, **maximized** → grounding.
- **Double Entropy (η=0.03)** — penalizes entropy on both branches; anti-collapse; **Arm B only**.
- **low_var_kl** — Schulman k3 estimator `exp(s)−s−1`, used for both KLs.
- **RECOMPUTE_AUG_LOG_PROBS** — True ⇒ extra grad-enabled masked forward (Arm B); False ⇒ detached snapshot (A/C).
- **Bug B** — released `main_qwen3` binds `loss` before the PAPO terms → they get no gradient; fixed in B/C.
- **reward_score vs accuracy_reward** — overall (0.9·acc+0.1·format) vs pure accuracy; compare arms on the latter.
