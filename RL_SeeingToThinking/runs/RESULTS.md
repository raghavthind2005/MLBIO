# Stage-1 RLVR — Results & Configuration Log

Master record for the Stage-1 (Visual Perception) GRPO runs on Qwen3-VL-4B-Instruct.
Each run gets a section; **Run 1 = Condition 1 (full fine-tune, LLM + ViT)**, which is also the
faithful reproduction of the paper's Stage 1.

Engine internals are explained in `../RL_MASTERY.md` (Ch. refs below point there).
Status: **config frozen, not yet executed** (sanity run pending).

---

## 0. Run identity & environment

| Item | Value |
|---|---|
| Paper | "From Seeing to Thinking…", UCSC-VLAA, ICML 2026, arXiv:2605.20177 |
| Stage / condition | Stage 1 (Visual Perception) · Condition 1: full fine-tune (LLM + ViT trainable) |
| Backbone | Qwen3-VL-4B-**Instruct** (`/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Instruct`) |
| Paper backbone | Qwen3-VL-**8B**-Instruct (we use 4B — smaller, what's on cluster) |
| Engine | EasyR1 (verl fork), commit `dd71bbd`, mounted via PYTHONPATH |
| Container | `easyr1_vllm0112.sqsh` — torch 2.9.0+cu129, vLLM 0.11.2, transformers 4.57.3, flash-attn 2.8.3 |
| Hardware | 1× Clariden node, **4× GH200** (96 GB), aarch64 (paper: 8× H200) |
| Train data | `UCSC-VLAA/VLM-CapCurriculum-Perception-Data` — 3,360 DOCCI MCQs, 1 image each |
| Val data | `hiyouga/geometry3k@test` (paper's Stage-1 val — a visual-reasoning set, used as a progress signal) |
| Reward | rule-based: `math.py:compute_score` = 0.9·accuracy + 0.1·format |
| Run script | `runs/stage1_full.sh` |

---

## 1. Data config

| Param | Value | Source | Matches paper? | Intended effect |
|---|---|---|---|---|
| `train_files` | perception jsonl (3360) | ours (same HF dataset) | ✅ | the Stage-1 perception MCQs |
| `val_files` | `hiyouga/geometry3k@test` | stage1 override | ✅ | periodic progress eval |
| `image_dir` | `$DATA/images` | ours (resolves `DOCCI/…`) | ✅ (functionally) | root for relative image paths |
| `prompt_key` | `problem` | config | ✅ | which jsonl field is the prompt |
| `answer_key` | `answer` | config | ✅ | ground-truth field for reward |
| `image_key` | `images` | config | ✅ | which field holds image paths |
| `format_prompt` | `math.jinja` | config | ✅ | appends `<think></think>…\boxed{}` instruction (Ch 3.1) |
| `max_prompt_length` | 2048 | stage1 override | ✅ | truncate/ filter long prompts |
| `max_response_length` | 2048 | config | ✅ | cap generation length |
| `rollout_batch_size` | 512 | config | ✅ | **prompts per training step** (affects math, Ch 6.8) |
| `val_batch_size` | 1024 | config | ✅ | prompts per val pass |
| `min_pixels` / `max_pixels` | 262144 / 4194304 | config | ✅ | image-resolution bounds fed to the ViT |
| `filter_overlong_prompts` | true | config | ✅ | drop prompts over `max_prompt_length` |
| `shuffle` / `seed` | true / 1 | config | ✅ | data order |

---

## 2. Algorithm config (GRPO)  — RL_MASTERY Ch 4–7

| Param | Value | Source | Matches paper? | Intended effect |
|---|---|---|---|---|
| `adv_estimator` | `grpo` | config | ✅ | group-relative advantage, no value net (Ch 5) |
| `disable_kl` | false | config | ✅ | keep the reference model active |
| `use_kl_loss` | **true** | config | ✅ | KL applied as a **loss term**, not in reward (Ch 7.5 Mode B) |
| `kl_penalty` | `low_var_kl` | config | ✅ | k3 estimator `e^x−x−1` (Ch 7.3) |
| `kl_coef` | `1e-2` | config | ✅ | strength of the KL leash to π_ref |
| `kl_type` | `fixed` | config default | ✅ | constant kl_coef (Ch 7.6) |
| `online_filtering` | false | config | ✅ | keep all groups incl. dead ones (Ch 5.3) |
| `gamma` / `lam` | 1.0 / 1.0 | config default | ✅ | unused by GRPO (GAE-only) |

---

## 3. Actor / optimizer / FSDP — RL_MASTERY Ch 6, 8

| Param | Value | Source | Matches paper? | Intended effect |
|---|---|---|---|---|
| `rollout.n` (group size) | **5** | config | ✅ | responses per prompt = the GRPO baseline (Ch 5) |
| `global_batch_size` | 128 | config | ✅ | samples per optimizer minibatch (affects math, Ch 6.8) |
| `micro_batch_size_per_device_for_update` | 16 | stage1 override | ✅ | memory-only chunking for the update pass (Ch 6.8) |
| `micro_batch_size_per_device_for_experience` | 32 | stage1 override | ✅ | memory-only chunking for log-prob pass |
| `ppo_epochs` | 1 | engine default | ✅ | one pass over each rollout batch (Ch 6.8) |
| `optim.lr` | `1e-6` | config | ✅ | learning rate |
| `optim.weight_decay` | `1e-2` | config | ✅ | AdamW weight decay |
| `optim.strategy` | `adamw_bf16` | stage1 override | ✅ | bf16 AdamW states (memory) |
| `optim.lr_warmup_ratio` | 0.0 | config | ✅ | no warmup |
| `max_grad_norm` | 1.0 | config | ✅ | gradient clipping (Ch 6.6 `_optimizer_step`) |
| `clip_ratio_low` | 0.2 | **engine default** | ⚠️ verify | PPO lower clip (Ch 6.5) |
| `clip_ratio_high` | 0.3 | **engine default** | ⚠️ verify | DAPO asymmetric upper clip (Ch 6.6) |
| `clip_ratio_dual` | 3.0 | **engine default** | ⚠️ verify | dual-clip floor for negative-advantage tokens |
| `loss_type` | `default` | engine default | ✅ | per-token PPO clip (vs gspo/cispo/sapo) |
| `loss_avg_mode` | `token` | engine default | ✅ | average loss over all tokens |
| `padding_free` | true | config | ✅ | drop padding in the forward (throughput, Ch 1.6/Part III) |
| `dynamic_batching` | true | config | ✅ | pack micro-batches by token count |
| `ulysses_size` | 1 | config | ✅ | no sequence parallelism |
| `fsdp.torch_dtype` | bf16 | stage1 override | ✅ | bf16 params |
| `fsdp.enable_full_shard` | true | config | ✅ | shard params across the 4 GPUs (Part III) |
| `offload.offload_params` | false | stage1 override | ✅ | keep params on GPU (speed > memory) |
| `offload.offload_optimizer` | false | stage1 override | ✅ | keep optimizer on GPU |
| `model.enable_gradient_checkpointing` | true | config | ✅ | recompute activations to save memory |
| `model.freeze_vision_tower` | **false** | stage1 override | ✅ | **ViT trainable** (Condition 1) |
| `model.freeze_language_model` | **false** | ours (default) | ✅ | **LLM trainable** (Condition 1); our patched flag |
| `model.trust_remote_code` | false | config | ✅ | Qwen3-VL is native in transformers 4.57 |

---

## 4. Rollout config (vLLM) — RL_MASTERY Ch 1.4, 8

| Param | Value | Source | Matches paper? | Intended effect |
|---|---|---|---|---|
| `temperature` (train) | 1.0 | config | ✅ | exploratory sampling for diverse group (Ch 1.4) |
| `top_p` (train) | 1.0 | config | ✅ | no nucleus truncation |
| `val_override: temperature/top_p/n` | 0.6 / 0.95 / 1 | config | ✅ | near-greedy single sample at eval |
| `gpu_memory_utilization` | 0.7 | stage1 override | ✅ | vLLM KV-cache fraction |
| `tensor_parallel_size` | 1 | stage1 override | ✅ | one model replica per GPU (4B fits) |
| `limit_images` | 0 | config | ✅ | `0` ⇒ unset ⇒ vLLM default **1 image/prompt** (`vllm_rollout_spmd.py:122`); matches our single-image data |
| `enforce_eager` | false | config | ✅ | allow CUDA graphs |
| `max_num_batched_tokens` | 8192 | engine default | ✅ | vLLM batching cap |

---

## 5. Reward structure — RL_MASTERY Ch 3

- **File:** `VLM-CapCurriculum/training/reward_functions/math.py`, function `compute_score` (unchanged from paper).
- **Accuracy** (`accuracy_reward`): `extract_boxed_content(response)` → `grade_answer(answer, ground_truth)` → 1.0 if the boxed letter matches `answer`, else 0.0.
- **Format** (`format_reward`): 1.0 iff response matches `<think>.*</think>.*\boxed{.*}` else 0.0.
- **Overall:** `0.9·accuracy + 0.1·format` (`format_weight=0.1`).
- **Reward type:** `batch` — declared via module attribute `REWARD_TYPE = "batch"` we appended (our EasyR1 reads this from the module, `function.py:130`; the paper passed `reward_type=batch` as a CLI key that no longer exists → **behaviorally identical**).
- Sparse/terminal layout: the scalar is placed on the **last response token**, rest zeros (`function.py:100`, Ch 3.3).

---

## 6. Trainer / schedule / observability

| Param | Value | Source | Matches paper? | Notes |
|---|---|---|---|---|
| `total_epochs` | 16 | stage1 override | ✅ | ~90–105 steps over 3360 prompts |
| `n_gpus_per_node` | **4** | ours (HW) | ⚠️ forced | paper used 8; math-neutral (same global/rollout batch) |
| `val_freq` | 6 | stage1 override | ✅ | val every 6 steps |
| `val_before_train` | true | config | ✅ | gives **step-0 baseline** eval |
| `max_try_make_batch` | 20 | config | ✅ | cap on regeneration (only matters with filtering) |
| — **observability (no gradient effect)** — | | | | |
| `save_freq` | 6 | **ours** | ⚠️ obs | paper 12 → we save denser for the trajectory |
| `save_limit` | **-1** | **ours** | ⚠️ obs | paper 8 → **keep ALL** checkpoints (the key change: paper would prune the early ones our analysis needs) |
| `save_model_only` | true | **ours** | ⚠️ obs | weights only (~8 GB/ckpt); enough for all offline analyses; no mid-optimizer resume |
| `val_generations_to_log` | 64 | **ours** | ⚠️ obs | paper 3 → capture reasoning traces across checkpoints |
| `logger` | console, wandb (offline) | **ours** | ⚠️ obs | wandb store = analyzable metric curves |

---

## 7. Fidelity summary (read this for the "exact reproduction" question)

**Mirrors the paper exactly** — every gradient-affecting knob: GRPO, n=5, lr 1e-6, KL (low_var_kl,
1e-2, loss-side), rollout_batch_size 512, global_batch_size 128, 16 epochs, reward fn, data,
pixel/length bounds.

**Forced, math-neutral deviations:**
1. **4 vs 8 GPUs** (`n_gpus_per_node`): changes per-device sharding + wall-clock, not the optimization (we hold global/rollout batch fixed). Ch 6.8.
2. **`reward_type` CLI key dropped**: engine-version drift; same behavior via `REWARD_TYPE` module attr.
3. **micro-batch sizes** are memory chunking only; kept at paper's 16/32, will shrink **only** if OOM (does not change the gradient).

**Observability overrides** (Section 6): checkpoint/logging only — zero gradient effect.

**⚠️ BRING-UP DEVIATIONS (discovered 2026-06-26, NOT all math-neutral):**
- **`data.max_pixels` 4194304 → 262144 — FORCED, and this one DOES affect fidelity** (it changes the
  image resolution the model sees). Root cause: at the paper's 4194304, the HuggingFace multimodal
  *training*-forward hits a single-threaded Python hotspot (vision index computation) that pins one CPU
  core for 30+ min per forward on our GH200/ARM node — effectively a hang (diagnosed via thread states:
  main thread `R` 95% CPU, all NCCL/CUDA threads idle; vLLM generation unaffected because it uses fused
  kernels). 262144 (~330 vision tokens/image) unblocks it. **The paper's value is untenable for the HF
  FSDP forward on this node.** Open question for the real campaign: find the largest `max_pixels` that
  still trains in acceptable time, and quantify the perception impact of the resolution drop.
- **`use_torch_compile` true → false** — reliability during bring-up; does not change the math. The
  compiled op is `log_probs_from_logits`, not the bottleneck, so no real speed loss.
- **`save_model_only` true → false** for the real run — saves optimizer state too, so a wall-clock
  time-out is **resumable** (`find_last_checkpoint`). Disk is plentiful (scratch 451 TB free). Weights
  for offline analysis are still present.
- **in-run validation disabled** (`val_freq=-1`, `val_before_train=false`) — observability only; we
  evaluate checkpoints offline. Maximizes training throughput for the unattended run.

### 7.1 Root-cause of the slowness + the fix — vision patch-embed `Conv3d` → matmul  ✅ RESULT-NEUTRAL

**The overnight full run (06-26) failed two ways**, both now understood:
1. **CUDA OOM** in the policy update at `max_response_length=2048` (2× the activations vs the sanity's
   1024, which fit). Memory-only — fixed by shorter response and/or smaller update micro-batch.
2. **~400× too slow forward** — the real wall. `compute_log_probs` ran at **1830 s/micro-batch**
   (≈11 s/sample; a 4B forward is ~0.03 s of GPU compute). 0 steps completed in 2 h.

**Diagnosis (systematic, evidence at each step):** `ps -T` thread states → main thread `R` at 95% CPU,
all NCCL/CUDA threads idle (⇒ not NCCL, not GPU, not deadlock; CPU/launch-bound). In-container `py-spy`
(ptrace works inside the job's own container; blocked only cross-container) → the active stack sits in
the **vision tower**, called from EasyR1's `verl/models/transformers/qwen3_vl.py:_get_input_embeds:148`
`model.visual(...)`, which **re-encodes every image on every forward** (×3/step × micro-batches ×
epochs). Aggregate of innermost frames → **8/11 samples in `_conv_forward`** (the patch-embed `Conv3d`).

**Why:** Qwen3-VL's `Qwen3VLVisionPatchEmbed` is an `nn.Conv3d` with **kernel == stride** (1×1×1 output)
— i.e. a per-patch linear projection *expressed* as a convolution. On **aarch64/cuDNN** that degenerate
Conv3d over a large patch batch has no good kernel and falls back to something pathological. Controlled
micro-benchmark at the real vision dims (`patch=16, temporal=2, in_ch=3, embed=1024`):

| N (patches) | `conv3d` | equivalent `matmul` | speedup | `maxdiff` |
|---|---|---|---|---|
| 10,000 | **56,708 ms** | **0.17 ms** | **337,569×** | **0.0000** |

**The fix (result-neutral, applied 2026-06-28):** replace the patch-embed conv with the algebraically
identical matmul: `linear(x.reshape(N,-1), proj.weight.reshape(embed,-1), proj.bias)`. cuBLAS GEMM is
first-class on aarch64; **output is bit-identical (`maxdiff=0.0`)** so the trained model is unchanged —
this is purely *how the same arithmetic executes*, like swapping BLAS backends. **No effect on results.**
- Implemented as a monkey-patch alongside EasyR1's existing Qwen3-VL patches:
  - `verl/models/transformers/qwen3_vl.py` — new `qwen3_vl_patch_embed_forward`.
  - `verl/models/monkey_patch.py` — `Qwen3VLVisionPatchEmbed.forward = qwen3_vl_patch_embed_forward`
    (in the `QWEN3_VL_MODELS` branch of `apply_ulysses_patch`, which already runs at model build).
- Local copy + cluster clone (`/iopsstor/.../code/EasyR1`) both patched.

**Fidelity upside:** the conv got *worse* at higher resolution (more pixels → more patches → slower
conv) — which is the **only** reason we were forced to `max_pixels=262144` (a result-affecting
deviation, §7 bring-up). With the conv fixed, **we restore `max_pixels` (and
`max_response_length`) to the paper's original values** and stay within memory/time.

**✅ REALIZED 2026-06-28 — paper config confirmed feasible.** A 2-step measurement at the FULL paper
config (`max_pixels=4194304`, `max_response_length=2048`, `rollout_batch_size=512`) ran: log-probs
1.26 s/it (23 its = 29 s), update 2.39 s/it (12 its = 26 s), **NO OOM** (with result-neutral
`micro_batch_size_per_device` 4/8 instead of paper's 16/32, since 96 GB GH200 < 141 GB H200). Per-step
≈ 2–3 min → the full 16-epoch run (~96 steps) ≈ **3–5 h, one 12h job**. So the final Stage-1 run keeps
**every result-affecting knob at the paper's value**; the only deviations are result-neutral
(conv→matmul, micro-batch chunking, expandable_segments, 4-GPU sharding). `runs/stage1_full.sh` updated
to this locked config.

**⚠️ Open verify item:** `clip_ratio_low/high/dual = 0.2/0.3/3.0` come from EasyR1's *engine defaults*
(the paper's `config.yaml` doesn't set them). If the paper's EasyR1 commit used vanilla PPO
(`0.2/0.2`), our asymmetric DAPO defaults are a minor uncontrolled difference. **Action:** if exact clip
fidelity is required, pin them explicitly; otherwise accept EasyR1 defaults and note it.

---

## 8. Derived quantities

- Steps/epoch ≈ ⌈3360 / 512⌉ = 7 → **~112 steps** over 16 epochs (minus overlong-filtered rows; paper reports ~90).
- Per step: 512 prompts × n=5 = **2,560 generations**.
- Checkpoints: every 6 steps, kept all → **~18 checkpoints** × ~8 GB = **~145 GB** on scratch (451 TB free ✓).

---

## 9. RESULTS — training dynamics

### Condition 1 (full, LLM+ViT) — job 2640662, 2026-06-29 ✅ COMPLETE
- **96/96 steps in 5h 43m** (214 s/it), clean exit, no OOM/errors. **16 checkpoints** saved
  (`global_step_6 … 96`, every 6 steps, all kept) → `runs/stage1_full/checkpoints/`.
- **accuracy reward 0.365 → 0.746** — strong, sustained climb (perception ~doubled from near-random;
  still rising at 16 epochs). overall reward 0.329 → 0.672. Approx curve: ~0.37 [s1] → ~0.50 [s20] →
  ~0.60 [s50] → ~0.69 [s75] → ~0.75 [s96].
- **Headline:** RLVR substantially improves visual perception on DOCCI MCQs at the paper-faithful config.

### Condition 2 (llm_only) / Condition 3 (vit_only) — *(fill after each run)*
- Compare final accuracy + curve vs Condition 1 → hypothesis H: is `llm_only ≈ full ≫ vit_only`?
- Per-condition checkpoints (own dirs) feed the offline analyses (weight-delta MLP-vs-attn, depth-probing).

## 10. ANALYSES toward the research question

> **Full from-scratch explanation of every analysis + result → [`analysis/FINDINGS.md`](analysis/FINDINGS.md).**
> Design spec → [`analysis/ANALYSIS_DESIGN.md`](ANALYSIS_DESIGN.md). Code → `runs/analysis/`.

_Offline on the saved checkpoints + base:_
- ✅ **Weight-delta MLP-vs-attention by layer** (`weight_delta.py`, tests S2) — **DONE for Condition 1.**
- ⏳ Depth-resolved perception decodability (`depth_probe.py`, S4/S5) — needs babyVision probe set.
- ⏳ Causal module-graft (`module_graft.py`, S3) — needs babyVision + eval.
- ⏳ CKA / modality-gap base-vs-trained (`cka_geometry.py`, S1) — control.
- ⏳ Image-utility vs reasoning-position — the temporal "perception as reasoning proceeds" curve (weak in
  Stage-1-Instruct; wants Stage-3 or long-reasoning eval probes — see FINDINGS Part 0/5).

### 10.1 Weight-delta — Condition 1 (full), 2026-06-29 ✅
*(Full explanation: FINDINGS.md Parts 2–5. Headline only here.)*
- **The RL weight change is tiny: mean rel_fro ≈ 5e-4 = 0.05%** (LLM), max 0.1%. Yet accuracy went
  0.365→0.746. **A 0.05% weight change ≈ doubled perception** → "surgical edit, not re-learning."
  Confirms senior's S2 "tiny change" + supports the S5 "re-access" thesis. This tininess is the *expected*
  fingerprint of our **lr=1e-6 + KL leash** (§2/§3).
- **MLP-biased over attention at every depth** (rel_fro mlp/attn = 1.59× early, 1.40× mid, 1.37× late) →
  S2 "MLP not attention" confirmed *directionally*.
- **NOT late-layer-concentrated** (mlp roughly flat / slightly early-peaked) → S2 depth-localization **not
  reproduced** in this 4B / 96-step / perception-only regime. Honestly reported; leading explanation =
  regime difference vs senior's full-curriculum 930-step 8B analysis.
- **Smooth monotonic drift** step 6→96 → stable training (process check; not yet evidence of re-surfacing).
- **Caveats:** attn bucket mixes q/k/v/o with tiny q_norm/k_norm (refine to projections-only); `full`
  condition only (cross-condition + freeze-at-weight-level pending Cond 2/3); correlational not causal
  (S3 graft pending). See FINDINGS Part 4.

---

## 11. Conditions 2 & 3 — LLM-only / ViT-only (the freeze ablation)

**One parameterized script** (`runs/stage1.sh <cond>`) drives all three conditions so they are
byte-identical except **two freeze flags** — a clean ablation by construction.

| Condition | `freeze_vision_tower` | `freeze_language_model` | learns | run dir |
|---|---|---|---|---|
| `full` (Cond 1, running) | false | false | LLM + ViT | `runs/stage1_full` |
| `llm_only` (Cond 2) | **true** | false | LLM only | `runs/stage1_llm_only` |
| `vit_only` (Cond 3) | false | **true** | ViT only | `runs/stage1_vit_only` |

**Why this is a clean ablation (the subtle points):**
- **Freezing changes only the backward, not the forward.** A frozen component still runs in the
  forward, so **generation, reward, and log-probs use the full model in all three conditions**; only
  *which params receive gradients* differs. Same rollouts mechanism, same data (`seed=1`), same init.
- **Same everything else** — lr, KL, n=5, batch sizes, epochs, reward, `max_pixels=4194304`,
  `response=2048`, the conv→matmul patch. The conv patch trains the ViT correctly (grads flow to
  `proj.weight`), so it's transparent for `vit_only`.
- `freeze_language_model` is our added flag (`fsdp_workers.py`, mirrors `freeze_vision_tower`): freezes
  `model.model.language_model` + `lm_head`, sets `use_orig_params=True`. Optimizer's `requires_grad`
  filter drops frozen params. Verify each run logs the freeze: *"Vision tower is set to not trainable."*
  (llm_only) / *"Language model is set to not trainable (ViT-only condition)."* (vit_only).
- **Data collection identical to Cond 1** (`save_freq=6`, `save_limit=-1`, `save_model_only=false`,
  console+wandb) into a **per-condition checkpoint dir** → full per-condition trajectory for the offline
  analyses (weight-delta MLP-vs-attn, depth-probing, perception-survival, CKA).

**What they answer (hypothesis H, proposal §5):** if `llm_only ≈ full ≫ vit_only` in perception gain,
the fix lives in the LLM (late MLPs), confirming the senior's finding *prospectively* (by what we let
train) rather than only post-hoc weight-grafting. Frozen conditions also use less memory (fewer
trainable params) — `micro=4/8` kept identical for exact mirroring (could be raised, result-neutral).

**Run order:** sequential (one node).

**✅ Freeze verified (2026-06-29, `runs/verify_freeze.py`).** Module sizes: total **4.438B** =
language_model **4.022B** + visual **0.415B** (lm_head 0.389B is weight-tied into the LLM). Trainable
params per condition match exactly: full **4.438B** (all), llm_only **4.022B** (ViT's 0.415B frozen),
vit_only **0.415B** (LLM frozen). 3-part check on the running `llm_only` job (2642908): config diff vs
Cond 1 shows only `freeze_vision_tower`/`experiment_name`/`save_checkpoint_path` differ (no training
hyperparameter changes); log confirms *"Vision tower is set to not trainable."*; `Total training steps:
96` + dataloader 6 = Cond 1; `grad_norm > 0` (LLM learning), reward/KL sane, no OOM.
