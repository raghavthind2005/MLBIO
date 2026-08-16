# VLM-CapCurriculum Stage 1 (perception RLVR) — authoritative reference

**Every value below was read from the authors' own GitHub / HuggingFace on 2026-08-16, not from our local
clone.** Sources are cited per-section. This file is a *reference*, not a decision — nothing here is adopted.

**Paper:** "From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of
Vision-Language Models" — Wu, Chen, Tu, Tang, Shi, Liu, Lu, Xie, Zhou (UCSC-VLAA). arXiv:**2605.20177**.

> **Venue correction:** the repo's `CITATION.cff`/README bibtex say **ICML 2026** (`booktitle = {…International
> Conference on Machine Learning (ICML)}`), the HF collection slug is `…-icml-2026-…`, and the dataset card tags
> `icml-2026`. Not ICLR. Worth fixing wherever we cite it.

Repo: `github.com/UCSC-VLAA/VLM-CapCurriculum` (paper-side assets only — EasyR1 and VLMEvalKit are **not**
vendored; the repo points at them).

---

## 1. The three stages

```
Stage 1                Stage 2               Stage 3
Visual Perception  →   Textual Reasoning  →  Visual Reasoning
(D_perc, RLVR)         (D_text, RLVR)        (D_vis, RLVR)
```

Headline claim: **86.9%** of Qwen3-VL-8B's wrong answers are perception errors that more thinking cannot fix.
Staged beats merged on 4 backbones; on Qwen3-VL-8B **+1.46% accuracy with 20.8% shorter traces**.
Steps per stage: **90 / 375 / 465**, merged = 930 (`training/README.md`, "Sec 4.1").

---

## 2. Stage-1 training config

Source: `training/examples/qwen3_vl_8b/stage1_perception.sh` + `training/configs/config.yaml` (raw.githubusercontent, main).

| Knob | Value | Where set |
|---|---|---|
| Engine | EasyR1 (verl fork), `python3 -m verl.trainer.main` | script |
| **Algorithm** | **GRPO** (`adv_estimator: grpo`) | config |
| KL | `use_kl_loss: true`, `kl_penalty: low_var_kl`, `kl_coef: 1.0e-2`, `disable_kl: false` | config |
| DAPO group filtering | `online_filtering: false` | config |
| **Backbone (primary)** | **Qwen3-VL-8B-Instruct** | `_env.sh` |
| Other backbones | Qwen2.5-VL-7B-Instruct, InternVL3-8B-hf, InternVL3_5-8B-HF | `_env.sh` |
| **Train data** | `perception_difficulty_curriculum.jsonl` (D_perc) | script via `VLMCC_STAGE1_TRAIN` |
| **Val data** | **`hiyouga/geometry3k@test`** | script (hardcoded) |
| `prompt_key` / `image_key` | `problem` / `images` | script |
| max_prompt_length | 2048 (Qwen) / 4096 (InternVL) | script |
| max_response_length | 2048 | config |
| rollout_batch_size | 512 | config |
| global_batch_size | 128 | config |
| **rollout n (group size)** | **5** | config |
| rollout sampling | temperature 1.0, top_p 1.0 | config |
| val sampling override | temperature 0.6, top_p 0.95, n=1 | config |
| lr / wd / warmup | 1.0e-6 / 1.0e-2 / 0.0 | config |
| optim strategy | `adamw_bf16` (Stage-1 script overrides config's `adamw`) | script |
| max_grad_norm | 1.0 | config |
| **freeze_vision_tower** | **false** (open in Stage 1 and 3; **frozen in Stage 2**) | script + `training/README.md` |
| min_pixels / max_pixels | 262144 / 4194304 | config |
| **total_epochs** | **16** (Stage 1); 15 for Stages 2/3 | script |
| val_freq / save_freq / save_limit | 6 / 12 / 8 | script |
| micro-batch update / experience | 16 / 32 | script |
| offload params / optimizer | false / false | script |
| gpu_memory_utilization | 0.7 | script |
| tensor_parallel_size | 1 | script |
| GPUs | 8 (`VLMCC_GPUS_PER_NODE` default), **8× H200**; full 3-stage ≈ 24 GPU-hours | `_env.sh`, README |
| seed | 1 | config |

**Reward** (`training/reward_functions/math.py:compute_score`, `reward_type=batch`):

```
overall = 0.9 · accuracy + 0.1 · format
format   = fullmatch(r"<think>.*</think>.*\boxed\{.*\}.*", DOTALL)     → 1.0 / 0.0
accuracy = mathruler grade_answer(extract_boxed_content(response), gt) → 1.0 / 0.0
```
with a `too_complex()` guard (len>200, >10 powers, >40 brackets, 50+-digit ints) and a 1 s subprocess timeout
wrapper. Responses are pre-cleaned with `re.sub(r"\s*(<|>|/)\s*", r"\1", …)` (a Qwen2.5-VL-32B format fix).

**Prompt** — one unified `math.jinja` for *all* stages, train and eval:

```
{{ content | trim }} You FIRST think about the reasoning process as an internal monologue and then provide
the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST
BE put in \boxed{}. i.e. <thinking> reasoning here </thinking> \boxed{final answer here}
```

`training/README.md` explicitly notes per-stage formats (e.g. `<description>` tags for Stage 1) were tried
internally but are **not** what produced the paper's numbers.

---

## 3. Stage-1 dataset D_perc — construction and stats

Sources: `data_pipeline/README.md`, HF card `UCSC-VLAA/VLM-CapCurriculum-Perception-Data`.

| | |
|---|---|
| Rows (`train`) | **3,360** |
| Format | 4-way MCQ, `"Respond using only the letter corresponding to the correct answer."` |
| Image source | **DOCCI**, downsampled 2× — 14,847 jpgs, ~5 GB, shipped as one `.tar.gz` |
| Difficulty signal | `pass_rate` ∈ [0,1] from **16 rollouts of Qwen3-VL-8B-Instruct** (the base model) |
| Fields | `index`, `problem`, `answer`, `images` (paths relative to `images/`) |

**Construction rule — the important part:**

```
keep iff   Â_img(Q | I) ≠ A    ∧    Â_cap(Q | C) = A
```

i.e. keep only MCQs the VLM **gets wrong from the image** but **gets right from the caption**. QA is generated
by Qwen2.5-72B from DOCCI captions (`prompts/docci_mcq_generation.txt`); filtering uses **two** VLMs
(Qwen2.5-VL-7B and Qwen2.5-VL-32B) and keeps the **intersection**, to suppress filter-model artefacts.

Stage 2 = ORZ-Math-13k. Stage 3 = CLEVR-Math + GeoQA170K + Math PUMA + ArxivQA. Neither is generated by this
pipeline.

---

## 4. Evaluation (how the paper's numbers are produced)

Source: `evaluation/README.md`, `evaluation/run_eval.sh`.

- Harness: **VLMEvalKit** (not vendored), served checkpoints via vLLM (Qwen) / LMDeploy (InternVL).
- **Judge: `bedrock-claude-haiku-4.5`** via AWS Bedrock. ⚠️ We have no Bedrock access — an exact eval repro
  would need either AWS credentials or a substitute judge (a deviation to declare).
- `MAIN_BENCH`: `MathVista_MINI`, `MathVision_MINI`, `MathVerse_MINI_Vision_Intensive`, `WeMath`,
  `RealWorldQA`, `MMStar`, `POPE` (README also groups A-OKVQA under "Perception AVG").
- `EXTENDED_BENCH` adds `MathVerse_MINI_Vision_Only` among others.
- Grouping: Visual Math = MathVista / MathVision / MathVerse(VI) / WeMath; Perception = A-OKVQA / RealWorldQA /
  MMStar / POPE.

---

## 5. Parallels and tensions with our caption-distortion Stage 1

**Parallel (supports the framing).** Their Stage 1 and ours occupy the same slot: train *perception* before
reasoning, with RLVR, on a perception-isolating dataset, then hand the checkpoint to a later stage. Their
result that this ordering beats merged training is the strongest published argument that a dedicated
perception stage is worth running at all.

**Tension 1 — D_perc is adversarial to our objective, by construction.** ⚠️ This is a hard incompatibility,
not a nuance. Our `J_cap` minimizes `KL(π(·|c,x) ‖ π(·|I,x))` — it trains the caption so the blind answer
**matches the image-conditioned answer**. But D_perc keeps only items where `Â_img(Q|I) ≠ A`: the
image-conditioned answer is **wrong by construction on every row**. Training `J_cap` on D_perc would therefore
optimize captions to reproduce a wrong answer on 100% of the data. **D_perc must not be used as the `J_cap`
training set without changing the objective's target.** (It remains an excellent *evaluation* set, and an
excellent set for the task-reward `J_success`.)

**Tension 2 — their Stage-1 validation does not measure perception.** Stage 1 trains on DOCCI perception MCQs
but validates on `hiyouga/geometry3k@test`, a geometry-math set. It is a generic "is training healthy" monitor,
not a perception metric. If we mirror their setup we inherit a validation signal that cannot detect the thing
we care about. Our perception measurement has to be a separate, deliberately chosen eval.

**Tension 3 — caption sufficiency is assumed by their filter, and is exactly what we are trying to train.**
D_perc's second condition `Â_cap(Q|C) = A` uses *human-written DOCCI captions*. That is close to Track T's
oracle-text arm (+7.5) — evidence the ceiling is real — but it is a ceiling produced by human captions, not by
the model's own. Our method's whole claim is that the model can be trained toward that ceiling itself.

**Parallel worth exploiting.** Their `pass_rate` (16 base-model rollouts per item) is a ready-made difficulty
axis, and their Sec 4.5 result is that capability and difficulty curricula **stack additively** (+4.43 vs
+1.97 / +1.80 alone). If we ever want a difficulty schedule, the labels already exist.
