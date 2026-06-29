# Offline Analysis Design — "Where Perception Lives Under RL Reasoning"

> **Status:** design spec for the offline analysis suite. Written on Opus for handoff to a
> coding pass (Sonnet). Everything here is grounded in the real EasyR1 code at commit `dd71bbd`
> and the checkpoints produced by `runs/stage1.sh`. See `EXPERIMENT_PROPOSAL.md` §2/§7/§8 for the
> science and `RESULTS.md` for the run configs.

---

## 0. The targets these scripts must hit

The senior compared the paper's **base vs. staged-RL** checkpoint and made five claims. Our job is to
**reproduce them on *our* model (4B-Instruct, Stage-1), across the training trajectory, and under the
3-condition freeze ablation** — turning his 2-point post-hoc analysis into a prospective, controlled one.

| ID | Claim | Type | Script that tests it |
|---|---|---|---|
| **S1** | representation geometry + modality gap **unchanged** base→trained | control / null | `cka_geometry.py` |
| **S2** | weight change **tiny, concentrated in late-layer MLPs, not attention** | localization | **`weight_delta.py`** |
| **S3** | graft **MLP** deltas onto base → perception ↑; **attention** deltas → nothing | causal | `module_graft.py` |
| **S4** | in the **base** model perception is decodable **mid-layer, decays toward the top** | precondition | `depth_probe.py` |
| **S5** | the late-MLP edit **re-surfaces mid-layer perception the upper layers were destroying** (adds no new info) | mechanism | `depth_probe.py` (trajectory) |
| **H** | headline ablation: **`llm_only ≈ full ≫ vit_only`** (fix lives in the LLM, not the encoder) | forward | `accuracy_curve.py` |

**Working hypothesis H = S5.** "Perception degrading as reasoning goes by" is operationalised as
**perception degrading as signal flows up the residual stack** (the depth axis). S2/S3 = *what/where*
the fix is; S4 = the *precondition*; S1 = the *control* ruling out "it just reshaped the representation."

---

## 1. What delivers DIRECT results NOW vs. what waits

**Two axes — keep them separate or the story gets muddy:**

### Depth axis (computed mid-stack, lost going up) — **DIRECT, NOW, from Stage-1**
The senior's actual mechanism lives here, and every piece is an **offline analysis of checkpoints we
already have** (Condition-1 done; Condition-2/3 finishing):
- **`weight_delta` (S2)** — runs **today** on Condition-1 + base. Real localization result.
- **`depth_probe` (S4/S5)** — runs as soon as coded, using **babyVision (388 items) as the labeled probe set**.
- **`module_graft` (S3)** — the senior's key causal test, reproduced on our model.
- **`accuracy_curve` (H)** — the ablation headline, the moment Condition-2/3 finish.

These are not Stage-3 things. They are the mechanistic core that *justifies the methodology*.

### Temporal axis (perception dies as the CoT gets longer) — **DEFERRED / weak now**
- Stage-1 on an **Instruct** backbone → short answers → little reasoning to decay through → **weak dynamic
  range**. (Documented in project memory: HONEST RESEARCH-SIGNAL FRAMING, 2026-06-25.)
- The clean version wants **Stage-3 training**, OR — cheaper — **eval-only long-reasoning probes on our
  Stage-1 checkpoints** (run the saved checkpoints on long-CoT visual benchmarks; no extra training).
- `perception_vs_length.py` (from rollout traces) gives a *preview* now but expect a flat-ish curve.

### Validation track (optional, de-risks the analysis code) — **available now**
**Reproduce-first:** run `weight_delta` / `depth_probe` / `module_graft` / `cka` on the **paper's released
8B base + staged checkpoints** (HF) before trusting them on our 4B. Confirms our code reproduces the senior
on the original model. Needs downloading the released checkpoints; not core, but cheap insurance.

**Bottom line for the question "can we support the methodology now":** yes — the depth/localization/causal/
ablation results are all directly obtainable from Stage-1. Only the *temporal-degradation-with-range* and
*downstream-reasoning-transfer* genuinely wait for Stage-3 or long-reasoning eval probes.

---

## 2. The data substrate (read this before coding anything)

### 2.1 Checkpoint layout
`runs/stage1_<cond>/checkpoints/global_step_<N>/` contains:
- `actor/model_world_size_4_rank_{0,1,2,3}.pt` — **sharded** model state dict; each parameter is a
  `torch.distributed._tensor.DTensor` holding **only that rank's slice**.
- `actor/optim_world_size_4_rank_*.pt`, `extra_state_*` — optimizer / lr / RNG (resume only; ignore for analysis).
- `actor/huggingface/` — config + tokenizer/processor ONLY (no weights). Needed to instantiate a model class.

Saved by `verl/utils/checkpoint/fsdp_checkpoint_manager.py:90`. `save_freq=6` → steps 6,12,…,96. **There is
no step-0 checkpoint: step 0 = the base model** at
`/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Instruct`.

### 2.2 The no-merge reconstruction trick (linchpin)
Full tensor = `cat([shard_rank0..shard_rank3], dim=placement.dim)`, computable **single-process, CPU-only,
no process group** — exactly as `scripts/model_merger.py:36-44,160-202` does. Therefore:
- **`weight_delta` needs NO merge and NO extra disk** — reconstruct each param transiently, diff, free.
- The **probe-dependent scripts** (which must run inference) DO want a one-time merge to a standard HF dir
  via `merge_checkpoints.py` (§7), because instantiating + running the model is cleanest from `from_pretrained`.

Reference loading logic (mirror it):
```python
sd_r = torch.load(f"model_world_size_4_rank_{r}.pt", map_location="cpu", weights_only=False)  # per rank
# for a key k: shards = [sd_r[k] for r in range(4)]
# each is a DTensor; full = torch.cat([s._local_tensor for s in shards], dim=shards[0].placements[0].dim)
# replicate placement -> just take shards[0]._local_tensor ; non-DTensor -> already full
```
DTensor `.placements` / `._local_tensor` are accessible without a live process group (the merger proves it).

### 2.3 Memory budget
A 4B bf16 model ≈ 8.8 GB; one rank file ≈ 2.2 GB; 4 rank files ≈ 8.8 GB RAM. `weight_delta` holds the 4 rank
files + one reconstructed param at a time → comfortable on a compute node. Merged HF dirs are ~8.8 GB each →
**merge selectively** (don't materialise all 16×3).

---

## 3. Shared helper — `qwen3vl_param_map.py` (build ONCE, used by weight_delta + module_graft)

The MLP-vs-attention definition MUST be identical across the *localization* test (S2) and the *causal graft*
test (S3), or the two results aren't comparable. One classifier, imported by both.

**`classify(param_name) -> (component, module, layer_idx)`**
- `component ∈ {"vision", "llm", "head", "embed", "other"}`
- `module ∈ {"attn", "mlp", "norm", "patch_embed", "merger", "other"}`
- `layer_idx: int | None` (the decoder/vision block index, for the "late-layer" axis)

**Expected Qwen3-VL HF naming** (⚠️ **Step 0 for the coder: print `base.named_parameters()` keys and
`safe_open(...).keys()` and confirm these before classifying — do not ship on assumption**):
```
model.visual.patch_embed.proj.{weight,bias}                         -> vision / patch_embed / None
model.visual.blocks.{i}.attn.{qkv,proj}.*                           -> vision / attn  / i
model.visual.blocks.{i}.mlp.*                                       -> vision / mlp   / i      (verify: linear_fc1/2 vs gate/up/down)
model.visual.blocks.{i}.norm{1,2}.*                                 -> vision / norm  / i
model.visual.merger.* , model.visual.deepstack_merger_list.{j}.*    -> vision / merger / None
model.language_model.embed_tokens.weight                           -> embed  / other / None
model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.*         -> llm / attn / i
model.language_model.layers.{i}.self_attn.{q,k}_norm.*             -> llm / attn / i          (Qwen3 QK-norm: attn bucket)
model.language_model.layers.{i}.mlp.{gate,up,down}_proj.*          -> llm / mlp  / i
model.language_model.layers.{i}.{input,post_attention}_layernorm.* -> llm / norm / i
model.language_model.norm.weight                                   -> llm / norm / None
lm_head.weight                                                     -> head / other / None
```
Regex sketch: `r"\.layers\.(\d+)\."` for layer_idx; `"self_attn"`/`".attn."`→attn; `".mlp."`→mlp;
`"norm"`/`"layernorm"`→norm. Anything unmatched → `other` (and **log it** so nothing is silently dropped).

---

## 4. `weight_delta.py` — FLAGSHIP (tests S2; doubles as weight-level freeze proof)

**Why first:** (a) directly tests the senior's late-MLP localization on *our* model; (b) it's the *lever* for
the broader tool-call vision (where the fix is → what to intervene on); (c) **bonus gold-standard freeze check** —
`vit_only` must show LLM-delta ≈ 0, `llm_only` must show vision-delta ≈ 0. One script, three payoffs. Runs today
on Condition-1, CPU-only.

**CLI:** `python weight_delta.py --base <MODEL> --ckpt <…/global_step_N/actor> --condition <name> --step <N> --out deltas.csv`
(append mode so multiple ckpts/conditions accumulate into one tidy CSV).

**Algorithm:**
1. Print/confirm param naming (§3 Step 0), once.
2. Open base via `safetensors.safe_open` (lazy); load the 4 rank `.pt` files.
3. For each base key:
   - reconstruct full ckpt tensor (§2.2), cast both → fp32.
   - `abs_fro = ‖Δ‖_F`; `rel_fro = ‖Δ‖_F / max(‖W_base‖_F, eps)` ← **primary**; `mean_abs = mean|Δ|`;
     `cos = F.cosine_similarity(vec(W_base), vec(W_ckpt), dim=0)`.
   - classify(key); free tensors.
4. Append rows.

**CSV schema** (`deltas.csv`, long format):
```
condition, step, key, component, module, layer_idx, n_params, base_fro, abs_fro, rel_fro, mean_abs, cos
```

**Decisive plot** (separate `plot_weight_delta.py`, pandas+matplotlib):
- x = `layer_idx`, y = `rel_fro`, **one line per `module` (mlp vs attn)**, faceted by `condition`,
  filtered to `component=="llm"`. **H predicts:** mlp curve rises toward high layer indices; attn flat/low.
- Secondary bar: total `abs_fro` summed by `component` per condition → the freeze sanity check
  (`llm_only`: vision≈0; `vit_only`: llm≈0; `full`: both > 0).
- Trajectory variant: for a fixed late layer, `rel_fro` vs `step` (does the MLP edit grow monotonically? = S5 dynamics).

**Run:** short CPU `sbatch` (no GPU), loop over `global_step_*` for each condition.

---

## 5. `accuracy_curve.py` — QUICK WIN (the ablation headline = H)

**CLI:** `python accuracy_curve.py --logs full=<log> llm_only=<log> vit_only=<log> --out curves.csv`

**⚠️ Step 0 for the coder:** dump one metric block from the Condition-1 log
(`grep -nE "step|reward|accuracy|kl|entropy|grad_norm" <log> | head`, and `grep -A30 "'step': 6" <log>`)
to read the **literal metric key strings** in this EasyR1 build. Do NOT assume `reward/overall`; verify, then
write the regex. Console (slurm log) is the source; the `wandb/offline-run-*` dirs are a fallback but the
`.wandb` binary is heavier to parse — prefer console.

**Output CSV:** `condition, step, reward_overall, reward_accuracy, reward_format, kl, entropy, grad_norm, resp_len_mean`
**Plot:** multi-line `reward_accuracy` vs `step`, one line per condition → the **`llm_only ≈ full ≫ vit_only?`** verdict.
(Condition-1 reference already known: accuracy 0.365 → 0.746, RESULTS.md §9.)

---

## 6. Probe-dependent trio (design now; depends on babyVision as bucket D)

Shared probe set = **babyVision (388 vision-primitive VQA items, labeled)** — already on disk under `babyVision/`.
Front-end = `merge_checkpoints.py` (§7) → `extract_activations.py` → the three analyses.

### 6.1 `extract_activations.py` (shared substrate)
Run a merged checkpoint on the 388 items; forward-hook **each decoder layer's residual output**; save
`acts[n_items, n_layers, hidden]` (fp16) + `labels[n_items]` + `meta` (item ids, answer span). One file per
(condition, step). This is the input to depth_probe and cka.

### 6.2 `depth_probe.py` (S4 + S5)
For each layer, fit a **logistic linear probe** (sklearn, cross-validated) predicting the perception label from
that layer's pooled activation; record accuracy. Plot **probe-accuracy vs layer**:
- **S4** = base model shows mid-layer peak + top-layer decay.
- **S5** = across the trajectory the **top-layer decay flattens** (re-surfacing). Overlay base vs steps 6→96.
- Faceted by condition (does `vit_only` fail to flatten? = corroborates "fix is in the LLM").

### 6.3 `module_graft.py` (S3 — the senior's key causal test)
`W_grafted = W_base + mask ⊙ (W_ckpt − W_base)`, where `mask` selects **MLP-only** or **attention-only** params
(via the §3 classifier — *same definition as weight_delta*). Load grafted weights into the model, eval babyVision
accuracy. **Predict:** MLP-graft recovers most of the perception gain; attn-graft ≈ none. Also do a layer-band
variant (late-MLP-only vs early-MLP-only) to sharpen S2/S5.

### 6.4 `cka_geometry.py` (S1 control)
Linear CKA between base and ckpt activations per layer + modality-gap (image-token vs text-token centroid
distance). **Predict:** high CKA / unchanged gap → deltas are *functional*, not representational.

---

## 7. `merge_checkpoints.py` (front-end for the inference-based scripts)

Thin wrapper over `scripts/model_merger.py`. **Disk-aware:** merge only a **specified sparse set** of steps
(default `{6, 24, 48, 96}` + base) — each merged HF dir ≈ 8.8 GB, so do NOT merge all 16×3 = 48.
`--keep/--clean` to delete merged dirs after activation extraction. Output: standard `from_pretrained`-loadable
HF dirs. (`weight_delta` does NOT use this — it reconstructs in-place.)

---

## 8. File layout & coding order

```
runs/analysis/
  qwen3vl_param_map.py     # shared classifier (build FIRST)
  weight_delta.py          # (1) S2 — runs today on Cond-1, CPU-only
  plot_weight_delta.py     # (1) the decisive MLP-vs-attn-vs-depth plot
  accuracy_curve.py        # (2) H — ablation headline, parse the 3 logs
  merge_checkpoints.py     # (3) front-end for inference scripts
  extract_activations.py   # (3) shared activation substrate (babyVision)
  depth_probe.py           # (4) S4/S5
  module_graft.py          # (5) S3 — reuses qwen3vl_param_map
  cka_geometry.py          # (6) S1
```

**Order rationale:** 1–2 produce direct results from data we already have (S2 + H), CPU-cheap, no probe set
needed. 3–6 share the merge→activation substrate and deliver S4/S5/S3/S1 on babyVision. Build the shared
classifier first so localization (S2) and causal graft (S3) use a byte-identical MLP/attn definition.

**Coder's mandatory Step 0 in each script:** verify real artifact shapes/keys before computing — print
`named_parameters()` keys (weight_delta/graft), dump a real metric block (accuracy_curve), confirm hook points
(extract_activations). No assumptions; this is a no-black-box project (see `feedback-master-rl-no-blackbox`).
