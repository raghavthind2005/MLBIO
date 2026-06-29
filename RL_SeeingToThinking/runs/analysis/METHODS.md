# METHODS — Exactly How Every Number Was Computed (from scratch, master-level)

> The definitive "how" reference. For each experiment: the precise inputs, the exact math, the code
> mechanism (tensor ops, hooks, reconstruction), every parameter, and the output schema — written so a
> newcomer can follow each step and an expert can reproduce it. **Companion docs:** `FINDINGS.md` (results
> narrative), `PRESENTATION.md` (the talk), `RESULTS.md` (training config), `RL_MASTERY.md` (RL internals
> from scratch), `EXPERIMENT2_PLAN.md` (activation-patch design).
>
> **Headline results (so this file is self-contained):** RL took DOCCI perception 0.365→0.746 (training
> reward); on our 300-item direct probe, base 0.377 → trained 0.657. The fix is **tiny** (0.05% weight
> change), **MLP-dominant** (graft: 63% vs 30% attn), **distributed across depth** (late_mlp graft 3.6%),
> **manifests late** (depth probe diverges at layer ~24–25), and is **LLM-internal** (llm_only ≈ full, ViT
> weight-change = exactly 0).

---

## Part A — How the model was trained (the thing we then dissect)

### A.1 The objective, mechanically (RLVR / GRPO — see RL_MASTERY.md for the full derivation)
The model is a **policy** `π_θ`: given a prompt (image + question), it generates a token sequence (the
answer/reasoning). Training is **Reinforcement Learning with Verifiable Rewards**:
1. For each prompt, sample **n = 5** completions from `π_θ`.
2. Score each with a **reward**: accuracy (is the boxed/letter answer correct?) + a format term.
3. **GRPO** (Group Relative Policy Optimization) computes each completion's **advantage** as a *group-relative
   z-score*: `A_i = (r_i − mean(r_1..r_5)) / std(r_1..r_5)`. No value network — the group mean is the baseline.
4. The policy-gradient update raises the probability of above-average completions and lowers below-average
   ones, clipped (PPO-style) to prevent large steps, with a **KL penalty** keeping `π_θ` near the frozen base
   policy `π_ref`.

**Why this matters for the analyses:** the **learning rate (1e-6)** and the **KL leash (coef 1e-2)** make the
weight change *necessarily tiny* — this is why weight_delta finds ~0.05% (Part C). The reward is on
*perception correctness*, so the thing being optimized is exactly what the probe measures.

### A.2 Every result-affecting knob (values; full list in RESULTS.md)
`lr=1e-6`, `kl_coef=1e-2`, `kl_penalty=low_var_kl` (the unbiased k3 estimator `e^x−x−1`), rollout group
`n=5`, `rollout_batch_size=512`, `global_batch_size=128`, `total_epochs=16` (→ 96 optimizer steps),
`max_prompt_length=2048`, `max_response_length=2048`, `max_pixels=4194304`, `seed=1`. Result-neutral
deviations from the paper: 4 GPUs (vs 8), micro-batch 4/8 (gradient identical), a Conv3d→matmul vision
patch-embed fix (bit-identical output, pure speed), `use_torch_compile=false`. **All three conditions share
these exactly** — only the freeze flags differ (A.4).

### A.3 The data
`perception_difficulty_curriculum.jsonl` — 3360 DOCCI perception MCQs. Each: `problem` (a question + 4 lettered
options + "respond using only the letter"), `answer` (a letter), `images` (one image). The model trains to emit
the **bare correct letter** — *direct-answer, no reasoning required*. (This is why our direct-readout probe is
the faithful measurement; Part D.)

### A.4 The three conditions (the experimental lever)
One script, identical except two flags. Freezing = `requires_grad_(False)` on a submodule → the optimizer's
`filter(p.requires_grad)` drops it → those weights **never update** (literally Δ=0).
| condition | `freeze_vision_tower` | `freeze_language_model` | trains |
|---|---|---|---|
| full | false | false | ViT (0.42B) + LLM (4.0B) |
| llm_only | **true** | false | LLM only |
| vit_only | false | **true** | ViT only |

---

## Part B — The data substrate: checkpoints and how we read them

### B.1 What is saved
Every 6 steps (`save_freq=6`) the trainer dumps the full model → 16 checkpoints (`global_step_6…96`). **Step 0 =
the base model** (no separate file). Each `global_step_N/actor/` contains **4 files**
`model_world_size_4_rank_{0,1,2,3}.pt` — because we trained on 4 GPUs with **FSDP**, which splits every weight
matrix into 4 slices (one per GPU). Each slice is a **DTensor** that remembers *how* it was split (its
"placement": `Shard(dim)` = a slice along dimension `dim`, or `Replicate` = a full copy).

### B.2 Reconstructing a full weight (the engineering backbone)
To analyze a weight we stitch the 4 slices back together — single-process, CPU, no GPUs, no process group:
```
load all 4 rank files → for a key k, gather the 4 DTensors → full = torch.cat([d._local_tensor for d in ds], dim=placement.dim)
# Replicate placement: just take slice 0.  Non-DTensor: already full.
```
This mirrors the official `model_merger.py` exactly. Code: `weight_delta.reconstruct_param` /
`ckpt_model.reconstruct_full_state_dict`. **We never write merged checkpoints to disk** — reconstruction is
transient (one matrix at a time for weight_delta; the full dict in RAM for inference scripts; RAM budget ~9 GB,
node has 450 GB). Edge case: `lm_head.weight` is **tied** to `embed_tokens.weight` (the base stores it once, 713
keys; the checkpoint re-saves it, 714 keys) — the duplicate is bit-identical, so we ignore it.

---

## Part C — Analysis 1: weight-delta (where did the weights change?)  [`weight_delta.py`]

### C.1 Inputs
Base model (HF safetensors) + one checkpoint's `actor/` (4 shards). 713 weight tensors.

### C.2 The exact computation, per tensor
Let `W_base`, `W_ckpt` be a tensor before/after; `Δ = W_ckpt − W_base` (cast to fp32). We compute:
- **Frobenius norm of the change** `abs_fro = ‖Δ‖_F = sqrt(Σ_ij Δ_ij²)` — the Euclidean length of Δ flattened.
- **Relative change** `rel_fro = ‖Δ‖_F / max(‖W_base‖_F, 1e-12)` — *fraction of itself the matrix moved*
  (primary metric; comparable across shapes). *Worked example:* a 100×100 all-0.1 matrix has `‖W_base‖_F = 10`;
  a per-entry shift of 5e-4 gives `‖Δ‖_F ≈ 0.05`, so `rel_fro = 0.005 = 0.5%`.
- **Mean absolute change** `mean_abs = mean(|Δ|)`.
- **Cosine** `cos = cos(vec(W_base), vec(W_ckpt))` — ≈1 means "nudged, not rebuilt."

### C.3 The classifier (so we can group by part of the model)  [`qwen3vl_param_map.classify`]
Each key → `(component, module, layer_idx)` by regex on the name. Verified exhaustively: all 713 keys classified
(0 unclassified); counts match the architecture exactly — LLM 36 layers × {attn 6 tensors (q,k,v,o + q_norm,
k_norm), mlp 3 (gate,up,down), norm 2} + final norm + embed; vision 24 blocks × {attn,mlp,norm = 4 each} +
merger/patch_embed. `q_norm`/`k_norm` go in the **attn** bucket. "Late layer" = layer_idx in the top third
(≥24 of 36).

### C.4 Output & key results
CSV: `condition, step, key, component, module, layer_idx, n_params, base_fro, abs_fro, rel_fro, mean_abs,
cos_sim`. **Findings:** mean rel_fro ≈ **5e-4 (0.05%)**, MLP/attn ratio 1.4–1.6× at every depth, roughly uniform
across depth. **Freeze proof:** `llm_only` → vision rel_fro = **exactly 0.000** (all 315 vision tensors); the
optimizer never touched them.

---

## Part D — The perception probe (how we score "perception" deterministically)  [`mc_eval.run_mc_probe`]

This is the measuring stick reused by depth_probe, module_graft, activation_patch.

### D.1 The mechanism, step by step
For one multiple-choice item:
1. Build the chat input: `messages=[{user: [image, question+lettered-options+"answer with the letter"]}]` →
   `processor.apply_chat_template(..., add_generation_prompt=True)` produces the prompt up to where the
   assistant's answer begins.
2. `processor(text, images)` → input tensors (token ids + `pixel_values` + `image_grid_thw`).
3. **One forward pass** → `logits = out.logits[0, -1, :]` — the next-token distribution at the **answer
   position** (the last prompt token).
4. **Restrict to the option letters.** We pre-compute the token id of each letter: `A→32, B→33, …` (verified by
   decoding the id back to the character). For an item with `k` options, take the logits at those `k` ids,
   `argmax` → predicted letter.
5. **Grade:** predicted letter == gold letter? (gold = `chr(65 + choiceAns)`.) Deterministic; no generation, no
   LLM judge.

### D.2 Why this is the *right* probe
The model was trained to answer DOCCI with a **bare letter** (Part A.3), so reading the answer-position letter
logit *is* how it answers. We use **greedy/direct** readout (no reasoning) → measures **direct perception**, the
perception-not-reasoning axis. Caveat: gold-letter skew in the 300-sample (B-heavy) → majority baseline ~49%;
we check the model beats it and isn't position-biased (base 0.377 is *below* majority while making varied
predictions; trained 0.657 *above*).

---

## Part E — Validating the probe (babyVision → DOCCI)

A measuring stick must detect a *known* signal. babyVision (388 vision-primitive items, 135 MC) gave base 32.6%
≈ trained 33.3% — *no* detectable gain. Diagnosis: not a probe bug (DOCCI training format is native direct-answer
MC, so the design is correct) but **out-of-distribution** (babyVision ≠ DOCCI; harder, adversarial). We switched
the probe to a **300-item sample of the training distribution** (DOCCI), where base 0.377 → trained 0.657 (+0.28)
— calibrated (base ≈ training-reward start 0.365), above majority. *Only then* did we trust depth_probe/graft.
**Contamination note:** the model trained on DOCCI, so trained numbers are *train-accuracy*; fine for
*localization* (we dissect the learned mechanism), base numbers are clean, and we don't claim generalization (the
babyVision miss shows generalization is limited — a finding in itself).

---

## Part F — Analysis 2: depth-probe / logit-lens (where does the answer become readable?)  [`depth_probe.py`]

### F.1 Concept from scratch
As the model runs, each layer outputs a **hidden state** (a vector = its "notes" at that depth). Normally only
the **top** layer's notes are turned into an answer, via the **output head** = final RMSNorm then the
unembedding matrix `lm_head` (vector → score per vocabulary token). The **logit-lens** applies that *same* head
to *any* layer's notes — "as if the model had to answer there."

### F.2 The exact computation
One forward with `output_hidden_states=True` → `hs` = tuple of length 37 (`hs[0]` = embeddings, `hs[L]` = output
of layer L). For each L: take the answer-position vector `h = hs[L][0, -1, :]`; compute
`logits_L = lm_head(final_norm(h))`; restrict to the option letters; record `P(correct letter)` (softmax over the
option logits) and argmax-accuracy. We locate `final_norm` defensively (`model.language_model.norm`) and use
`model.get_output_embeddings()` for `lm_head`. Run for base and each checkpoint; overlay.

### F.3 Results & the caveat
Final layer reproduces `mc_eval` exactly (0.377 / 0.657) → lens calibrated. **Layers 0–23 identical** base vs
trained; **sharp divergence at L24–25**; trained sustains ~0.62–0.66 to the top. The sub-chance dip at L19–23 is
a **logit-lens artifact** (mid-layer states aren't in the output basis) and is *identical* in both models → no
signal there; the robust signals are the late layers + base-vs-trained differences. (A *tuned lens* — a learned
per-layer affine map — would clean the middle.)

---

## Part G — Analysis 3: module-graft (which weights *cause* the fix?)  [`module_graft.py`]

### G.1 Concept: counterfactual weight transplant
weight_delta/depth_probe are observational. To get **causation** we *construct* models: start from base, copy
**only a chosen subset** of the trained weights, leave the rest at base, and measure perception. If a subset
recovers the gain, it is **sufficient** → causal.

### G.2 The exact mechanism
1. Load base model once; reconstruct the trained checkpoint's full state dict (Part B.2).
2. **Snapshot** base values for all graftable keys: `base_snapshot = {k: model.state_dict()[k].clone()}`.
3. For each mode, build the overwrite set via the **same classifier as weight_delta** (so the mlp/attn boundary
   is identical across the two analyses): `mask(k, mode)` selects which keys get the checkpoint value.
   - `mlp` = LLM mlp keys; `attn` = LLM attn keys; `late_mlp` = LLM mlp with layer_idx ≥ 24; `early_mlp` = < 12;
     `full` = all; `base` = none.
4. Apply: `model.load_state_dict(base_snapshot, strict=False)` (reset), then
   `model.load_state_dict({k: ckpt[k] for masked k}, strict=False)` (graft), then run the MC probe (Part D).
   Restore base between modes.

### G.3 Results
base 0.377, full 0.657; **mlp 0.553 (recovers 63%), attn 0.460 (30%)**, early_mlp 0.430 (19%), **late_mlp 0.387
(3.6%)**. → MLP-dominant (causal S3), but **distributed** (full-mlp ≫ sum of thirds = synergy; early > late).
**Replicated in Cond 2** (per-graft accuracies near-identical; ViT frozen → unambiguously LLM-internal).
*Caveat:* grafts test **sufficiency**, not a clean additive decomposition — non-additivity is expected.

### G.4 Reconciling with depth-probe (subtle, important)
depth-probe (through the output head) shows **where the answer becomes readable** (late); graft shows **which
weights cause it** (distributed MLP). The logit-lens only sees the *output-aligned* part of the representation
(identical until late); the early/mid MLP edits change *other* residual directions (invisible to the lens early)
that propagate up and let the late layers read out. **Manifests late, caused throughout.** (Hypothesis,
consistent with all data; the activation patch — Part H — tests the residual-direction story directly.)

---

## Part H — Analysis 4: activation-patch (are the representations RE-USABLE?)  [`activation_patch.py`]

### H.1 Why & concept
The program's goal is representation-space (find a better representation, re-use it in a tool-call). Patch =
inject the **trained model's residual** into the **base model** at inference and see if accuracy recovers. This
is a **true causal test** — real residuals, real output head, *no* logit-lens approximation.

### H.2 The exact mechanism (forward hooks)
"Residual at layer L" = output of decoder layer L (0-indexed) at the **answer position** (last token). A PyTorch
**forward hook** on `model.…layers[L]` can read or replace `output[0][:, -1, :]` (the layer's output tuple;
handled tensor-safe). Three hooks:
- **capture**: store `out[0][:, -1, :]` (no modification).
- **patch**: `out[0][:, -1, :] = vec` (overwrite with the trained residual; return modified output).
- **steer**: `out[0][:, -1, :] += α·vec` (add a fixed direction).

Returning the modified output from the hook replaces what flows to layer L+1. (We run under `torch.no_grad()`.)

### H.3 The phased procedure (memory-safe; one model at a time)
1. **Phase 1** — load trained; one forward per item with capture hooks on *all* layers → cache `r^trained[item][L]`
   (answer position; ~28 M floats total) + trained accuracy. Free trained.
2. **Phase 2** — load base; capture → `r^base[item][L]` + base accuracy. Steering vector
   `v_L = mean_items(r^trained_L − r^base_L)`.
3. **Sanity** — self-patch base→base at one L must reproduce base accuracy (proves the hook/position mechanics).
4. **Phase 3 (patch sweep)** — for each L, per item, base forward with patch hook@L from `r^trained` → accuracy@L.
5. **Phase 4 (steer sweep)** — for each (L, α), base forward with steer hook@L using `v_L` → accuracy@L,α.

Inputs are pre-processed once (chat template + image) and reused across all phases. Recovery %
`= (acc − base)/(trained − base)`.

### H.4 Decision criteria (what the result *means*)
| outcome | meaning for the tool-call methodology |
|---|---|
| patch@L≥24 recovers most of the gain | the late representation **is** the carrier → portable, substrate confirmed |
| only very-late (≈35) patch works | trivial (already the output rep) → weak for re-injection |
| partial recovery, plateau | distributed cause (Part G) → rep *partly* portable; quantify the ceiling |
| steering vector lifts base | a **deployable fixed direction** exists → strongest signal |
| only per-item patch works, steer doesn't | rep is input-specific → the tool must **re-derive it from the image** (= "re-inspect") — itself a justification for the tool-call |

---

## Part I — Statistics, scope, and honesty (read before presenting)
- **Sample size:** the probe uses 300 DOCCI items → ±~3% (≈ ±1 item per 0.3%). Treat sub-~3% differences as
  noise (e.g. full 0.657 vs llm_only 0.593 is partly real, partly noise; the conditions are equal on the
  training reward).
- **Single model / seed / stage:** Qwen3-VL-4B, seed 1, Stage-1 only. This is a **mechanism study**, not a 1:1
  paper reproduction. Findings are about *our* model's mechanism.
- **Contamination:** probe = training distribution → trained numbers are train-accuracy (fine for localization;
  base is clean; generalization is *not* claimed and is shown limited by the babyVision miss).
- **Logit-lens approximation:** Part F mid-layers unreliable; late layers + deltas are robust.
- **Graft non-additivity:** sufficiency, not decomposition.
- **Residual-direction reconciliation (G.4) is a hypothesis** that activation-patch (H) tests directly.

---

## Appendix — the analysis code (all in `runs/analysis/`, run in the container, Python 3.12)
| file | what it computes |
|---|---|
| `qwen3vl_param_map.py` | param → (component, module, layer_idx) classifier (shared by weight_delta + graft) |
| `weight_delta.py` + `summarize_deltas.py` | per-tensor Frobenius deltas (S2) + text readout |
| `babyvision_data.py` / `docci_data.py` / `probe_loader.py` | probe-set loaders → shared `MCItem` |
| `ckpt_model.py` | load base + reconstruct ckpt (FSDP shards) in-memory; conv→matmul patch |
| `mc_eval.py` | the MC perception probe (`run_mc_probe`) |
| `probe_debug.py` | top-k next-token + greedy-generation diagnostic |
| `depth_probe.py` | logit-lens decodability per layer (S4/S5) |
| `module_graft.py` | counterfactual weight grafts + probe (S3) |
| `activation_patch.py` | residual capture/patch/steer (Experiment 2) |
| `run_*.sh` | sbatch wrappers (weight_delta = CPU; the rest = 1 GPU) |
