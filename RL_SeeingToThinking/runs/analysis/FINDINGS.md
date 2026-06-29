# FINDINGS — From-Scratch Explanation of Every Analysis & Result

> **Purpose.** This document explains, *from absolute scratch but with full technical rigor*, what
> each analysis computes, why it is designed that way, how it connects to our specific RL training
> choices, and exactly what each number means. It is written so a beginner can follow it, yet contains
> enough precision to present professionally. It grows as each analysis lands.
>
> **Companion docs:** `RESULTS.md` (per-knob training config), `ANALYSIS_DESIGN.md` (the analysis spec),
> `RL_MASTERY.md` (RL internals from scratch), `EXPERIMENT_PROPOSAL.md` (the science & the senior's claims).

---

## Table of contents
- [Part 0 — The big picture: what are we even asking?](#part-0)
- [Part 1 — Background you need (weights, layers, checkpoints)](#part-1)
- [Part 2 — Analysis #1: Weight-Delta (tests claim S2)](#part-2)
- [Part 3 — RESULT #1: Weight-Delta on Condition 1 (full)](#part-3)
- [Part 4 — Honest caveats & limitations of this result](#part-4)
- [Part 5 — What this does and does NOT prove; what's next](#part-5)

---

<a name="part-0"></a>
## Part 0 — The big picture: what are we even asking?

### 0.1 The phenomenon
Vision-Language Models (VLMs) — models that take an **image + a text question** and produce a **text
answer** — make a lot of mistakes on *perception*: they literally mis-see the image (miscount objects,
miss a small difference, misread a position). The paper we reproduce ("From Seeing to Thinking", UCSC-VLAA)
showed **86.9% of VLM errors are perception errors** that *more reasoning cannot fix* — if the model
mis-saw the image, thinking longer about the wrong percept doesn't help.

### 0.2 The fix the paper proposes
Train the model with **Reinforcement Learning with Verifiable Rewards (RLVR)** specifically on perception
tasks (their "Stage 1"). The model generates an answer, gets **+reward if correct, 0 if wrong**, and the
RL algorithm (GRPO) nudges the weights to make correct answers more likely. We reproduced this: our
**Condition 1** run took perception accuracy from **0.365 → 0.746** in 96 steps (RESULTS.md §9).

### 0.3 The deeper question (our actual research)
A senior researcher looked *inside* the trained model and made five mechanistic claims (S1–S5; see
EXPERIMENT_PROPOSAL.md §2). The headline one (**S2**) is:

> *"When you compare the base model's weights to the RL-trained model's weights, the change is **tiny**
> and **concentrated in the MLP sub-layers of the late transformer layers**, NOT in the attention
> sub-layers."*

If true, this is a profound statement: it says RL doesn't rebuild the model's vision — it makes a
**small, surgical edit in one specific place**. That, in turn, supports his bigger intuition (**S5**):
the model already computes the right perception mid-way through its layers, then *loses* it in the upper
layers, and RL's small MLP edit just **stops that loss** ("re-surfacing", not "re-learning").

**Our job:** test whether the *same mechanism* appears when *we* do the RL, on our own model, across the
whole training trajectory, under controlled conditions. The first tool for that is the **weight-delta
analysis**, which directly tests S2. That is what this document's first result covers.

---

<a name="part-1"></a>
## Part 1 — Background you need (weights, layers, checkpoints)

### 1.1 What is a "weight"?
A neural network is, concretely, a big collection of **numbers arranged in matrices** (grids of numbers).
These numbers are called **weights** or **parameters**. When the network processes an input, it multiplies
the input by these matrices (plus some additions and nonlinear functions). **Learning = changing these
numbers.** Our model, Qwen3-VL-4B, has ~4.4 **billion** of them, organized into **713 named tensors**
(matrices/vectors), e.g. `model.language_model.layers.5.mlp.gate_proj.weight` is one such matrix.

### 1.2 What is a transformer "layer", and what are "attention" vs "MLP"?
The language part of our model is a stack of **36 identical-in-shape blocks** ("transformer layers" /
"decoder layers"), numbered 0 (bottom, closest to the input) to 35 (top, closest to the output). Each
block has two main sub-parts:

- **Attention (`self_attn`)** — lets each token (word/piece) *look at and pull information from other
  tokens*. This is the "mixing across positions" machinery. Parameters: `q_proj, k_proj, v_proj, o_proj`
  (four big matrices), plus in Qwen3 two small normalization vectors `q_norm, k_norm`.
- **MLP (`mlp`, a.k.a. feed-forward)** — processes *each token independently*, transforming its
  representation through a wider hidden space. Parameters: `gate_proj, up_proj, down_proj` (three big
  matrices). The MLP is widely believed (in interpretability research) to be where the model stores and
  retrieves **factual / feature knowledge** — which is *exactly* why the senior's "the fix is in the MLP"
  claim is interesting: it would mean RL edits the model's *knowledge/feature readout*, not its
  *attention/routing*.

There are also tiny **normalization** vectors (`input_layernorm`, `post_attention_layernorm`) that just
rescale activations. Above the 36 blocks sits a final `norm` and the output projection (which in our model
is **tied** to the input word-embeddings — see 1.5).

The **vision tower** (`model.visual`) is a separate, smaller transformer (24 blocks) that converts the
image into a sequence of "image tokens" the language model can read.

> 🌐 **Generalizes:** every modern transformer decouples *position-mixing* (attention) from
> *per-token feature transformation* (MLP). "Where did learning happen — attention or MLP?" is a
> standard mechanistic-interpretability question.
> 📦 **Ours:** Qwen3-VL-4B = 36 LLM layers (attn: q/k/v/o + q_norm/k_norm; mlp: gate/up/down) + a 24-block
> vision tower. Confirmed exactly by our parameter census (Part 3.2).

### 1.3 Base model vs. trained model
- **Base model** = Qwen3-VL-4B-Instruct, downloaded, *before any of our training*. Its weights are
  `W_base`. We treat this as **"step 0"**.
- **Trained model** = the same model *after* RL. Its weights are `W_trained`.
- **What the model "learned" is entirely captured by the difference** `Δ = W_trained − W_base`. Studying
  Δ is studying the learning itself. This is the foundation of the weight-delta analysis.

### 1.4 What is a "checkpoint", and why do we have 16?
A **checkpoint** is a saved copy of all the weights at a moment during training. We configured
`save_freq=6` (RESULTS.md §6), meaning **every 6 training steps** the trainer dumps the full model to disk.
Over 96 steps that gives checkpoints at steps 6, 12, 18, …, 96 → **16 checkpoints**. This lets us watch the
weights move *over time* (the **trajectory**), not just compare start vs end. The senior only had 2 points
(base + final); we have 17 (base + 16). That extra time-resolution is one of our contributions.

### 1.5 Two model-specific facts that shape the analysis
1. **Sharded checkpoints (because we trained on 4 GPUs).** We used **FSDP** (Fully Sharded Data Parallel),
   which *splits* each weight matrix into 4 pieces, one per GPU, to fit memory. So each checkpoint is saved
   as **4 files** (`model_world_size_4_rank_{0,1,2,3}.pt`), each holding one slice of every weight. To
   analyze a weight we must **stitch the 4 slices back together** (Part 2.3). This is a direct consequence
   of our hardware choice (4×GH200 instead of the paper's 8×H200).
2. **Tied embeddings.** Qwen3-VL-4B uses `tie_word_embeddings=true`: the matrix that turns word-IDs into
   vectors at the input (`embed_tokens`) is the **same matrix** (transposed) used to turn vectors back into
   word-scores at the output (`lm_head`). So the base model stores it **once** (713 tensors, no separate
   `lm_head`). The checkpoint redundantly re-saves it as `lm_head.weight` (714 tensors); since it's identical
   to `embed_tokens`, our analysis correctly **ignores the duplicate** (the "1 extra ckpt key" warning you saw).

---

<a name="part-2"></a>
## Part 2 — Analysis #1: Weight-Delta (`weight_delta.py`, tests S2)

### 2.1 The question, made precise
S2 says the change is "tiny, in late-layer MLPs, not attention." To test that we must, for **every one of
the 713 weight tensors**, measure **how much it changed** from base to trained, then **group** those changes
by (a) which component — vision vs LLM, (b) which module — attention vs MLP vs norm, (c) which layer index
(0–35), and ask: *is the MLP change bigger than the attention change, and does it grow toward the late
layers?*

### 2.2 How we measure "how much a matrix changed" — the four metrics

For a single weight tensor, let `W_base` be its base value and `W_ckpt` its checkpoint value, and let
`Δ = W_ckpt − W_base` be the change. We "flatten" each matrix into one long list of numbers and use:

**(a) Frobenius norm of the change, `abs_fro = ‖Δ‖_F`.**
The Frobenius norm of a matrix is simply **the square root of the sum of the squares of all its entries** —
i.e. treat the matrix as one long vector and take its ordinary (Euclidean) length:
```
‖Δ‖_F = sqrt( Σ_ij Δ_ij² )
```
So `abs_fro` is the **total geometric distance** the weight moved. Bigger = moved more.

**(b) Relative Frobenius norm, `rel_fro = ‖Δ‖_F / ‖W_base‖_F`.** ← **our primary metric**
Raw `abs_fro` is unfair to compare across tensors: a big matrix has a big norm just because it has more
numbers. So we **divide by the original matrix's norm** to get the **fraction** by which it moved.
```
rel_fro = ‖Δ‖_F / ‖W_base‖_F
```
*Worked example:* a 100×100 matrix whose entries are all ≈0.1 has `‖W_base‖_F = sqrt(10000 × 0.1²) = 10`.
If every entry shifts by 0.0005, then `‖Δ‖_F = sqrt(10000 × 0.0005²) ≈ 0.05`, so `rel_fro = 0.05/10 = 0.005`
= **0.5% of the matrix's size**. This is the natural "what fraction did it move" number, comparable across
tensors of any shape. **All our headline numbers are rel_fro.**

**(c) Mean absolute change, `mean_abs = mean(|Δ|)`.** The average size of a single entry's change — a
human-readable "typical per-number nudge."

**(d) Cosine similarity, `cos_sim = cos(vec(W_base), vec(W_ckpt))`.** Treats the whole matrix as a vector
and measures the **angle** between before and after. `cos_sim ≈ 1.0` means the weight barely changed
*direction* (it was nudged, not rebuilt). This is the complementary view to rel_fro (magnitude): together
they say "tiny nudge in nearly the same direction."

> 🌐 **Generalizes:** Frobenius distance + relative-Frobenius + cosine is the standard toolkit for
> "how far did the weights move" in any fine-tuning / model-diffing study.
> 📦 **Ours:** we compute all four per-tensor and store them long-format, so any later grouping
> (by module, by layer, by condition, by step) is a simple filter.

### 2.3 How we reconstruct a weight from 4 GPU shards (the engineering core)
Because of FSDP (1.5), each weight lives in 4 pieces across `rank_0..rank_3.pt`. Each piece is a **DTensor**
("distributed tensor") that remembers *how* it was split (its "placement": either **Shard(dim)** = this is
slice along dimension `dim`, or **Replicate** = every rank has a full copy). To get the full weight we:
1. read the same key from all 4 rank files,
2. if **Replicate** → just take rank 0's copy,
3. if **Shard(d)** → **concatenate** the 4 slices back along dimension `d`.

This mirrors the official `scripts/model_merger.py` and runs **single-process on CPU** — no GPUs, no
distributed setup needed. We do it **one tensor at a time** (reconstruct → measure → free) so memory stays
bounded (~9 GB), which is why the whole job runs as a small CPU sbatch in minutes.

### 2.4 How we label each weight (the classifier, `qwen3vl_param_map.py`)
For each tensor name we assign `(component, module, layer_idx)` by pattern-matching the name, e.g.
`model.language_model.layers.27.mlp.gate_proj.weight` → `(llm, mlp, 27)`. This is the **grouping key** that
lets us average rel_fro over "all MLP tensors in late layers" vs "all attention tensors in late layers".
**Crucially, the same classifier is reused by the causal-graft test (module_graft.py)** so the MLP-vs-attn
boundary is *identical* across the correlational (S2) and causal (S3) tests — otherwise the two results
wouldn't be comparable.

### 2.5 How the training choices *predetermine* what we'll see
Before looking at results, note what our **RL setup** implies for weight deltas:
- **Learning rate = 1e-6** (RESULTS.md §3) — extremely small. Each gradient step moves weights by a
  minuscule amount. Over 96 steps the total move is necessarily small. → **We should expect tiny rel_fro.**
- **KL-regularization toward the base model** (`use_kl_loss=true`, `kl_coef=1e-2`, RESULTS.md §2) — the loss
  *actively penalizes drifting away from the base model's behavior*. This is a leash that keeps `W_trained`
  near `W_base` *by design*. → Another reason to expect **small** deltas, and a near-1 cosine.
- **GRPO, 96 steps, Stage-1-only** — short training, single capability (perception). → We're seeing an
  *early, perception-only* snapshot of learning, not the full multi-stage curriculum the senior analyzed.

So if we see "tiny weight change," that is **not a surprise or an artifact** — it is the *expected*
fingerprint of low-LR, KL-leashed RL, and it is exactly the regime in which the senior's "tiny surgical
edit" claim is meaningful.

---

<a name="part-3"></a>
## Part 3 — RESULT #1: Weight-Delta on Condition 1 (full LLM+ViT), 2026-06-29

**Command:** `run_weight_delta.sh full` looped all 16 checkpoints → `deltas.csv` (713 × 16 = 11,408 rows).
**Readout:** `summarize_deltas.py --csv deltas.csv`.

### 3.1 Parameter census (sanity that we're measuring the right thing)
The classifier tagged **all 713 tensors with zero unclassified**, and the counts exactly match the known
architecture (proof the labeling is correct, not approximate):

| Component / module | count | arithmetic |
|---|---|---|
| llm / attn | 216 | (q,k,v,o + q_norm,k_norm) = 6 × 36 layers |
| llm / mlp | 108 | (gate,up,down) = 3 × 36 |
| llm / norm | 73 | 2 × 36 + 1 final norm |
| embed | 1 | tied input/output embedding |
| vision / attn | 96 | (qkv w+b, proj w+b) = 4 × 24 blocks |
| vision / mlp | 96 | (fc1 w+b, fc2 w+b) = 4 × 24 |
| vision / norm | 96 | (norm1 w+b, norm2 w+b) = 4 × 24 |
| vision / merger, patch_embed, other | 27 | merger(24) + patch_embed(2) + 1 |
| **total** | **713** | ✓ |

### 3.2 The numbers (step 96 = final)

**Freeze / component check — did both parts train?**
| component | mean rel_fro | max rel_fro | n tensors |
|---|---|---|---|
| vision | 3.66e-04 | 3.57e-03 | 315 |
| llm | 4.86e-04 | 1.09e-03 | 397 |
| embed | 3.15e-04 | — | 1 |

**S2 localization — LLM, mean rel_fro by module × layer-band:**
| band | attn | mlp | mlp/attn |
|---|---|---|---|
| early (layers 0–11) | 4.94e-04 | 7.86e-04 | **1.59×** |
| mid (layers 12–23) | 5.71e-04 | 7.98e-04 | **1.40×** |
| late (layers 24–35) | 4.84e-04 | 6.64e-04 | **1.37×** |

**S5 dynamics — late-layer LLM/mlp rel_fro across training:** rises **smoothly and monotonically**
from 9.3e-05 (step 6) → 6.6e-04 (step 96), roughly linearly.

### 3.3 What each finding means, connected to our setup

**Finding A — the weight change is TINY. (S2, first half: CONFIRMED.)**
Mean rel_fro ≈ **5e-4 = 0.05%**. The LLM's weights moved by *five-hundredths of one percent* of their size.
The single most-changed tensor moved only 0.1% (`llm max = 1.09e-3`). This is a **strong, clean confirmation
of the senior's "tiny change" claim** — and, as Part 2.5 predicted, it is the direct fingerprint of our
**lr=1e-6 + KL leash**. The striking scientific point to put in front of your bosses:

> **A 0.05% change in the weights nearly DOUBLED perception accuracy (0.365 → 0.746).**
> RL is not rebuilding the model's vision; it is making a *minuscule, targeted adjustment*. This is the
> empirical heart of the senior's S5 intuition ("re-access, not re-learning") and of our broader thesis
> (the perception information is *already in the model*; training just changes how it's read out).

**Finding B — the change is bigger in MLPs than in attention, at every depth. (S2, "MLP not attention":
DIRECTIONALLY CONFIRMED.)**
MLP tensors moved **1.37×–1.59× more** (relative) than attention tensors in every band. So on our model too,
RL preferentially edits the **MLP / feed-forward** machinery — the part associated with feature/knowledge
readout — over the **attention / routing** machinery. This matches the *direction* of the senior's claim.

**Finding C — the change is NOT concentrated in late layers. (S2, "late-layer" part: NOT reproduced here.)**
The senior found the MLP edit piling up in the *top* layers. We find the opposite-ish: MLP rel_fro is
**roughly flat, even slightly higher in early/mid layers** (early 7.86e-4 ≥ late 6.64e-4). We **report this
honestly as a divergence.** The most likely reasons, all tied to our deliberate scope choices:
- **Different training regime.** The senior analyzed the **full 3-stage curriculum (~930 steps, 8B model)**.
  We analyzed **Stage-1 only (96 steps, 4B model)**. Late-layer specialization may be something that emerges
  *later* in training or specifically during the *reasoning* stages (2/3), not the perception stage.
- **Much earlier point on the trajectory.** Our total move is still tiny and growing roughly uniformly
  (Finding D); a late-layer *concentration* may simply not have formed yet at step 96.
- **Model scale.** Depth-localization patterns can differ between 4B (36 layers) and 8B (more layers).

This is not a failure — it's a *precise, publishable* statement: *"On a 4B model after 96 steps of
perception-only RL, the update is MLP-biased but depth-uniform; the senior's late-layer concentration does
not yet appear in this regime."* It also sets up a clean follow-up question for Stages 2/3.

**Finding D — training is smooth and stable. (process check.)**
The late-MLP rel_fro grows **monotonically and almost linearly** (no spikes, no plateaus, no collapse) from
step 6 to 96. This tells us the optimization was healthy — the model drifted steadily away from base, never
diverged or oscillated. It is consistent with the steadily-rising accuracy curve (RESULTS.md §9). Note: this
*time-growth* is **not** itself evidence of S5 "re-surfacing" — it only shows *that* weights moved over time,
not *what that movement did to the model's internal perception*. Testing re-surfacing requires the
**depth-probe** analysis (S4/S5), which reads *activations*, not weights.

---

<a name="part-4"></a>
## Part 4 — Honest caveats & limitations of this specific result

A professional presentation states these up front:

1. **The attention bucket mixes big projections with tiny norm vectors.** Our `attn` group includes the four
   large matrices (q/k/v/o) **and** the two small RMSNorm gain vectors (q_norm/k_norm). RMSNorm gains can have
   very different rel_fro behavior; averaging them in could bias the MLP/attn ratio. **Planned refinement:**
   recompute the S2 ratio over *projection matrices only* (q/k/v/o vs gate/up/down), excluding all norms.
   We expect the MLP>attn direction to survive, but the exact ratio may shift.
2. **rel_fro is averaged unweighted across tensors.** A large matrix and a small one count equally in the
   mean. A parameter-count-weighted version (weight each tensor by its number of entries) answers a slightly
   different question ("where did the most *total* movement go") and is worth reporting alongside.
3. **This is the `full` condition only.** The real S2 story needs the **three conditions overlaid**
   (`full`, `llm_only`, `vit_only`). Those runs also serve as the **weight-level freeze proof**: `llm_only`
   must show vision rel_fro ≈ 0, `vit_only` must show llm rel_fro ≈ 0. Until then, the freeze is only
   confirmed behaviorally (log message + param counts + grad norms), not yet at the weight level.
4. **Correlation, not causation.** Weight-delta shows *where the weights moved*. It does **not** show that
   moving the MLPs is what *caused* the perception gain. The senior's S3 graft test (`module_graft.py`) —
   write only-MLP deltas onto the base and see if perception recovers — is what establishes causation. That
   is the next analysis to build.

---

<a name="part-5"></a>
## Part 5 — What this proves, what it doesn't, and the next steps

**Proven on our own model, today, directly from Stage-1 data:**
- ✅ The RL weight change is **tiny** (0.05%) yet behaviorally large (≈2× accuracy) — S2 first half + the core
  "surgical edit" thesis.
- ✅ The change is **MLP-biased over attention** at every depth (1.37–1.59×) — S2 direction.
- ✅ Training was **smooth and stable** (monotonic drift).

**Not (yet) supported / open:**
- ❌ **Late-layer concentration** (S2 depth part) — not present in this 4B/96-step/perception-only regime;
  honestly reported as a divergence, with regime-difference as the leading explanation.
- ⏳ **Causation** (S3) — needs the module-graft test.
- ⏳ **Re-surfacing** (S4/S5) — needs the depth-probe (reads activations, not weights).
- ⏳ **Cross-condition / freeze-at-weight-level** — needs `llm_only` & `vit_only` delta runs.

**Immediate next actions:**
1. When **Condition 2 (llm_only)** finishes → run its weight-delta. Expect **vision rel_fro ≈ 0** (frozen ViT)
   = the gold-standard freeze proof, plus a second S2 profile to overlay.
2. Same for **Condition 3 (vit_only)** → expect **llm rel_fro ≈ 0**.
3. Build **`module_graft.py`** (S3 causation) and **`depth_probe.py`** (S4/S5 re-surfacing) using babyVision
   (388 labeled items) as the probe set.
4. Refinement: projections-only S2 ratio + parameter-weighted variant (Part 4.1–4.2).

---

*Last updated: 2026-06-29 after Condition-1 weight-delta. This document is appended to as each analysis lands.*
