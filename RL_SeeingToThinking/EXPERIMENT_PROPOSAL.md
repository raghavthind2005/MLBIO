# Experiment Proposal — Where Perception Lives Under RL Reasoning

**Author:** Raghav Thind · **Date:** 2026-06-24 · **Status:** for review / approval
**Engine:** EasyR1 (verl fork) GRPO · **Backbone:** Qwen3-VL-4B · **Hardware:** 1× Clariden node (4×GH200, aarch64)

---

## 1. One-line goal

Understand **how perception behaves as reasoning proceeds** — specifically, to characterize and combat the *degradation of perception utility as a VLM reasons longer* — and to localize **where** in the network RL fixes perception.

This builds on the UCSC-VLAA "Seeing→Thinking" paper (arXiv:2605.20177): 86.9% of VLM errors are perception errors that longer reasoning cannot fix, and staged RLVR (perception → text-reasoning → visual-reasoning) repairs perception better than caption SFT.

---

## 2. Motivating finding (senior's analysis) → our hypothesis

A senior compared the paper's **base vs. staged-RL** checkpoint (released on HF) and reported **five distinct claims**:

| ID | Claim | Type |
|---|---|---|
| **S1** | Representation **geometry and modality gap are unchanged** base vs. trained | control / null |
| **S2** | The weight change is **tiny and concentrated in late-layer MLPs**, *not* attention | localization |
| **S3** | **Causal graft:** writing back **only the MLP** deltas onto the base → perception improves; **only attention** → no improvement | causal |
| **S4** | In the **base** model, perception is **more decodable at mid-layers and degrades toward the top** | precondition |
| **S5** | *(the hypothesis proper — his "feeling")* the late-MLP edit **adds no new information; it re-surfaces mid-layer perception that the upper layers were destroying** | mechanism |

**Working hypothesis H = S5:** perception is computed mid-stack and degraded on the way up; RL's gain is a late-MLP edit that **stops that degradation / re-surfaces mid-layer evidence**. S2+S3 say *what/where* the fix is; S4 is the *precondition* that makes "re-surfacing" meaningful; S1 is the *control* ruling out "it just reshaped the representation." This *is* our research question stated mechanistically — "perception degrading as reasoning goes by" = perception degrading as signal flows **up the residual stack**.

**Important scoping note:** S1–S4 were derived on the paper's *released* checkpoints and are **fully offline weight/activation analyses — confirmable today, no training required** (see §8). Our training run's value is to test whether the **same mechanism arises when *we* do the RL** (4B / Thinking / Stage-1), **across the trajectory** (he had only 2 points), and **under controlled freezes** — i.e. we both *reproduce* and *extend* him, testing H **prospectively** (by what we let train) rather than only post-hoc.

**Caveat on the freeze conditions (§5):** full / LLM-only / ViT-only cuts **LLM vs. vision encoder** — a *coarser* axis than S2/S3's **MLP vs. attention *inside* the LLM**. The conditions corroborate the "it's in the LLM, not the encoder" half; **only the offline graft (§8.5) resolves MLP-vs-attention.** An optional finer condition (train MLP-params-only vs. attention-only) would convert S2/S3 into a *forward* causal test — see §5.

---

## 3. Current state (preliminaries — DONE)

- ARM EasyR1 container built + verified: `easyr1_vllm0112.sqsh` (torch 2.9.0+cu129, vLLM 0.11.2, transformers 4.57.3, flash-attn 2.8.3, ray 2.55.1). EDF `~/toml/verl_easyr1.toml`.
- **Full GRPO loop validated** on Qwen3-VL-4B-Thinking, 4×GH200 (Ray + vLLM rollout + FSDP 4-GPU update + checkpoint). 2 steps / ~68s on synthetic data.
- EasyR1 (main) + VLM-CapCurriculum on cluster scratch; reward fn / prompt / config in place.
- **Not yet downloaded:** paper's 3 HF datasets (Perception / TextReasoning / VisualReasoning). This is the next practical step once scope is approved.

---

## 4. Scope decision A — which stage(s)? (Stage 1 vs Stage 3, not all three)

We propose running **one stage**, not the full 3-stage curriculum, to keep the campaign cheap and interpretable.

| | **Stage 1 — Visual Perception** (recommend) | **Stage 3 — Visual Reasoning** |
|---|---|---|
| Reward signal | perception correctness (direct) | reasoning-over-image correctness |
| Cost (paper) | ~90 steps (cheapest) | ~465 steps (most expensive) |
| Isolates perception? | **Yes** — perception *is* the objective | No — perception + reasoning entangled in reward |
| Tests hypothesis H? | **Directly** (perception is the lever) | Indirectly |
| Where degradation lives | n/a (we *fix* perception here) | this is where long reasoning *causes* the drop |

**Recommendation: Stage 1 as the primary run.** Rationale:
1. It is the cheapest and isolates perception as the learning signal — the cleanest probe of H and of the 3 conditions below.
2. **Because we use a *Thinking* backbone, even Stage-1 rollouts contain long reasoning chains.** So we still observe the "accuracy vs. reasoning-length" degradation signal *within* Stage 1 — we get the phenomenon and the fix in one run.
3. The "does reasoning training harm perception" angle is then obtainable cheaply as **eval-only** (run saved Stage-1 checkpoints on visual-reasoning benchmarks; no Stage-3 training needed for v1).

> Open question for you: accept Stage-1-only for v1, or do you want a Stage-3 contrast run (≈5× cost)?

---

## 5. Scope decision B — three finetuning conditions

Run Stage 1 under three freeze regimes. This is a **forward (training-time) test of the senior's post-hoc copy result**.

| Condition | Trains | EasyR1 support | Prediction under H |
|---|---|---|---|
| **LLM + ViT** (full) | vision tower + LLM | native (`freeze_vision_tower=false`) — paper's Stage-1 setting | upper bound |
| **LLM only** | LLM (ViT frozen) | native (`freeze_vision_tower=true`) | **≈ full** — confirms fix lives in the LLM (late MLPs) |
| **ViT only** | vision tower (LLM frozen) | **not a native flag** — needs custom `requires_grad=False` on LLM params ⚠️ | weak — encoder alone shouldn't recover the gain |

**If `LLM-only ≈ full ≫ ViT-only`, H is supported**: the perception repair is an LLM-internal (late-MLP) operation, not better encoding. This is the headline result the campaign is built to produce.

> ⚠️ ViT-only requires verifying we can freeze the LLM in EasyR1 (no stock flag). Fallback if infeasible: drop ViT-only and report LLM-only vs full only.

> **Optional 4th/5th condition (forward test of S2/S3):** within the LLM, train **MLP-params-only** vs **attention-params-only** (custom `requires_grad` masks). If MLP-only recovers the perception gain and attention-only doesn't, that confirms the senior's MLP localization *prospectively*, not just by post-hoc graft (§8.5). Higher implementation cost; propose as a v2 add-on once the freeze mechanism is proven.

---

## 6. Backbone — Instruct vs. Thinking (what we have vs. the paper)

- Paper used **Qwen3-VL-8B-Instruct**. On the cluster we have **Qwen3-VL-4B-Thinking** (smaller + thinking-tuned).
- **Why Thinking is a feature for us:** our object of study is perception-under-reasoning. A Thinking model reasons by default → it is exactly the regime where "perception degrades as reasoning goes by" should be strongest. Stage-1 perception MCQs answered with long CoT give us the degradation signal for free.
- **Why it is also a deviation to watch:**
  - The format reward (`math.py:format_reward`) requires the *literal* string `<think>…</think> … \boxed{}`. Qwen3-Thinking may emit its reasoning via its own template/channel, not literal `<think>` text → format reward could misfire. **Accuracy reward only needs `\boxed{}` (robust).** Mitigation: verify decoded rollout format on a few samples before the full run; relax `format_weight`/regex if needed (we already debugged Thinking-output parsing in babyVision).
  - 4B < 8B: weaker base perception → more RL headroom, but noisier.

> Open question for you: proceed on 4B-**Thinking** (aligns with the reasoning-degradation theme), or should we request **4B/8B-Instruct** for a cleaner paper-faithful repro?

---

## 7. Single-run instrumentation — "train once, analyze forever"

Goal: capture enough during each run that **every downstream analysis is offline** (no re-training). Most depth analysis runs on saved checkpoints, so the run itself only needs rich logging + frequent full checkpoints + rollout dumps.

| Bucket | What to collect | How |
|---|---|---|
| **A. Step metrics** | reward (overall/accuracy/format), KL, entropy, grad-norm, response length (mean/max), clip-frac, per-group pass-rate | EasyR1 native; `logger=["file","wandb"]` (wandb offline) |
| **B. Checkpoints** | **full model**, frequent (`save_freq` small, `save_limit=-1` if disk allows), incl. step-0 base | enables all offline probing + weight-delta analysis |
| **C. Rollout traces** | generated responses + per-sample reward components + group pass-rate, dumped periodically (not just `val_generations_to_log=3`) | the actual reasoning chains; needed for §8.1/8.2 |
| **D. Frozen probe set** | one fixed held-out perception set, **labeled** (answers retained so layer-wise linear probes can be fit), scored at **every checkpoint** | perception-accuracy trajectory + S4/S5 depth probing |
| **E. Activations** | hidden states per layer on the probe set | offline on checkpoints — no inline cost |

Inline cost is therefore low; the expensive part (probing, geometry, weight deltas) is fully offline on B/D/E.

---

## 8. Offline analyses these enable (the actual science)

1. **Perception vs. reasoning-length** (cheap, from C alone): bin rollouts by CoT length → does longer reasoning lower perception accuracy *at fixed model/step*? Direct measure of the degradation phenomenon. *(Mirrors the babyVision/HallusionBench result.)*
2. **Depth-resolved perception probing across training → tests S4 + S5** (from B/D/E): fit a layer-wise linear probe on the labeled set at *every* checkpoint. S4 = base model shows mid-layer peak, top-layer decay; S5 = RL **flattens the top-layer decay** over the trajectory (he had only base vs. final).
3. **Weight-delta localization → tests S2** (from B): per-layer, per-module (MLP vs. attention) ‖ckpt − base‖ → confirm changes concentrate in late MLPs; compare concentration across the three conditions (§5).
4. **Perception "utility" vs. reasoning position** (from B + E, offline replay): attention mass on image tokens / image-ablation Δ as a function of token position in the CoT → does the model stop *using* the image as it reasons, and does RL change that?
5. **Causal module graft → tests S3 (the senior's key experiment)** (from B, offline, no training): take a trained checkpoint, write back **only the MLP** deltas onto the base weights and eval the probe set; then **only the attention** deltas and eval. Predict MLP-graft recovers most of the perception gain, attention-graft ≈ none. This — not the freeze conditions — is what resolves MLP-vs-attention.
6. **Geometry / modality gap → tests S1** (from E): CKA + modality-gap base-vs-trained, reproducing the "geometry unchanged" control → confirms our deltas are *functional*, not representational.
7. **Transfer eval (Stage-3 question, no Stage-3 training)**: score Stage-1 checkpoints on visual-reasoning benchmarks → did fixing perception alone move downstream reasoning?

> **Reproduce-first:** analyses 2/3/5/6 can be run **right now on the paper's released base + staged checkpoints** to confirm S1–S4 on the original 8B model before we spend any GPU hours — cheap de-risking and a direct check that our analysis code reproduces the senior.

---

## 9. Approval checklist (decisions I need from you)

1. **Stage:** Stage-1-only for v1 (recommended), or add a Stage-3 contrast (~5× cost)?
2. **Conditions:** all three (full / LLM-only / ViT-only), accepting the ViT-only freeze risk — or full + LLM-only only?
3. **Backbone:** 4B-Thinking (theme-aligned, what we have) or request an Instruct variant for paper fidelity?
4. **Probe set for D:** reuse babyVision, or do you want a specific perception benchmark?
5. **Reproduce-first:** want me to run the S1–S4 analyses on the paper's *released* base/staged checkpoints before our run (cheap, validates our analysis code against your result)?
6. **Forward MLP-vs-attn test:** include the optional MLP-only/attention-only freeze (§5) in v1, or defer to v2?
7. Anything in §7/§8 you'd add now so we never have to re-run.

## 10. Known risks

- **ViT-only freeze** not natively supported in EasyR1 (§5 ⚠️) — needs a code check before committing to 3 conditions.
- **Thinking-format reward** mismatch (§6) — verify rollout format pre-run; accuracy reward is safe regardless.
- **4B≠8B / Thinking≠Instruct** — results are about *our* model's mechanism, not a 1:1 paper reproduction (acceptable for a mechanism study; flagged for fidelity-minded reviewers).
- **Checkpoint disk** — frequent full saves of a 4B model add up; confirm scratch quota before `save_limit=-1`.
