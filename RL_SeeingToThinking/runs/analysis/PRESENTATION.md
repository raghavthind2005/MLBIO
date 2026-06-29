# How RL Fixes Perception in a Vision-Language Model — and How We Know

> **Purpose.** A presentation companion. For every result it explains, from scratch but precisely,
> **what we found** *and* **how we computed it** — so a newcomer can follow the logic and an expert can
> check the mechanics. Reading order is linear; each section = one method + one finding.
>
> Source-of-truth lab log: `FINDINGS.md`. Training config: `RESULTS.md`. Code: `runs/analysis/`.

---

## 0. The setup (what we're even looking at)

**The model.** Qwen3-VL-4B-Instruct — a *vision-language model* (VLM): it takes an **image + a text
question** and produces a **text answer**. Internally it has two parts:
- a **vision tower** (~0.42B parameters, 24 transformer blocks) that turns the image into a sequence of
  "image tokens," and
- a **language model / LLM** (~4.0B parameters, **36 transformer layers**) that reads those image tokens
  plus the question and writes the answer.

**The training.** We post-trained it with **Reinforcement Learning with Verifiable Rewards (RLVR)**, using
the GRPO algorithm, on *perception* multiple-choice questions: the model answers, gets **+reward if correct,
0 if wrong**, and the optimizer nudges the weights to make correct answers more likely. Result: perception
accuracy climbed **0.365 → 0.746**.

**The question this project answers.** *Where, and how, does that improvement live inside the 4.4 billion
numbers of the model?* Did it learn to **see** better (vision tower), or to **read what it already saw**
better (LLM)? Is the change big or small? Spread out or localized? And is it *causal*?

**The experimental lever — three training conditions**, identical in every setting except which part is
allowed to change:
| condition | what trains | what's frozen |
|---|---|---|
| **full** | vision tower + LLM | nothing |
| **llm_only** | LLM | vision tower |
| **vit_only** | vision tower | LLM |

Comparing these isolates *which component* carries the fix.

---

## 1. Foundations you need (5 concepts, then everything follows)

**(a) A "weight" / "parameter."** A neural network is, concretely, billions of **numbers arranged in
matrices**. When it processes an input it multiplies the input by these matrices (with some additions and
nonlinearities). **Learning = changing these numbers.** Our model has ~4.4B of them, grouped into **713
named matrices** (e.g. `...layers.5.mlp.gate_proj.weight`).

**(b) A transformer "layer," and "attention" vs "MLP."** The LLM is a stack of **36 layers**, numbered 0
(bottom, near the input) to 35 (top, near the output). Each layer has two engines:
- **Attention** — lets each word/token *look at other tokens and pull information across positions* (the
  "routing / mixing" machinery).
- **MLP** (feed-forward) — processes *each token on its own*, transforming its representation through a wider
  space. Interpretability research associates the MLP with **stored features / knowledge readout**. (This is
  why "is the fix in the MLP?" is a meaningful question — it would mean RL edits the *knowledge readout*, not
  the *routing*.)

**(c) The "residual stream."** Each layer *adds* its output to a running vector that flows up the stack. So
the representation at layer L is the accumulation of everything below it. "Information flows up the residual
stream" is literal.

**(d) Base model vs checkpoints.** The **base model** = the weights *before* our training (call it "step 0").
A **checkpoint** = a saved copy of all weights *during* training; we saved one every 6 steps → **16
checkpoints** (steps 6, 12, …, 96). Everything the model "learned" is captured by the **difference** between
trained weights and base weights. Studying that difference *is* studying the learning.

**(e) Why checkpoints come in 4 pieces (and how we reassemble them).** We trained on **4 GPUs** using FSDP
(Fully Sharded Data Parallel), which *splits* every weight matrix into 4 slices, one per GPU, to fit memory.
So each checkpoint is 4 files. To analyze a weight we **stitch the 4 slices back together** by concatenating
them along the split dimension — done once, on a CPU, no GPUs needed. (This is the engineering backbone of
every analysis below.)

---

## 2. Investigation #1 — *Where did the weights change?* (the "weight-delta")

### The question
Of the 713 matrices, **which ones moved**, by **how much**, and are the movers concentrated in the **MLP** or
the **attention**, in **early** or **late** layers?

### How we computed it — from scratch
1. **Get before and after.** Load the base weights `W_base` and a checkpoint's weights `W_ckpt` (reassembling
   the 4 FSDP slices, §1e).
2. **Measure how much a matrix moved.** For each matrix, the change is `Δ = W_ckpt − W_base`. To put a single
   number on "how big is Δ," we use the **Frobenius norm** — *treat the matrix as one long list of numbers and
   take its ordinary length*: `‖Δ‖ = sqrt(sum of every entry squared)`. Bigger = moved more.
3. **Make it comparable across matrices.** A big matrix has a big norm just from having more numbers, so we
   divide by the original size: **relative change `= ‖Δ‖ / ‖W_base‖`** — *"what fraction of itself did this
   matrix move?"* (Worked example: a matrix of size ~10 that shifts by ~0.05 has relative change 0.005 = 0.5%.)
4. **Tag every matrix** by which part of the model it belongs to — *vision* vs *LLM*, *attention* vs *MLP* vs
   *norm*, and *which layer number*. Now we can average the relative change over, say, "all MLP matrices in
   late layers" and compare to "all attention matrices in late layers."

### What we found
- **The change is astonishingly tiny: ~0.05% on average** (largest single matrix: 0.1%). Yet that 0.05%
  nudge nearly **doubled** perception accuracy.
- **MLP matrices moved ~1.4–1.6× more than attention matrices**, at every depth.
- The change was **roughly uniform across depth** (not bigger in late layers, in terms of *size*).

### What it means
RL did **not** rebuild the model; it made a **tiny, surgical edit**, biased toward the MLP. The tininess is
exactly what our training settings predict (a very small learning rate plus a regularizer that keeps the model
close to its starting point) — so this is the expected fingerprint of a *gentle* edit, not a rewrite. This is
the first hint that *the perception ability was largely already in the model* and RL just adjusted how it's used.

---

## 3. Investigation #2 — *Which component carries the fix?* (the freeze ablation)

### The question
Is the improvement about **seeing better** (vision tower) or **reading better** (LLM)?

### How we computed it — from scratch
We ran the *same* training **three times**, byte-identical except the freeze flags (full / llm_only /
vit_only). Two checks:
1. **Behavioral:** compare final accuracy across conditions.
2. **Weight-level "freeze proof":** re-use the weight-delta (Investigation #1) — a frozen part must show
   relative change **exactly 0** (its weights literally never updated).

We also verified the three runs were truly identical except the freeze (same data seed, learning rate,
batch sizes, etc.) by diffing their configs.

### What we found
- **`llm_only` reached the same accuracy as `full`** (0.749 vs 0.746) — freezing the vision tower **cost
  nothing**.
- The weight-level proof was perfect: in `llm_only`, **every vision-tower matrix had relative change exactly
  0.000** (bit-identical to base) while the LLM moved normally.
- (`vit_only` — training only the vision tower — is the final comparison, completing soon.)

### What it means
The perception fix is an **LLM operation, not better seeing**. Training the LLM alone reproduces the entire
gain; the vision tower contributes (essentially) nothing. This is direct support for *"re-access, not
re-representation"* — the image features were already good enough; the bottleneck was downstream, in the LLM.

---

## 4. Investigation #3 — *Where in the 36 layers does the answer become readable?* (the "logit-lens")

### The question
As information flows up the 36 layers, **at which depth is the correct answer actually decodable**, and how
does training change that?

### How we computed it — from scratch
1. **Hidden states.** As the model processes an item, each layer produces a **hidden state** — a vector that
   is the model's internal "notes" at that depth.
2. **The output head.** Normally only the *top* layer's notes get turned into an answer, via the model's
   **output head** (a matrix that maps an internal vector to a score for every possible next word).
3. **The logit-lens trick.** Take the notes from *any* layer L and run them through that **same output head**
   — as if the model had to answer right there. Read off the probability it assigns to each option letter
   (A/B/C/D). If the correct letter scores highest, the answer is **"readable" at layer L**.
4. Do this at **every layer** → a curve of *decodability vs. depth*. Run it for the base model and the trained
   model and overlay them. (No extra training needed — we use the model's own head; this is the standard
   "logit-lens." Caveat: the head is tuned for the top layer, so *middle*-layer readings are approximate — we
   rely on the late layers and on base-vs-trained *differences*.)

### What we found
| layers | base | trained |
|---|---|---|
| 0–23 (early/mid) | identical to trained | identical to base |
| ~24–25 (the split) | recovers to ~0.37 | **jumps to ~0.62** |
| 36 (final) | 0.377 | **0.657** |

- **Layers 0–23 are identical** in base and trained — RL leaves the early/mid representations untouched.
- **The two curves diverge sharply at layer ~24–25** and stay split: the trained model reads the answer far
  better in the **late** layers.
- The final-layer numbers exactly match our direct accuracy measurement — confirming the lens is calibrated.

### What it means
**RL's functional effect appears entirely in the late layers.** The model computes the same things as before
through layer 23, then — in the trained model only — the late layers convert that into a readable answer. This
*reconciled* a puzzle from Investigation #1: the weight change was uniform in **size** across depth, but its
**effect** is late-specific. "Uniform in magnitude, late-specific in effect."

---

## 5. Investigation #4 — *Which weights actually CAUSE the fix?* (the "graft")

### The question
Investigations #1 and #3 are *correlational* — they show *where* things changed/appeared. They don't prove the
MLP change *causes* the improvement. For causation we need an intervention.

### How we computed it — from scratch (this is the key method)
A **graft** (weight transplant) builds a **counterfactual model**: start from the untrained base, and copy
over **only part** of the trained changes, leaving the rest at base. Formally, for each weight:
`W_grafted = W_ckpt` if it's in the chosen part, else `W_base`. Then measure perception. We built:
- **mlp** — only the LLM's MLP weights are trained, rest base → *"what if ONLY the MLPs had changed?"*
- **attn** — only the attention weights → the control.
- **early_mlp / late_mlp** — only the MLPs in the bottom third / top third of layers.
- **base** and **full** as sanity bounds (should reproduce 0.377 and 0.657).

If a graft recovers most of the accuracy gain, **that subset is *sufficient*** to produce the fix — a causal
claim, because we *constructed* the model and measured its behavior (an intervention, not an observation).

### What we found (Condition 1; **replicated** in Condition 2)
| graft | accuracy | % of the +0.28 gain recovered |
|---|---|---|
| base | 0.377 | 0% |
| full | 0.657 | 100% |
| **mlp** | 0.553 | **63%** |
| **attn** | 0.460 | **30%** |
| early_mlp | 0.430 | 19% |
| **late_mlp** | 0.387 | **3.6%** |

- **MLP dominates attention, causally** — the MLP transplant recovers ~2× what attention does. (Senior's
  hypothesis supported. Honest nuance: attention isn't *zero* — 30% — so "MLP-dominant," not "MLP-only.")
- **The fix is DISTRIBUTED across depth, not late-localized** — and this *corrected our own prediction*. From
  Investigation #3 we expected the *late* MLPs to carry it; instead **late_mlp recovers almost nothing (3.6%)**
  while **early_mlp recovers more (19%)**, and the full MLP (63%) is far more than the sum of its parts
  (synergy — no single third suffices).

### Reconciling #3 and #4 (the subtle, important point)
These look contradictory but aren't — they answer different questions:
- The **logit-lens (#3)** reads through the *output head*, so it only sees the part of the representation
  **aligned with the answer** — which is identical until late. → it shows **where the fix *manifests*** (late).
- The **graft (#4)** shows **which weights *cause* it** (MLPs, spread across depth, early-leaning).

The likely mechanism (a hypothesis consistent with all data): the **early/mid MLP edits change *other*
directions** in the residual stream — invisible to the logit-lens early on — that propagate upward and let the
**late layers read out the answer**. Grafting *only* late MLPs fails because they need those earlier changes
feeding them. **The fix appears late but is caused throughout.**

---

## 6. A crucial methods point — *we validated our measuring stick first*

Before trusting any of #3–#4, we had to be sure our **perception probe** could actually *detect* the known
improvement. The probe: show an item, do **one forward pass**, read the model's probability over the option
letters at the answer position, pick the highest → a deterministic, judge-free accuracy.

We first tried it on an external set (babyVision) — and it showed **no improvement** (base 32.6% ≈ trained
33.3%). Rather than trust or discard it, we diagnosed: the training data is *native* direct-answer multiple
choice, so the probe *design* is correct; babyVision is simply **out-of-distribution** (a different, harder
benchmark). We switched the probe to a held-out sample of the **training distribution**, where:

| | direct-readout accuracy |
|---|---|
| base | 0.377 |
| trained | 0.657 |

The probe now sees a **+0.28 gain**, is **calibrated** (base 0.377 ≈ the training reward's start), and beats
the majority-guess baseline. *Only then* did we run the depth-probe and graft. **Lesson for the talk:** a null
result can be a measurement problem; we tested the instrument before trusting it — and the babyVision miss is
itself a finding (*the gain is distribution-specific; it doesn't transfer to adversarial vision-primitives*).

---

## 7. The synthesis — one story, four independent angles

> **RL makes a tiny set of MLP-dominated weight edits, distributed across the LLM stack (not the vision
> encoder), that don't change the model's early/mid representations but reorganize the residual stream so the
> late layers can *read out* perception the base model already computes. The improvement therefore appears in
> the late layers but is caused throughout — and it is entirely an LLM operation, not better seeing.**

| angle | method | result | role |
|---|---|---|---|
| **Where weights moved** | weight-delta (Frobenius) | tiny (0.05%), MLP-biased, uniform in size | the edit is small & surgical |
| **Which component** | 3-condition freeze ablation | LLM-only = full; ViT change = 0 | it's the LLM, not the encoder |
| **Where it shows** | logit-lens by layer | identical early; late layers light up | effect manifests late |
| **Which weights cause it** | counterfactual graft | MLP-dominant, distributed, synergistic | causal localization |

Two things make this robust: it **replicated** under the ViT-frozen condition, and the data **corrected a
prediction** of ours (late-MLP did *not* dominate) — intellectual honesty that strengthens, not weakens, the result.

---

## 8. Honest caveats (state these — they build credibility)
- **Contamination:** the perception probe uses the training distribution, so trained-model numbers are
  *train-accuracy*. This is fine for *localization* (which weights carry the competence); the base numbers are
  uncontaminated, and we are not claiming generalization (the babyVision miss shows generalization is limited).
- **Non-additivity:** grafts measure *sufficiency* of a subset, not a clean decomposition; the strong synergy
  (full-MLP ≫ sum of thirds) is real and expected in neural nets.
- **Logit-lens at mid layers is approximate** (the head is tuned for the top layer); the robust signals are the
  late layers and base-vs-trained differences. A "tuned lens" would clean up the middle.
- **The residual-direction reconciliation (§5) is a hypothesis** consistent with the data, not yet directly proven.
- Single model (4B), single seed, Stage-1-only — a mechanism study, not a 1:1 reproduction of the paper.

## 9. The bridge to the real goal — *are the better representations re-usable?*

The program's actual question is not "how does RL change weights," but: **can we find internal representations
that make the model understand better, and re-use them — e.g. in a mid-reasoning tool-call — to improve
accuracy on demand?** Everything above establishes the *premise* of that idea, but in the wrong *space*:

- **What we've shown (weight-space):** RL finds a tiny, LLM-internal edit that produces a **more
  answer-decodable late-layer representation** (depth probe: 0.37 → 0.66). The "good understanding" the model
  acquires lives in *how the answer-bearing tokens are represented in the late layers.*
- **What the goal needs (representation-space):** to use this in a tool-call, the improvement must be a
  **representation we can *extract* and *re-inject* at inference** — not just something baked into weights.

**The bridge experiment: activation patching / steering.** Take the **difference in the residual stream**
between the trained and base models at the late layers (the "good direction" RL found), and **inject it into
the base model at inference**. Then measure accuracy:
- **Per-item patching** (copy trained's late representation into base) → *is the late representation the
  carrier of the answer?* (upper bound on what representation-injection can recover).
- **A fixed steering vector** (the mean trained−base difference) → *is there a portable direction a tool could
  deploy?* (the deployable artifact).

**If accuracy recovers, the better representation is portable** — extractable and re-injectable — which is
exactly the substrate a re-inspection tool-call would exploit. This is the experiment that turns "RL found a
better representation" into "we can *use* that representation later to improve accuracy." (Caveat from §5: the
cause is *distributed and synergistic*, so single-point injection may recover only part of the gain — the
layer sweep quantifies how much, and even partial recovery is informative.)

## 10. Remaining loose ends
- Close the freeze ablation with **vit_only** (training only the vision tower — expected to recover little),
  plus its weight-level freeze proof.
- **Finer layer-band grafts** to map the distribution precisely and test the §5 mechanism.
- **Tuned lens** for a clean mid-stack readout.
