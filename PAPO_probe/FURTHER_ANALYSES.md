# Deeper Analyses Beyond the Probe (#2 Intra-chain decay, #3 Accuracy-under-masking, #4 Attention-to-image)

**Status:** DESIGN sketches (companions to [`PROBE_DESIGN.md`](PROBE_DESIGN.md)). These are the analyses
that turn the offline perception-KL probe from "a grounding number" into an answer to the *mechanistic*
research question: **how does visual perception change across the reasoning chain, and how does PAPO's
perception loss shift it?** Each is designed to *triangulate* the same phenomenon through a different
observable, so a conclusion is believed only when independent measures agree.

**Why triangulation matters (the logic that ties all four together).** The probe (#1) gives a
*distributional* grounding scalar G, but G is a proxy (threat T4: high G could be useful grounding *or*
brittleness). The three analyses below convert G into a defensible mechanistic claim:
- **#2** asks *where in the chain* grounding lives (position-resolved G) — the actual "degrades across the
  chain" question.
- **#3** asks whether grounding is *behaviorally real* (does masking actually change answers) — resolves
  the brittleness-vs-grounding ambiguity.
- **#4** asks whether grounding has a *mechanistic correlate* (attention to image tokens) — an independent
  observable of the same construct.
A PAPO grounding effect is convincing iff it shows up as: higher/flatter G along the chain (#2) **and**
appropriate accuracy loss under masking with clean accuracy preserved (#3) **and** more/steadier
attention-to-image (#4). Convergence across these is the standard of proof.

---

## Analysis #2 — Intra-chain perception decay (the mechanistic centerpiece)

**Question.** Does the model's reliance on the image **fade as the reasoning chain gets longer** (a
"reason-longer, see-less" effect), and does PAPO **flatten** that decay?

**Measurement (nearly free — reuses the probe).** The probe's Pass-2 already computes and **saves the full
per-token KL vector** for every response (§5.8c of PROBE_DESIGN). Here we simply *do not collapse over
position*:
1. For each response, take its per-token KL vector `kl_t` (real-vs-masked, low_var_kl).
2. Map each token to its **relative position** in the response `p_t = t / T ∈ [0,1]` (relative, because
   responses vary in length — absolute position would confound with length).
3. Bin into deciles (or 20 bins); average `kl_t` within each bin, across samples and prompts → a curve
   **G(p)** for each checkpoint/arm.
4. Also segment by **structural region**: the `<think>…</think>` CoT vs. the post-`</think>` answer region
   (split at the `</think>` token id) → G_think vs G_answer. This separates "grounding while reasoning"
   from "grounding while answering."

**Pre-registered hypotheses.**
- **H2a (decay in GRPO):** for Arm A, G(p) **decreases** with p (grounding fades deeper into the chain).
- **H2b (PAPO flattens):** Arms B/C have a **flatter / higher** G(p), especially in the late chain
  (large p) and in the answer region → PAPO keeps the model looking at the image throughout reasoning.
- **Decision:** compare the **slope** of G(p) (and G_answer − G_think) across arms with bootstrap CIs over
  prompts. A significant reduction in decay slope for C vs. A supports H2b.

**Confounds & controls.**
- *Length ↔ position:* use relative position; additionally report G(p) **within a fixed length band**
  (e.g., responses 2–4k tokens) so the curve isn't a length artifact.
- *Content ↔ position:* later tokens are systematically different content (answers, numbers). The
  **fixed-token mode** of the probe gives G(p) on *identical* tokens across checkpoints, isolating the
  policy effect from content drift — run #2 in both on-policy and fixed-token modes and require agreement.
- *Token-type:* optionally tag numeric/answer tokens; report whether decay is content-type driven.

**Why it's the centerpiece.** It is the *only* analysis that directly measures the paper-motivating claim
("perception degrades across reasoning") at the resolution of the chain, and it costs almost nothing on top
of the probe (the per-token vectors already exist). **Threats:** single seed (T1); relative-position binning
assumes comparable chain structure across arms (checked via fixed-token mode).

---

## Analysis #3 — Accuracy-under-masking (behavioral grounding; required companion to the probe)

**Question.** When we corrupt the image, does the model's **answer actually change** — and does it change
*more* for the PAPO arms? This is the behavioral counterpart to G and the direct resolution of threat T4
(is G real grounding or mere brittleness?).

**Measurement.** For each checkpoint (subset — see cost): generate answers under the **masked** image (in
addition to the probe's real-image generation) and grade both.
- `acc_real` = accuracy on the intact image (the probe already has this).
- `acc_mask` = accuracy on the masked image.
- **Image-reliance gap** Δ = `acc_real − acc_mask`.
- **Answer-flip rate** φ = P(wrong under mask | correct under real) — cleaner than Δ because it conditions
  on items the model got right *because of* (or despite) the image.

**Pre-registered hypotheses.**
- **H3a (behavioral grounding):** a more distributionally-grounded arm (higher G) shows a **larger** Δ and
  φ — masking hurts more because the model was actually using the image.
- **H3b (grounding is useful, not brittle):** the PAPO arm preserves `acc_real` (≈ or ≥ GRPO's clean
  accuracy at matched steps) **while** having larger Δ/φ. If instead `acc_real` is *also* degraded, the
  high G is **brittleness**, not useful grounding — a decisive, disconfirming outcome we must be willing to
  report.
- **Joint decision:** classify each arm in the (acc_real, Δ) plane. Grounded-and-useful = high acc_real +
  high Δ; brittle = low acc_real + high Δ; image-ignoring = any acc_real + Δ≈0.

**Confounds & controls.**
- *OOD generation:* the model never *generated* under masks during training (it only *scored* masked
  tokens). Generating under a mask is out-of-distribution and may yield degenerate text — that is a valid
  behavioral readout, but we note it and inspect samples.
- *Mask identity:* use the **same frozen mask bank** as the probe (K masks; report mean over masks) so Δ is
  not a mask-luck artifact.
- *Same prompts/samples* as the probe → paired comparison; bootstrap over prompts.

**Cost.** This one **requires extra generation under masks** (unlike #2). To bound cost: run at a
**subset of checkpoints** (e.g., base, 30, 60) and K=1–2 masks first; expand only if the signal warrants.
Prioritize step 60 (the endpoint decision) + base (anchor).

**Decision-relevance.** #3 is the arbiter of the whole study's interpretation: only #3 can certify that a
probe/#2 grounding difference is *useful* grounding rather than sensitivity noise. Treat it as **required**,
not optional, for any causal-sounding claim about PAPO improving perception.

---

## Analysis #4 — Attention-to-image mass (independent mechanistic correlate; stretch)

**Question.** Does the model literally **attend more to the image tokens** under PAPO — and does
attention-to-image decay along the chain the way grounding does? An independent observable of the same
construct as G, from the model's internals rather than its output distribution.

**Measurement.** Run the HF model with `output_attentions=True` (requires **eager** attention, not
flash-attn) on (prompt, real response) for a small stimulus subset:
1. Identify image-token positions (Qwen3-VL marks vision tokens by id / grid — resolve at build time).
2. For each response token, compute the **fraction of its attention mass on image tokens** (vs. text/system),
   aggregated across heads and a chosen set of layers (report per-layer and a summary).
3. Resolve by **position-in-chain** (as in #2) → Attn_img(p) per checkpoint/arm.

**Pre-registered hypotheses.**
- **H4a:** PAPO arms allocate **more** attention mass to image tokens than GRPO, especially late in the
  chain (large p).
- **H4b (cross-validation):** Attn_img(p) **correlates** with G(p) from #2 across positions/arms. Agreement
  triangulates the mechanism ("grounding = looking"); **disagreement is itself a finding** (grounding via
  non-attention pathways, e.g., value-side routing) worth reporting.
- **Decision:** report Attn_img(p) trajectories + the per-position correlation with G; this is
  **exploratory/mechanistic**, not a confirmatory gate.

**Confounds & threats.**
- *Attention ≠ attribution:* attention weights are a **contested** proxy for information use; a null or
  positive here is suggestive, not dispositive. This is why #4 is a *correlate*, never the primary evidence.
- *Aggregation choices:* head/layer aggregation is lossy; pre-register the layers/heads summary (e.g.,
  mean over all layers + a mid-layer readout) to avoid post-hoc cherry-picking.
- *Vision-token count varies* per image → normalize by image-token count.
- *Cost/feasibility:* eager attention is slower and memory-heavy at 8192 tokens → run on a **small subset**
  (few prompts, few checkpoints: base/30/60) and short responses, or a capped context. Lowest priority of
  the three; do only if #2/#3 leave the mechanism ambiguous.

---

## Priority & sequencing (recommendation)

1. **Build the probe (#1)** — backbone; produces the per-token vectors #2 needs.
2. **#2 intra-chain** — essentially free once #1 runs; answers the core mechanistic question. Do next.
3. **#3 accuracy-under-masking** — required to interpret #1/#2 causally; moderate extra cost, run on a
   checkpoint subset. Do before making any "PAPO improves perception" claim.
4. **#4 attention-to-image** — stretch/triangulation; run only if the mechanism remains ambiguous after
   #1–#3, and treat as suggestive.

**The one-paragraph scientific throughline.** #1 says *whether* the output distribution depends on the
image; #2 says *where in the reasoning* that dependence lives and whether PAPO stops it fading; #3 says
*whether that dependence is behaviorally real and useful* (not brittleness); #4 says *whether it has an
attentional mechanism*. A PAPO perception effect is established only where these converge — and each is
designed so that a null is as informative and reportable as a positive.
