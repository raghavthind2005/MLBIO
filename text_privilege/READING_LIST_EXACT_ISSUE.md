# Reading List — Work That Tackles Our Exact Issue

**Date:** 2026-08-04. **Status:** reading guide + synthesis. Nothing normative.
**Companion:** [`LITERATURE_SCAN_01.md`](LITERATURE_SCAN_01.md) (the broader "train articulation" field).

**Our exact issue, stated precisely so we can match against it:**

> The visual information is demonstrably accessible to the model (Set-2 probe reads the CLEVR scene at
> **0.918** from image tokens; Track-T shows the model *uses* perceptual facts when handed them as text,
> **+0.075** MC). But the model does not bring that information into its own reasoning: its
> self-description recovers **none** of the oracle benefit (recovery **−0.41**).

Papers below are ranked by how directly they hit *that*, not by how well-known they are.
Tags: `[FETCHED]` = I retrieved the abstract/HTML this session · `[SNIPPET]` = search summary only, unverified.

---

## Read in this order

### 1. VDGD — *Visual Description Grounding Reduces Hallucinations and Boosts Reasoning in LVLMs*
arXiv **2405.15683** · **ICLR 2025** · code `github.com/Sreyan88/VDGD` · project page `sreyan88.github.io/VDGD` · `[FETCHED]`

**Read this first. It is the closest paper to us that exists, and it contradicts our result.**

Their diagnosis, verbatim from the abstract: *"We identify the core issue as a lack of true visual
perception in LVLMs: although they can accurately recognize visual elements, they struggle to fully
interpret these elements in the context of the input prompt and effectively link this recognition to
their internal knowledge, which is critical for reasoning."*

That is Track T's diagnosis, published fifteen months earlier, in different words.

**Their method (VDGD):** generate a detailed description of the image → **append it as a prefix to the
instruction** → during generation, sample tokens by **KL divergence to the description**, favouring
low-divergence candidates. Training-free. Reports **2–33%** gains across visual-reasoning benchmarks.
Also introduces **VaLLu**, a benchmark for cognitive/reasoning prompts.

**Why it matters to us — two things, and the second is the important one:**

- Their prefix *is* Track T's `self` arm. They report it works; we measured `self − base = −0.030`.
- **But VDGD is not prefix-only.** The prefix is half the method; the other half is a *decoding
  constraint* that keeps pulling every generated token back toward the description. **Track T tested
  only the prefix.** The active ingredient may be the binding, not the text.

Also read their stated limitation: self-generated descriptions carry hallucinations, so VDGD is *worse*
on POPE-adversarial. That is the precision-vs-coverage trade, in their own results.

**Read for:** (a) exactly what the description prompt is; (b) the ablation of prefix-only vs
prefix+KL-decoding — if they ran it, it answers our central question directly; (c) which base models
(I expect non-thinking LLaVA-class — see §5).

---

### 2. Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?
arXiv **2404.06510** · **CVPR 2025** · `[FETCHED]`

**The closest published analogue to our recovery ratio.**

They compare **oracle feedback** against **self-generated feedback** for correcting grounding errors:
- oracle binary feedback: **+6 to +17** points
- self-generated feedback: **+0.4 to +4.4** points (ADE20k)
- → roughly a **3–4× gap**, i.e. a recovery fraction of about **0.25–0.30**.

Models: LLaVA-1.5, ViP-LLaVA, CogVLM, GPT-4V, GPT-4o. Data: ADE20k, COCO panoptic.

**Why it matters:** independent corroboration that self-supplied signal is worth a fraction of oracle
signal — the same *shape* as Track T, on a different task (grounding *correction* rather than
*articulation*). Their recovery ≈ 0.25–0.30; ours ≈ 0. **Our result is the more extreme version of a
known effect, not an isolated anomaly.** That is good for defensibility and bad for novelty; both need
stating.

**Read for:** how they define and measure the oracle-vs-self gap — it is the prior art for our recovery
metric and we should adopt their framing where it is compatible.

---

### 3. Caption This, Reason That: VLMs Caught in the Middle
arXiv **2505.21538** · **NeurIPS 2025** · `[FETCHED abstract]`

Verbatim: *"models struggling with direct visual reasoning show marked improvement when reasoning over
their own generated text captions."* Their conclusion: the bottleneck is **not** reasoning capacity and
**not** low-level perception, but **the integration of visual features into the reasoning process**.

New details from this pass `[SNIPPET — verify]`:
- self-captioning gain is **+18.31%** on *location-only* tasks vs **+3.54%** on *category* tasks;
- the large gain is in the **image-free** condition; adding the image back **shrinks** it (attributed to
  attention capacity / interference);
- base model for the decoupling analysis: Qwen2.5-VL-7B.

**Why the spatial split matters:** Track-T's perceptual scope is **DI+IP** — descriptive and *implicit
spatial* givens. This paper predicts self-captioning should help **most** on exactly that content. Track
T found the opposite. That does not resolve the conflict — it **sharpens** it, and it is why this paper
must be read carefully rather than cited in passing.

**Read for:** the exact image-free vs image-present numbers, and their integration argument — it is the
EXTRACTED-BUT-NOT-INTEGRATED hypothesis with data behind it.

---

### 4. Seeing but Not Believing: Probing the Disconnect Between Visual Attention and Answer Correctness
arXiv **2510.17771** · `[FETCHED]`

*"Deep layers often lock onto the correct evidence even when the final answer is wrong."* Measured on
**VisualCoT** (human-annotated evidence regions) across **8 VLMs / 4 families** (LLaVA-Next, Qwen2.5VL,
Gemma3, InternVL3.5) on InfoVQA, DocVQA, SROIE, TextVQA. Proposes **Vea**, an inference-time
evidence-highlighting method, **+5.67 EM**.

**Read it, but read it critically.** Their instrument is **attention**, which this project has already
ruled out twice — the literature reason (Jain & Wallace) and our own decisive one (Set-2 E3/A2: an
injected image is *attended* yet rarely changes the answer; attention ≠ causal use). So they make our
claim with weaker evidence than we have. **This is a positioning asset:** Track T establishes the same
disconnect *behaviourally and causally, with a placebo gate*, which is a stronger instrument than
attention mass.

---

### 5. The synthesis these four force (the important part)

**Three independent papers report that the model's own description helps. We measured that it doesn't.**
That is now a real conflict and it must be resolved before any method is designed. Four candidate
explanations, ranked by my confidence:

1. **Prefix alone is not the method — binding is.** VDGD prepends the description *and* constrains every
   decoding step toward it. Track T prepended and let the model run free. If the description's benefit
   requires continuous re-anchoring, prefix-only would null exactly as we observed. **Directly testable
   and cheap.**
2. **Thinking vs non-thinking models.** VDGD, Caption This Reason That, Seeing-but-Not-Believing, and the
   self-correction paper all use **non-thinking** VLMs (LLaVA-1.5/Next, Qwen2.5-VL, GPT-4o). We use
   **Qwen3-VL-4B-Thinking**. A long chain gives far more opportunity to drift away from a prefix.
   **Our own Set-3 data supports this independently:** the same enumeration gave **+0.114** placed
   *before* reasoning but **+0.007** placed mid-think — the benefit decays as the chain proceeds.
3. **Image-free vs image-present.** The reported gains are largest without the image; we kept it.
4. **Ceiling compression.** Our MathVerse MC base is 0.808; their tasks are harder.

**(1) and (2) combine into a thesis worth having:**

> Description-grounding works for short-chain VLMs and fails for long-chain *thinking* VLMs, because the
> reasoning chain washes out the prefix — which is why VDGD needs a per-token decoding constraint to make
> it stick.

If that is right, the problem is **not** that the model cannot articulate perception. It is that
articulated perception **does not survive contact with a long reasoning chain**. That is a different
project from "train Stage-1 articulation" — and it is better positioned: it explains a published method's
mechanism, it uses the thinking-model regime nobody in this cluster has tested, and our Set-2/Set-3
instrument history is built for exactly this question.

**It also means the deferred checks are no longer optional bookkeeping — they are the experiment.**
Grading our self-descriptions distinguishes "can't articulate" (explanation 4-ish) from "articulates fine,
doesn't bind" (explanations 1–2). That single number picks which project we are in.

---

## Second tier — read after the four above

| Paper | arXiv | Why | Tag |
|---|---|---|---|
| **Prism** — decoupling perception & reasoning | 2406.14544 (NeurIPS 2024) | The measurement framework for our construct; Track T is a within-model Prism. Must be cited. | `[SNIPPET]` |
| **Caption Bottleneck Models** | 2607.00578 | Caption as an explicit information bottleneck — the formal version of "text-privilege". | `[SNIPPET]` |
| **The Perceptual Bandwidth Bottleneck in VLMs** | 2605.01345 | Argues the bottleneck is fixed encoder token capacity — a *competing* explanation for our gap. | `[SNIPPET]` |
| **See It, Say It, Sorted** | 2602.21497 | Iterative training-free grounded reasoning; reportedly critiques VDGD's "fixed logit replacement tied to a single static caption". Likely the current SOTA in this line. | `[SNIPPET]` |
| **Self-Introspective Decoding** | 2408.02032 | Decoding-side hallucination control; adjacent mechanism. | `[SNIPPET]` |
| **CAPEval** | 2608.02589 (2026-08-04) | Caption **Precision** predicts downstream performance & hallucination robustness; Coverage predicts general understanding. One day old. | `[SNIPPET]` |

---

## What this changes

The Stage-1-then-Stage-2 plan assumed the deficit is **articulation**. Four papers on the exact issue
locate it at **integration / binding** instead, and one of them (VDGD) already ships a training-free fix
whose mechanism is a decoding-level constraint, not better descriptions.

That does not kill the direction. It relocates it, and arguably improves it — from "train a model to
describe better," which is crowded (see Scan 01), to "**why does articulated perception fail to bind
during long-chain reasoning, and what makes it stick**," which nobody in this cluster has studied because
nobody in this cluster uses thinking models.

**Next concrete step, once the four papers are read:** decide whether the VDGD prefix-vs-binding ablation
exists in their paper. If it does, it may already answer our question. If it doesn't, running it on a
thinking model is a clean, cheap, publishable experiment — and it is a straight extension of Set-3's
`V_self` / `V_self_pre` machinery, which is already written and audited.
