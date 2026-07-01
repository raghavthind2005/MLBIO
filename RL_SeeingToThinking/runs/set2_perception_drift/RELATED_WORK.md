# Set 2 — Related Work & Novelty Lock (Step 0a)
**Date:** 2026-07-01 · **Method:** *adversarial* literature pass — we searched to **refute** our novelty, not confirm it. ~12 targeted web searches (2024–2026) + 3 deep paper reads.
**Confidence:** focused, not a full systematic survey. Re-run before any submission. All claims below cite a real arXiv ID; where we characterize a paper in detail we read its HTML.

---

## TL;DR — the honest bottom line

1. **The phenomenon is NOT novel.** "Visual perception/faithfulness decays over long reasoning chains" is an **active 2025–26 topic**. One paper literally reports *"temporal context drift in hallucination over long reasoning chains."* **We must not claim to have discovered it.**
2. **The measurement style in that work is exactly what we reject.** The closest papers quantify with **LLM/VLM judges**, analyze **reasoning-step text only**, and stop at **correlation**.
3. **Our lane is open:** a **judge-free (scene-graph GT), capability-gated, causally-verified, mechanistically-localized** characterization that **derives the intervention point**. No paper we found does this combination.
4. **The headline novelty must ultimately live in the METHODOLOGY** (a learned representation that beats raw image re-injection at the localized point). Set 2's diagnosis is the **rigorous evidence base** that justifies and designs it — a legitimate *measurement + mechanism + causality* contribution, but **not** a "new problem."

---

## 1. The phenomenon is already established (do not claim discovery)

| Paper | arXiv | What it establishes |
|---|---|---|
| Journey Before Destination: Visual Faithfulness in Slow Thinking | [2512.12218](https://arxiv.org/abs/2512.12218) | reasoning-trained VLMs reach right answers via **visually unfaithful** steps; **context drift over long chains** (App. D) |
| On the Faithfulness of Visual Thinking | [2510.23482](https://arxiv.org/html/2510.23482v1) | visual evidence in MCoT is "**largely ignored**"; defines reliability & sufficiency of visual thoughts |
| The Hidden Life of Tokens / VISTA | [2502.03628](https://arxiv.org/abs/2502.03628) | via logit-lens, **visually-grounded tokens decline across generation** while hallucinated tokens surface |
| Vision Inference Former | [2605.18160](https://arxiv.org/html/2605.18160) | names "**visual consistency decay**"; injects visual signals into hidden states each step to fix it |
| Visual attention decay (many) | [2505.21472](https://arxiv.org/abs/2505.21472), IKOD [2508.03469](https://arxiv.org/pdf/2508.03469), MCA-LLaVA [2507.09184](https://arxiv.org/pdf/2507.09184) | visual attention ratio **drops during decoding**; hallucinations cluster in low-attention regions (the "simple attention graph" we deliberately avoid) |

**Implication:** our contribution is *how* we measure and *what we prove causally*, not *that the problem exists*.

---

## 2. The three closest papers — precise characterization

Read in full (HTML). Columns are the exact axes on which we differ.

| Axis | **Journey Before Destination** (2512.12218) | **Faithfulness of Visual Thinking** (2510.23482) | **Hidden Life of Tokens / VISTA** (2502.03628) | **OURS (Set 2)** |
|---|---|---|---|---|
| Measures | step-level visual faithfulness | reliability + sufficiency of visual thoughts | token-ranking decline over generation | **RIPE + internal percept flip + causal recovery** |
| How | **VLM judge** (Claude Sonnet 4) | **GPT-4o judge** | GPT-4o oracle categories, **aggregate** stats | **scene-graph exact-match, no judge** |
| Internal probe (layer × token)? | ✗ text only | ✗ text only | partial (logit-lens, **aggregate**, captioning) | **✓ per-item (layer × reasoning-token)** |
| Causal patch (within network)? | ✗ | text-level only (inject text mistakes / noise crop) | ✗ (correlation only) | **✓ restore own early percept** |
| Capability gate (`D=1`)? | ✗ | ✗ | ✗ | **✓ isolates drift from perception-inability** |
| Separates drift vs reasoning-logic error? | partially (step labels) | ✗ | ✗ | **✓ E2 flip vs stay + E3 falsification** |
| Task type | perception benchmarks | V*Bench/HR-Bench | **open-ended captioning** (CHAIR/POPE) | **compositional reasoning w/ verifiable GT** |
| Localizes injection point? | ✗ (text regeneration) | ✗ | ✗ (all-layer steering vector) | **✓ (layer×token sweep)** |
| Models | 4× 7B reasoning VLMs | DeepEyes/Pixel-Reasoner/Qwen2.5-VL-7B | LLaVA-1.5/MiniGPT-4/etc. | Qwen3-VL-4B-Thinking |

**Per-paper one-liners (for the talk):**
- **Journey Before Destination** — closest in *spirit* (perception-vs-reasoning step split; notes drift over length), but **judge-based, text-only, no causal, no gate**. Confirms our problem is real; leaves the rigorous mechanism open.
- **Faithfulness of Visual Thinking** — measures whether visual thoughts matter via **text/image-level** interventions and a **GPT-4o judge**; **no internal probing, no capability gate.**
- **Hidden Life of Tokens (VISTA)** — closest in *tooling* (logit-lens across tokens×layers), but **aggregate ranking on captioning, correlation-only**, and VISTA is a **generic all-layer steering vector**, not a localized restoration of the model's **own early percept** on a **verifiable** reasoning item.

---

## 3. Adjacent buckets (comprehensiveness)

- **Mechanistic tools we *reuse* (not claim):** logit-lens object decodability peaks ~L25 — Neo et al., *Towards Interpreting Visual Information Processing in VLMs* [2410.07149](https://arxiv.org/html/2410.07149) (ICLR'25); "visual enrichment → semantic refinement" two-stage. Causal tracing of object tokens (single forward): FCCT [2511.05923](https://arxiv.org/abs/2511.05923) (AAAI), *Devils in Middle Layers* [2411.16724](https://arxiv.org/pdf/2411.16724), *What's in the Image* [2411.17491](https://arxiv.org/pdf/2411.17491), *Dual-Pathway Circuits of Object Hallucination* [2605.13156](https://arxiv.org/pdf/2605.13156). → All **single-forward**, not across a reasoning chain, and about **object hallucination**, not capability-gated reasoning drift.
- **Re-injection / visual re-access methods (crowded — we do NOT claim re-injection as method):** Visual Perception Token [2502.17425](https://arxiv.org/html/2502.17425v1), TVI-CoT [2606.08464](https://arxiv.org/pdf/2606.08464), v1 [2505.18842](https://arxiv.org/html/2505.18842v4), Latent Visual Reasoning [2509.24251](https://arxiv.org/html/2509.24251), Machine Mental Imagery [2506.17218](https://arxiv.org/pdf/2506.17218), Visual CoT (438k bbox). → We use re-injection as **causal evidence + localization**, not as the deliverable.
- **Belief tracking across CoT tokens (text-only):** Reasoning Theater [2603.05488](https://arxiv.org/html/2603.05488v2), Hidden Error Awareness [2605.09502](https://arxiv.org/html/2605.09502). → probe belief across reasoning tokens, but **language reasoning**, never the **visual percept** in a VLM.
- **Atomic / compositional decomposition for eval:** Atomic Visual Skills [2505.20021](https://arxiv.org/html/2505.20021v1), Self-Rewarding via Reasoning Decomposition [2508.19652](https://arxiv.org/abs/2508.19652), Compositional Visual Reasoning survey [2508.17298](https://arxiv.org/pdf/2508.17298). → we borrow atomic-fact decomposition as our **judge-free GT source**.
- **Video adjacent:** CircuitProbe (visual temporal evidence flow in **video** LMs) [2507.19420](https://arxiv.org/pdf/2507.19420) — tracks visual evidence over time, but video frames, not reasoning-chain percept drift.
- **Controlled datasets (our GT source):** CLEVR (Johnson et al., CVPR'17, [1612.06890](https://arxiv.org/abs/1612.06890)) — synthetic scenes, scene graphs, functional-program questions, open generation engine. GQA (Hudson & Manning, CVPR'19, [1902.09506](https://arxiv.org/abs/1902.09506)) — real images, scene-graph functional programs.

---

## 4. What is genuinely ours (the differentiators)

1. **Judge-free, verifiable quantification.** RIPE and the E2 probe are scored by **scene-graph exact-match**, not an LLM/VLM judge — directly fixing the subjectivity of 2512.12218 / 2510.23482.
2. **Capability gate (`D=1`).** We only count items the model **provably perceives in isolation**, separating drift from perception-inability. No close paper does this.
3. **Specific verifiable percept flipping.** We track *one enumerable atomic fact* (from the functional program) flip **correct→wrong** across (layer × reasoning-token) on a **reasoning** task — vs Hidden Life's **aggregate** ranking on **captioning**.
4. **Causal proof on capability-gated reasoning items.** Activation-patch **restores the model's own early percept**; if the answer flips back, drift **caused** the error. Prior causal work is single-forward object tokens or text-level edits — not this.
5. **Drift vs reasoning-logic error, explicitly split** (E2 flip-vs-stay + E3 falsification: patching reasoning-errors must NOT help).
6. **Derives the intervention point** (layer×token sweep) and compares **self early-percept vs raw image re-injection** — directly informing the methodology, vs VISTA's generic all-layer vector.

---

## 5. The exact claim (say this) vs. what we must NOT say

**SAY (defensible):**
> "Recent work shows visual faithfulness decays over long VLM reasoning, but measures it with **LLM judges on step text** and stops at **correlation**. We give the first **judge-free, capability-gated, causal** account: on scene-graph-verifiable items the model can perceive in isolation, we track a **specific percept flipping correct→wrong** inside the chain, **causally** restore it by re-injecting the model's **own early representation**, and thereby **localize where/what to inject** — turning a loosely-measured phenomenon into a quantified, mechanistically-localized, causally-verified one that specifies an intervention."

**Do NOT say (false / overclaim):**
- ✗ "We discovered that perception degrades during reasoning." (2512.12218, 2510.23482, 2502.03628 predate us.)
- ✗ "First to track visual information across generation." (Hidden Life of Tokens did, in aggregate.)
- ✗ "First to re-inject visual features." (crowded — §3.)
- ✗ "Our method fixes the problem." (Set 2 is **diagnosis**; the method is future work.)

---

## 6. How a reviewer attacks — and our defense

| Attack | Defense |
|---|---|
| "Just visual-faithfulness decay again (2512.12218)." | Theirs is **VLM-judge, text-only, correlational, no capability gate**. Ours is **exact-match GT, internal, causal, gated** — and separates drift from reasoning-logic error. |
| "Hidden Life of Tokens already logit-lensed tokens×layers." | **Aggregate** ranking on **open-ended captioning**, **correlation-only**, generic all-layer steering. We do **per-item verifiable percept flip on a reasoning task** + **causal self-percept restoration**. |
| "CLEVR is synthetic/toy." | GQA real-image slice replicates it; synthetic gives **unambiguous GT + enumerable atomic facts** — a *feature* enabling per-fact probing/patching. |
| "Re-injection is well-trodden." | We do **not** claim re-injection as a method; it's **causal evidence + localization**. The method (learned rep > raw) is explicitly scoped as future. |
| "Probes read info the model doesn't use." | That's **exactly** why E3 (causal patch) is mandatory (guard F5); the probe alone never carries a claim. |

---

## 7. Verdict for the plan

- **Green-light the design** — the judge-free + causal + gated + localized combination is unclaimed.
- **Reframe every slide/claim** per §5 (diagnosis with rigor, not discovery).
- **Push the novelty headline downstream** to the methodology (learned representation, localized injection) — Set 2 is the evidence that earns the right to build it.
- **Before submission:** re-run this pass (esp. new 2026 arXiv) and add a systematic related-work table.
