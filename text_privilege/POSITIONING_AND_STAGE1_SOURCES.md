# Positioning vs. the Staged Papers + Sources for Building Stage 1

**Date:** 2026-08-04. **Status:** working notes. Nothing normative.
**Companions:** [`LITERATURE_SCAN_01.md`](LITERATURE_SCAN_01.md), [`READING_LIST_EXACT_ISSUE.md`](READING_LIST_EXACT_ISSUE.md).

Tags: `[FETCHED]` = retrieved this session · `[SNIPPET]` = search-summary only, **unverified** · `[LOCAL]` = from our own repo records.

---

## Part A — The staged paper you're thinking of, and whether we differ

### A.1 Identification

**"From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language
Models"**, UCSC-VLAA, arXiv **2605.20177**. Project page `ucsc-vlaa.github.io/VLM-CapCurriculum`. `[FETCHED]`

Three sequential RLVR stages: **Visual Perception → Textual Reasoning → Visual Reasoning.** That matches
your description exactly.

**Venue — flagging a discrepancy rather than resolving it:** our own record
`RL_SeeingToThinking/runs/RESULTS.md:16` says **ICML 2026** `[LOCAL]`; you said ICLR; my search returned
no venue confirmation either way. **Do not cite a venue until one of us checks the paper page.**

**This is our own founding paper** — `RL_SeeingToThinking/` is its reproduction (Stage-1 Condition-1 run
took perception accuracy 0.365 → 0.746 over 96 steps) `[LOCAL]`.

Their stated findings `[FETCHED, from search summary of the abstract/page]`:
- visual perception is a *dominant limiting factor* for visual reasoning; longer reasoning cannot
  compensate for perceptual errors;
- perception (a) needs targeted optimization with specialized data, (b) should be **solidified before**
  refining visual reasoning, (c) **is more effectively learned via RL than caption-based SFT**;
- result: **+1.5% reasoning accuracy with 20.8% shorter reasoning traces.**

### A.2 Is our framing different? — Yes, on three specific axes

| | From Seeing to Thinking | Ours |
|---|---|---|
| **What Stage 1 optimizes** | *answer accuracy on perception questions* (3,360 DOCCI MCQs) | *the quality of a textual articulation* that a downstream consumer must use |
| **What Stage 1 outputs** | a better-perceiving policy | a better-perceiving policy **plus an artifact (text) that is consumed downstream** |
| **How stages combine** | a **data curriculum** — same algorithm (RLVR), different task types, run sequentially | a **different algorithm** — on-policy self-distillation where the teacher is the same policy conditioned on the Stage-1 artifact |
| **Self-conditioning** | never; the model's perception output is not fed back to itself | central — the articulation *is* the privileged channel |

So the difference is real and statable in one sentence: **they sequence task types; we make Stage 1
produce the privileged signal that Stage 2 internalizes.** Their "combination" step is "train on visual
reasoning tasks last." Ours is "remove the need for the articulation by distilling its effect into the
weights."

### A.3 Two honest problems with the positioning

**Problem 1 — their finding (c) is a negative result about caption-based perception training.**
"Perception is more effectively learned via RL than caption-based SFT" means **they already tested a
caption-based route to perception and found it worse.** Before committing, read exactly what their
caption-SFT baseline was:
- if it was *SFT on generic captions*, our RL-with-utility-reward Stage 1 is untouched by it;
- if they also tried *caption RL*, that is a much bigger problem and the design has to answer it directly.

This is the single most important thing to check in that paper, and it is checkable in an afternoon.

**Problem 2 — "stage it" is not a differentiator; the staged pattern is crowded.** `[SNIPPET — all of these need verification]`
- **LMM-R1** — two-stage: text-only reasoning → multimodal generalized reasoning.
- **ReVisual-R1** — three-stage: text-centric cold start → multimodal RL → text-only RL refinement.
- **Open Vision Reasoner** — two-stage; explicitly reports that text-only RLVR on VL tasks "exposes a
  perception bottleneck… correctness-only objectives are insufficient."
- **Advancing Multimodal Reasoning: From Optimized Cold Start to Staged RL** — arXiv 2506.04207.
- **Seeing with You: Perception-Reasoning Coevolution for Multimodal Reasoning** — arXiv 2603.28618.
- **SPARC: Separating Perception And Reasoning Circuits for Test-time Scaling of VLMs.**

**Implication:** the novelty cannot be "we stage perception before reasoning." It has to be **what the
stages exchange** (a textual articulation artifact) and **that the second stage is distillation rather
than more RL**. State it that way from the beginning or a reviewer will file it under staged curricula.

**One useful datum from their results:** better perception gave **20.8% shorter reasoning traces**. That
is independent support for the long-chain-washout hypothesis in `READING_LIST_EXACT_ISSUE.md` §5 —
shorter chains mean less opportunity for articulated perception to drift out of use.

---

## Part B — Sources for building Stage 1

Ranked by direct reusability. **The top two are the ones to read.**

### B.1 CapRL / CapRL++ — the closest working recipe, on our model family
- **CapRL**, arXiv **2509.22647**, ICLR 2026, code `github.com/InternLM/CapRL` `[FETCHED]`
- **CapRL++**, arXiv **2606.09393** `[SNIPPET]`

RLVR for captioning. The reward is **downstream utility**: a *separate, vision-free LLM* answers
multiple-choice questions using only the generated caption; its accuracy is the reward. No human
annotation, fully verifiable, no judge model.

CapRL++ details worth having `[SNIPPET — verify]`:
- it is **Qwen3-VL-based** — our exact family;
- its reward combines a **visual utility reward**, a format reward, and a **length-aware penalty** —
  i.e. they hit and solved the length-hacking problem that any utility reward creates;
- it is described as a decoupled two-stage VQA reward paradigm positioned explicitly as more objective
  and verifiable than earlier question-aware captioning methods including PromptCap.

**Why it is the best starting point:** it is a published, code-released, verifiable-reward recipe for
exactly "train a VLM to articulate an image well," validated by downstream utility, on our model family.
If we want a method to test whether the thought process leads to accurate results, this is the one to
reproduce first and then modify — rather than inventing a reward from scratch.

**What we would change (our contribution surface):** CapRL's consumer is an *external* vision-free LLM
and its output is a *pretraining dataset*. It never closes the loop back into the same model's reasoning.
That loop is Stage 2, and it is open.

### B.2 PromptCap — the prompt-relevant description paper (your bonus ask)
arXiv **2211.09699** · **ICCV 2023** · demo `yushi-hu.github.io/promptcap_demo` · HF `tifa-benchmark/promptcap-coco-vqa` `[FETCHED]`

**This is the canonical work on describing only what the prompt needs.** From the paper page: PromptCap
*"takes a natural-language prompt to control the visual entities to describe in the generated caption,"*
where *"the prompt contains a question that the caption should aid in answering."* Its stated motivation
is exactly your concern: *"when summarizing an image in a single caption sentence, which visual entities
to describe are often underspecified, and generic image captions often miss visual details essential for
the language model to answer visual questions correctly."*

- Trained on examples **synthesized with GPT-3** from existing datasets — no extra human annotation.
  **The data-synthesis trick is the transferable part.**
- OK-VQA **60.4**, A-OKVQA **59.6** (SOTA at the time); generalizes zero-shot to WebQA.

**Honest caveats:** 2023, pre-dates modern VLMs, SFT not RL, and the caption is a single sentence rather
than a dense articulation. Take the *idea* and the *data-synthesis method*, not the model.

**Note the tension it does not resolve:** prompt-conditioning is what makes relevance possible, and it is
also what makes answer-leakage possible. PromptCap does not have to care (its consumer is a black-box
LLM and the goal is accuracy). We do, because our Stage 1 would be RL and leakage is the reward-hacking
optimum. Our own Set-3 precedent: `V_viz2` was dropped because the genuinely-relevant crop leaked the answer.

### B.3 Second tier — reward-shaping ideas `[SNIPPET, all unverified]`
| Paper | arXiv | What it offers Stage 1 |
|---|---|---|
| **BalCapRL** | 2605.07394 | "Balanced framework" for RL-based MLLM captioning — likely addresses the coverage/precision trade directly |
| **Claim-Level Rubric Rewards for Video Caption RL** | 2607.05150 | per-*claim* reward granularity rather than whole-caption utility |
| **Perception-R1** | 2506.07218 | dense perception reward built specifically to fix **reward sparsity**; validated on MathVerse incl. Vision-Only |
| **Perceval** | 2604.24583 | a trained per-claim verifier (image-grounded) — a ready-made *precision* signal to pair with utility's *coverage* |
| **CAPEval** | 2608.02589 | reports caption **Precision** predicts downstream performance; **Coverage** predicts general understanding — i.e. which reward term to weight for which goal |

### B.4 Prompt-relevant description, other lines `[SNIPPET]`
- **GeReA** — prompts a multimodal model for **question-aware captions**, then reasons over them for
  knowledge-based VQA. Training-free.
- **ChatCaptioner** — an LLM progressively asks questions, a VLM answers, answers are summarized into a
  description. A different route to question-relevance.
- **QA-guided image description generation** (seen via the Memory-QA paper) — generate candidate recall
  questions for an image, then produce a description that lets you answer them **without the image**.
  **This is a clean, cheap data-synthesis recipe for prompt-relevant articulation that does not require
  the target question at training time — which is also a partial answer to the leakage problem.**

---

## Part C — Suggested reading order across all three scans

1. **VDGD** (2405.15683) — closest to our issue; contradicts Track T; check the prefix-only vs
   prefix+KL-decoding ablation.
2. **From Seeing to Thinking** (2605.20177) §on caption-SFT vs RL — Problem 1 above. We already hold
   this paper locally (`Papers/FromSeeingToThinking.pdf`).
3. **CapRL** (2509.22647) + **CapRL++** (2606.09393) — the Stage-1 recipe to reproduce and modify.
4. **PromptCap** (2211.09699) — prompt-relevance, and the annotation-free synthesis trick.
5. **PTD-PO** (2606.07000) — nearest neighbour for Stage 2; Qwen3-VL-Thinking, MathVerse + MMK12.
6. *(then)* Can-LVLMs-Self-Correct (2404.06510) and Caption This Reason That (2505.21538) for the
   oracle-vs-self gap framing.

**Everything marked `[SNIPPET]` above is a lead, not evidence.** None of it should enter a design
document or a related-work section until read first-hand.
