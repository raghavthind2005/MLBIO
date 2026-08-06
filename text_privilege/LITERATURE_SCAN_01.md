# Text-Privilege — Literature Scan #1: Is "training perception articulation" a live, measurable field?

**Date:** 2026-08-04. **Status:** SCAN, not a design. Nothing here is normative.
**Question asked:** find work that trains a VLM to produce *good textual articulation of an image* —
improving perception + description, benchmarked and measurable — ideally validated on the datasets we
already use for perception hallucination.

**Method:** web search + direct retrieval of abstracts / HTML. **Confidence tags on every entry:**
- `[FETCHED]` — I retrieved the paper page / HTML / PDF myself and the claims below come from that text.
- `[SNIPPET]` — the claim comes only from a search-result summary. **Treat as unverified.** Must be
  read first-hand before it enters any design document.

**Honest limitation stated up front:** most 2026 arXiv IDs below post-date my training data, so I have
no independent knowledge of them — everything is from retrieval this session. I read abstracts and
method sections, **not full papers**. No number below should be quoted in a write-up until the paper is
read end-to-end. Several entries I could only see as snippets; they are marked and should be treated as
leads, not evidence.

---

## 0. Verdict

**The field is real, active, and measurable — and it is crowded.** Both halves of the proposed
pipeline exist separately in the 2026 literature, on our exact model family and our exact benchmarks.

- **Stage 1 (train the model to articulate perception): occupied.** At least five works, one of which
  (`Perception-R1`) rewards perception articulation and reports on MathVerse including its *Vision-Only*
  subset.
- **Stage 2 (privileged self-distillation from a same-model teacher): occupied.** At least six works.
  One (`PTD-PO`) runs on **Qwen3-VL-2B/4B/8B-Thinking** and evaluates on **MathVerse and MMK12** — our
  model family and both of our substrates.
- **The specific triple we would claim is, as far as this scan goes, unoccupied** (§5).

So: the waters are worth testing, but the direction cannot be pitched as "nobody has trained perception
articulation." It has to be pitched on the one asset nobody else has — Track T's **recovery ≈ 0**.

---

## 1. Stage-1 analogues — training articulation quality

### 1.1 CapRL — *Stimulating Dense Image Caption Capabilities via Reinforcement Learning*
arXiv **2509.22647**, ICLR 2026. Code: `github.com/InternLM/CapRL`. `[FETCHED]` (ICLR poster page)

RLVR applied to captioning. Decoupled two-stage: an LVLM writes a caption; the **reward is the accuracy
of a separate, vision-free LLM answering multiple-choice questions from that caption alone**. Caption
quality is redefined as *utility*. CapRL-3B annotates a 5M-caption dataset; gains across 12 benchmarks;
within the Prism framework, comparable to Qwen2.5-VL-72B, +8.4% average over baseline.

**Relevance:** this is the utility-based articulation reward, already published. The consumer is an
*external* vision-free LLM, and the output is a *pretraining dataset* — the loop is never closed back
into the same model's reasoning. That gap is ours to take.
**Not yet verified:** which 12 benchmarks. Needed before positioning.
**Follow-on:** CapRL++ (arXiv 2606.09393), video extension, same reward principle. `[SNIPPET]`

### 1.2 Perception-R1 — *Advancing Multimodal Reasoning via Visual Perception Reward*
arXiv **2506.07218**. `[FETCHED]` (HTML v3) — **closest existing Stage 1 on our benchmark.**

Adds a **visual perception reward** to RLVR: consistency between the model's response and *textual
visual annotations*. Annotations are built by generating correct-answer CoT with Gemini-2.5-Pro, then
having a text-only LLM (Qwen2.5-32B-IT) extract "atomic visual annotations" (~96% accurate by manual
check). A judging LLM scores what fraction of annotations appear in the response.

- Base: **Qwen2.5-VL-7B-IT**. Training data: **1,442 samples** (~100× less than Vision-R1's 200K).
- **MathVerse testmini: 54.3%** (vs Vision-R1-7B 52.4%). Reported best on **vision-only subsets**.
- Note: it does *not* train the model to emit a standalone description — it rewards perception facts
  appearing *within* the reasoning.

**Why this matters to us:** it is the existence proof that perception-articulation reward improves
MathVerse. It is also a direct competitor for Stage 1, and its annotation pipeline needs an external
frontier model + a judge — both of which our CLEVR/Track-T setup could avoid.

### 1.3 VTPerception-R1 — *Explicit Visual and Textual Perceptual Grounding*
arXiv **2509.24776**. `[FETCHED]` (PDF; extraction quality was poor — treat as partial)

Separate **visual** and **textual** perception rewards, and — importantly — the framework **requires the
model to emit perception statements before reasoning**. Evaluated on MathVista and MathVerse.

**This is "describe-then-reason, trained" — the closest published thing to our Stage-1 output format.**
Must be read in full; my extraction did not recover base models or headline numbers.

### 1.4 Perceval — *Perception-centric Process Reward Models*
arXiv **2604.24583**. `[FETCHED]` (HTML)

A **process** reward model that extracts image-related claims from a response and verifies each one
against the image. Backbone Qwen2.5-VL 3B/7B; annotations from Gemini-2.5-Pro; SFT-trained verifier
(LLM-based, **not rule-based**). In RL it masks hallucinated spans and modifies GRPO advantages
(`Â' = Â − α·m·|Â|`, α=0.1). Evaluated on **V\***, BLINK, MMStar, MathVista, MATH-Vision, ChartQA,
MME-RealWorld, RealWorldQA. ~4% over GRPO on visual search, ~3% on math/chart.

**Relevance:** this is a ready-made **per-fact articulation grader** — exactly the instrument our Track-B
Step 0 and any Stage-1 reward would need. It also uses **V\***, our planned Track-I substrate.

### 1.5 Others in this cluster `[SNIPPET]` — leads only
- **CPPO** (arXiv 2601.00501, ICML 2026) — contrastive perception loss applied selectively to percept tokens.
- **CycleCap** (arXiv 2603.18282) — self-supervised cycle-consistency fine-tuning for captioning.
- **VCap** (arXiv 2605.28023) — hypergeometric rewards for weak-to-strong visual captioning.
- **Perception Verified Self-Training** (arXiv 2606.22158).
- **Self-Rewarding VLM via Reasoning Decomposition** (arXiv 2508.19652).

---

## 2. Stage-2 analogues — privileged same-model self-distillation

### 2.1 PTD-PO — *Teaching the Way, Not the Answer*
arXiv **2606.07000**. `[FETCHED]` (HTML) — **the highest-overlap paper in this entire scan.**

Teacher = **the same frozen model conditioned on hints** via in-context learning (`xʰ = [x; h]`);
student stays in the original answer-free context; token-level distillation on failed rollouts.
Hints are two kinds: **spatial** (relevant regions/objects/relations, distractors to ignore) and
**textual reasoning guidance** (high-level intermediate directions), under a **zero-spoiler rule**.

- Base models: **Qwen3-VL-2B / 4B / 8B-Thinking** ← our exact family and sizes.
- Benchmarks: **MathVerse**, **MMK12**, Geo3K, MathVista, We-Math, MMMU-Pro, Counting, MathVerseV,
  LogicVista ← **both of our substrates**.
- 4B: GRPO 68.06% → **PTD-PO 71.23%**; beats answer-conditioned HDPO (68.29%).
- **Hints are generated offline by external models** (Qwen3-VL-235B-A22B-Thinking or Gemini-3.0-Pro)
  **using the ground-truth answer.**

**Read this one first, in full.** It is the nearest neighbour on every axis. Our differentiators against
it are precise and defensible: our privileged channel would be (a) **self-generated**, (b) **purely
perceptual** (PTD-PO explicitly includes reasoning directions), (c) **requiring no GT answer**.

### 2.2 VCSD — *Visual Contrastive Self-Distillation*
arXiv **2607.21556** (local copy: `Papers/2607.21556v1.pdf`). `[FETCHED — read locally]`

EMA teacher evaluates the student's own prefix under **two visual conditions** (real image vs
content-erased control); the token-wise log-prob contrast sharpens the real-image distribution; distilled
by forward KL. Trains on **ViRL39K** (our PAPO training set), **Qwen3-VL 2B/4B/8B**. Reports
62.27→67.04 (2B), 71.30→73.16 (4B), 72.51→76.26 (8B) on a seven-benchmark aggregate. No external
teacher, no privileged answers, **no additional inference-time cost**.

Its related-work names three more privileged-asymmetry OPSD works: privileged answers/reasoning traces
(Zhao et al. 2026), evidence-centered visual views (Yuan et al. 2026), paired evidence for trajectory
selection (Sun et al. 2026). `[SNIPPET, via VCSD's own citations]`

### 2.3 Others `[SNIPPET]` — leads only
- **Vision-OPD / Visual-OPSD** (arXiv 2606.18974) — *crop-conditioned* privileged self-teacher distilled
  into a full-image student. Structurally our Stage 2 with crops instead of text.
- **HDPO** (arXiv 2603.23871) — answer-conditioned privileged self-distillation (PTD-PO's baseline).
- **NOPD** (arXiv 2607.23125) — noisy-student OPSD, no ground-truth answers, covers VQA/chart.
- **AVSD** (arXiv 2605.20643) — adaptive-view self-distillation.
- **Privileged Information Distillation for LMs** (2602.04942) — text-only framing of the same idea.

---

## 3. Measurement frameworks — how this gets scored

### 3.1 Prism — *A Framework for Decoupling and Assessing the Capabilities of VLMs*
arXiv **2406.14544**, NeurIPS 2024. `[SNIPPET]` — but this is a well-established paper.

Two stages: a **perception stage where a VLM extracts and articulates visual information in text**, and
a **reasoning stage where an LLM answers from that text alone**. Enables separate measurement of
perception and reasoning. A 2B LLaVA + GPT-3.5 matches VLMs 10× larger on MMStar.

**This is the measurement framework for our exact construct, and it predates us by two years.** Track T
is essentially a *within-model* Prism experiment (privileged = oracle articulation, self = the model's
own). Any write-up must cite and position against Prism. CapRL already uses it as its eval harness.

### 3.2 CAPEval — *A Decoupled Caption Evaluation across Understanding and Generation*
arXiv **2608.02589**, dated **2026-08-04** (today). `[SNIPPET]`

Reported finding: general understanding tracks caption **Coverage**; **hallucination robustness tracks
caption Precision**; and **Precision is a positive, highly significant predictor of downstream
performance**.

**If this holds, it is a partial answer to the fidelity dose-response question** — i.e. articulation
quality does translate to downstream gains, and precision matters more than completeness for
hallucination. Worth fetching immediately; it is one day old.

### 3.3 Caption This, Reason That — *VLMs Caught in the Middle*
arXiv **2505.21538**, NeurIPS 2025. `[FETCHED]` (verbatim abstract) — **read §7, this one challenges us.**

Cognitive-science framing (Perception / Attention / Memory). Verbatim from the abstract: *"models
struggling with direct visual reasoning show marked improvement when reasoning over their own generated
text captions."* Base model in the decoupling analysis: **Qwen2.5-VL-7B** `[SNIPPET]`.

Two further reported details `[SNIPPET — must verify]`:
- the large gain comes from the **image-free** self-captioning condition;
- **adding the image back alongside the caption shrank the gain**, attributed to attention capacity /
  interference; on category-only tasks, adding images *decreased* performance;
- their conclusion: the bottleneck is **not** reasoning capacity and **not** low-level perception, but
  **the integration of visual features into the reasoning process**.

---

## 4. Benchmark overlap with our own substrates

| Paper | MathVerse | MMK12 | V\* | ViRL39K | CLEVR | Model family |
|---|---|---|---|---|---|---|
| **PTD-PO** | ✅ | ✅ | — | — | — | **Qwen3-VL-2B/4B/8B-Thinking** ✅ |
| **Perception-R1** | ✅ (incl. Vision-Only) | — | — | — | — | Qwen2.5-VL-7B |
| **VTPerception-R1** | ✅ | — | — | — | — | unverified |
| **Perceval** | — | — | ✅ | — | — | Qwen2.5-VL-3B/7B |
| **VCSD** | — | — | — | ✅ | — | Qwen3-VL-2B/4B/8B ✅ |
| **CapRL** | unverified (12 benchmarks) | — | — | — | — | 3B annotator |
| **Caption This, Reason That** | — | — | — | — | — | Qwen2.5-VL-7B |

**Nobody in this scan uses CLEVR.** That is consistent with our own read — CLEVR perception is easy by
design (D1) — but it also means our exact-GT articulation instrument (`score_enum`, multiset-exact,
self-tested 599/599) is an unusual asset in this field, where everyone else grades articulation with a
frontier-model annotator plus an LLM judge.

---

## 5. What is actually unoccupied

Each leg is taken. The **combination** is not:

1. the privileged text is **self-generated by the same model** (PTD-PO: external 235B/Gemini; Perception-R1: Gemini + judge);
2. it is **purely perceptual** — no reasoning directions, no answer (PTD-PO's hints explicitly carry "intermediate reasoning directions");
3. it needs **no ground-truth answer at hint time** (PTD-PO's hint generator is given the GT answer);
4. and it is then **internalized by self-distillation**, so it costs nothing at inference (CapRL never closes this loop — it exports a dataset).

**Honest risk:** a paper whose novelty is "combination of two existing legs" is incremental unless the
*science* carries it. The science we have that nobody else in this scan has is Track T's **recovery
≈ 0** — a controlled, placebo-gated measurement that the model can *use* perceptual text it cannot
*supply*. That is the asset. Lead with the diagnostic; the method is the consequence.

---

## 6. Two literature findings that challenge our premise

**C1 — Self-captioning is reported to HELP, and we measured it not helping.**
Caption This, Reason That (NeurIPS 2025) reports marked improvement from reasoning over self-generated
captions. Track T measured `self − base = −0.030` with recovery **−0.41**. Both cannot be
context-free-true. Candidate reconciliations, in order of my confidence:
- **(a) The image-free condition.** Their headline gain is reported in the *image-free* setting; adding
  the image back shrinks it. **Track T's self arm keeps the image and adds the description — exactly the
  configuration they report as weakest.** This is directly testable and turns our earlier "no-image
  control" from a caveat into a possible *positive* experiment.
- (b) Different model (Qwen2.5-VL-7B vs Qwen3-VL-4B-Thinking) and different task suite.
- (c) Ceiling compression: our MathVerse MC base is 0.808; their tasks are harder.

**C2 — Their stated bottleneck is integration, not perception.**
"Not fundamental reasoning capacity, nor purely low-level perception, but the effective integration of
visual features into the reasoning process." That is the **EXTRACTED-BUT-NOT-INTEGRATED** mode already
named in `TRACK_B_DESIGN_NOTES.md` §4.3. If it is right, **Stage 1 as conceived optimizes a
non-deficit.** Grading our own self-descriptions against the delta settles this for our setting, and the
data is already on disk.

---

## 7. Assessment

**Is the field worth pursuing?** Yes — it is demonstrably measurable (Prism, CAPEval, the Vision-Only
MathVerse subset), actively funded, and publishing at ICLR/ICML/NeurIPS. It is not a dead end and it is
not a backwater.

**Is our specific plan safe?** Not yet. Three things must be settled before any design freeze:

1. **Read PTD-PO (2606.07000) in full.** Same model family, same sizes, both our benchmarks, and a
   privileged same-model teacher. If our contribution cannot be stated as a clean delta against it in
   one sentence, there is no project.
2. **Grade our self-descriptions against the delta.** Free, on disk. Settles C2 — whether Stage 1 has a
   real target — and it is the single cheapest decisive measurement available.
3. **Re-run Track T's self arm image-free.** Settles C1. If self-descriptions help once the image is
   removed, the diagnosis changes from "cannot articulate" to "cannot integrate while attending," which
   is a *different and arguably more interesting* paper — and one that points at a different method.

**What we should not do yet:** design the Stage-1 reward. Four different reward constructions are
already published (vision-free-LLM utility, GT-annotation coverage, per-claim process verification,
contrastive percept-token loss). Choosing among them before knowing which deficit is real would be
picking an instrument before knowing the disease.

---

## Appendix — search log (for reproducibility)

Queries run 2026-08-04, via WebSearch:
1. `reinforcement learning train VLM to generate better image descriptions perception 2026`
2. `describe-then-reason caption-then-answer VLM self-generated caption improves VQA accuracy`
3. `CapRL dense image caption reinforcement learning vision-free LLM reward arxiv`
4. `Prism framework decoupling perception reasoning VLM caption LLM solver benchmark`
5. `training VLM image description reward evaluated POPE HallusionBench MMVP V*Bench perception hallucination 2026`
6. `self-distillation VLM own caption privileged information improve reasoning same model 2026`
7. `caption-conditioned privileged teacher self-distillation VLM textual description internalize perception`
8. `"self-captioning" OR "self-generated caption" VLM hurts helps contradictory evidence reasoning ceiling`
9. `RL perception grounding reward image description diagram geometry MathVerse improve VLM 2026`
10. `VLM caption quality training evaluated CHAIR POPE object hallucination downstream reasoning gains 2026 RL`

Direct retrievals: ICLR poster 10010207 (CapRL); arxiv.org/abs/2509.22647; arxiv.org/html/2606.07000v1
(PTD-PO); arxiv.org/abs/2505.21538 (Caption This, Reason That); arxiv.org/html/2506.07218v3
(Perception-R1); arxiv.org/html/2604.24583 (Perceval); arxiv.org/pdf/2509.24776 (VTPerception-R1,
degraded extraction); local `Papers/2607.21556v1.pdf` (VCSD).

**Not yet retrieved, flagged for the next pass:** CAPEval (2608.02589), Prism full paper (2406.14544),
CapRL benchmark list, Vision-OPD (2606.18974), CPPO (2601.00501), and VCSD's three cited
privileged-asymmetry works (Zhao / Yuan / Sun 2026).
