# Probe A — Does a Strong External Articulation Help? (design notes)

**Status:** WORKING DESIGN NOTES — **NOT FROZEN, NOT PRE-REGISTERED.** No bar or verdict rule here is
normative (`feedback_normative_must_be_frozen`). A `PROBE_A_PREREGISTRATION.md` with artifact hashes gets
frozen before any confirmatory generation.
**Date:** 2026-08-05.

## 0. What this probe is, and what it is not

**Is:** an *upper-bound* test. We borrow a released, RL-trained articulator (CapRL) as a drop-in
"already-trained Stage 1" and ask whether its descriptions improve a reasoner's accuracy.

**Is not:** a test of whether *training our own* articulation helps. Scope the claim accordingly.

**Directional asymmetry — pre-state it:**
- **A null is strong.** If a SOTA released articulator doesn't help, training our own 2B to articulate is
  very unlikely to pay. Stage 1 is not the lever; the deficit is binding/integration.
- **A positive is weaker.** CapRL-4B may articulate better than anything our model could learn, and an
  external captioner is not self-articulation. A positive licenses "worth training," not "will work."

## 1. Validation dataset — **MMStar** (primary)

`[verified this session]` 1,500 human-curated multiple-choice samples, 6 core capabilities / 18
fine-grained axes; NeurIPS 2024; `github.com/MMStar-Benchmark/MMStar`.

Why it wins on every stated criterion:

| Criterion | MMStar |
|---|---|
| **Rule-based scoring** | *"accuracy … calculated by some heuristic rules"* — no LLM judge |
| **Vision-dependence** | a **construction criterion**: *"the collected samples can be correctly answered only based on understanding the visual content"* — this is the control Track T lacked (§11.2) and CapRL enforces via its QA filter |
| **Contamination** | second construction criterion is *"minimal data leakage"* — addresses Track T's §11.1 top caveat |
| **Size** | n=1,500 → paired McNemar/bootstrap on modest effects |
| **Perception isolation** | has **fine-grained perception** and **coarse perception** capability axes → we can claim a perception gain without claiming a reasoning gain, the way RH-Bench's 450/450 split would have, but judge-free |
| **Literature usage** | widely used — **and it is one of CapRL's own 12 evaluation benchmarks**, so there is a published external reference point for the captioner |

**Secondary (optional):** **BLINK** (perception-specialised, large headroom) and **MMVP** (CLIP-blind
*pairs* — the pairing is a built-in consistency control). Both MC, both judge-free.

**Demoted, with reasons:**
- **POPE** — judge-free, but binary yes/no with a well-known **yes-bias** axis. A question-blind caption
  that omits the queried object will push the model toward "no", which *raises* accuracy on adversarial
  splits and *lowers* it elsewhere. That is a **response-bias shift, not a perception gain**, and it
  would be easy to misread. If used at all, POPE's yes-ratio / FP-ratio must be reported alongside
  accuracy. Also near-ceiling for modern VLMs.
- **RH-Bench** — judge-dependent as run (Qwen3-32B). MC-only branch exists
  (`compute_scores.py:16`) giving n=228 on the halu half; keep as a secondary endpoint only.
- **HallusionBench** — official scorer is GPT-based. Ideal paired design, but needs a custom FP-audited
  extractor. Defer.
- **MathVerse** — keep as the **anchor** where the oracle delta and +0.075 effect size live, not as the
  perception benchmark (ceiling-compressed, contamination-flagged).

## 2. Fixed elements

- **Captioner (constant):** **CapRL-Qwen3VL-4B** — their "High Performance, Advanced Captioning Ability"
  tier. Deliberately the strongest, so a null is a strong upper bound. Instruct-class (verified §7b of
  `CAPRL_TECHNICAL_READ.md`) — fine, it is a captioner, not a reasoner.
- **Reasoners:** **Qwen3-VL-4B-Thinking** and **Qwen3-VL-4B-Instruct** — *same family, same size*.
  Pinning size+family is what makes the thinking-vs-instruct contrast interpretable rather than a
  confound with training data or capacity.
- **Payload placement:** reuse Track T's frozen mechanics verbatim — wrapper
  `"From the figure, I can see the following:\n<payload>"`, prefilled into the assistant reasoning turn.
  For Thinking, the chat template auto-opens `<think>` (`mv_gen.py:78-81` asserts this). For Instruct,
  there is no `<think>`; placement must be defined explicitly and logged (assistant prefill, not user
  turn, to keep the roles parallel).
- Decode, K draws, two-level bootstrap, truncated=wrong + concluded-only: inherit Track T's protocol.

## 3. Arms

**Core factorial — 2 reasoners × 3 caption conditions:**

| | none (baseline) | CapRL caption | **placebo caption** |
|---|---|---|---|
| **Qwen3-VL-4B-Thinking** | T0 | T1 | T2 |
| **Qwen3-VL-4B-Instruct** | I0 | I1 | I2 |

**Plus:**
- **P — Prism arm:** text-only LLM answers from the CapRL caption **alone, no image**.
- *(optional)* **C — ceiling arm:** Thinking + caption from a much stronger captioner (e.g. a frontier
  model), giving a practical articulation ceiling so a recovery fraction can be computed.

### 3.1 Changes from the proposed design, and why

**(a) ADD the placebo arms (T2, I2) — this is the one non-negotiable addition.**
Without a content-mismatched, length-matched caption from a *different* image, "caption helps" cannot be
separated from "any long, authoritative-sounding prefill changes behaviour." This is not hypothetical in
our own data: Track T measured placebo **−0.029** and self **−0.030** — *indistinguishable*. The placebo
is the only reason that reading was safe. Machinery exists (`mv_placebo.py`, deterministic donor
assignment, `placebo_assignment.json` as the pattern).
Placebos are needed on **both** rows, because the placebo effect may itself differ by model type — which
is precisely what the factorial is for.

**(b) REPLACE "captioner answers the validation set itself" with the Prism arm (P).**
As proposed, that arm is close to uninformative: CapRL was RL-trained *for captioning*, so its VQA
ability may have degraded, and its accuracy measures residual QA skill, not caption value. It controls
for nothing in the main comparison.
What we actually want is the caption's **information content, measured independently of any VLM's ability
to integrate it** — i.e. Prism's construct, and CapRL's own reward construct: a text-only LLM answering
from the caption alone. This single arm resolves the ambiguity that has dogged the whole project:

> If **P is high** (the caption carries the information) but **T1 ≈ T0** (the thinking model doesn't
> benefit), the deficit is **integration/binding, not articulation** — and Stage 1 is the wrong lever.
> If **P is low**, the caption genuinely lacks the needed content — articulation *is* the lever.

**(c) PIN the Instruct model** to Qwen3-VL-4B-Instruct (§2).

**(d) Report per-arm truncation.** CapRL++ regularises captions at τ1=2048/τ2=3072 tokens. A ~2k-token
caption plus a long thinking chain will push cap hits. PAPO Arm C is the precedent: perception pressure
moved truncation 7.2% → 16.6%. Truncated=wrong primary, concluded-only sensitivity, per-arm rates
reported — Track T's protocol, carried over unchanged.

## 4. The contrasts that matter

| Contrast | Question |
|---|---|
| **T1 − T0** | does a strong articulation help the *thinking* model? |
| **I1 − I0** | does it help the *instruct* model? |
| **(I1−I0) − (T1−T0)** | **the interaction — the washout test.** A positive interaction is the finding: articulation helps short-chain models and is washed out by long chains |
| **T1 − T2**, **I1 − I2** | content-specific effect (vs wrapper + length) |
| **T2 − T0**, **I2 − I0** | the cost of *any* prefill |
| **P** | caption information content, integration-free |

The interaction term is the scientifically interesting quantity, and it is the one the proposed design
already had the right instinct about. It explains why VDGD and *Caption This, Reason That* report gains
while Track T measured none — **if** the interaction is real.

## 5. Known limitations to pre-state

1. **The caption is question-blind.** CapRL never sees the question (that is its leak-proof property,
   `CAPRL_TECHNICAL_READ.md` §4). On fine-grained perception items a generic dense caption may simply not
   mention the queried detail. This is a **scope mismatch, not a bug** — and it is the honest reason a
   null would not kill the direction: it would instead motivate the prompt-conditioned variant.
2. **No true oracle on MMStar** → no recovery fraction unless the optional ceiling arm (C) is run.
   MathVerse remains the anchor where recovery is defined.
3. **External articulator ≠ self-articulation** (§0).
4. **CapRL-Qwen3VL-4B is Instruct-class and was never trained/validated in the thinking regime**
   (`CAPRL_TECHNICAL_READ.md` §7b) — its captions are not tuned for consumption by a long-chain reasoner.

## 6. Open decisions

1. Text-only LLM for the Prism arm P — a small one (CapRL uses Qwen2.5-3B-Instruct as `M_L`) or a strong
   one? A small one matches CapRL's construct; a strong one gives a cleaner information-content read.
2. Run the optional ceiling arm C, or accept a no-recovery design on MMStar?
3. K draws per item and whether to reuse Track T's decode settings verbatim.
4. Whether to add BLINK/MMVP now or hold them as replication substrates for a positive result.
