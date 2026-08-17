# Caption-Distortion Stage 1 — The Method, Explained From Scratch

*Written to the standard set by `PAPO_fixed/PAPO_THREE_RUNS_EXPLAINED.md` and
`RL_SeeingToThinking/runs/RESULTS.md`: state the method in one paragraph, define every term before using
it, give every parameter a value **and a source**, and separate what is decided from what is not.*

**Status: the method below is decided; the training loop is NOT.** Part 6 lists exactly what is missing.
Part 5 is an honest audit of whether this objective can achieve the goal — read it before anything else.

---

# PART 0 — The one-paragraph summary

One model plays two roles. As **captioner** it sees an image and a question and writes a description. As
**answerer** it then answers that question from the description alone, **with the image taken away**. We
score the caption by how little the answer distribution moves when the caption replaces the image — a KL
divergence we call the *distortion*. Captions with low distortion are reinforced. The claim being tested is
that a caption trained this way is one that carries the perceptual information the model's own reasoning
actually uses, which is precisely the ability our prior work found the model to lack.

---

# PART 1 — The goal, and the chain that leads here

**The program's question.** When a VLM answers a perception-dependent question wrongly, is that an
*extraction* failure (the information is in the image but never reaches the reasoning) or a *reasoning*
failure (the information arrives and is misused)?

**The program's answer, already measured.** Track T (`TRACK_T_SIGNAL_REPORT.md`, MathVerse, n=497,
Qwen3-VL-4B-Thinking) found **extraction deficit**:

| arm | Δ vs base (MC) | reading |
|---|---|---|
| privileged — dataset ground-truth perceptual facts injected as text | **+0.075** | the model **can use** serialized perception |
| self — the model's own question-aware description | −0.030 (n.s.) | the model **cannot self-serialize** it |
| recovery = (self−base)/(privileged−base) | **−0.41** | none of the oracle benefit is self-recovered |

**Therefore the target capability is self-serialization**: get the model to write, itself, the kind of text
that helps it. Probe A then showed an *external* specialist captioner (CapRL), even question-targeted, only
recovers baseline (T3−T0 = +0.006, p=0.945) — so buying a better off-the-shelf captioner is not the answer
either.

**This project's bet.** Train self-serialization directly, with a supervision signal that needs no human
captions: the model's own image-conditioned behaviour. That is the caption-distortion objective.

---

# PART 2 — The method from scratch

## 2.1 Vocabulary

- **Policy `π_θ`** — the model as a function assigning a probability to every next token. One set of
  weights, used in both roles. There is no second network.
- **Captioner role** — `c ~ π_θ(· | I, x, q_cap)`: image `I`, question stem `x`, caption instruction `q_cap`.
- **Answerer role** — `ỹ ~ π_θ(· | c, x)`: caption and question, **no image**. This is the *blind* pass.
- **Reference distribution** — `π_θold(· | I, x)`: what the model does with the image. Never sampled from;
  only scored against. This is the *target behaviour*.
- **Distortion `D(c)`** — how far the blind answer distribution sits from the image-conditioned one.

## 2.2 The objective

```
D(c) = KL( π_θold(·|c,x)  ‖  π_θold(·|I,x) )          ← minimise this
J_cap(θ) = − E[ sg[ D̂(c) ] ]                          ← so reward = −D̂(c)
∇J_cap  = − E[ sg[ D̂(c) ] · ∇_θ log π_θ(c | I,x,q_cap) ]
```

`sg[·]` is stop-gradient: the answerer and the reference are *evaluated*, never trained. **The gradient
reaches the caption tokens only.** Stage 1 trains `J_cap` alone — no task reward, no `λ` (D1).

## 2.3 How the distortion is estimated

The spec estimates `D̂` by summing the log-ratio of the *sampled* answer tokens. We use the
Rao-Blackwellised form instead, justified by the chain rule for KL:

```
KL(p‖q) = E_{y~p} [ Σ_j KL( p(·|y_<j) ‖ q(·|y_<j) ) ]
```

At each answer position we compute the **exact full-vocabulary KL** between the two contexts and sum over
positions. Same estimand, unbiased, strictly lower variance, and the logits are already computed (D9).

Three properties of that choice, each load-bearing:

1. The trajectory supplying the positions **must** be sampled from `π(·|c,x)` — the identity fails
   otherwise (gate **G-SAMPLED**).
2. The sum **must** include the EOS position, or we estimate a truncated KL (gate **G-EOS**).
3. Every per-position term is a KL, hence **≥ 0 by construction**. Any negative value is *proof* of a bug —
   misalignment, wrong context, or swapped `p`/`q` (gate **G-FINITE**). The spec's signed one-sample
   estimator would not have given us this check.

## 2.4 Why the two prompts must be twins

`D` isolates "caption vs image" **only if** the two scored prompts are identical in every respect except
the evidence. Same question text, same answer-format suffix, same chat template. Any other difference means
we measure template effects and report them as perceptual distortion (D17, gate **G-PARITY**). The answerer
must additionally be provably blind — no vision tokens, no `pixel_values` (gate **G-BLIND**).

---

# PART 3 — The pipeline as built

```
ViRL39K row
   │
   ├─ parse ──► stem  (options removed)  ─────► CAPTIONER prompt = [image] + stem + q_cap
   │            full_text (options kept) ──┬──► ANSWERER  prompt = caption + full_text + SUFFIX   (blind)
   │                                       └──► REFERENCE prompt = [image] + full_text + SUFFIX
   │
   └─ answer ─► used for accuracy readouts only; NOT in the Stage-1 gradient
```

Step 1 (pool construction) is **complete and gated**: 27,326 eligible of 38,870; 200 drawn (76 letter /
124 numeric) with a nested 50-item subset; manifest `1b28495bd9151e54…`; all 11 pool gates pass.

---

# PART 4 — Parameter table (value · source · does it affect the result?)

| Parameter | Value | Source | Result-affecting? |
|---|---|---|---|
| Backbone | `Qwen3-VL-2B-Instruct` | D4 | **yes** |
| Train pool | `PAPO_ViRL39K_train` @ `ff6996d5` | D5 | **yes** |
| Answer shapes kept | letter, numeric (free text dropped) | D22 | **yes** — drops 7,892 rows |
| Answer form | short answer, no chain | D6 | **yes** — defines the estimand |
| `SHARED_SUFFIX` | `Answer with only the final answer, in \boxed{}.` | D25 | **yes** |
| Captioner sees options? | **no** (stem only) | D18 | **yes** — leak channel |
| Estimator | per-position exact full-vocab KL, summed | D9 | **yes** |
| Length normalisation | sum, not mean | D7 | **yes** |
| Advantage | GRPO group-relative, `(−D̂−mean)/std` | D8 | **yes** — deviates from spec |
| Group size `G` | 5 | D10 (matches VLM-CapCurriculum `rollout.n`, PAPO) | **yes** |
| Answers per caption `M` | 1 (3 on the pilot subset) | D10, D21 | **yes** |
| Sampling, both roles | temp 1.0, `top_p` 1.0, `top_k` −1 | D23 (amended) | **yes** |
| Reference KL | `low_var_kl`, `kl_coef` 1e-2 | D13 (matches VLM-CapCurriculum, PAPO β) | **yes** |
| `max_pixels` / `min_pixels` | 4,194,304 / 262,144 | D24 (EasyR1 + VLM-CapCurriculum default) | **yes** |
| Vision tower | open (`freeze_vision_tower=false`) | D12 (matches their Stage 1) | **yes** |
| Caption length cap | **undecided** — set by Pilot 0 (b) | D14 | **yes** |
| Engine | EasyR1/verl, container `easyr1_vllm0112` | D2 | no (infrastructure) |
| Seed | 0 (pool draw) | D20 | no (reproducibility) |

**Deliberate deviations from the source spec**, all recorded: GRPO baseline instead of bare REINFORCE (D8),
Rao-Blackwellised estimator instead of the one-sample form (D9). Both are argued in `DECISIONS.md`.

---

# PART 5 — Alignment audit: can this objective reach the goal?

Three concerns. The first is structural and I think it is the most important thing in this document.

## 5.1 The objective's optimum is *parity with the image*, never better than it

`D` is minimised when blind behaviour **matches** image-conditioned behaviour. So:

- A caption carrying the correct perceptual fact, which makes the blind model answer **correctly** where
  the image-conditioned model answers **wrongly**, receives a **large** `D̂` and is **penalised**.
- The objective's global optimum is `π(·|c,x) = π(·|I,x)` — perfect mimicry, including mimicry of the
  model's perceptual errors and of its uncertainty.

So `J_cap` alone is a **distillation objective with the current image-conditioned policy as its ceiling.**
It cannot, even in principle, produce captions that beat looking at the image.

Whether that is a problem depends on which goal we mean:

| goal | does J_cap serve it? |
|---|---|
| "make the model able to **serialize what it already perceives**" | **yes** — this is exactly that, and Track T says the gap is real and large |
| "make serialized text **more useful than the image**, as Track T's +7.5 showed" | **no** — that gain came from *ground-truth* text **added alongside** the image; here text **replaces** the image and parity is the ceiling |

This is not a reason to abandon the design, but it must be stated in any writeup, and it bounds what a
positive result can claim. It also means the honest success criterion is *"blind-from-caption accuracy rises
toward image-conditioned accuracy"*, never *"exceeds it"*.

## 5.2 Items where the image-conditioned policy is wrong invert the training signal

On such rows the target behaviour is a wrong answer, so we train captions to faithfully reproduce a
perceptual error — the same defect that disqualified `D_perc` (D3), arriving through a different door.
Track T's whole premise is that this class is **large**. D11 (filter to image-correct items) is still
undecided, and Pilot 0 measurement (a) exists precisely to size it. **This is the measurement that should
decide whether the filter is mandatory.**

## 5.3 The goal is about reasoning; the current estimand is about answers

Track T's extraction deficit was measured inside a Thinking model's reasoning chain. Stage 1 judges captions
by the short-answer behaviour of an Instruct model (D6). The step "a caption that supports the right short
answer also supports the right reasoning" is an **assumption we have not tested**. D6 bought a clean,
low-variance estimator by narrowing the estimand; that trade is defensible but it is a trade, and the gap
between what we measure and what we care about should be closed later — by evaluating trained captions in a
reasoning setting, not by assuming transfer.

---

# PART 6 — What is NOT decided (the honest gap list)

The single largest gap: **the training loop does not exist yet.** Everything decided so far concerns data
and the estimator. Undecided:

1. **How `D̂` becomes a gradient inside verl** — which file, which hook, how the two-context scoring pass is
   scheduled relative to rollout. PAPO's perception-KL is the precedent but the wiring is not written.
2. `θ_old` refresh cadence / `ppo_epochs` — on-policy or off-policy (OPEN-3).
3. Batch shapes, steps, epochs, learning rate (OPEN-7).
4. Caption length cap (D14) — awaiting Pilot 0 (b).
5. Whether to filter to image-correct items (D11) — awaiting Pilot 0 (a); see §5.2.
6. **Evaluation set and success criteria (OPEN-8)** — must be frozen *before* any full run, per
   `feedback_normative_must_be_frozen`. Currently nothing defines what "it worked" means.
7. Arm/condition naming. PAPO has A/B/C; we have one unnamed configuration and no control arm. At minimum a
   Stage-1 run needs something to be compared against.

**Process note, stated plainly:** we built the data pipeline before writing this document. The parser work
was necessary and it caught five real leak channels, but the correct order is method → data, and this file
should have existed first.

---

# PART 7 — Glossary

| term | meaning |
|---|---|
| `I`, `x`, `q_cap` | image, question, caption instruction |
| `c` | the generated caption — the object being trained |
| `ỹ` | the answer sampled **blind**, from the caption only |
| `D(c)`, `D̂(c)` | distortion, and its estimate |
| `θ_old` | frozen weights used for all scoring inside `D̂` |
| stem / full_text | question **without** options (captioner) / **with** options (answerer + reference) |
| dead group | all `G` captions score alike ⇒ zero advantage ⇒ no gradient |
| G-\* | the pre-registered correctness gates (PARITY, BLIND, ALIGN, EOS, SAMPLED, FINITE, PARSE) |
