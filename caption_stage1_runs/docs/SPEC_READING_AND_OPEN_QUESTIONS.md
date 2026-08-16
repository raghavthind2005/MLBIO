# Caption-distortion spec — my reading, and what must be resolved before code

**STATUS: WORKING NOTES. NOT FROZEN, NOT PRE-REGISTERED. No decision here is binding.**
Written 2026-08-16 against `SOURCE_SPEC_hackmd.md` (SHA-256 `5bb96f47…5c35a2d0`, 170 lines, read in full).

---

## 1. What the spec says (restated, no additions)

One policy `π_θ` plays **two roles**:

| role | conditioning | sampled |
|---|---|---|
| **captioner** | `I, x, q_cap` (image + question + caption instruction) | `c ~ π_θ(·\|I,x,q_cap)` |
| **answerer** | `c, x` (caption + question, **no image**) | `ỹ ~ π_θ(·\|c,x)` |

Because `c` is generated conditioned on the question `x`, it is a *question-conditioned textual representation
of the image*. The target property is `π(·|c,x) ≈ π(·|I,x)`.

**Distortion** (spec §"Caption-distortion objective"):

```
D(c) = KL( π_θold(·|c,x)  ||  π_θold(·|I,x) )
```

Both sides use `θ_old`; the spec states this explicitly — the distortion loss is taken w.r.t. the *captioning*
process only. Spec notes forward-KL / JS are open alternatives.

**Estimator** — a single answer trajectory `ỹ=(y_1..y_T) ~ π_θold(·|c,x)`:

```
D̂(c) = Σ_{j=1..T} [ log π_θold(y_j | c,x,y_<j) − log π_θold(y_j | I,x,y_<j) ]
```

**Caption objective and its gradient:**

```
J_cap(θ)  = − E[ sg[ D̂(c) ] ]
∇J_cap    = − E[ sg[ D̂(c) ] · ∇_θ log π_θ(c | I,x,q_cap) ]
```

i.e. **REINFORCE on the caption tokens with scalar reward `−D̂(c)`**. Policy evaluations inside `D̂` are fixed;
gradient reaches the caption only.

**Task objective:** `J_success(θ) = E[R(ỹ)]`, with `c ~ π_θ(·|I,x,q_cap)`, `ỹ ~ π_θ(·|c,x)`.

**Final:** `J(θ) = J_success(θ) + λ·J_cap(θ)`, `λ ≥ 0`. Spec also offers `J_cap` **as a pretraining objective
before RL** as an alternative to the joint form.

**Central claim:** *"A good visual caption is one that preserves the model's downstream reasoning behavior"* —
so the full-image policy supervises task-relevant perception with no human captions.

---

## 2. What the spec does NOT specify (must be decided, not assumed)

These are genuine gaps in the source, not disagreements with it.

| # | Gap | Why it is load-bearing |
|---|---|---|
| G1 | **No baseline / advantage in `∇J_cap`** | Written as bare REINFORCE. Determines variance and whether training is stable at all — see §3.1. |
| G2 | **`D̂` is a SUM over answer tokens, not a mean** | Makes the reward scale with answer length ⇒ a length-hacking channel — see §3.2. |
| G3 | **Where `J_success`'s gradient flows** | `ỹ` and `c` are *both* sampled from `π_θ`. Gradient can reach answer tokens, caption tokens, or both. Completely changes what is trained. |
| G4 | **Nothing forbids the caption from containing the answer** | The stated objective is *satisfied* by an answer-leaking caption — see §3.3. |
| G5 | KL estimator family | Spec's `Σ log(p/q)` is the unbiased-but-high-variance k1 form; can be negative per-sample. Our PAPO stack uses k3 (`low_var_kl`). |
| G6 | Whether `q_cap` sits in a separate turn, and whether the answerer sees `q_cap`/`I` at all | Determines prompt construction and whether the "blind" condition is truly blind. |
| G7 | `λ`, group size, #answer samples per caption, `θ_old` refresh cadence | Standard RL knobs, all absent. |
| G8 | Reward `R(ỹ)` definition | Verifier? Format term? Ours must match the benchmark's scoring, not a proxy. |

---

## 3. Issues I consider serious (raised as head engineer, before implementation)

### 3.1 REINFORCE with no baseline, on a reward that is ≤ 0 in expectation

`D(c)` is a KL, so `E[D̂] ≥ 0`, so the reward `−D̂(c)` is **non-positive in expectation** (individual k1 samples
can be positive, since single-sample `log p/q` is signed). The gradient as written is *unbiased*, but with a
systematically negative coefficient on `∇log π_θ(c|·)` every sampled caption gets pushed **down** in
probability, differing only in degree. That is the textbook motivation for a baseline; without one the run
risks entropy collapse and is in any case very high variance, because `D̂` is an unbounded sum of log-ratios.

**Standard fix that fits our existing stack:** sample a group of `G` captions per `(I,x)` and use the
group-relative advantage `A_i = (−D̂_i − mean)/std` — i.e. GRPO on the caption. This is a *deviation from the
written spec* and needs explicit sign-off.

### 3.2 Length hacking: shorter answers mechanically reduce distortion

`D̂` sums over the `T` tokens of `ỹ`, and `ỹ` is sampled from `π(·|c,x)` — so **the caption influences `T`**.
A caption that induces a 1-token answer has a 1-term sum; a caption that induces a 500-token chain has 500.
Minimizing `E[D̂]` is therefore partly served by *making the answerer terminate early*, independent of any
perceptual fidelity.

This is not hypothetical for us. Probe A (`text_privilege/PROBE_A_FULL_RUN_REPORT.md`) measured exactly this
failure mode on this model family: injecting a text block into the reasoning prefill drove **premature
`</think>` closure in 55% (blind caption) and 84% (placebo) of generations**, and short-circuited generations
were **11–13 accuracy points worse**. Here the answerer is caption-only *by construction*, which is the regime
where we already documented that pathology. A length-summed reward would actively reward it.

Options: per-token mean instead of sum; length-normalized advantage; fixed/capped answer budget; explicit
length control in the gate set. Each changes the estimand — needs a decision, not a default.

### 3.3 The objective is satisfied by a caption that states the answer

`D(c)` is minimized when `π(·|c,x)` matches `π(·|I,x)`. If the caption simply asserts the conclusion, the
caption-conditioned answer distribution becomes sharply peaked on that conclusion — and if the image-conditioned
policy is also peaked there, **distortion goes to ≈0 with zero perceptual content transferred.**

So `J_cap` as literally written optimizes *behavioral agreement*, which does not distinguish "the caption
carried the visual facts" from "the caption carried the verdict." The paper's own framing ("task-relevant
perception without human captions") assumes the former. We have prior machinery for this: Track T ran an
answer-free audit (0 leaks) and Probe A measured a 0.36% format-leak rate with explicit gates. **We should
treat leak-rate as a first-class gate, not a footnote.**

### 3.4 `J_cap` optimizes self-consistency, not correctness

The supervision target `π_θold(·|I,x)` is the *current model's own* image-conditioned behavior, which is often
wrong. On those items `J_cap` trains captions to faithfully preserve a **wrong** answer. This is presumably why
`J_success` is in the objective — but it means `λ` is not a mere strength knob, it trades *fidelity-to-self*
against *correctness*, and the two objectives can point in opposite directions on the same item. Worth measuring
directly (distortion vs accuracy on items the image-policy gets wrong), not just tuning.

### 3.5 This is close to published work — check before investing

The answerer is **blind** (`c,x`, no image). That is structurally close to **Vision-SR1** (arXiv 2508.19652,
ICLR 2026): model emits a self-contained visual description, is re-prompted blind on that description, and is
rewarded for sufficiency. Our existing scoop warning for `text_privilege` applies here with more force, because
this design shares the blind-re-prompt core. The *distinguishing* element is the KL-distortion signal (a dense,
distributional criterion) versus Vision-SR1's sufficiency reward. That distinction is defensible but must be
verified against their paper before we commit compute.

### 3.6 Prior first-party evidence bearing on the expected outcome

Not an objection — context for what we should predict, so we do not fool ourselves later:

- **Track T:** the model *can* use serialized perception (oracle text **+7.5 pts**) but *cannot self-serialize*
  it (self-description recovery **−0.41**). This method is a direct attempt to train that gap away — a good
  motivation.
- **Probe A:** an external, question-targeted, RL-trained captioner (CapRL) recovered baseline but added
  **nothing** over it (T3−T0 = +0.006, p=0.945), with 71 fixes and 71 breaks exactly cancelling.
- Implication: a credible success bar must beat *caption-mediated* baselines, and the headline metric must be
  chosen so that "recovers what deliberation already achieved" cannot be misread as a gain.

---

## 4. Engineering notes (efficiency, for when we cost this out)

- `log π_θold(y_j|c,x,y_<j)` are **exactly the generation logprobs** of the answer rollout — no extra forward
  pass if we capture them at sampling time.
- Only `log π_θold(y_j|I,x,y_<j)` needs an additional scored forward: same continuation `ỹ`, different prefix
  (image instead of caption). This is structurally the **same operation as PAPO's perception-KL** (score the
  real-image rollout tokens under a corrupted-image context), so we have a verified precedent in
  `PAPO_fixed/.../dp_actor.py` for the token-alignment and masking.
- Per `(I,x)`: 1 caption generation (with image) + 1 answer generation (no image) + 1 scoring forward (with
  image) + the training forward on caption tokens. Group sampling multiplies the first three.
- Prefix lengths differ between the two scorings, so **token alignment must be asserted**, not assumed — the
  continuation must be byte-identical and the log-prob index offsets independently verified.

---

## 5. Open decisions — owner: user

Nothing below is decided. Listed in the order they change the most downstream code.

1. **Regime:** `J_cap` alone as Stage-1 pretraining, or the joint `J_success + λ·J_cap`?
2. **Backbone**, and whether it must be a Thinking model.
3. **Dataset / task substrate**, and whether it overlaps our existing pools (contamination + comparability).
4. **Codebase:** extend our verified verl/EasyR1 PAPO stack, or build standalone?
5. Baseline/advantage for `J_cap` (§3.1); length normalization (§3.2); leak gates (§3.3).
6. `R(ỹ)` definition; `λ`; group size; `θ_old` refresh; KL estimator family.
7. What the evaluation is, and what would count as success — frozen **before** any full run.
