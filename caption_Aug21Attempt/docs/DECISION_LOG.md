# Caption-Distortion — Decision Log

**STATUS: WORKING LOG. Nothing is frozen. No pre-registration exists. No training code written, no
training runs launched.**

*As of 2026-08-23: substrate, backbone, prompts and estimand are settled (S1–S10); the PROVISIONAL tier is
empty. **S8 departs from `SOURCE_SPEC.md`** — the spec writes reverse KL, we adopt forward. That disagreement
is recorded here, never patched into the spec (§0). Open before any full run: **O3** (estimator family, next
up), O4/O2/S3b/L1, then RL config and the frozen success criterion.*

Append-only in spirit. Every entry records who decided, on what evidence, and — where it matters — what was
argued on the other side, so a decision can be revisited without re-deriving the argument from scratch.

Legend: **[U]** user decision · **[CC]** my recommendation, awaiting sign-off · **[V]** verified fact, with
the source of verification named.

Status tiers, used deliberately:
- **SETTLED** — decided; changing it is a change of experiment.
- **PROVISIONAL** — we are proceeding this way, but it is explicitly *not* frozen and is under active review.
- **OPEN** — not decided. §3 orders these by when they must be solved.
- **REJECTED** — considered and declined, recorded so it is not re-litigated.

---

## 0. Provenance and scope

- **Spec authority:** [`SOURCE_SPEC.md`](SOURCE_SPEC.md), SHA-256
  `c1c48cec9fac0f261cfc828f74612cbae706e85752fa701d128fcec75f1ec923`, 169 lines, stored byte-exact.
  It is never edited. Any disagreement with it is recorded here, not patched into it.
- **Clean break [U, 2026-08-21]:** this attempt supersedes `caption_stage1_runs/` entirely. **Nothing from
  that project's decision chain is inherited.** Where a question was answered there, it is re-derived here on
  its own merits or it is open. Its measured *facts* may be cited as evidence (they were paid for); its
  *decisions* carry no authority.
- **[V]** The current spec differs from the earlier one in **exactly one place** — the `J_success` block, now
  sighted. Verified by diff against `caption_stage1_runs/docs/SOURCE_SPEC_hackmd.md` @ git HEAD.

---

## 1. SETTLED

### S1 — `J_success` rewards the **sighted** answer **[U, 2026-08-21]**

`J_success(θ) = E[R(y)]` with `y ~ π_θ(·|I,x)`. Task accuracy on image + question.

*Why it matters beyond bookkeeping.* `π_θ(·|I,x)` now plays three roles at once: what the accuracy reward
improves, the reference `q` inside the distortion, and the model we deploy and evaluate. One distribution,
three jobs, no mismatch — this is what licenses the central mechanism claim: **the accuracy term raises the
teacher while the caption term closes the gap to it.** It is the clean inversion of PAPO, whose repulsive KL
to a deliberately-blinded teacher names a direction to flee rather than a destination (our measurement:
C-pure perception-KL 0.047 → 0.102 while validation sat at 0.536 vs GRPO's 0.540).

*Consequence:* the caption earns **no** reward for being correct — only for being faithful to sighted
behaviour. A sharper factorisation than Vision-SR1, where both terms are answer-correctness and the
description reward can therefore be raised by getting better at answering from thin descriptions.

*Consequence:* the caption term reaches sighted VQA **only through shared weights.** The bet is that training
the model to serialise what it sees reshapes representations the sighted pathway reuses. Plausible, motivated
by Track T — and **unproven. This is the load-bearing step of the whole design.**

### S2 — Captioner sees the **entire** question, including options **[U, 2026-08-23]**

*Argument for (user's, and sound):* symmetry. A caption generated for a different `x` than it is evaluated on
can omit precisely the distinction that separates the options. It also deletes the options-parser bug surface
— a real, measured cost (the prior attempt found five undocumented formats, 9.2% unparseable rows, and a
full-scale gate failure on a histogram legend that was option-*shaped* without being options).

*Cost, accepted knowingly:* leakage is no longer prevented structurally. **Leakage moves from prevention to
instrumentation, and the instruments are not optional.** See §5 and roadmap item **L1**.

*Partially relieved by S8 [2026-08-23].* This entry originally read "combined with reverse KL (P1) — which is
itself the leak-attractive direction — the two stack." **That stacking no longer applies.** S8 adopts forward
KL, which *penalises* a verdict-asserting caption rather than rewarding it, so the objective now works against
leakage instead of with it. The instruments remain mandatory — forward disfavours leaking, it does not make it
impossible, and L1 is also how we will adjudicate the S8 head-to-head.

### S3 — `q_cap` is permissive **[U, 2026-08-23]**

A single "do not state the answer" clause; otherwise freedom to report whatever is derivable from the image.
Deliberately not over-constrained: Track T's precedent is an over-restrictive caption instruction ("do not
infer relationships") that suppressed legitimate content and biased the result. *Exact wording still to be
drafted — roadmap item S3b.*

### S4 — Substrate: **Vision-SR1-47K**, pinned at revision `2900b038` **[U, 2026-08-23]**

Apache-2.0, 47,628 rows, 20 parquet shards.

*Decisive reason — the control arm gets a published anchor.* S1 makes our primary comparison *accuracy-only
GRPO* vs *accuracy + caption-KL* on identical data, which is **exactly** the comparison Vision-SR1 reports
(+1.7 / +1.5 / +3.5 over answer-reward-only GRPO on this data). If our Arm A lands far from their
answer-reward-only numbers we learn the setup is wrong *before* interpreting Arm B. That is a free correctness
check on the most expensive part of the experiment, and ViRL39K cannot provide it.

*Secondary:* options ship pre-parsed; `data_source` and the `path` prefix give principled stratification
without inventing categories.

*Costs accepted:* ViRL39K was already cluster-verified and we had measured its 34% text-solvable rate
first-hand; Vision-SR1-47K's equivalent number is unmeasured (roadmap **M1**).

*Vision-necessity is not invented by us:* **PAPO already splits its evaluation into "General Reasoning" and
"Vision-Dependent Reasoning"** and hand-built `MathVerse_V` (verified in their `data/DATA.md`). We apply an
established criterion from this literature to the training side.

### S5 — Pool construction rules **[U, 2026-08-23]** *(executed; see §4 for results)*

1. Drop `problem_type == regression`.
2. Keep rows where `grade_answer(a, a)` is True under the container's `mathruler`. **[V]** Derived from
   Vision-SR1's own reward, read from source: `vision_sr1/reward_function/self_reward.py` computes
   `grade_answer(extract_boxed_content(response), ground_truth)`, exception → 0.
3. **Exactly one row per image, globally** — not merely disjoint across splits. Rows sharing an image are not
   independent *within* a split either, so group statistics and paired tests would be quietly clustered.
4. Stratified proportional draw across the five `path` categories, targeting the **eligible** population.
5. Sizes: trial 5,000 · eval 1,000 · dev 300. Seed 0. Manifest hashed.

### S6 — Backbone: **Qwen2.5-VL-3B-Instruct**, pinned at `66285546` **[U, 2026-08-23]** *(resolves O1)*

**[V] Qwen2.5-VL has no Thinking variant.** `Qwen2.5-VL-3B` and `Qwen2.5-VL-3B-Thinking` do not exist on
HuggingFace; `-Instruct` is the only release. So "Instruct vs Thinking" was never a live choice for this
family.

**[V] The whole direction is Instruct.** Vision-SR1 (Qwen2.5-VL-3B/7B, `train.sh` defaults to
`Qwen/Qwen2.5-VL-7B-Instruct`) · PAPO paper primaries (Qwen2.5-VL-3B/7B) · DeepEyes (Qwen2.5-VL-7B-Instruct) ·
VLM-CapCurriculum (Qwen3-VL-8B-Instruct primary). The **single** Thinking exception anywhere is PAPO's
`main_qwen3` port (Qwen3-VL-2B-Thinking, asserted at `monkey_patch.py:59`) — **which is the branch we used,
and exactly where our truncation catastrophe occurred.**

*Truncation was self-inflicted, not inherited.* PAPO ships `max_response_length=2048` for its Qwen2.5-VL
primaries. **[V]** Running Qwen3-VL-2B-Thinking on that same config gave **82–95% truncation** with
format = accuracy = 0.047 (the model never finishes); raised to 8192 it still truncated **44%** at a 5,627-token
mean. **[V]** Qwen3-VL-4B-Instruct reaches **EOS on only 68.3%** of generations, and the long-chain behaviour is
documented upstream and unfixed (QwenLM/Qwen3-VL#1922, no budget knob in the chat template). Meanwhile
**[V] Vision-SR1 trains at `max_response_length=4096` for a format containing TWO chains**
(`<visual_reasoning>` and `<think>`) plus the answer. A two-chain format fits their 4096; our one-chain model
did not fit 8192. That is a model-family property, not a budget shortfall.

*Why 3B over the already-present 7B:* the 7B costs ~2.3x per step, and the reason for a small backbone is that
full-CoT KL adds scoring passes per caption per group. The 3B also carries the tighter anchor — Vision-SR1
Table 2, averaged over 7 benchmarks, **trained on our exact dataset**:

| backbone | zero-shot | answer-reward-only GRPO (**= our Arm A**) | their method |
|---|---|---|---|
| Qwen2.5-VL-**3B** | 35.5 | **47.1** | 48.8 (+1.7) |
| Qwen2.5-VL-7B | 41.5 | 50.7 | 52.2 (+1.5) |

*Costs accepted:* no first-party chain-length measurement yet — and since a non-thinking model must be
*prompted* into a chain, its length depends on our prompt, so their config is strong evidence and not proof
for our setting (**M2** measures it, reporting EOS rate first). We also lose continuity with all our own
mechanistic work, which is Qwen3-VL (Set 2/3, Track T). Qwen2.5-VL is 2025-era, so benchmark contamination
applies.

*Location:* the durable `/capstor` store, matching where `Qwen2.5-VL-7B-Instruct` already lives. Scratch lost
this project's data once and the backbone is what every later measurement is defined against.

### S7 — Prompts: `think_answer.jinja` verbatim for **both** scored arms **[U, 2026-08-23]**

Four prompts, in `code/ca21_prompts.py`, with three gates that are unit-tested to fire on planted violations.

The load-bearing constraint is that `π(·|I,x)` plays **three** roles under S1 — the thing `J_success` rewards,
the KL reference, and what we deploy. So the shared instruction must be **evidence-agnostic**: it has to
survive the image → caption swap unchanged. That single requirement eliminates Vision-SR1's `see_think.jinja`
("analyzing an image/video…") and converges on `think_answer.jinja`, which names no modality:

> You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The
> reasoning process MUST BE enclosed within `<think> </think>` tags. The final answer MUST BE put in `\boxed{}`.

**[V] This is byte-identical to the prompt behind our 47.1 Arm A anchor.** Read from source:
`vision_r1/config.yaml:15` → `think_answer.jinja`, `:91` → `accuracy_reward.py`, whose `compute_score` returns
`overall = acc`. **The anchor arm carries NO format reward** — which resolves the open "format reward or not"
question: no. Their *full method* does use one (`self_reward.py`, weight 0.1), which is why the distinction had
to be checked rather than assumed.

*Gates, all tested to fire:* **G-PARITY** (the two scored prompts differ only by removal of the evidence
block) · **G-BLIND** (the answerer carries no image part and no textual `<image>` placeholder) ·
**G-PIXELS** (the configured resolution cap is actually in force).

*Compliance — measured, and one gate FAILED. See §4.5.* `\boxed{}` **89.3%** against a pre-committed ≥90%
bar. Recorded as a **knowing override**, not as a pass — reasoning in §4.5.

### S8 — KL direction: **FORWARD** `KL(π(·|I,x) ‖ π(·|c,x))` **[U, 2026-08-23]** *(supersedes P1)*

Reverses the spec, and the earlier provisional decision, **on argument** — with an empirical test scheduled
because argument is not measurement.

*Notation, since the letters are a trap:* below `sighted = π_θold(·|I,x)` and `blind = π_θold(·|c,x)`. The
spec writes `D(c) = KL(blind ‖ sighted)` — the **reverse** direction. We adopt `D(c) = KL(sighted ‖ blind)`.

**1. Variance — forward gets an exact cancellation reverse cannot.** Draw `y ~ sighted` **once** and share it
across all `G` captions in the group:

```
D̂_i = (1/T) Σ_t [ log sighted(y_t) − log blind_i(y_t) ]
```

`log sighted(y_t)` is identical for every `i`, so under GRPO's group-centred advantage it cancels **exactly**:

```
A_i ∝ D̂_i − D̄ = −(1/T)Σ_t log blind_i(y_t) + (1/G)Σ_j (1/T)Σ_t log blind_j(y_t)
```

The advantage reduces to *how well caption i predicts the shared sighted trajectory, relative to its
group-mates* — teacher-forced scoring with **zero sampling noise** given `y`. Two consequences: the estimate is
deterministic, and `log sighted` is not needed for the gradient at all, only for logging.

Reverse gives each caption its own `ỹ_i ~ blind_i`. Nothing cancels, every `D̂_i` carries independent sampling
noise, and it is a one-sample estimate of a heavy-tailed quantity. Worse, early in training captions are poor,
so `ỹ_i` lands off-distribution and `log sighted(ỹ_i)` swings hardest exactly when the policy is most fragile.

**2. Leakage — the directions have opposite signs, and S2 makes this decisive.** Reverse is zero-forcing: a
caption asserting *"the answer is B"* collapses `blind` onto B, and if `sighted` favours B too, `D → 0` **with
zero perceptual content**. Forward is zero-avoiding: the same spike incurs `log blind → −∞` wherever `sighted`
retained mass, and `D` blows up. Since S2 gives the captioner the full question **including options**, reverse
would have paired the leak-attractive direction with maximal leak opportunity.

**3. Why this bears on validation accuracy, not just tidiness.** S1's load-bearing bet is that the caption term
reshapes representations the sighted pathway reuses. Under reverse the cheapest descent direction is
answer-copying — the caption degenerates into an answer-repeater, no perceptual learning occurs, and Arm B
collapses toward Arm A. Worst case it *teaches the shortcut*, which is the exact pathology this programme
exists to remove (Probe A's premature closure; Track T's extraction deficit).

**4. Cost, now secondary.** Forward needs one scoring pass; reverse needs `G` generations plus scoring. With
chains measured at p99 = 770 (§4.5) reverse is affordable — roughly 2–3× per step, not prohibitive. **Cost is
no longer what decides this**, which is why the decision was correctly scheduled after §4.5.

*Honest case for reverse, retained:* it is what the spec says; its estimand is closer to Vision-SR1's
sufficiency notion; and mode-seeking is often more stable early since it never forces coverage of a
poorly-estimated distribution. Against that: **we deploy `sighted`, not `blind`.** We want fidelity to sighted
behaviour, not sufficiency-to-answer — which *is* the sharper factorisation S1 claims over Vision-SR1.

*Real risk accepted, and it is not small.* Sharing one `y` trains captions to faithfully support **one
trajectory, including a wrong one**. At a measured 30.7% sighted accuracy (§4.5), most trajectories *are*
wrong. Two containments: **O4** (gate the caption term on `1[R(y)>0]`) — this decision makes O4 **more**
urgent, not less, and sharpens its starvation concern since it would fire on under a third of samples — and
**resampling `y` each step** so captions optimise against the sighted *distribution* rather than one draw.
Forward also carries a tail risk where `log blind → −∞` on sighted samples; mitigated by both being the *same
weights* under different context, but it needs per-token clamping and monitoring, not trust.

*Settled by experiment, not left to argument.* Built as a config switch, then run head-to-head on the trial
pool — same data, seed and steps — comparing validation accuracy **and** L1's strongest leak instrument
(answer from `c` with `x` removed). If reverse leaks it will show there before it shows as a flat curve.

### S9 — `y` = full CoT **+** answer **[U, 2026-08-23]** *(supersedes P2)*

*The reason that holds is leak-resistance,* not brevity. If `y` is the answer alone, a caption stating the
verdict achieves parity with zero perceptual content; reproducing a **chain** requires supplying the facts the
chain reasons over. Under S8 this compounds: forward KL asks how likely the caption-conditioned model finds the
sighted model's *actual reasoning*, which is close to a direct measurement of the programme's question.

*The originally stated reason was falsified and is recorded in §6.* "Instruct implies short answers" is false
for Qwen3-VL. It happens to be **true for this backbone** — §4.5 measures mean 191, p50 133, p99 770, max 1,092
— but that is a property of Qwen2.5-VL-3B-Instruct measured first-hand, not an inference from "Instruct".

### S10 — Container: vLLM's ViT attention override is patched, and the patch is gated **[CC, 2026-08-23]**

`easyr1_vllm0112.sqsh` cannot run Qwen2.5-VL at all without this. Qwen2.5-VL's vision tower has
`head_dim = 1280/16 = 80`; the bundled `flash_attn` 2.8.3 was compiled for a reduced head-dim set and rejects
anything not a multiple of 32. Qwen3-VL's ViT has different head geometry, which is why the container never
exposed this before.

**Setting `mm_encoder_attn_backend="TORCH_SDPA"` is not sufficient, and this is the part worth remembering.**
vLLM accepts the value, echoes it in its own non-default-args log line, and reverts it internally: the CUDA
branch of `maybe_get_vit_flash_attn_backend` never consults `attn_backend_override`, though the ROCm branch ten
lines above does. Job 3167568 therefore failed **identically** to 3167519 while every log line claimed
`TORCH_SDPA`. **A silently-ignored setting is indistinguishable from a working one.**

The patch copies the ROCm guard onto CUDA — one logical line, resolving an inconsistency inside a single
function. No upstream fix exists to adopt (`vllm#27821` lists honouring the flag as a future goal;
`vllm#38411` is the same bug class, closed as not planned).

*Result-neutral:* SDPA and FlashAttention both compute exact softmax attention, differing only at bf16
rounding; `TORCH_SDPA` is vLLM's own class default for `Qwen2_5_VisionAttention`. Comparability is unaffected
because we never test against 47.1 directly — we run our own control arm in this container, so the kernel
cancels in the contrast. There is no more faithful option: this FA2 build cannot execute this ViT at all.

*Controls, because a bind-mount overrides silently:* the patch is version-controlled · `ORIGINAL.sha256` pins
the file it was derived from so an image rebuild fails loudly · **gate G-VITATTN** asserts identity *and*
behaviour before the engine is built · a separate `ca21_vllm.toml` keeps the mount away from the PAPO and
DeepEyes lines. Full write-up: `patches/vllm_0_11_2/README.md`.

**[V] The HF training path needs the same treatment, and it is cheaper.** `Qwen2_5_VLVisionAttention` computes
`head_dim = 80` (`modeling_qwen2_5_vl.py:187`) and dispatches on `config._attn_implementation` with an explicit
`flash_attention_2` branch (`:216-219`) into the same package — so verl's global
`attn_implementation="flash_attention_2"` (`fsdp_workers.py:215`) will hit this identically. But transformers
4.57 accepts a **per-subconfig dict** (`modeling_utils.py:2802-2838`), so the training fix is a config change,
not a patch:

```python
attn_implementation={"": "flash_attention_2", "vision_config": "sdpa"}
```

This keeps FA2 on the language model (`head_dim = 128`, works and is faster) and drops to SDPA only in the ViT.

---

## 2. PROVISIONAL — proceeding this way, explicitly not frozen

**Currently empty.** P1 and P2 were the only entries; both were settled on 2026-08-23 after §4.5 supplied the
measured chain lengths they were waiting on. P1 → **S8** (forward, reversing the spec). P2 → **S9**.

### P1 — *superseded by S8 on 2026-08-23.* Retained: the comparison that drove it

The distinction is *not* mainly cost. Both directions evaluate both contexts; they differ in which must be
**sampled** from versus merely **scored** under. Reverse needs `ỹ ~ π(·|c,x)` — an autoregressive generation
per caption. Forward needs `y ~ π(·|I,x)` — which S1 already generates for `J_success` — then only a
teacher-forced scoring pass per caption. Reverse buys back blind-answer accuracy for free; under forward that
becomes a separate instrumentation pass (roadmap **L1**).

*In this table `p = π(·|c,x)` (blind) and `q = π(·|I,x)` (sighted)* — the opposite lettering to S8, which
spells both out to avoid exactly this trap.

| | **Reverse** `KL(p‖q)` *(spec; rejected)* | **Forward** `KL(q‖p)` *(adopted, S8)* |
|---|---|---|
| character | mode-seeking / zero-forcing | mode-covering / zero-avoiding |
| optimum | caption makes the blind model collapse confidently onto **one** high-probability region of sighted behaviour | caption must make the blind model assign mass to **everything** the sighted model does |
| **leakage** | **attractive** — a caption asserting the verdict makes `p` a spike; if `q` favours it too, `D → 0` with zero perceptual content | **resistant** — the same spike is heavily penalised, since `q` has mass across the reasoning and `p` has none there |
| uncertainty | **discarded** — caption becomes a sufficient statistic of the image's *argmax*; expect under-dispersion, worst on ambiguous items where perception is hardest | preserved |
| with full-CoT `y` | asks whether the blind model's own chain would look plausible to the sighted model | asks *how likely the caption-conditioned model finds the sighted model's actual chain* — i.e. does the caption contain the facts the reasoning invokes. Near-direct measurement of the programme's question |
| group variance | each caption scored on its **own** trajectory ⇒ per-caption lexical noise | all `G` captions scored on the **identical** trajectory ⇒ stylistic variation is common-mode and cancels; the `log q` term is a per-item constant that cancels exactly |

*Honest case for reverse, and it is real:* at deployment one samples **from** the blind model, so matching in
the direction one samples is operationally relevant; and mode-seeking is typically more stable early, since it
never forces coverage of a poorly-estimated `q`. **Answered in S8:** we deploy the *sighted* model, not the
blind one — the blind pathway is an instrument, never a deliverable — so the first half does not apply here.
The second half stands and is the reason S8 schedules a head-to-head rather than declaring victory.

### P2 — *superseded by S9 on 2026-08-23.*

---

## 3. ROADMAP — everything still open, in the order it must be solved

Ordered by dependency, not by importance. Each entry says what it blocks and why it cannot come earlier.

### Stage 0 — housekeeping (non-blocking)

| id | item |
|---|---|
| **H1** | `_env.sh` documents `HF_HOME` as `.../hf_cache`, but the cluster environment already sets it and the snapshot landed in `.../huggingface/`. **[V]** job 3163760. The comment is now inaccurate; correct the file rather than leave a doc that lies. No re-download needed. |
| **H2** | `format_check` persists only the first 20 of *n* records, so 3 of the 32 `\boxed{}` failures were characterised rather than all 32 (§4.6). One-line change; land it with the next run rather than spending a job on it. Full responses are needed for **L1** regardless. |
| **H3** | Apply the training-path attention fix from **S10** — `attn_implementation={"": "flash_attention_2", "vision_config": "sdpa"}` at verl's load site. Config change, no patch. Blocks the first training smoke, so it is only "non-blocking" until then. |

### Stage 1 — ~~the one hard blocker~~ **RESOLVED: see S6**

### Stage 2 — substrate characterisation ~~(go/no-go)~~ **M2 DONE, M1 still open**

| id | item | status |
|---|---|---|
| **M2** | **Sighted pass rate + termination.** | ✅ **DONE — §4.5**, job 3168210, full dev split. EOS 100%, truncation 0%, max 1,092, accuracy 30.7%. The Qwen3-VL head-to-head was dropped as moot once S6 settled. R2 satisfied with room to spare. |
| **M1** | **Vision-necessity rate** — question text only, no image, n draws, on the trial pool. **Reported, not used as a filter.** | ⬜ **OPEN, off the critical path.** Decides whether `D` is vacuous and how often. Falsifiable self-check: Knowledge should shed far more than Chart. `build_no_evidence_messages` + `assert_no_evidence` already exist and are tested, so this is a runner away. Worth doing in parallel — a largely vacuous pool would reopen the substrate question, and R1 is constitutive (§4.1). |

### Stage 3 — the estimand ~~(needs M2)~~ **P1/P2 DONE, O3 open**

| id | item | status |
|---|---|---|
| **P2** | `y` scope. | ✅ **S9** — full CoT + answer. |
| **P1** | Forward vs reverse KL. | ✅ **S8** — forward, reversing the spec. Settled on variance (exact common-mode cancellation), leak asymmetry, and estimand fit; a head-to-head on the trial pool is scheduled to convert argument into result. |
| **O3** | Estimator family: the spec's one-sample signed `Σ log(p/q)` vs a Rao-Blackwellised per-position exact KL. | ⬜ **OPEN — now the first thing to settle.** S8 changes its character: under forward with a shared `y`, the one-sample form is already low-variance because `log sighted` cancels in the centred advantage, which weakens the case for the exact form. But the exact form still buys a **free correctness oracle** — per-position KL is ≥ 0 by construction, so a negative value proves a bug — and that is worth a great deal in a project where three of four instrumentation failures printed healthy output right up to the point of failure (§4.6). Memory is `vocab × T × batch` across two contexts; at the measured `T` (p99 = 770) this is far cheaper than the earlier `T ≈ 1,100` assumption implied. |

### Stage 4 — reward shape (needs stage 3, and M2)

| id | item | why here |
|---|---|---|
| **O4** | **Conditional caption reward:** gate the caption term on the sighted rollout being correct, `1[R(y)>0]·(−D̂)`. | ⬆️ **PRIORITY RAISED by S8.** Removes "train captions to faithfully reproduce a wrong chain" at the root — and S8's shared-`y` forward estimator makes that failure mode *direct* rather than diffuse. **M2 has now quantified the tension:** at 30.7% sighted accuracy the gate would fire on under a third of samples, so the starvation concern is real and measured, not hypothetical. Decide alongside **resampling `y` per step**, which attacks the same problem without discarding two thirds of the batch. Precedent: DeepEyes' conditional tool bonus, whose Table 5 ablation shows conditionality is what makes the behaviour emerge at all. |
| **O2** | Reward composition: a single `λ` vs two separately group-normalised advantages. | The terms live on incompatible scales — `R(y)` is bounded in [0,1], a sequence-level KL is unbounded and length-dependent. **[V] Corrected 2026-08-23 — see §4.7.** Vision-SR1 *does* use separately z-scored per-component advantages, but at weights `accuracy 1.0 / format 0.1 / description_accuracy 1.0 / description_format 0.1` — **not** λ=0.5 each, as this log previously claimed. Their description term carries the same weight as the sighted accuracy term. Still needs O3 first, since the scale of `D̂` depends on the estimator. |
| **S3b** | Exact `q_cap` wording. | Cheap, but it is a result-affecting string and belongs under sign-off, not improvisation. Current draft is in `code/ca21_prompts.py` and unit-tested for its load-bearing clauses; it needs freezing, not writing. |
| **L1** | **Leak instrumentation spec.** Three instruments: gold-string containment in `c`; verdict-assertion phrasing; and the strong one — **answer from `c` with `x` removed** (a caption carrying evidence cannot answer a question it cannot see; one carrying a verdict can). | Made load-bearing by S2. **S8 changes its job but not its necessity:** forward KL penalises leaking rather than rewarding it, so L1 is no longer guarding against an objective that actively pulls the wrong way — but it is now also the **adjudicator of the S8 head-to-head**, since "does reverse leak more than forward" is the question that decision defers to evidence. Must exist before the first training run, not after a suspicious result. **[V] Add a fourth instrument from §4.7:** Vision-SR1's `extract_description` falls back to the whole response when tags are missing, making their `description_accuracy` trivially satisfiable exactly when format compliance fails — our two-context design makes that impossible, and saying so requires having measured it. |

### Stage 5 — RL configuration (needs stage 4)

| id | item |
|---|---|
| **O5** | `θ_old` refresh cadence / `ppo_epochs`. Non-routine here: the accuracy term is *deliberately* moving `π(·|I,x)`, so the caption chases a shifting reference by design. |
| **O6** | Group size `G`, batch shapes, steps, learning rate, seed. |

### Stage 6 — frozen before any full run

| id | item |
|---|---|
| **O7** | **Evaluation set and success criterion.** **This is the rule the previous attempt broke** — five GPU jobs ran before anyone had written down what winning looked like. Must be frozen with a hash before a single training step. |
| **O8** | Control-arm specification. S1 already determines its shape: accuracy-only GRPO vs accuracy + caption-KL, identical data/steps/seed — the same axis Vision-SR1 reports. Needs O6 to be written precisely. |

---

## 4. Substrate: requirements and what was measured

### 4.1 Derived requirements

Each falls out of a specific term in the objective, not from taste.

| # | Requirement | Derivation |
|---|---|---|
| **R1** | **Vision-necessary** — `x` alone must not determine `y` | If it does, `π(·|c,x) ≈ π(·|I,x)` for *every* caption ⇒ `D ≈ 0` ⇒ zero group variance ⇒ **zero gradient**. Not weak — **vacuous**. |
| **R2** | **Intermediate difficulty** — sighted pass rate strictly in (0,1) | GRPO's advantage is group-relative; all-wrong and all-correct groups both give zero advantage. |
| **R3** | **Captionable** — question-relevant content finitely statable in text | **[V]** Set 3: text payloads "enumerate objects but not spatial relations; CLEVR is relational, so objects-as-text cannot restore the layout." Unserialisable content gives `D` a floor no training removes. |
| **R4** | **Rule-verifiable answers** | `J_success` computable without an LLM judge. |
| **R5** | **Uncontaminated** | A memorised item is answered from parametric knowledge — violating R1 by another door. The M1 screen catches it for free. |
| **R6** | **Diverse perceptual operations** | Needed only for a *general* perception claim. |
| **R7** | **Images within the pixel budget** | Our localisation says the deficit is LLM **read-out** of adequate visual tokens; downscaled-past-resolvable items test *encoding* instead. |
| **R8** | **Single image per item** | Otherwise `c` has an undefined referent. |
| **R9** | **Survives filtering at volume** | ≥ ~2K supports 30–60 trial steps at rollout batch 128–256. |

### 4.2 Measured — download + verification **[V] job 3163760, 2026-08-23, 52 s**

Every characterisation before this came from the HF datasets-server API. This re-derived it from the parquet.

- rows on disk **47,628**, exact; schema carries all 10 required columns.
- `problem_type` marginals match the API exactly: multiple choice 29,702 · numerical 12,586 · **regression 5,340**.
- `path` partitions **totally** into five categories — the dataset's own taxonomy, which differs from a
  hand-grouping by `data_source`: **CLEVR sits under `./Math/`, IconQA under `./General/`.**
  Knowledge 12,019 (25.2%) · Math 11,812 (24.8%) · Spatial 10,380 (21.8%) · Chart 9,066 (19.0%) · General 4,351 (9.1%).
- **Image reuse is large:** 37,138 distinct images behind 47,628 rows; 8,886 images back more than one row
  (TabMWP up to ×5) = **19,376 rows, 40.7%**. This was the unknown that decided S5.3.
- ⚠️ **The released artifact does not match the paper's Table 1.** Table 1 lists Math 30.5% / Science
  Knowledge 30% / General Visual Reasoning 39.5%; the artifact contains **Multimath-300k (5,000 rows, 10.5%)**,
  which appears nowhere in that table. Claims of the form "same data as Vision-SR1" need care.
- ⚠️ The artifact is **block-grouped, not shuffled**, so any single parquet shard is a biased single-source
  slice. Subsets must be seeded stratified draws.

### 4.3 Measured — pool build **[V] job 3163976, 2026-08-23, 24 s**

- **`regression` is 100% Spatial** — `{'Spatial': 5340}`. A prediction, checked and confirmed. This settles the
  stratification-target question: `--target raw` would have demanded 21.8% Spatial (1,373 images) from only
  5,040 eligible, over-sampling ~37% against availability.
- **`ungradeable` = 0.** No answer failed `grade_answer(a, a)`. Honest reading: the filter is currently a
  **no-op on this dataset**. It is a *necessary-condition* check that eliminates one specific failure mode and
  found no instances — **not** evidence the answers are otherwise well-formed. Unlike a vacuous assertion it
  demonstrably *can* fire (unit-tested); it simply found nothing.
- Eligible: 42,288 rows → **31,798 distinct images**. Drew 6,300 = ~20% utilisation, leaving ample headroom to
  enlarge the trial or draw a second disjoint pool without touching eval.
- **Unpredicted finding:** the one-row-per-image collapse shifts shares too, because reuse is very uneven.
  Rows-per-image by category: Spatial **1.00** (no reuse at all) · Math 0.82 · General 0.79 · Knowledge 0.68 ·
  **Chart 0.61**. So the collapse costs Chart most and Math least, which is why Math *rises* 24.8 → 30.3%.

| category | raw % | eligible % | drawn % |
|---|---|---|---|
| Knowledge | 25.2 | 25.5 | 25.5 |
| Math | 24.8 | **30.3** | 30.3 |
| Spatial | 21.8 | **15.9** | 15.9 |
| Chart | 19.0 | 17.5 | 17.5 |
| General | 9.1 | 10.8 | 10.8 |

- Splits exact: **trial 5,000 · eval 1,000 · dev 300**; 6,300 images, each in exactly one split, each once.
- **Manifest SHA-256** `63164939e6ca0ef58026fac8bc690e7fc217dabb06ad52570a1e510acfcbfe57`.

### 4.4 Prior measured evidence carried forward (facts only, not decisions)

- **[V]** ViRL39K: **34% of rows answered 5/5 from question text alone** (first-party no-evidence control,
  2026-08-18). The reason R1 is treated as constitutive.
- **[V]** No literature dataset validates vision-necessity at *training* scale. Those that do are benchmarks:
  MMStar 1,500 · NaturalBench 1,900 · CV-Bench 2,638 · BLINK ~3.8K. Hallucination-specific sets (POPE,
  HallusionBench, AMBER, MMHal-Bench, RH-Bench) are all eval-sized.

### 4.5 Measured — sighted prompt compliance **[V] job 3168210, 2026-08-23, full 300-item dev split**

Qwen2.5-VL-3B-Instruct, `think_answer.jinja`, one draw, temperature 1.0 / top_p 0.99 (Vision-SR1's rollout
setting, **not** the model card's 1e-6, which is effectively greedy and would make every rollout in a GRPO
group identical — zero variance, zero advantage, no gradient). Measured at an 8,192 budget deliberately: a
censored sample cannot report its own tail, so measuring at 4,096 and concluding "4,096 suffices" is circular.

| metric | result | pre-committed bar | |
|---|---|---|---|
| `\boxed{}` present | **89.3%** (268/300) | ≥ 90% | ❌ **FAILED** |
| EOS reached | **100.0%** | ≥ 95% | ✅ |
| truncated | **0.0%** | — | ✅ |
| `<think>` present | 81.3% | ≥ 90% | ❌ (convention only) |
| `</think>` present | 50.3% | — | benign |
| length | mean 191.1 · p50 133 · p90 444 · p99 770 · **max 1,092** | — | |
| exceeds a 4,096 budget | **0.0%** | — | ✅ |
| accuracy | 30.7% (16.7% Spatial → 50.0% General) | never a criterion | |

**The truncation problem is solved, and this is the measurement that proves it.** Against Qwen3-VL's 68.3% EOS
and 44% truncation at 8,192, this backbone reaches EOS on **every one** of 300 generations across five
categories, longest chain 1,092 tokens. `max_response_length: 4096` (Vision-SR1 parity) gives ~4× headroom over
the observed maximum and 5× over p99. **No increase is needed or warranted.** Budget check: largest dev image
5,220 visual tokens + a caption at the 4,096 ceiling + question still fits `max_prompt_length: 12800`.

**`\boxed{}` failed its bar and is recorded as a knowing override, not as a pass.** The evidence either way:

- Wilson 95% CI **[0.853, 0.923]** contains 0.90 — *n* = 300 cannot resolve this bar, and the failure is
  uniform across categories (87.5%–90.6%), so it is a property of the model, not of any data slice.
- The failures are **real, not a detector artifact**: one invents `<end-think>` then gives a bare `1.84`; one
  writes `< think >` with spaces and answers `Correct Answer: B. Yes`; one closes `</think>` and stops without
  answering. All `finish_reason=stop`. Vision-SR1's own `re.sub(r"\s*(<|>|/)\s*", ...)` normalisation cannot
  rescue any of these — it touches angle brackets, and `\boxed{` has none.
- **Unboxed rollouts are self-correcting under GRPO** in a way low accuracy is not: they score 0 against
  group-mates scoring 1, so the group-relative advantage pushes directly toward boxing. At `G=8` that is ~0.86
  format-zeros per group — added variance, not a broken signal, and a signal the model can climb.
- **[V] Vision-SR1 started from exactly here** — same model, same prompt, same accuracy-only reward (S7) — and
  went 35.5 → 47.1. Whatever their init compliance was, this configuration produced it and training worked.

*Consequence, binding:* **boxed-rate becomes a tracked training metric.** If it does not rise in the first
steps that is a finding about the backbone surfaced early, not a surprise at step 40. The override is thereby
falsifiable rather than an excuse. More sampling was explicitly declined: it would tighten a CI around a number
we have already decided how to act on.

*Also observed:* the model emits `<ref>` and `<code>` grounding tags mid-chain — in-distribution Qwen2.5-VL
behaviour, worth knowing before computing a token-level KL over these sequences. `</think>` at 50.3% is why the
two requirements were reported **separately**; merged into one "format rate" it would have read ~50% and looked
like failure, when nothing in the design reads `</think>` at all.

### 4.6 Instrumentation failures found and fixed (recorded, not hidden)

| # | failure | how it surfaced | fix |
|---|---|---|---|
| 1 | `images` treated as a list of structs; Vision-SR1-47K declares a **singular** `Image` feature, so `cell[0]` raised `KeyError: 0` | job 3167490, 55 s of GPU | `extract_image_bytes` handles both shapes; multi-image is a hard **R8** error, never a silent `[0]` pick |
| 2 | ViT `head_dim=80` vs a restricted FA2 build | job 3167519 | → **S10** |
| 3 | the S10 fix was *accepted, logged, and silently reverted* by vLLM | job 3167568, failed **identically** to 3167519 | patched `layer.py` + gate **G-VITATTN**, which asserts the outcome instead of trusting the log line |
| 4 | **`get_split` took a head slice.** `build_pool` writes each split sorted by image path, which groups categories, so `items[:50]` on the 300-row dev split returned **50 of the 52 Chart rows** while logging "50 items from split 'dev'" | job 3168166 — the skew appeared only in the by-category table *after* the GPU time was spent | `get_split` defaults to **stratified**, largest-remainder, seeded; `head` must be named explicitly. `format_check` prints composition against the full split **before** generating and refuses a sample missing a category. `--limit` defaults to 0 = whole split |

**Common thread, and the reason all four are logged here:** every one produced *correct-looking output* right up
to the point of failure. #3 and #4 are the same disease — a claim believed because it was printed. That is the
PAPO lesson (§5 preamble) recurring inside our own instrumentation, and it is why gates in this project must be
shown to **fire on a planted violation** rather than merely to pass.

*Not fixed, deliberately:* only 20 of 300 records are persisted, so 3 of the 32 `\boxed{}` failures were
characterised rather than all 32. The one-line change lands with the next run; a dedicated re-run to inspect
formatting errors we have already decided not to act on was declined as cost without information.

### 4.7 Facts established from Vision-SR1 source **[V] read 2026-08-23, commit `85b7c6a`**

Read because three roadmap items rested on claims about their code that had never been checked.

- **The 47.1 anchor arm carries no format reward.** `vision_r1/config.yaml:15` → `think_answer.jinja`, `:91` →
  `accuracy_reward.py`, `compute_score` returns `overall = acc`. Their full method does use one
  (`self_reward.py`, `format_weight=0.1`). → **S7**.
- **Anchor hyperparameters:** `max_response_length` 4096 · `max_prompt_length` 12800 · `rollout_batch_size` 512
  · `n` 8 · temperature 1.0 / top_p 0.99 · `adv_estimator` grpo · `use_kl_loss` true · `kl_penalty` low_var_kl
  · `kl_coef` 1e-2 · `total_epochs` 1 · val on `zli12321/mmstar@test`. Our sampling already matches exactly.
  Note `train.sh` defaults to **7B**; the 3B numbers come from passing the argument.
- **CORRECTION to O2 [CC].** The log claimed *"Vision-SR1 uses separate z-scored advantages at λ=0.5 each."*
  The **structure** is right — `ray_trainer.py:120-134` z-scores each component independently per group, then
  weighted-sums. The **weights are not**:
  `{"accuracy": 1.0, "format": 0.1, "description_accuracy": 1.0, "description_format": 0.1}`.
  `description_accuracy` carries **1.0**, equal to the sighted accuracy term, not half of it. Note also that
  `self_reward.py` computes an `overall` scalar that the trainer then **does not use** for advantages.
- **They make no attempt at prompt parity.** Their second-hop wrapper (`ray_trainer.py:39`) is an entirely
  different prompt — *"You are provided a text description of a problem and a question…"* — with no
  relationship to the first-hop prompt. **G-PARITY is now a source-verified differentiation, not an asserted
  one.** (This also corrects `TALK/VISION_SR1_DIFFERENTIATION.md`, which describes our leak prevention via the
  abandoned spec's D18 — superseded by S2 and not authoritative for this project.)
- **A leak channel in their reward.** `extract_description` (`ray_trainer.py:56`) falls back to returning the
  **entire response** when `<description>` tags are absent. So a format-noncompliant rollout feeds its own
  `<think>` and `\boxed{}` into the second hop as the "description", making `description_accuracy` trivially
  satisfiable *exactly when format compliance fails*. Structurally impossible in our two-context design. → **L1**.
- **Trainer structure:** `SelfRewardTrainer(RayPPOTrainer)` overrides six methods, and `fit()` is a **verbatim
  copy of the base loop with one section swapped** — verl exposes no plug-point for custom advantage
  computation. Our estimator extends by the same pattern; their code is precedent, not a dependency.

---

## 5. Predicted failure modes, to instrument from day one

Named in advance so a healthy-looking run cannot be mistaken for a correct one — the PAPO lesson, where every
logged metric looked fine while the perception loss received no gradient at all.

1. **Answer leakage into the caption.** Structural, given S2 — the captioner sees the options. **S8 makes the
   objective push against it** rather than with it, but "disfavoured" is not "impossible". Instruments in **L1**.
2. ~~**Under-dispersion** of `π(·|c,x)`~~ — *this was the predicted signature of **reverse** KL and no longer
   applies under S8.* **Replaced by its mirror image: over-dispersion.** Forward KL is zero-avoiding, so the
   pathological caption is now one that makes the blind model *diffuse* — hedging across possibilities to avoid
   the `log → −∞` penalty — rather than one that collapses. A caption reading "the chart shows several values,
   any of which could be highest" is bland, contentless, and scores well. **Instrument the entropy of
   `π(·|c,x)` against `π(·|I,x)` in both directions**, and treat a rising gap as the forward-specific failure.
3. **Length hacking.** A KL summed over `T` tokens rewards shorter continuations mechanically. **[V]** Probe A
   documented exactly this on this model family: injected text drove premature `</think>` closure in 55% / 84%
   of generations, and short-circuited generations were 11–13 accuracy points worse. **Note the interaction with
   §4.5:** 50.3% of sighted generations already omit `</think>`, so premature-closure metrics need a clean
   baseline from *this* backbone before they can be read as a training effect.
4. **Dead groups** from vacuous items — measurable as the fraction of groups with ~zero `D̂` variance.
   Quantified in advance by **M1**, which is why M1 stays on the list even though it is off the critical path.
5. **Fidelity-to-a-wrong-chain** on items the sighted model gets wrong. **Elevated by S8** — a shared `y` makes
   this direct, and §4.5 measured sighted accuracy at 30.7%, so most trajectories are wrong. Mitigations:
   **O4**, and resampling `y` per step.
6. **Format collapse.** `\boxed{}` at 89.3% (§4.5) is below the bar we set and was overridden knowingly. If
   boxed-rate does not rise under training, `J_success` is partly scoring formatting rather than perception.
   **Tracked as a training metric from step 1** — this is the override's falsification condition.

---

## 6. REJECTED

- **DeepEyes-Datasets-47k as substrate.** Its published perception-utility filter keeps samples "where the
  ground-truth crop provably helps" — i.e. it selects for the **encoding-limited** regime. The captioner sees
  the same un-zoomed image, so where information is not resolvable at full resolution, no caption built from
  it can carry that information. Selected for exactly the case the method cannot serve.
- **VLM-CapCurriculum `D_perc`.** Keeps rows iff `Â_img(Q|I) ≠ A` — the sighted answer is **wrong by
  construction on every row**. Under S1 the accuracy term would have no correct behaviour to reinforce; under
  O4 the caption gate would never fire.
- **CoSyn-400K and VisOnlyQA_Train.** Both are training-scale with vision-necessity by construction, and CoSyn
  additionally ships an oracle caption (`data` = the CSV that generated the chart, plus the rendering `code`),
  which would have given a no-training headroom probe. **Declined on positioning, not properties [U]:** the
  paper sits in the PAPO / Vision-SR1 / DeepEyes lineage, and training on a Molmo-lineage synthesis resource
  forfeits comparability and invites "why this data?". Recorded because the oracle-caption idea may be worth
  revisiting as a *diagnostic* even if not as a substrate.
- **"Instruct implies short answers."** Falsified — see P2.
- **My own overstatement, recorded as a correction [CC]:** I asserted that "every dataset in the literature
  validating vision-necessity is a benchmark, sized 1.5–4K." That was wrong at the training-scale end —
  VisOnlyQA_Train (70K) and CoSyn-400K (408K) both exist. The claim was stated categorically without checking
  that end of the range, and the user was right to challenge it before we committed to a filtering route.

---

## 7. Chronology

| date | event |
|---|---|
| 2026-08-21 | Sub-repo created. Prior attempt reviewed end-to-end; clean break declared. |
| 2026-08-21 | `J_success` fixed to the sighted answer (S1). New spec stored + hashed. |
| 2026-08-23 | Reverse KL adopted provisionally (P1); forward-KL argument accepted as substantive, left unfrozen. |
| 2026-08-23 | Captioner given the full question (S2); leakage moved from prevention to instrumentation. |
| 2026-08-23 | Dataset requirements R1–R9 derived; CoSyn/VisOnlyQA surfaced then declined on positioning. |
| 2026-08-23 | Substrate settled: Vision-SR1-47K @ `2900b038` (S4). Downloaded and verified on disk (job 3163760). |
| 2026-08-23 | Pool built (job 3163976): trial 5,000 / eval 1,000 / dev 300, manifest `63164939…`. |
| 2026-08-23 | Backbone settled: Qwen2.5-VL-3B-Instruct @ `66285546` (S6, resolves O1). |
| 2026-08-23 | Prompts reviewed and frozen (S7). `think_answer.jinja` chosen because S1 forces the shared instruction to be **evidence-agnostic**, which rules out `see_think.jinja`. |
| 2026-08-23 | Vision-SR1 source read (`85b7c6a`). Anchor arm confirmed **accuracy-only, no format reward** → resolves the open format-reward question. O2's weights **corrected**; two leak facts added to L1 (§4.7). |
| 2026-08-23 | Container defect found and patched (S10). vLLM 0.11.2 accepts `mm_encoder_attn_backend` and silently reverts it; job 3167568 failed *identically* to 3167519 while logging success. Gate **G-VITATTN** added to assert the outcome. |
| 2026-08-23 | `get_split` head-slice bug: job 3168166's compliance numbers were **Chart-only**. Sampler fixed to stratified-by-default; composition now asserted before generation (§4.6). |
| 2026-08-23 | **M2 done** (job 3168210, full dev split): EOS **100%**, truncation **0%**, max 1,092, accuracy 30.7%. The truncation problem that shaped this whole programme is **resolved by the backbone switch**, and `max_response_length: 4096` needs no increase. |
| 2026-08-23 | `\boxed{}` **89.3%** vs a pre-committed ≥90% bar → **FAILED, overridden knowingly**, with boxed-rate made a tracked training metric as the falsification condition (§4.5). |
| 2026-08-23 | **P1 reversed: forward KL adopted (S8)**, against the spec, on variance + leak-asymmetry + estimand-fit grounds; head-to-head scheduled. **P2 settled (S9).** PROVISIONAL tier now empty. |
