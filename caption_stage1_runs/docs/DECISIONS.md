# Decision log — caption-distortion Stage 1

Append-only record of every setting, who decided it, and why. **This is a log, not a pre-registration** —
nothing here is frozen until it appears in a `PREREG_FREEZE` file with a hash. Where a decision deviates from
`SOURCE_SPEC_hackmd.md`, that is stated explicitly.

Legend: **[U]** = user decision · **[CC]** = my recommendation, pending user sign-off · **[V]** = verified fact.

---

## D1 — Training regime: `J_cap` alone (Stage 1) **[U]**

Train only the caption-distortion objective. No task reward, no `λ`.

*Why:* isolates whether distortion alone teaches useful perception. With `J_success` present, any gain is
attributable to the task RL rather than the distortion signal. Spec explicitly offers this
("Alternatively, `J_cap` can be used as a pretraining objective before RL training").

*Consequence:* `R(ỹ)` is **not** needed for training. Accuracy is still needed for evaluation and for any
data filtering, so a scorer is still required — it just does not enter the gradient.

## D2 — Codebase: extend the verl/EasyR1 PAPO stack **[U]**

*Why:* PAPO's perception-KL already performs the exact operation this method needs — score rollout tokens
under a second, different-length visual context — so token alignment and masking have an audited precedent.
Reuses the verified GH200 container (`easyr1_vllm0112`: vLLM 0.11.2, transformers 4.57.3, torch 2.9.0+cu129).

## D3 — Distortion target stays as written; use a different train set **[U]**

Keep `D(c) = KL(π_old(·|c,x) ‖ π_old(·|I,x))` — the caption is trained to preserve the **image-conditioned**
behavior.

*Why this needed deciding:* the UCSC-VLAA Stage-1 set `D_perc` keeps rows **iff** `Â_img(Q|I) ≠ A`, so the
image-conditioned answer is wrong **by construction on every row**. Training `J_cap` there would optimize
captions to reproduce a wrong answer 100% of the time, with a healthy-looking loss curve. `D_perc` is
therefore excluded as a `J_cap` training set. See `VLMCC_STAGE1_REFERENCE.md` §5.

## D4 — Backbone: `Qwen3-VL-2B-Instruct` **[U]**

*Considered and rejected:* `Qwen2-VL-2B-Instruct` (user's initial proposal). Same parameter count but released
2024-08-28 vs 2025-10-19; would break comparability with every first-party result we hold (all Qwen3-VL);
would require a new container bring-up rather than reusing the proven one; and its weaker base policy raises
the GRPO dead-group rate (zero-variance groups ⇒ zero advantage ⇒ no gradient) on hard reasoning data.

*Rejected:* `Qwen3-VL-8B-Instruct` (the paper's primary) — 4×GH200 memory; 4B-Thinking already OOMed under PAPO.

**[V]** Not currently on the cluster — needs a ~4.3 GB download. Present: `Qwen3-VL-2B-Thinking` (scratch),
`Qwen3-VL-4B-Instruct` / `-Thinking` (a0174 store).

*Instruct over Thinking is also a methodological choice, not just convenience:* `D̂` sums over answer tokens,
so a Thinking model's long, high-variance CoT directly inflates the length-hacking channel (§3.2 of
`SPEC_READING_AND_OPEN_QUESTIONS.md`). Short Instruct answers keep `D̂` bounded and low-variance.

## D5 — Train set: `PAPOGalaxy/PAPO_ViRL39K_train` **[U]**

**[V] Verified 2026-08-16.** On cluster at
`hf_cache/hub/datasets--PAPOGalaxy--PAPO_ViRL39K_train/snapshots/ff6996d5cdd0e5fc12c01f3dab96f1af37453ceb/`,
6 parquet shards. Schema `images` (List) / `problem` (string) / `answer` (string) — matches our verl config
keys with no remap. 38,870 rows (per prior PAPO verification).

**[V] Content (600-row sample across 6 offsets):** answers are short — median **2** chars, p90 **8**, max 90.
~29% are single-letter MCQ answers (A/B/C/D); the remainder are short numeric or short text (`"100"`,
`"30 minutes"`, `"Rectangles"`). Mixed MCQ / open-ended.

*Open:* whether to pre-filter to items the image-policy answers correctly — see OPEN-1.

## D6 — Answer form: short answer only **[U]**

The answerer `π_θ(·|c,x)` emits the final answer without a reasoning chain.

*Why:* makes `T` small and near-constant, which is what makes the faithful **sum** estimator (D7) safe. This
is native to ViRL39K, whose answers are 2 chars at the median — not a constraint we are imposing on the data.

*Cost to state honestly:* this measures caption sufficiency **for the answer**, not for a reasoning process.
The spec's language ("preserve the model's reasoning behavior") is more literally served by free CoT. We are
trading estimand breadth for estimator cleanliness, deliberately.

## D7 — Length normalization: sum over answer tokens, as written **[U]**

`D̂ = Σ_j [·]`, no division by `T`. Faithful to the spec and the correct estimator of the *sequence-level* KL.
Safe here only because D6 controls `T`.

*Standing requirement:* answer length must still be instrumented per-condition; a systematic length drift
would reopen the hacking channel even under D6.

## D8 — Advantage: GRPO group-relative **[U]**

Sample `G` captions per `(I,x)`; advantage `A_i = (−D̂_i − mean) / std` within the group.

*Deviation from spec, deliberate.* The spec's `∇J_cap = −E[sg[D̂]·∇log π_θ(c|I,x,q_cap)]` is bare REINFORCE
with no baseline. Since `D` is a KL, the reward `−D̂` is ≤0 in expectation, so every sampled caption is pushed
down in probability — unbiased but high-variance and entropy-collapse-prone. Group-relative centering makes
the signal zero-mean by construction and reuses the estimator already in our verl stack.

*Known caveat to carry:* std-normalization is the debated part of GRPO (it up-weights low-variance groups).
Recorded so it can be revisited; not currently changed.

## D9 — Estimator: per-position exact full-vocab KL **[U]**

At each answer position `j` along a trajectory sampled from `π_old(·|c,x)`, compute the exact full-vocab
`KL( π_old(·|c,x,y_<j) ‖ π_old(·|I,x,y_<j) )`, and sum over positions.

*Justification (why this is not a change of estimand).* By the chain rule for KL divergence,

```
KL(p‖q) = E_{y~p} [ Σ_j KL( p(·|y_<j) ‖ q(·|y_<j) ) ]
```

so this is an unbiased estimator of the **same sequence-level KL** the spec defines — it replaces each sampled
log-ratio with its exact conditional expectation (Rao-Blackwellization), giving strictly lower variance at
essentially no extra cost, since both forward passes already produce the logits.

**Implementation gates (both are correctness-critical, not optional):**
- The trajectory supplying the positions **must** be sampled from `π_old(·|c,x)` (the first KL argument). The
  chain rule does not hold for positions drawn from any other distribution.
- The sum **must** include the EOS position. Omitting it silently estimates a truncated KL.

*Tractable only because of D6:* with `T`≈2–8, full-vocab logits are ~1.5 GB per context at bf16 — versus the
32.6 GiB single-logits tensor that OOMed 4B-Thinking under PAPO at 8192 tokens.

## D10 — Sampling: G=5 captions per item, M=1 answer per caption **[U]**

`G=5` matches both the paper's `worker.rollout.n=5` and PAPO's group size. `M=1` is the spec's estimator.

## D11 — Data filtering: measure before deciding **[U]**

No filter is adopted yet. A pre-flight measurement of `Qwen3-VL-2B-Instruct`'s image-conditioned accuracy on a
ViRL39K sample decides it. Do not assume the rate. → resolved by Pilot 0.

## D12 — Vision tower: open (`freeze_vision_tower=false`) **[U]**

Matches the paper's Stage 1 and Stage 3.

*Noted counter-evidence, not acted on:* our own S2T campaign found `llm_only ≈ full` (0.749 vs 0.746) with the
ViT provably unchanged (`mean_rel_fro = 0.000`), i.e. freezing the encoder cost nothing there.

## D13 — Reference KL: `low_var_kl`, `kl_coef = 1e-2` **[U]**

Matches their Stage 1 (`use_kl_loss=true`, `kl_penalty=low_var_kl`, `kl_coef=1.0e-2`) and PAPO's β=0.01.
Applied to the caption tokens — the surface being trained. Guards against degenerate drift, which matters more
here than usual because `J_cap` alone has no task reward anchoring the policy.

## D14 — Caption length cap: from a pilot **[U]**

Sample untrained captions from the base model under the real `q_cap` on ViRL39K, then set the cap so
truncation is negligible. → resolved by Pilot 0.

## D15 — `q_cap` style: describe-only, minimally restrictive **[U]**

User's wording: a plain "do not give the answer" is enough; the prompt should *encourage* the model to report
every piece of information or relation derivable from the image that could help solve the question. Explicitly
**not** to be over-constrained — Track T's bug (an over-restrictive "do not infer relationships" clause
suppressed legitimate content and biased recovery) is the precedent to avoid. Exact text: pending approval.

## D16 — Leak gate: blind-no-image control arm **[U]**

Detect verdict-carrying captions behaviorally: measure whether the caption alone determines the answer at a
rate rising above the image-conditioned policy over training. Judge-free and cheap.

## D17 — Prompt parity between the two scored contexts **[CC]** — invariant, not a preference

`D(c) = KL(π(·|c,x) ‖ π(·|I,x))` isolates "caption vs image" **only if** the two prompts are identical in every
respect except the evidence source. Same question text, same answer-format instruction, same system prompt,
same chat template — differing only in caption text vs image tokens. Any other difference means we are
measuring template effects and reporting them as perceptual distortion. **Must be enforced by construction and
verified by an audit gate that diffs the two rendered prompts.**

## D18 — Captioner does not see MCQ options **[U]**

`q_cap` receives the question **stem only**; the answerer receives the full problem including options.

*Why:* removes the strongest leak channel — aligning a described detail to a specific option string. Probe A's
T3 did exactly this and measured a 0.36% leak rate.

*Cost, stated plainly:* this requires an options parser over the `problem` field, which is a new bug surface.
**[V]** Format variety is real and already demonstrated: a `Choices:|Options:` regex matched only 9.7% of a
600-row sample while 28.7% of answers are single letters — so a naive parser would silently miss most MCQ
rows. The parser needs its own validation gate (see OPEN-9).

## D19 — Answer format: `\boxed{answer}` via a suffix shared by both contexts **[U]**

The identical format instruction is appended to the caption-context and image-context prompts, satisfying D17.
Reuses the `mathruler` `extract_boxed_content` / `grade_answer` scorer already present in both the PAPO and
VLM-CapCurriculum stacks, so evaluation needs no new grader. Wrapper tokens are near-deterministic under both
contexts and so contribute ~0 to the KL.

## D20 — Pilot 0 approved: n=200 items, G=5 captions, all four measurements **[U]**

Base-model inference only, no training. Resolves D11 (filter) and D14 (cap), and establishes whether `D̂`
carries usable signal before the training loop is built. Full design in `PILOT_0_DESIGN.md`.

## D21 — Pilot variance decomposition: M=3 on a 50-item subset **[U]**

Main pilot runs at `M=1`; a nested 50-item subset re-runs at `M=3` purely to estimate answer-sampling noise.

*Why this was needed:* my original Pilot 0 proposal asked for a caption-vs-answer variance decomposition at
`M=1`, which is **unidentifiable** — with one answer draw per caption there is no within-caption replication.
Corrected before any code was written.

## D22 — Pool restricted to gradeable answers **[U]**

An answer is gradeable **iff `grade_answer(a, a) is True`** under the container's exact `mathruler`.

*Why the self-match rule:* if the grader cannot match an answer to itself, it can never credit a correct
response, so accuracy on that row is meaningless. This is a property of the *actual scorer*, not a regex proxy.
**[V]** A regex estimate put letter+numeric at ~76.6% of a 1,000-row sample (≈29.8K of 38,870), but the
`"other"` bucket contains items `mathruler` may well handle (`"1:1"`, `"2:1"`, `"4:05"`), so the true retained
fraction must be **measured in-container, not assumed**.

## D23 — Sampling: temperature 1.0, untruncated (`top_p=1.0`, `top_k=-1`), for BOTH caption and answer **[U]**

> **⚠️ AMENDED — supersedes the earlier "model-card `generation_config.json`" decision.** The original was
> recommended by me and approved; I then read the actual card values (`temp 0.7, top_p 0.8, top_k 20`) and
> withdrew the recommendation. Recorded rather than overwritten, per decision-log discipline.

*Why the amendment is a correctness matter, not a preference:* D9's estimator relies on
`KL(p‖q) = E_{y~p}[Σ_j KL(p(·|y_<j) ‖ q(·|y_<j))]`, which holds **only** for `y` sampled from the true policy
`p`. Truncated sampling draws from `p̃ ≠ p`, so `D̂` becomes silently biased. The same applies to the caption
rollout: GRPO's policy gradient assumes samples come from `π_θ`.

*Why PAPO's lesson does not transfer:* there, unbounded sampling made a **Thinking** model loop to a 8192-token
cap. Here the model is **Instruct**, answers are ~8 tokens, and captions are length-capped. Degeneration risk is
low and monitored; estimator bias would have been certain.

**[V]** `Qwen3-VL-2B-Instruct` `generation_config.json`: `temperature 0.7, top_p 0.8, top_k 20,
repetition_penalty 1.0, eos_token_id [151645, 151643]`. Recorded for provenance; deliberately not used for
rollouts. `repetition_penalty 1.0` (a no-op) is retained.

## D24 — `max_pixels = 4,194,304`; `min_pixels = 262,144` **[U]**

**[V]** `Qwen3-VL-2B-Instruct`: `patch_size 16`, `merge_size 2` ⇒ **1,024 px² per visual token** ⇒ 4194304 ≈
**4,096 visual tokens**. (Model's own preprocessor default is `longest_edge 16777216` ≈ 16,384 tokens.)

*Field practice at RL-training time:* EasyR1 default **4194304**; VLM-CapCurriculum Stage 1 **4194304**; PAPO
**1003520**; DeepEyes served at 12845056 (inference, not training). We take the top of the training band.

*The decisive reason, specific to this method:* the image-conditioned distribution `π(·|I,x)` **is the
supervision target**. In ordinary RLVR, low resolution merely costs accuracy; here it degrades the ground truth
itself — we would train captions to faithfully preserve the behavior of a model that could not see properly,
with a perfectly healthy-looking loss. Same class of defect that disqualified `D_perc` (D3), different door.
Reinforced by substrate: ViRL39K is chart/diagram/K12-science, where text-in-image and fine structure dominate.

*Explicitly not carried forward:* our S2T campaign's `262144`. That was a workaround for the aarch64 conv3d
slowness bug, since fixed (conv3d→matmul, bit-identical); honoring it now would respect a dead constraint.

## D25 — `SHARED_SUFFIX` **[U]**

```
Answer with only the final answer, in \boxed{}.
```

Appended **identically** to both scored prompts (D17). Short, invites no reasoning, and its near-deterministic
wrapper tokens contribute ≈0 to the KL — concentrating the signal on content tokens.

## D26 — Measurement (a) at n=5 draws per item **[U]**

Yields a per-item pass rate (0/5…5/5), the natural input to a D11 filter, and mirrors how VLM-CapCurriculum
built its `pass_rate` difficulty signal (they used 16 rollouts). Symmetric with `G=5` on the caption side.

## D27 — Pilot 0 gets its own smoke **[U]**

A small-scale run to prove the harness and every gate fires **before** spending on the real measurement.

## D28 — GOAL RESTATED (user, 2026-08-17) — supersedes the framing in earlier entries **[U]**

Train the model to **articulate the regions of the image that bear on the question**. Hypothesis: doing so
improves the model's **general perception**, so the trained model should beat the base model on a **held-out
VQA benchmark**, evaluated at **every checkpoint**. Explicitly a hypothesis, not an assumption.

**What this changes.** The `J_cap` parity ceiling (METHOD.md §5.1) bounds *blind-from-caption* performance
only. It does **not** bound image-conditioned VQA, which is the actual outcome. The concern is therefore
resolved for the headline metric.

**What it relocates.** `J_cap`'s gradient reaches **caption tokens only** (answerer and reference are
stop-grad), so any VQA change arrives as a **side effect of shared weights**. The hypothesis rests entirely
on that indirect pathway. It is plausible — this is effectively self-supervised perception training — but
unproven, and drift toward caption-style outputs could equally *degrade* direct QA. This makes the
reference-KL leash (D13) far more load-bearing than it appeared when set. **This is now the central risk.**

**Prior-work caution.** VLM-CapCurriculum's Table 6 ablation
(`training/examples/ablations/qwen3_vl_8b_sft_perception_then_stage23.sh`) is exactly "caption training
replaces Stage-1 perception RLVR", and their headline is that perception learns better via RLVR. Their
caption arm was **next-token SFT on generic human DOCCI captions** ("Please generate a detailed caption") —
imitative, not question-conditioned, and with nothing tying the caption to usefulness. Our objective differs
on all three axes, so their negative result arguably identifies the missing ingredient we supply. Carry it
as motivation *and* as a warning.

## D29 — Validation: MMK12_test **+ MMStar**, direct VQA only **[U]**

Per-checkpoint validation of the **image-conditioned** model (image + question → answer). MMK12 for
comparability with the PAPO arms; **MMStar** added because MMK12 is K12 science/math from the same curator
and family as ViRL39K — near-in-distribution, dominated by math-reasoning ability, and therefore a weak
instrument for a *general-perception* claim. MMStar is the instrument that can detect the effect, and Probe
A baselines already exist on it.

**Contamination check required** on both before use.

**Caption-mediated evaluation is deferred, not abandoned** — it can be run offline from preserved
checkpoints, which is one reason D31 is mandatory.

**Statistical power is a first-class requirement, not a detail.** VLM-CapCurriculum's *entire three-stage
recipe on 8B* moved overall accuracy **+1.46%**; a Stage-1-only run on 2B plausibly moves less. On 1,500
items the SE of a single accuracy is ~1.2 points, so a 1-point effect is **undetectable** by independent
comparison. Validation must therefore use **paired, same-item comparison with McNemar** (base vs
checkpoint, identical items and prompts), as Track T did — roughly an order of magnitude more sensitive, at
no extra cost.

## D30 — Placebo control: sequenced, revisit after the first analysis **[U]**

Run real-vs-base first. **If VQA moves, the placebo arm must be run before any claim is made** — otherwise
"is this just RL pressure on caption tokens?" is unanswerable. A null needs no control. User asked to be
consulted after the run's analysis. Placebo design (if run): identical loop, `D̂` computed against a
*different item's* image, so the reward carries no item-specific perceptual signal.

## D31 — Checkpoint preservation is MANDATORY **[U, emphatic]**

Every checkpoint is copied to `/capstor/store/cscs/swissai/a0174/caption_stage1_ckpts/` and **verified**, as
it is produced. Never left only on scratch.

**[V] Cause:** all 21 PAPO checkpoints were lost from scratch on 2026-08-11. Directory skeletons survived, so
the loss was invisible to `ls` — `global_step_60/actor/huggingface/` existed and was empty. The frozen
perception-KL probe is now unrunnable without ~3 twelve-hour slots per arm of retraining.

**[V] Target verified 2026-08-17:** `/capstor/.../a0174` is writable, **753 GB free of 1.0 TB**. Different
filesystem, untouched by that event (models there date from June and survive). It is a **shared** store, so
size discipline applies — the preserve script enforces a 50 GB headroom reserve.

Mechanism: `runs/preserve_checkpoint.sh`. Builds a sha256 manifest **from the source** before copying,
copies, then verifies every digest and the file count. Refuses to preserve a zero-file checkpoint — that is
the exact PAPO failure signature. Adds a replica; never moves or deletes.

## D32 — D11 threshold: decided from the measured pass-rate histogram **[U]**

Filter the training pool to rows the base model answers correctly from the image — but the **threshold** is
chosen once Pilot 0 measurement (a) reports the distribution.

*Why the threshold is not cosmetic:* the objective matches **distributions**, not modal answers. At
`pass_rate = 3/5` roughly **40% of the target probability mass sits on wrong answers**, so we would train
captions to make the blind model wrong 40% of the time on those rows. The filter's purpose was
"parity ⇒ correct"; at 3/5 it delivers "parity ⇒ 60% correct". The threshold directly sets how much
wrongness is in the supervision target.

---

## Open items (not yet decided)

- **OPEN-9** Options-parser validation: how we prove stem-extraction is correct across ViRL39K's format
  variety, and what happens to rows the parser cannot confidently handle. *Handled operationally by gate
  G-PARSE (drop + count + dump); the acceptance threshold for dropped rows is still unset.*

*Resolved:* ~~OPEN-1~~ → D11 + Pilot 0 · ~~OPEN-2~~ → D9 · ~~OPEN-5~~ → D16 · ~~OPEN-6~~ → D12 ·
~~OPEN-10~~ → D21 · ~~OPEN-11~~ → D22 · ~~OPEN-12~~ → D23.

**Still open, but NOT blocking Pilot 0** (training-loop items, to be decided before the loop is built):
- **OPEN-3** `θ_old` refresh cadence / `ppo_epochs` (on- vs off-policy).
- **OPEN-4** Caption length cap — resolved by Pilot 0 measurement (b).
- **OPEN-7** Batch sizes, steps/epochs, lr, seed.
- **OPEN-8** Evaluation set and success criteria — **must be frozen before any full run**, per
  `feedback_normative_must_be_frozen`.

- **OPEN-1** Pre-filter the train pool by image-conditioned correctness? Requires a pre-flight measurement of
  `Qwen3-VL-2B-Instruct` image-conditioned accuracy on ViRL39K — do not assume it.
- **OPEN-2** Estimator: spec's 1-sample k1 vs per-position exact full-vocab KL (Rao-Blackwellized, same
  estimand, strictly lower variance). **[CC]** recommends the latter.
- **OPEN-3** Group size `G`; answer samples per caption `M` (spec uses 1); `θ_old` refresh cadence.
- **OPEN-4** `q_cap` caption instruction wording; whether the answerer sees `q_cap` or `I` at all (must be
  provably blind); caption length cap.
- **OPEN-5** Answer-leak gates (a caption asserting the conclusion drives `D→0` with zero perceptual content).
- **OPEN-6** Vision tower frozen or open (their Stage 1: open).
- **OPEN-7** Batch sizes, steps/epochs, lr, KL-to-reference, seed.
- **OPEN-8** Evaluation set and success criteria — to be frozen **before** any full run.
