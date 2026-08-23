# Caption-Distortion — Decision Log

**STATUS: WORKING LOG. Nothing is frozen. No pre-registration exists. No training code written, no
training runs launched.**

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

*Cost, accepted knowingly:* leakage is no longer prevented structurally. Combined with reverse KL (P1) — which
is itself the leak-attractive direction — the two stack. **Leakage moves from prevention to instrumentation,
and the instruments are not optional.** See §5 and roadmap item **L1**.

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

---

## 2. PROVISIONAL — proceeding this way, explicitly not frozen

### P1 — KL direction: **start with reverse**, forward under active deliberation **[U, 2026-08-23]**

`D(c) = KL(π(·|c,x) ‖ π(·|I,x))`, as the spec writes. **Not frozen — the user has agreed the forward-KL
argument is substantive and wants further deliberation before committing.** Scheduled at roadmap **stage 3**;
§3 explains why it cannot responsibly be settled before stage 2.

The distinction is *not* mainly cost. Both directions evaluate both contexts; they differ in which must be
**sampled** from versus merely **scored** under. Reverse needs `ỹ ~ π(·|c,x)` — an autoregressive generation
per caption. Forward needs `y ~ π(·|I,x)` — which S1 already generates for `J_success` — then only a
teacher-forced scoring pass per caption. Reverse buys back blind-answer accuracy for free; under forward that
becomes a separate instrumentation pass.

| | **Reverse** `KL(p‖q)` | **Forward** `KL(q‖p)` |
|---|---|---|
| character | mode-seeking / zero-forcing | mode-covering / zero-avoiding |
| optimum | caption makes the blind model collapse confidently onto **one** high-probability region of sighted behaviour | caption must make the blind model assign mass to **everything** the sighted model does |
| **leakage** | **attractive** — a caption asserting the verdict makes `p` a spike; if `q` favours it too, `D → 0` with zero perceptual content | **resistant** — the same spike is heavily penalised, since `q` has mass across the reasoning and `p` has none there |
| uncertainty | **discarded** — caption becomes a sufficient statistic of the image's *argmax*; expect under-dispersion, worst on ambiguous items where perception is hardest | preserved |
| with full-CoT `y` | asks whether the blind model's own chain would look plausible to the sighted model | asks *how likely the caption-conditioned model finds the sighted model's actual chain* — i.e. does the caption contain the facts the reasoning invokes. Near-direct measurement of the programme's question |
| group variance | each caption scored on its **own** trajectory ⇒ per-caption lexical noise | all `G` captions scored on the **identical** trajectory ⇒ stylistic variation is common-mode and cancels; the `log q` term is a per-item constant that cancels exactly |

*Honest case for reverse, and it is real:* at deployment one samples **from** the blind model, so matching in
the direction one samples is operationally relevant; and mode-seeking is typically more stable early, since it
never forces coverage of a poorly-estimated `q`.

*Decision rule going in:* run reverse with under-dispersion and leak rate instrumented **as predicted failure
modes from day one**, so that if reverse misbehaves we switch on evidence rather than on argument.

### P2 — `y` = full CoT **+** answer *(leaning; not settled)*

*User's position:* full CoT + answer is the better estimand. **I agree, but not for the stated reason.**

*The stated reason does not survive our own measurements.* "Use an Instruct model to reduce the load" assumes
Instruct answers are short. **[V]** Qwen3-VL-4B-Instruct measured **median 1,125 tokens** (job 3105710); at 2B,
median 3,072 with 75% truncation. Documented, unfixed upstream behaviour (QwenLM/Qwen3-VL#1922 — their median
1,899 vs our 1,959), and `chat_template.json` exposes no thinking/budget knob. Instruct here is *relatively*
shorter than Thinking, not short.

*The reason that does hold:* full CoT is **leakage-resistant**. If `y` is the answer alone, a caption stating
the verdict achieves parity with zero perceptual content. Reproducing a *chain* requires supplying the facts
the chain reasons over.

---

## 3. ROADMAP — everything still open, in the order it must be solved

Ordered by dependency, not by importance. Each entry says what it blocks and why it cannot come earlier.

### Stage 0 — housekeeping (non-blocking)

| id | item |
|---|---|
| **H1** | `_env.sh` documents `HF_HOME` as `.../hf_cache`, but the cluster environment already sets it and the snapshot landed in `.../huggingface/`. **[V]** job 3163760. The comment is now inaccurate; correct the file rather than leave a doc that lies. No re-download needed. |

### Stage 1 — the one hard blocker

| id | item | why it is first |
|---|---|---|
| **O1** | **Backbone.** Instruct vs Thinking; size. | Everything downstream is **model-relative**. Vision-necessity (M1) and difficulty (M2) are properties of *a model on this data*, not of the data alone — they cannot even be measured before this is chosen. It also fixes chain length, which sets the estimator's memory bill. Instruct is tractable and still emits a real chain (~1,125 tokens median). Thinking is where the phenomenon the programme documented actually lives (Track T, RH-Bench, HallusionBench length-collapse); choosing Instruct buys tractability at the price of a declared estimand-vs-phenomenon gap. |

### Stage 2 — substrate characterisation (needs O1; cheap; go/no-go)

| id | item | why here |
|---|---|---|
| **M1** | **Vision-necessity rate** — question text only, no image, n draws, on the trial pool. **Reported, not used as a filter.** | Decides whether `D` is vacuous and how often. Falsifiable prediction to check the instrument against itself: Knowledge should shed far more than Chart. Needs O1 because it is model-relative. |
| **M2** | **Sighted pass rate** — image + question, n draws. | Decides whether `J_success` has room to grow (R2). Also yields the **chain-length distribution for free**, which is the input to P2 and P1. |

These two together answer "does this substrate support the experiment at all" for roughly two generation
passes. If M1 shows the pool is largely vacuous, the substrate question reopens before any training code
exists — which is the point of running them here rather than later.

### Stage 3 — the estimand (needs O1 and M2's measured chain lengths)

| id | item | why it must wait |
|---|---|---|
| **P2** | Finalise `y` scope: full CoT + answer, answer-span only, or full chain with the divergence restricted to the answer span. | The cost and variance arguments are all functions of `T`. Deciding before M2 measures the actual chain-length distribution would be guessing at the one number that drives the choice. |
| **P1** | **Forward vs reverse KL.** | Depends on P2. With short `T` the extra blind generation reverse requires is cheap and the choice is nearly free; with `T ≈ 1,100` it is a large recurring cost, and forward's common-mode cancellation across a shared trajectory becomes a serious variance advantage. The leakage asymmetry argues for forward, S2 sharpens that, and reverse is what the spec says — this is the most consequential open decision in the document and it is deliberately scheduled after the evidence that bears on it. |
| **O3** | Estimator family: the spec's one-sample signed `Σ log(p/q)` vs a Rao-Blackwellised per-position exact KL. | Determined jointly by P1 and P2. Memory is `vocab × T × batch` across two contexts for the exact form — chunkable, but only worth the complexity if the variance reduction is needed. The exact form gives a free correctness oracle: per-position KL is ≥ 0 by construction, so a negative value proves a bug. |

### Stage 4 — reward shape (needs stage 3, and M2)

| id | item | why here |
|---|---|---|
| **O4** | **Conditional caption reward:** gate the caption term on the sighted rollout being correct, `1[R(y)>0]·(−D̂)`. | Removes "train captions to faithfully reproduce a wrong chain" at the root. Precedent: DeepEyes' conditional tool bonus, whose Table 5 ablation shows conditionality is what makes the behaviour emerge at all. **Needs M2** — if the sighted pass rate is very low the gate rarely fires and starves the caption term. |
| **O2** | Reward composition: a single `λ` vs two separately group-normalised advantages. | The terms live on incompatible scales — `R(y)` is bounded in [0,1], a sequence-level KL is unbounded and length-dependent. Vision-SR1 uses separate z-scored advantages at λ=0.5 each; likely ours, but it needs O3 first since the scale of `D̂` depends on the estimator. |
| **S3b** | Exact `q_cap` wording. | Cheap, but it is a result-affecting string and belongs under sign-off, not improvisation. |
| **L1** | **Leak instrumentation spec.** Three instruments: gold-string containment in `c`; verdict-assertion phrasing; and the strong one — **answer from `c` with `x` removed** (a caption carrying evidence cannot answer a question it cannot see; one carrying a verdict can). | Made load-bearing by S2 + P1. Must exist before the first training run, not after a suspicious result. |

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

---

## 5. Predicted failure modes, to instrument from day one

Named in advance so a healthy-looking run cannot be mistaken for a correct one — the PAPO lesson, where every
logged metric looked fine while the perception loss received no gradient at all.

1. **Answer leakage into the caption.** Structural, given S2 + P1. Instruments in roadmap **L1**.
2. **Under-dispersion** of `π(·|c,x)` relative to `π(·|I,x)` — the predicted signature of reverse KL.
   Distinguish from leak-induced sharpening; they are not the same thing.
3. **Length hacking.** A KL summed over `T` tokens rewards shorter continuations mechanically. **[V]** Probe A
   documented exactly this on this model family: injected text drove premature `</think>` closure in 55% / 84%
   of generations, and short-circuited generations were 11–13 accuracy points worse.
4. **Dead groups** from vacuous items — measurable as the fraction of groups with ~zero `D̂` variance.
5. **Fidelity-to-a-wrong-chain** on items the sighted model gets wrong (mitigation: **O4**).

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
