# Caption-Distortion — Decision Log

**STATUS: WORKING LOG. Nothing is frozen. No pre-registration exists. No code written, no runs launched.**

Append-only. Every entry records who decided, on what evidence, and — where it matters — what was argued on
the other side, so a decision can be revisited without re-deriving the argument from scratch.

Legend: **[U]** user decision · **[CC]** my recommendation, awaiting sign-off · **[V]** verified fact, with
the source of verification named.

Status tiers, used deliberately:
- **SETTLED** — decided; changing it is a change of experiment.
- **PROVISIONAL** — we are proceeding this way, but it is explicitly *not* frozen and is under active review.
- **OPEN** — not decided; listed with the live arguments.
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
  sighted. Verified by diff against `caption_stage1_runs/docs/SOURCE_SPEC_hackmd.md` @ git HEAD. Consequence:
  textual ambiguities noted before still exist here and are re-raised on their own merits below.

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
behaviour. This is a sharper factorisation than Vision-SR1, where both terms are answer-correctness and the
description reward can therefore be raised by getting better at answering from thin descriptions.

*Consequence:* the caption term reaches sighted VQA **only through shared weights.** The bet is that training
the model to serialise what it sees reshapes representations the sighted pathway reuses. Plausible, motivated
by Track T — and **unproven. This is the load-bearing step of the whole design.**

### S2 — Captioner sees the **entire** question, including options **[U, 2026-08-23]**

`x` handed to the captioner is the full problem text, identical to what the answerer and reference receive.

*Argument for (user's, and I agree it is sound):* symmetry. A caption generated for a different `x` than it is
evaluated on can omit precisely the distinction that separates the options. It also deletes the
options-parser bug surface entirely — a real, measured cost (the prior attempt found five undocumented
formats, 9.2% unparseable rows, and a full-scale gate failure on a histogram legend that was option-*shaped*
without being options).

*Cost, accepted knowingly:* leakage is no longer prevented structurally. Combined with reverse KL (P1) — which
is itself the leak-attractive direction — the two stack. **Leakage therefore moves from prevention to
instrumentation, and the instruments are not optional.** See §5.

### S3 — `q_cap` is permissive **[U, 2026-08-23]**

A single "do not state the answer" clause; otherwise the captioner is given freedom to report whatever is
derivable from the image. Deliberately *not* over-constrained: Track T's precedent is an over-restrictive
caption instruction ("do not infer relationships") that suppressed legitimate content and biased the result.

---

## 2. PROVISIONAL — proceeding this way, explicitly not frozen

### P1 — KL direction: **start with reverse**, forward under active deliberation **[U, 2026-08-23]**

`D(c) = KL(π(·|c,x) ‖ π(·|I,x))`, as the spec writes. **Not frozen — the user has agreed the forward-KL
argument is substantive and wants further deliberation before committing.**

The distinction is *not* mainly about cost. Both directions evaluate both contexts; they differ in which one
must be **sampled** from versus merely **scored** under. Reverse needs `ỹ ~ π(·|c,x)` — an autoregressive
generation per caption. Forward needs `y ~ π(·|I,x)` — which S1 already generates for `J_success` — and then
only a teacher-forced scoring pass per caption. Reverse buys back blind-answer accuracy for free; under
forward that becomes a separate instrumentation pass.

The training implications, which are the real argument:

| | **Reverse** `KL(p‖q)` | **Forward** `KL(q‖p)` |
|---|---|---|
| character | mode-seeking / zero-forcing | mode-covering / zero-avoiding |
| optimum | caption makes the blind model collapse confidently onto **one** high-probability region of sighted behaviour | caption must make the blind model assign mass to **everything** the sighted model does |
| **leakage** | **attractive** — a caption asserting the verdict makes `p` a spike; if `q` favours it too, `D → 0` with zero perceptual content | **resistant** — the same spike is heavily penalised, since `q` has mass across the reasoning and `p` has none there |
| uncertainty | **discarded** — caption becomes a sufficient statistic of the image's *argmax*, not of the image; expect systematic under-dispersion, worst on the ambiguous items where perception is hardest | preserved |
| with full-CoT `y` | asks whether the blind model's own chain would look plausible to the sighted model | asks *how likely the caption-conditioned model finds the sighted model's actual chain* — i.e. does the caption contain the facts the reasoning invokes. Near-direct measurement of the programme's question (cf. Track T) |
| group variance | each caption scored on its **own** trajectory ⇒ per-caption lexical noise | all `G` captions scored on the **identical** trajectory ⇒ stylistic variation is common-mode and cancels in the centred advantage; the `log q` term is a per-item constant that cancels exactly |

*Honest case for reverse, and it is real:* at deployment one samples **from** the blind model, so matching in
the direction one samples is the operationally relevant one; and mode-seeking is typically more stable early,
since it never forces coverage of a poorly-estimated `q`.

*Decision rule going in:* run reverse with under-dispersion and leak rate instrumented **as predicted failure
modes from day one**, so that if reverse misbehaves we switch on evidence rather than on argument.

### P2 — `y` = full CoT **+** answer *(leaning; not settled)*

*User's position:* full CoT + answer is the better estimand. **I agree, but not for the stated reason.**

*The stated reason does not survive contact with our own measurements.* "Use an Instruct model to reduce the
load" assumes Instruct answers are short. **[V]** Qwen3-VL-4B-Instruct measured **median 1,125 tokens**
(job 3105710); at 2B, median 3,072 with 75% truncation. It is documented, unfixed upstream behaviour
(QwenLM/Qwen3-VL#1922 — their median 1,899 vs our measured 1,959), and `chat_template.json` exposes no
thinking/budget knob. Instruct here is *relatively* shorter than Thinking, not short. Choosing Instruct on
that basis would re-adopt a premise that was already falsified once.

*The reason that does hold:* full CoT is **leakage-resistant**. If `y` is the answer alone, a caption stating
the verdict achieves parity with zero perceptual content. Reproducing a *chain* requires supplying the facts
the chain reasons over — you cannot shortcut a hundred steps by naming the conclusion.

*Open sub-question:* memory. Under the sampled-token (k1) estimator the spec writes, cost is O(T) per
sequence and full CoT is affordable. Under a Rao-Blackwellised per-position full-vocab KL it is
`vocab × T × batch` across two contexts, which becomes the binding constraint — mitigable by chunking over
positions and reducing to a scalar per chunk, never materialising `[T, V]` twice. **Estimator family is
itself open (O3).**

### P3 — Dataset: chart/figure/table block of **Vision-SR1-47K** **[CC, awaiting sign-off]**

Boundary is *their own* `path` category (`./Chart/`) plus TabMWP/IconQA — a column in the released artifact,
not a filter we invent. ≈9,000 rows: ChartQA 955, DVQA 1,623, PlotQA 885, FigureQA 1,254, MapQA 806,
TabMWP 3,543.

Rationale against the criteria in §4: chart values exist **only** in the image (R1); a chart is a discrete set
of labelled values, close to the ideal case for lossless textual serialisation (R3); answers short and
rule-gradeable (R4); modest resolution (R7); single-image (R8); right size for a trial (R9).

*Costs, stated plainly:* **R6 diversity is sacrificed** — one perceptual mode, so this is a trial substrate
and never a general claim. And chart QA may be **too easy** for a 4B backbone, saturating pass rates and
starving R2; this is the one property to check before committing.

---

## 3. OPEN — not decided

- **O1 — Backbone.** Instruct vs Thinking. Instruct is more tractable and, per P2, still emits a real chain
  (~1,125 tokens median). Thinking is where the phenomenon we are chasing actually lives — every diagnostic
  in the programme (Track T, RH-Bench, HallusionBench length-collapse) concerns degradation during *long*
  reasoning. Choosing Instruct buys tractability and incurs a declared estimand-vs-phenomenon gap.
- **O2 — Reward composition.** `λ` versus two separately group-normalised advantages. The terms live on
  incompatible scales: `R(y)` is bounded in [0,1]; a sequence-level KL summed over tokens is unbounded and
  length-dependent. Vision-SR1 uses separate z-scored advantages with λ=0.5 each; that is the field-standard
  answer and probably ours, but it is not yet decided.
- **O3 — Estimator family.** The spec's one-sample signed `Σ log(p/q)` versus a Rao-Blackwellised
  per-position exact KL. Interacts with P2 (memory) and gives a free correctness oracle (per-position KL is
  ≥ 0 by construction, so a negative value proves a bug).
- **O4 — Conditional caption reward.** Gate the caption term on the sighted rollout being correct:
  `1[R(y)>0] · (−D̂)`. Removes "train captions to faithfully reproduce a wrong chain" at the root. Precedent:
  DeepEyes' conditional tool bonus, whose ablation (their Table 5) shows the conditionality is what makes the
  behaviour emerge at all. **[CC]** I think this is likely right; not proposed formally yet.
- **O5 — `θ_old` refresh cadence.** Non-trivial here rather than routine: the accuracy term is *deliberately*
  moving `π(·|I,x)`, so the caption is chasing a shifting reference by design.
- **O6 — Group size `G`, batch shapes, steps, lr, seed.**
- **O7 — Evaluation set and success criterion.** **Must be frozen before any full run.** This is the single
  rule the previous attempt broke — five GPU jobs ran before anyone had written down what winning looked like.
- **O8 — Control arm.** S1 makes the natural primary comparison *accuracy-only GRPO* vs *accuracy + caption-KL*
  on identical data, steps and seed — a true one-variable ablation, and the same axis Vision-SR1 reports
  (+1.7 / +1.5 / +3.5 over answer-reward-only GRPO on identical data).

---

## 4. Derived dataset requirements

Each falls out of a specific term in the objective, not from taste.

| # | Requirement | Derivation |
|---|---|---|
| **R1** | **Vision-necessary** — `x` alone must not determine `y` | If it does, `π(·|c,x) ≈ π(·|I,x)` for *every* caption ⇒ `D ≈ 0` across the group ⇒ zero variance ⇒ **zero gradient**. The caption term is not weak here, it is **vacuous**. |
| **R2** | **Intermediate difficulty** — sighted pass rate strictly in (0,1) | GRPO's advantage is group-relative; all-wrong and all-correct groups both give zero advantage. |
| **R3** | **Captionable** — question-relevant content finitely and precisely statable in text | **[V]** Set 3: both text payloads "enumerate objects but not spatial relations; CLEVR is relational, so objects-as-text cannot restore the layout." Content that resists serialisation gives `D` a floor no training can remove. |
| **R4** | **Rule-verifiable answers** | `J_success` must be computable without an LLM judge. |
| **R5** | **Uncontaminated** | A memorised item is answered from parametric knowledge, not the image — violating R1 through another door. The R1 screen catches this for free. |
| **R6** | **Diverse perceptual operations** | Required only if the claim is *general* perception improvement. Explicitly sacrificed in P3. |
| **R7** | **Images within the pixel budget** (no forced downscaling) | Our own localisation says the deficit is LLM **read-out** of adequate visual tokens. Downscaled-past-resolvable items test *encoding* instead — a different paper. |
| **R8** | **Single image per item** | Otherwise `c` has an undefined referent and the two KL contexts diverge structurally. |
| **R9** | **Survives filtering at volume** | ≥ ~2K after screening supports 30–60 trial steps at rollout batch 128–256. |

**Measured evidence on candidate pools:**
- **[V]** ViRL39K: **34% of rows answered 5/5 from question text alone** (first-party no-evidence control,
  2026-08-18). On those rows the caption gap is structurally zero.
- **[V]** Vision-SR1-47K composition (HF datasets-server, 2026-08-23; 47,628 rows, 30 sources):
  Knowledge 25.2% · Spatial 21.8% · Chart/table/figure 21.6% · Math/geometry 20.1% · General 6.6% ·
  CLEVR 4.7%. `problem_type`: multiple choice 62.4%, numerical 26.4%, **regression 11.2%**.
- **[V]** **The released artifact does not match the paper's Table 1.** Table 1 lists Math 30.5% / Science
  Knowledge 30% / General Visual Reasoning 39.5% over a named source list; the artifact contains
  **Multimath-300k (5,000 rows, 10.5%)**, which appears nowhere in that table. Comparability claims of the
  form "same data as Vision-SR1" need care.
- **[V]** The artifact is **block-grouped, not shuffled** (offsets 12000 and 20000 are both contiguous
  Multimath-300k), so any single parquet shard is a biased single-source slice. Subsets must be seeded
  stratified draws.
- **[V]** No literature dataset validates vision-necessity at *training* scale. Those that validate it are
  benchmarks: MMStar 1,500 · NaturalBench 1,900 · CV-Bench 2,638 · BLINK ~3.8K.

---

## 5. Predicted failure modes, to instrument from day one

Named in advance so that a healthy-looking run cannot be mistaken for a correct one — the lesson from the
PAPO audit, where every logged metric looked fine while the perception loss was receiving no gradient at all.

1. **Answer leakage into the caption.** Structural, given S2 + P1. Instruments: (a) gold-string containment
   in `c`; (b) verdict-assertion phrasing; (c) the strong one — **answer from `c` with `x` removed**: a
   caption carrying *evidence* cannot answer a question it cannot see, while one carrying a *verdict* can.
2. **Under-dispersion** of `π(·|c,x)` relative to `π(·|I,x)` — the predicted signature of reverse KL (P1).
   Distinguish from leak-induced sharpening; they are not the same thing.
3. **Length hacking.** A KL summed over `T` tokens rewards shorter continuations mechanically. **[V]** Probe A
   documented this exact pathology on this model family: injected text drove premature `</think>` closure in
   55% / 84% of generations, and short-circuited generations were 11–13 accuracy points worse.
4. **Dead groups** from vacuous items (R1) — measurable directly as the fraction of groups with ~zero `D̂`
   variance.
5. **Fidelity-to-a-wrong-chain** on items the sighted model gets wrong (mitigation: O4).

---

## 6. REJECTED

- **DeepEyes-Datasets-47k as the substrate.** Its published perception-utility filter keeps samples "where
  the ground-truth crop provably helps" — i.e. it selects for the **encoding-limited** regime. The captioner
  sees the same un-zoomed image, so where the information is not resolvable at full resolution, no caption
  built from it can carry that information either. Selected for exactly the case the method cannot serve.
- **VLM-CapCurriculum `D_perc` as the training set.** Construction rule keeps rows iff `Â_img(Q|I) ≠ A` —
  the sighted answer is **wrong by construction on every row**. Under S1 the accuracy term would have no
  correct behaviour to reinforce, and under O4 the caption gate would never fire.
- **"Instruct implies short answers."** Falsified — see P2.

---

## 7. Chronology

| date | event |
|---|---|
| 2026-08-21 | Sub-repo created. Prior attempt reviewed end-to-end; clean break declared. |
| 2026-08-21 | `J_success` fixed to the sighted answer (S1). New spec stored + hashed. |
| 2026-08-23 | Reverse KL adopted provisionally (P1); forward-KL argument accepted as substantive, deliberately left unfrozen. |
| 2026-08-23 | Captioner given the full question (S2); leakage moved from prevention to instrumentation. |
| 2026-08-23 | Dataset requirements R1–R9 derived; Vision-SR1-47K characterised; chart block proposed (P3). |
