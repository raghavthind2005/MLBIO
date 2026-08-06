# Probe A — Every Setting, With Rationale. FOR APPROVAL BEFORE ANY CODE.

**Date:** 2026-08-05. **Status:** DECISION LIST AWAITING SIGN-OFF. No code written yet, by design
(`feedback_experiment_rigor_protocol`: approve every detail first).

**Design fixed by you:** dataset = **MMStar**; captioner constant = **CapRL-Qwen3VL-4B**; arms =
T0/T1/T2, I0/I1/I2, plus **A5 = the captioner answering MMStar itself** (Prism arm dropped at your
instruction — A5 measures the captioner's own VQA capability).

---

## D0. Codebase — build fresh from OUR code, do **not** vendor CapRL

**Decision: fork our own Track-T pipeline into `text_privilege/`; vendor nothing.**

Rationale:
- The experiment is **inference + rule-based scoring + paired stats**. No RL. CapRL's repo is
  OpenRLHF-based and needs *two incompatible vLLM versions in separate conda envs*
  (`CAPRL_TECHNICAL_READ.md` §8). Importing it buys nothing and costs a dependency war.
- `mv_gen.py` already implements this exact shape: vLLM load → build prompts with a frozen wrapper →
  prefill → K draws → per-draw logging (payload SHA, box, truncation, finish_reason, ntok) → provenance
  meta. `mv_gen_audit.py` runs 13 checks on the dump. `mv_score.py` is judge-free and 0-FP-asserted.
  `mv_analyze.py` has two-level bootstrap + McNemar + Holm. **All audited and known-good.**
- **We do read MMStar's official eval code** to match its scoring heuristic exactly — read, not vendor.

Honest caveat: this is a *port with heavy edits* (new loader, new scorer, three models instead of one,
seven arms), not a copy. Estimated ~5 new scripts.

---

## D1. Models & data — cluster inventory REQUIRED before anything

| Asset | Expected | Status |
|---|---|---|
| `Qwen3-VL-4B-Thinking` | `/capstor/store/cscs/swissai/a0174/models/` | **almost certainly present** (Set-2/3 + Track T used it) — VERIFY |
| `Qwen3-VL-4B-Instruct` | same path | **probably present** (`runs/RESULTS.md:20` references it for the Stage-1 RL work) — VERIFY |
| `CapRL-Qwen3VL-4B` | — | **almost certainly ABSENT → download (~8 GB bf16)** |
| MMStar (1,500 items + images) | — | **ABSENT → download** (HF; use `huggingface_hub`, not curl — Track T learned parquet is Xet-backed, commit `fac37fb`) |
| Container w/ vLLM + transformers ≥ 4.57.1 | — | VERIFY (CapRL's `config.json` declares 4.57.1) |

**Gate G0:** nothing proceeds until all five are confirmed on disk with byte sizes and SHAs logged.

---

## D2. Decode parameters — the most consequential block

### D2.1 Qwen3-VL-4B-Thinking → **reuse Track T's frozen recipe verbatim**
`temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0.0, repetition_penalty 1.0`, seed 0.

Rationale: this is the model card's recommended VL sampling, and **greedy was empirically rejected in our
own work** — Track-T prereg §3: greedy *"induces endless-loop degeneration (mechanics smoke — greedy
truncated ≥40% of seeded-arm draws at 40k tokens)"*, while recommended sampling drove truncation to ~0.
Re-deriving this would repeat a known failure.

### D2.2 Qwen3-VL-4B-Instruct → **its own card's recommended settings** (⚠ needs your call, see Q1)
The Instruct model has different recommended sampling. Using Thinking's settings on it would be
off-recommendation.

**Why this does not confound the headline comparison:** the caption effect is measured *within* each
model (`T1−T0`, `I1−I0`), and decode is held constant inside each contrast. The differencing cancels the
decode difference. Only a direct T-vs-I *level* comparison would be confounded — and we do not need one.

### D2.3 Captioner → **REVISED 2026-08-06: CapRL's own recommended sampling, not greedy**

**Superseded decision:** greedy, one frozen caption. **New decision:** `temperature 0.7, top_p 0.8,
top_k 20, repetition_penalty 1.0` (CapRL-Qwen3VL-4B's own `generation_config.json`, verified on disk),
fixed seed, captions frozen to disk and hashed after generation.

**Why greedy was wrong — and it is worse than a quality question, it is a validity question:**

1. **It would have biased the probe toward its own null.** Probe A is an *upper-bound* test: use the
   strongest available articulation so that a null is decisive (§0). Choosing an off-recommendation decode
   that risks degrading the caption **destroys that logic** — a null would then measure our decode choice,
   not the ceiling on articulation. A confound that pushes toward the null is fatal here specifically.
2. **Our own precedent says greedy degenerates in this family.** Track-T prereg §3: greedy *"truncated
   ≥40% of seeded-arm draws at 40k tokens with degenerate repetition"* on Qwen3-VL-4B-Thinking;
   recommended sampling drove truncation to ~0.
3. **CapRL is an RL-trained policy.** It was optimised by GRPO *under sampling*; its reward was computed
   on sampled rollouts. There is no guarantee its argmax is its best output — modes of RL-tuned policies
   can be degenerate. Evaluating it at a decode it was never optimised for is not "the strongest
   articulation."
4. **Length amplifies the risk.** Captions run to 2–4k tokens, and CapRL++ reports *">30% of responses
   are truncated due to excessive length"* even *under* sampling.

**The determinism argument for greedy was weak.** The requirement is that T1 and I1 consume a
byte-identical caption — and that is guaranteed by **freezing the artifact** (Pass 0 writes captions to
disk + SHA; Pass 2 reads that file), not by the decode. Sampling with a fixed seed is reproducible, and
the frozen file makes reproduction unnecessary.

### D2.3b Consequence — caption-sampling variance (**new open item, Q5**)

Sampling makes the caption *a draw from a distribution*. Freezing exactly one caption per image narrows
the estimand to "does **this particular** caption help," and the CI **under-counts caption variance**.

**This is precisely the bug Track T caught and fixed before its confirmatory run.** Prereg §4: the self
arm originally reused one stochastic description across all K answer-draws, which *"under-counts self
variance and biases the recovery CI narrow"*; fixed to K independent descriptions at commit `b9585bc`.
The signal report §5.4 records: *"The buggy version would have reported spurious positive recovery."*

**Proposed fix:** generate **M captions per item**, pair caption *i* with answer-draw *i*, and use the
**same caption set in T1 and I1** (rows stay comparable; variance is counted). Placebo donors draw from
the same set. Cost is small — captions are one forward pass per item; reasoner draws dominate.

**Proposed empirical resolution rather than an assumption:** measure caption variance at the smoke
(M=5 captions on 16 items; report length spread and content overlap). If caption variance is negligible,
M=1 becomes justified *by measurement*. See Q5.

### D2.4 K draws → **K=5** (see Q2 for the cost fork)
Matches Track T, so its variance-decomposition and two-level bootstrap carry over unchanged.
Cost: 7 arms × 1500 × 5 ≈ **52,500 generations**.

---

## D3. Context and length

- **Thinking:** `max_new_tokens=40960`, `max_model_len=49152` — Track T's proven values (~0 truncation).
- **Instruct:** `max_new_tokens=8192` (short answers; generous headroom).
- **Captioner:** `max_new_tokens=4096` (CapRL++ regularises captions at τ1=2048/τ2=3072).
- **Startup assert (Set-3 `p2_sweep.py` pattern):** `worst_prompt_tokens + image_reserve + max_new_tokens
  ≤ max_model_len`, failing loudly. **No silent prompt clamping — ever.**
- **Per-arm truncation reported; truncated = wrong (primary) + concluded-only sensitivity.** PAPO Arm C
  is the precedent: perception pressure moved truncation 7.2% → 16.6%. A ~2k-token caption in front of a
  long chain will move it here too.

---

## D4. Image resolution — pin it, or the arms aren't comparable

**Decision: pin `min_pixels` / `max_pixels` identically for the captioner and both reasoners; log the
realized image-token count per item per model and assert equality.**

Rationale: Qwen3-VL's processor turns an image into a variable number of tokens depending on these
bounds. If the captioner sees the image at a different resolution than the reasoner, "the captioner
described what the reasoner sees" is **false by construction** and the whole probe is uninterpretable.
This is exactly the class of silent setting that ruins an experiment. (PAPO used 200704 / 1003520; we use
the Qwen3-VL defaults unless the smoke shows a mismatch.)

---

## D5. Prompt construction

- **Wrapper (frozen, byte-identical across T1/T2/I1/I2), SHA logged:**
  `"From the image, I can see the following:\n<caption>\n"`
  Adapted from Track T's frozen `"From the figure…"` because MMStar includes natural images, not just
  figures.
- **Placement:** assistant-turn prefill in both rows. For Thinking the chat template auto-opens `<think>`
  so the payload lands inside the reasoning block (`mv_gen.py:78-81` asserts this — commit `1dfb5cb`
  fixed a double-open bug here, so the assert stays). For Instruct there is no `<think>`; payload goes
  directly into the assistant prefill.
  **Not the user turn** — Track-T prereg §3 rejected that as *"not a realistic privileged modification."*
- **Question prompt:** MMStar's **official** prompt/answer-format instruction, identical in all arms.
- **System prompt:** none, pinned explicitly (not "whatever the template defaults to").
- **Smoke dumps the exact prompt string for one item in every one of the 7 arms**, for eyeball
  verification.

---

## D6. Placebo construction (T2 / I2)

- Donor = another MMStar item's CapRL caption. Deterministic assignment, seed logged, donor IDs stored.
- **Length-matched** on caption token count (nearest-neighbour), tolerance reported.
- **Same-capability-category donor where possible** (see Q3) — a same-category placebo is the *harder,
  more conservative* control: it matches topic and style so only content differs. Cross-category would be
  trivially distinguishable.
- Asserts: donor image ≠ target image; donor caption ≠ target caption byte-wise; no self-assignment; no
  duplicate donors beyond a logged cap.

---

## D7. Scoring — judge-free, and the one confound our own history flags

- Match **MMStar's official heuristic** exactly (read their eval code first). MC letter extraction.
- Parse **post-`</think>`** for the Thinking model; full output for Instruct.
- Self-test to **0 false positives** on hand-built cases, the `mv_score.py` standard.

**⚠ The known trap — answer-format confound.** Track T §11.8 measured this: a model answering with the
option's *text* instead of its *letter* is scored wrong by a strict scorer, and **the rate is arm-dependent**
— base 0.045 vs privileged 0.159, a 3.5× difference. **A caption arm can look worse for pure formatting
reasons.** Mitigations, both pre-registered:
1. **primary = strict** letter scoring;
2. **secondary = format-tolerant** re-score crediting an option-text answer that matches the gold choice;
3. **per-arm non-letter rate reported** as a first-class number.

Also reported: per-option answer distribution per arm (detects a caption-induced choice-position bias —
CapRL shuffles options in its own reward for exactly this reason).

---

## D8. Statistics

- Per-item `p̂` = fraction of K draws correct. Paired **McNemar exact** on majority-vote binarisation +
  **10k two-level bootstrap** (resample items, then draws within item) — Track T's `mv_analyze.py`.
- **Pre-registered contrast family (7 tests), Holm-corrected:**
  `T1−T0`, `I1−I0`, **interaction `(I1−I0)−(T1−T0)`**, `T1−T2`, `I1−I2`, `T2−T0`, `I2−I0`.
- A5 (captioner VQA) is **descriptive only** — reported with a CI, not in the test family.
- Perception subsets (fine-grained / coarse) reported separately, pre-specified, not mined.

---

## D9. Execution, progress tracking, resume

Five passes, each writing a hashed artifact; **any pass resumable**:

| Pass | Output | Resume key |
|---|---|---|
| 0 · captions | `captions.jsonl` + SHA | item_id |
| 1 · placebo assignment | `placebo_assignment.json` + SHA | (deterministic, re-derivable) |
| 2 · generation | `gen_<arm>.jsonl`, **one file per arm** | `(item_id, arm, draw_idx)` |
| 3 · scoring | `scored.jsonl` | recomputable from Pass 2 |
| 4 · analysis | report | recomputable |

- **One arm per SLURM job** → a crash loses at most one arm, never the run.
- **Incremental append + flush per batch**; on restart, load completed keys and skip. Every row carries
  its resume key, so partial files are safe.
- Captions frozen in Pass 0 means a Pass-2 crash never regenerates stimuli — the stimulus set cannot
  drift mid-experiment.
- **Provenance in every output:** git SHA, model paths + SHAs, data SHA, wrapper SHA, realized
  `SamplingParams` dumped verbatim, image-token counts.
- **Captioner and reasoners never co-reside** in one process (PAPO's vLLM sleep/wake OOM lesson).

---

## D10. Smoke — the gate list (16 items, n=16 items × 7 arms × K=2)

Nothing runs at scale until **all** pass:

1. All 3 models + MMStar present; SHAs logged.
2. MMStar loads; **count == 1500**; every item has exactly 1 image; images decode; fields as expected.
3. **Chat-template asserts:** Thinking auto-opens `<think>`; Instruct does **not**. Fail loudly.
4. **Exact prompt string dumped for all 7 arms**, one item — payload present, correct position, correct wrapper.
5. **Image-token count equal** across captioner and both reasoners (D4).
6. Prompt-token + max_new_tokens ≤ max_model_len assert fires correctly when violated.
7. Caption generation: no degeneration, no empties, length distribution reported.
8. Placebo asserts (D6) all pass.
9. Scorer self-test: 0 FP; agreement with MMStar's official rule on a hand-checked sample.
10. **Per-arm non-letter-answer rate** (D7 confound) measured at smoke, not discovered at scale.
11. Per-arm truncation rate.
12. `SamplingParams` dumped and matched against the frozen spec.
13. **Resume test:** kill mid-pass, restart, verify skip-completed and identical output.
14. Determinism probe: re-run one cell; document any batch non-determinism (Set-3 one-run-rule basis).
15. Timing extrapolation → full-run wall-clock estimate before committing.
16. `gen_audit` equivalent passes on the smoke dump (Track T's 13-check pattern, adapted).

---

## D11. Open questions — **RESOLVED 2026-08-05 (user sign-off)**

| # | Question | Decision |
|---|---|---|
| **Q1** | Instruct decode settings | ✅ **Each model's own recommended values.** Thinking keeps Track-T's frozen recipe; Instruct uses its own card's. Justified because the caption effect is measured *within* each model, so decode is constant inside every contrast. **A direct T-vs-I level comparison is therefore NOT licensed and must not be reported as one** — only the within-model deltas and their interaction. |
| **Q2** | K draws | ⏳ **Deferred to the smoke.** K is fixed from the measured wall-clock extrapolation (smoke gate 15), then frozen before Pass 2. **Smoke gate 15 is now load-bearing, not informational.** |
| **Q5** | Captions per item **M** | ✅ **M captions per item, run with M>1 at the smoke.** `M` is a **command-line hyperparameter** (`--captions-per-item`), not a constant, so it can be retuned without touching code. Caption *i* pairs with answer-draw *i*; **the same caption set is shared by T1 and I1** so the rows stay comparable while caption-sampling variance is counted (the Track-T `b9585bc` lesson). Smoke measures caption variance so a later M=1 could be justified by data rather than assumed. |
| **Q6** | CapRL decode | ⏳ **Deferred to the smoke — no authoritative recommendation exists for this checkpoint (D11d).** Candidates generated and compared on length/repetition/degeneration before Pass 0 is frozen. |
| **Q3** | Placebo donor | ✅ **Same capability category**, length-matched. The harder, conservative control — topic and style held, only content differs. |
| **Q4** | A5 prompt | ✅ **Identical official MMStar prompt** as the reasoners, so A5 reads as "same task, different model" and is comparable to T0/I0. |

### Consequences of Q1 to carry into the write-up
The design supports `T1−T0`, `I1−I0`, and their interaction. It does **not** support "the Thinking model
is better/worse than the Instruct model," because those two levels differ in decode as well as in model.
Pre-state this so it cannot be misread later.

### Consequence of Q2
The smoke must emit a defensible per-arm timing extrapolation (generations/sec at realistic output
lengths, per model, including the caption-lengthened prompts), because **K is chosen from that number**.

---

## D11b. GATE G0 — cluster inventory RESULTS (2026-08-05, read-only)

**Present ✅**

| Asset | Path |
|---|---|
| `Qwen3-VL-4B-Thinking` | `/capstor/store/cscs/swissai/a0174/models/` |
| `Qwen3-VL-4B-Instruct` | same |
| Containers | `/capstor/store/cscs/swissai/a0174/ce-images/` — `vllm011.sqsh`, `vllm+latest.sqsh`, `easyr1_vllm0112.sqsh`, `verl-vllm.sqsh` |
| HF cache dirs | `$SCRATCH/hf_cache`, `$SCRATCH/huggingface` |
| Repo clone | `$SCRATCH/set2_pilot/MLBIO_repo` (git needs `GIT_DISCOVERY_ACROSS_FILESYSTEM=1` on the login node) |
| Scratch free | **753 TB** — no constraint |

**Absent ❌ → download required:** `CapRL-Qwen3VL-4B` (~8 GB bf16); **MMStar** (1,500 items + images).

### Finding 1 — Q1 instantiated from the actual checkpoints (`generation_config.json`)

| | temperature | top_p | top_k | repetition_penalty |
|---|---|---|---|---|
| **Qwen3-VL-4B-Thinking** | 1.0 | 0.95 | 20 | 1.0 |
| **Qwen3-VL-4B-Instruct** | **0.7** | **0.8** | 20 | 1.0 |

Thinking's values are **exactly** Track T's frozen recipe — confirming Track T simply adopted the
checkpoint's own `generation_config`, so our precedent and the model card agree.

**New consequence to record:** Instruct's lower temperature means **less decode variance per draw** than
Thinking. Within-item variance will therefore differ systematically between rows. This is not a bias in
any contrast (each is within-model), but the **interaction term's CI will be dominated by the noisier
Thinking row**. Pre-state it; do not discover it in the analysis.

### Finding 2 — D4 was worded against the wrong API, and the real risk is now sharper

Qwen3-VL does **not** use `min_pixels`/`max_pixels`. Its `preprocessor_config.json` uses
`size = {"shortest_edge": 65536, "longest_edge": 16777216}` (areas in px²), `patch_size 16`,
`merge_size 2`, `Qwen2VLImageProcessorFast`. **D4 is amended accordingly.**

Good news: **both reasoners carry byte-identical preprocessor configs**, so T-row and I-row see the same
image by default. The open risk narrows to one question: **does `CapRL-Qwen3VL-4B` ship the same
preprocessor config?** A captioner may well have been tuned at higher resolution — in which case "the
captioner described what the reasoner sees" is false and must be forced equal. **Checkable the moment it
is downloaded; added as smoke gate 5a.**

Note also `longest_edge = 16,777,216 px²` permits ~4096×4096 → up to ~16k image tokens for a large image.
MMStar images are modest, but the realized count must be **measured**, since it feeds the
`prompt + max_new_tokens ≤ max_model_len` assert (D3).

### Finding 3 — Track T's raw dumps are GONE

No `mv_gen*`, no `selfdesc`, no `pool_manifest` anywhere under `$SCRATCH` or the shared store. The
2,485 self-descriptions are not on disk.

**Consequence:** the "grade the self-descriptions at zero cost" fallback I recommended twice is **no
longer free** — it would require regenerating the Track-T self arm. This does not block Probe A (which
needs none of it), but it changes the cost of that diagnostic if we ever want it.

---

## D11d. Recommended-settings audit (2026-08-06) — **`generation_config.json` was NOT enough**

Read from each model card's **Multimodal** best-practice block (all our arms carry an image, so the
Multimodal block governs — the cards give different values for Text-only, which do **not** apply).

| | temperature | top_p | top_k | repetition_penalty | **presence_penalty** | **out_seq_length** |
|---|---|---|---|---|---|---|
| **Qwen3-VL-4B-Thinking** | 1.0 | 0.95 | 20 | 1.0 | **0.0** | **40960** |
| **Qwen3-VL-4B-Instruct** | 0.7 | 0.8 | 20 | 1.0 | **1.5** | **16384** |
| CapRL-Qwen3VL-4B | *(no model-specific recommendation — see below)* | | | | | |

**Two settings exist only in the model cards, not in `generation_config.json`:** `presence_penalty` and
`out_seq_length`. Relying on `generation_config.json` alone — which is what I had done — would have run
**Instruct at presence_penalty 0.0 instead of 1.5**, i.e. off-recommendation, and at **8192 instead of
16384** output tokens (my D3 draft). Both are now corrected. **D3 amended: Instruct
`max_new_tokens = 16384`.**

Also confirmed: Thinking's recommended block is **byte-identical to Track-T's frozen recipe**
(temp 1.0 / top_p 0.95 / top_k 20 / rep 1.0 / presence 0.0 / 40960). Our precedent was correct.

### CapRL-Qwen3VL-4B — no model-specific recommendation exists (**resolve at smoke, not by assumption**)

Two candidate sources, and neither is authoritative for this checkpoint:
- **Its `generation_config.json`** = temp 0.7 / top_p 0.8 / top_k 20 — **identical to Qwen3-VL-4B-Instruct**,
  i.e. almost certainly inherited from the base and never changed by the CapRL authors.
- **The README usage example** = temp 1.0 / top_p 1.0 / rep 1.0 — but that block is explicitly written
  for **CapRL-3B** (*"If you want to use CapRL-3B for captioning…"*, pointing at the Qwen2.5-VL series),
  a different base model.

**Open question about `presence_penalty` specifically:** the base's multimodal recommendation is 1.5, but
a presence penalty punishes token reuse, which is **actively harmful for dense captioning** — you
legitimately repeat object nouns, colours and positions. CapRL was RL-trained with no presence penalty
(GRPO rollouts). Applying 1.5 could degrade exactly the arm whose strength the probe depends on.

**Decision: measure it.** The smoke generates captions under the candidate settings and we pick by
observed quality — length distribution against CapRL++'s regularised band (τ1=2048/τ2=3072), repetition
rate, degeneration/truncation. Not an assumption; a number. **See Q6.**

### CapRL's own training prompt — use it verbatim

The `prompt` column of **CapRL-QA-75K** (the RL training set) is:

> **`Please describe this image in detail.`**

*(structured as `[{"role": "user", "content": "<image> Please describe this image in detail."}]`)*

**Decision: use this string verbatim as the captioning instruction.** Inventing our own prompt would put
the captioner off-distribution from its own RL training and weaken the upper bound the probe depends on.
This is "best outcome forward" in its most literal form.

Note the consequence: this prompt is **question-blind** — it confirms `CAPRL_TECHNICAL_READ.md` §4, and it
is the reason the caption cannot leak an answer. Also the reason a caption may simply omit the detail a
given MMStar item asks about (limitation §5.1 of `PROBE_A_DESIGN_NOTES.md`).

---

## D11c. MMStar — data verified + official scoring rule pinned (2026-08-06)

### Data (job 3018108, in-container; parquet **and** TSV read independently, they agree)

- **1,500 rows**, schema exactly as expected: `index, question, answer, category, l2_category, image,
  meta_info`. TSV cross-check: 1,500 rows, identical answer distribution. ✅
- **Categories perfectly balanced, 250 each** across 6 axes → **coarse perception 250 + fine-grained
  perception 250 = 500 perception items** for the pre-specified perception subset. 18 l2 axes.
- **⚠ Answer labels are NOT uniform: B 447 · A 429 · D 315 · C 309** (A+B = 58.4%). A model with an
  early-letter bias scores above chance. **Per-arm answer distribution is therefore mandatory reporting**
  (already D7) — a caption that shifts choice bias could move accuracy without moving perception.
- **Options are inline in the question**, format `"…question…\nOptions: A: …, B: …, C: …, D: …"` —
  note `A:` **not** `A.` (only 4/1500 contain `A.`). The scorer's regex must not assume `A.`.
- Question length 51 / **140** / 1802 chars (min/median/max).
- Images: W 89 / 512 / 3160, H 29 / 376 / 2560. **Estimated Qwen3-VL image tokens: median 171, mean 243,
  max 6,591** — cheap; the max is what the `prompt + max_new_tokens ≤ max_model_len` assert must clear.
- `meta_info` carries `source` (e.g. MMBench) — items are re-curated from other benchmarks; relevant to
  how we frame contamination.

### Official scoring rule (MMStar delegates to VLMEvalKit; its `eval/vlmeval/` embeds it)

`can_infer(prediction, choices)` = `can_infer_option(...)` **else** `can_infer_text(...)`:

- **`can_infer_option`** — strip punctuation `.()[],:;!*#{}`, split into words, count choice-letter
  occurrences. Match **iff exactly one letter is found AND it appears in the last 4 words**. Else if
  `'Z'`/empty → `'Z'`. Else regex `(?i)(?:correct\s+)?answer\s+is\s+\**([ABCD])\**`. Else `False`.
- **`can_infer_text`** — lowercase both sides; return a choice letter iff its text appears in the answer
  **and no other choice matches**; **rejected if the answer exceeds 2× the combined length of all choice
  texts**.
- **If `can_infer` returns `False`, the official pipeline falls back to an LLM extractor**
  (`model.generate(prompt)`).

**Our pinning (judge-free):** implement `can_infer_option` + `can_infer_text` faithfully; **do not invoke
the LLM fallback**. Un-inferable responses are **scored wrong (primary)** and the **per-arm un-inferable
rate is reported as a first-class number**.

**Two consequences worth recording:**
1. **The official rule already credits option-*text* answers** via `can_infer_text`. So Track-T's §11.8
   "format-tolerant sensitivity" is here the *official behaviour*, not an extension — that confound is
   substantially mitigated by simply adopting the official rule.
2. **Two residual arm-dependent risks remain, both measurable:** the last-4-words constraint and the
   2×-length rejection both **penalise verbose answers**. Thinking is more verbose than Instruct, and
   caption arms may be more verbose still. So the un-inferable rate can differ by arm for reasons
   unrelated to perception. **Measure it per arm at the smoke, not at scale.**

---

## D11e. Scoring — SUPERSEDES the scorer parts of D7 and D11c (2026-08-06, pre-outcome)

Three findings forced this, all from reading the **verbatim** cloned source (`$TP/vendor/`), not
web summaries:

1. **MMStar does not use VLMEvalKit's `can_infer` at all.** It ships `MMStar_eval`
   (`eval/vlmeval/evaluate/mmstar.py`): four prefix patterns anchored at **position 0**.
2. **An earlier reconstruction of `can_infer` (from web summaries) invented three rules that do not
   exist** — a last-4-words window, an `answer is ([ABCD])` regex, and a 2×-length rejection. All
   three penalise verbosity, which differs by arm; running with it would have produced
   arm-dependent error correlated with the treatment.
3. **The official scorer has a real defect**, found by unit-testing it: `answer == predict[0]`
   compares one character, so gold B + *"**B**ased on the image…"* is credited, as is
   *"**B**etween B and C…"*. With non-uniform gold labels this is not an edge case, and because
   arms differ in what precedes the answer, its false-positive rate can differ by arm.

**Resolution — `\boxed{}` becomes the primary, on the user's proposal.** Rationale: boxed has **no
silent false-positive mode**. A missing box is wrong *and countable*; a first-char artifact credits
a wrong answer invisibly. Measurable false negatives beat invisible false positives.

**Honest caveat, from our own history:** boxing does **not** remove the arm-dependent format effect.
Track-T §11.8 used boxed extraction and still measured non-letter-box rates of **base 0.045 vs
privileged 0.159**. It relocates the confound from "how the prose started" to "what went in the box".
Hence the same defences Track T needed:

| metric | role | value-box behaviour |
|---|---|---|
| **boxed letter, strict** | **PRIMARY** | not credited |
| + value→letter mapping, then `can_infer` | pre-registered **SENSITIVITY** | credited |
| `MMStar_eval` | **COMPARABILITY** | reported, never primary |

All three are computed from the **same generations** (zero extra cost) and Pass 4 runs under each.
Per arm we report `box_letter_rate`, `box_value_rate`, `box_missing_rate`, `firstchar_artifact_rate`.

**Legitimacy of changing the instrument now:** no outcome data exists. Freezing it *after* seeing
results would be the violation; fixing it before generation is when it is supposed to happen.

**A5 headline = `correct_tolerant`** (user decision, pre-outcome). CapRL is a captioner, never
trained to follow an MCQ format instruction, so under the boxed primary "cannot do VQA" would be
indistinguishable from "does not box". A5 remains descriptive only, outside the contrast family.

## D11f. Execution model

**Smoke — ONE job, all seven arms, serial.** Not one smoke per arm: the gates that matter most are
*comparative* (G17 payload-changes-output needs T0+T1; G19 ptok-matching needs T1 vs T2; G10b needs
the spread across arms; all of Pass 4). A per-arm smoke would pass and still miss the confounds it
exists to catch. Generation is grouped by model (3 loads, not 7) to fit the 1:30 debug window; the
resume test uses a **solo** arm so both invocation modes are exercised.
Scale: `--limit 48` → 8 per category → 240 captions + 7 × 48 × 3 = **1,008 generations**.

**Full run — one arm per job, seven in parallel** (`submit_full.sh`, SLURM dependency chain):
captions → placebo → {7 arms ∥} → score+analyse. Per-arm rather than per-model-group because
wall-clock becomes the slowest *arm* not the slowest *group*, crash isolation is finest, model-load
overhead is noise against hours of generation — and every arm is then produced by an identical
procedure, which matters when the entire result is a set of between-arm contrasts.

## D11g. What gets recorded

Principle: **store what exists only at generation time; anything derivable from stored text can wait.**
Full generation text is kept, so all textual quantities stay recoverable.

Engine-only, now captured: `img_tok` (realized, from `prompt_token_ids` − text `ptok`),
`cumlogprob`, per-draw `seed`, library versions (`vllm/transformers/torch/pandas/numpy`), and model
fingerprints (SHAs of `config.json`, `generation_config.json`, `preprocessor_config.json`,
`safetensors.index.json`). Set-2/3 had to log version strings as `[NOT IN RECORD]`; not repeated.

Cheap precomputes: `think_tok`/`answer_tok` split (the washout hypothesis predicts a caption
*shortens* the chain — S2T reports 20.8% shorter traces), `l2_category` (18 axes; MMStar's own
metric averages per l2), `caption_idx`/`donor_index` as fields.

Deliberately not stored: per-token logprobs (large, unused), full prompt strings (reconstructible —
G20 proves it), hidden states/attention (needs an HF pass, out of scope). Frozen captions + seeds
make any generation exactly reproducible for a later internals pass.

## D11h. Open items at the time of the peek

- **Q2 (K)** — fixed from smoke gate G15's timing extrapolation.
- **Q6 (CapRL decode)** — no model-specific recommendation exists for this checkpoint; being
  settled empirically by `peek_captions.sbatch` across all three candidates.
- Whether the boxed primary survives: depends on the per-arm `box_value_rate` / `box_missing_rate`
  spread observed at the smoke.

---

## D12. Status / next action

**BLOCKED on cluster access** — the CSCS cert expired (`Valid: … to 2026-08-05T14:17:01`).
Needs `bash cscs-keygen.sh` (MFA, so it has to be you).

On restore, in order:
1. **Gate G0 inventory** (D1): confirm/locate the three models, the container's vLLM + transformers
   versions, free scratch space; download CapRL-Qwen3VL-4B and MMStar as needed.
2. Read MMStar's official evaluation code and pin the scoring rule (D7).
3. Read `Qwen3-VL-4B-Instruct`'s model card for its recommended decode (Q1).
4. Only then write Pass 0–4 plus the smoke, and bring the smoke plan back for sign-off before running it.
