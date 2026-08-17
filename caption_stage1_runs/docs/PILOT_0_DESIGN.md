# Pilot 0 — design for approval

**STATUS: PROPOSAL. Not approved, no code written, nothing launched.**
Base-model inference only — **no training, no weight updates**. Approved in principle (D20); this document is
the detail that needs sign-off before any code exists.

Model: `Qwen3-VL-4B-Instruct` (D33, supersedes D4). Data: `PAPO_ViRL39K_train` (D5). Container: `easyr1_vllm0112.sqsh`
(vLLM 0.11.2, transformers 4.57.3, torch 2.9.0+cu129) via `~/toml/verl_easyr1.toml`. Execution: `sbatch`,
never interactive `srun` (SSH disconnect kills it — established run-ops lesson).

---

## 1. What this pilot exists to decide

| Measurement | Unblocks | Why it cannot be assumed |
|---|---|---|
| **(a)** image-conditioned accuracy on the pool | **D11** — whether to filter to image-correct items | If the base policy is mostly wrong, `J_cap` trains captions to preserve errors — the same defect that disqualified `D_perc` |
| **(b)** caption length distribution under real `q_cap` | **D14** — the caption length cap | Setting a cap without the distribution risks silent truncation of the object we are training |
| **(c)** blind-from-caption accuracy | baseline for the **D16** leak gate; headroom check | Need the pre-training value to detect a *rise* later |
| **(d)** `D̂` magnitude + variance decomposition | **D10** — confirms `G=5, M=1` is adequate | `D̂` is the entire learning signal; if it is noise-dominated the training loop cannot work |

If (d) shows no usable signal, **we do not build the training loop** — that is the point of running this first.

---

## 2. Pool construction (Step 1, CPU-only, in-container)

Runs before any GPU work.

1. Load all 6 parquet shards from the verified snapshot
   `hf_cache/hub/datasets--PAPOGalaxy--PAPO_ViRL39K_train/snapshots/ff6996d5cdd0e5fc12c01f3dab96f1af37453ceb/`.
2. **Gradeability filter (D22).** An answer is gradeable **iff `grade_answer(a, a) is True`** using the exact
   `mathruler` in the container. Rationale: if the grader cannot match an answer to itself, it can never
   credit a correct response, so accuracy on that row is meaningless. This replaces my earlier regex proxy,
   which estimated ~76.6% letter+numeric but was *not* a test of the grader.
   - Record the retained fraction and the per-format breakdown.
   - **Dump every rejected answer to disk** for inspection — a surprising rejection pattern would indicate a
     grader problem rather than a data problem.
3. **Options stripping (D18).** Split `problem` into stem and options. The captioner gets the stem; the
   answerer gets the full problem.
   - **Gate G-PARSE:** the parser must classify every row and, for MCQ rows, produce a stem containing **zero**
     option strings. Rows it cannot handle confidently are **dropped, counted, and dumped** — never silently
     passed through. The parser refuses rather than guesses.

   **[V] Measured on 1,400 real rows** (`code/virl_pool.py`, 31 unit tests in `code/test_virl_pool.py`):

   | outcome | share | note |
   |---|---|---|
   | `no_options` (free-form) | 54.6% | nothing to strip |
   | `mcq_labeled` | 36.1% | stem/options cleanly separated |
   | `unparseable` (dropped + dumped) | 9.2% | breakdown below |

   Unparseable breakdown: 78 rows where the **options are images** (`A. <image_1> B. <image_2>` — a blind
   answerer cannot answer these at all), 43 unlabeled bare-line rows (answer letters denote in-image labels),
   5 with untrustworthy label runs, 2 with prose-length bodies, 1 malformed.

   **Five option formats had to be handled; four were invisible until the data was inspected:**
   1. `A. text` at line start — the *dominant* style (451/541), not `(A) text` (90).
   2. Inline single-line options (`…choose from the options provided: A. 5 cm B. 10 cm …`) — 7.7% of
      letter-answer rows. Missing these handed the captioner the full option list.
   3. Stray prose labels *before* the options — this dataset is geometry-heavy, so "intersecting BC at point E"
      yields a raw match list of `['E','A','B','C','D']`. Assuming the first match starts the run broke 7.7% of
      MCQ rows; fixed by locating the **trailing canonical run**.
   4. Literal backslash-`n` instead of real newlines (~1.4% of rows).
   5. Lowercase labels `(a). … (b). …` (~0.07%) — detected only in order to **refuse**.

   Residual: exactly **1** letter-answer row in 1,400 still parses as free-form, and inspection confirms it is
   genuinely not multiple-choice ("…the hotel with better performance is hotel A. What is your reason?").
4. Deterministic seeded sample of **n=200** items; a nested **50-item subset** flagged for the M=3 arm.
5. Write `pool_manifest.json`: indices, answer formats, MCQ flags, seed, dataset revision SHA, code git SHA,
   and a content hash of the selected rows.

---

## 3. Prompts (the parity invariant, D17)

Three prompts are rendered. **Two of them must be identical except for the evidence span.**

**Captioner** — image + stem, `q_cap` (D15/D18):

> **`cs1_prompts.py` is authoritative for every prompt string; this block is a copy for reading.**
> It previously showed a superseded draft, and a frozen doc disagreeing with the code on a
> result-affecting string is worse than no copy at all. Synced 2026-08-17.

```
Look at the image carefully and describe what it shows, so that someone who cannot see the
image would have everything they need to answer the question below.

Report the concrete visual facts and the relationships between them — objects, attributes,
colours, counts, text and labels, positions, and how things relate to one another. Keep the
description compact and to the point, but do not leave out anything that could be useful.

Do not give the answer to the question.

Question: {stem}
```

*Clause removed (user-approved):* "Describe only what can be seen in the image." It stacked a second
prohibition on top of the answer ban and risked suppressing the relational content the caption exists to
carry — a caption forbidden from stating how things relate cannot substitute for the image. Track T's
over-restrictive `q_cap` is the precedent. Leak risk is handled by D16/leak-rate gates, not by this clause.

**Answerer (caption context)** — the `p` side of the KL:
`[caption text] + {full problem} + {SHARED_SUFFIX}` — **no image, no `q_cap`**.

**Reference (image context)** — the `q` side of the KL:
`[image] + {full problem} + {SHARED_SUFFIX}`.

`SHARED_SUFFIX` (D19/D25) is one string constant used by both, and is **`Put your final answer in
\boxed{}.`** ([cs1_prompts.py:57](../code/cs1_prompts.py#L57)).

> **Corrected 2026-08-17.** This line and §6 both still carried the superseded draft *"Answer with only the
> final answer, in `\boxed{}`."* — the wording that **fought the model and lost** (65–75% of answers ran to
> the cap, only 25–35% emitted `\boxed{}`). It was replaced precisely because it forbade the reasoning the
> model insists on doing; both reference pipelines permit reasoning and demand a parseable final slot. Two
> frozen locations were still specifying the abandoned string.

**Gates:**
- **G-PARITY:** render both scored prompts, strip the evidence span from each, assert the remainders are
  **byte-identical**. Fail loudly otherwise.
- **G-BLIND:** assert the answerer's inputs contain **no** vision tokens and **no** `pixel_values`. A blind
  arm that secretly sees the image would invalidate every number in this study.

---

## 4. Generation and scoring

**Sampling parameters (D23, as AMENDED):** `temperature 1.0`, `top_p 1.0`, `top_k -1` — **untruncated** — for
both caption and answer. Rationale is correctness, not preference: the D9 estimator identity holds only for
`y` drawn from the true policy `p`, and any `top_p`/`top_k` truncation draws from `p̃ ≠ p`, silently biasing
every `D̂`. GRPO's policy gradient makes the same on-policy assumption.

> **Corrected 2026-08-17.** This paragraph previously said to read the values from the model's
> `generation_config.json` and warned *"do not inherit VLM-CapCurriculum's `top_p=1.0`"* — i.e. it described
> the **pre-amendment** D23 and advised against precisely what the amended D23 requires. Stale text, now
> aligned. The card values (`temp 0.7 / top_p 0.8 / top_k 20`, identical on 2B and 4B) are still recorded in
> the run manifest **for provenance only** and are not used for rollouts.

*Independent corroboration (2026-08-17).* EasyR1's own default (`examples/config.yaml`) is `temperature 1.0`,
`top_p 1.0`, `top_k` unset — untruncated. Vision-SR1, whose method is the closest published analogue to ours,
overrides only to `temperature 1.0`, `top_p 0.99`, `n 8` (`vision_sr1/config.yaml`), and likewise never uses
its backbone's card — Qwen2.5-VL-7B-Instruct's card is `temperature 1e-6`, effectively greedy. Their `top_p
0.99` clips the extreme tail for stability at the cost of a small bias; we hold `1.0` for strict
unbiasedness, and treat `0.99` as the documented fallback if the D33 sweep shows degeneration.

**Two-pass structure** (the PAPO-probe pattern): generate with vLLM → persist to disk → score with HF in a
separate process, so vLLM and HF never co-reside and we avoid the sleep/wake OOM class.

| Pass | What | Count |
|---|---|---|
| 1 | Captions, with image, `G=5` per item | 200 × 5 = 1,000 |
| 2 | Blind answers from each caption, `M=1` | 1,000 |
| 2b | Blind answers, `M=3`, on the 50-item subset | 50 × 5 × 2 extra = 500 |
| 3 | Image-conditioned answers (for measurement **a**), n=5 per item | 1,000 |
| 4 | HF scoring forwards: full-vocab logits for each answer under **both** contexts | ~3,000 pairs |

**`D̂` computation (D9):** at each answer position `j`, exact full-vocab
`KL(π(·|c,x,y_<j) ‖ π(·|I,x,y_<j))`, summed over positions.

**Hard constraints, verified not assumed:**
- **G-SAMPLED:** the trajectory supplying the positions must be **sampled** from `π(·|c,x)`, never greedy. The
  chain rule does not hold otherwise and the estimator becomes a silently biased "modal-path KL".
- **G-EOS:** the sum must include the EOS position. Omitting it estimates a truncated KL.
- **G-ALIGN:** the two contexts have different prefix lengths. Assert the continuation token ids are identical
  under both renderings and that the logit index offsets are computed per-context. This is the PAPO
  perception-KL failure surface; it gets an explicit test, not a comment.

---

## 5. Readouts

- **(a)** image-conditioned accuracy, overall and **per answer format** (letter / numeric), with CI. Plus the
  per-item pass-rate distribution at n=5 — the input to any D11 filter.
- **(b)** caption token-length distribution (mean/median/p25/p75/p90/max) + truncation rate at candidate caps.
- **(c)** blind-from-caption accuracy, same breakdown. Also `(a) − (c)` = the caption's information loss.
- **(d)** `D̂` distribution; **variance decomposition** from the M=3 subset: caption-to-caption variance vs
  answer-to-answer variance within caption. Report `answer_frac = σ²_answer / σ²_total` — the same criterion
  that justified Track T's K=5.
- **Dead-group rate:** fraction of the 200 items where all G=5 captions yield near-identical `D̂` (zero-variance
  group ⇒ zero GRPO advantage ⇒ no gradient). A high rate predicts a stalled training run.
- **Degeneration check:** repetition/looping rate and truncation rate, to prove the sampling params are healthy.
- **Leak baseline:** blind-answer accuracy relative to image-conditioned accuracy, pre-training.

Artifacts: `captions.jsonl`, `answers.jsonl`, `dhat.jsonl` (per caption, with per-position KL retained),
`pilot0_report.json`, plus `_meta.json` carrying code SHA, dataset revision, model SHA, sampling params, seed.

---

## 6. Resolved knobs

All previously-open knobs are now decided (see `DECISIONS.md`):

| Knob | Value | Ref |
|---|---|---|
| `max_pixels` / `min_pixels` | **4,194,304** (≈4,096 visual tokens) / 262,144 | D24 |
| Sampling, captions **and** answers | temperature **1.0**, `top_p=1.0`, `top_k=-1` (untruncated) | D23 (amended) |
| `SHARED_SUFFIX` | `Put your final answer in \boxed{}.` | D25 |
| Measurement (a) draws | n=5 per item | D26 |
| Pool rule | `grade_answer(a, a) is True` | D22 |
| Variance decomposition | M=3 on a nested 50-item subset | D21 |

**Note for implementers — do not "fix" this later:** the image-context reference distribution is only ever
**scored**, never sampled from, so it takes no sampling parameters at all. Temperature/top-p must not be
applied to it; doing so would silently rescale `q` and corrupt every `D̂`.

## 7. Smoke (D27) — runs first, gates the real pilot

A ~10-item, G=2 run whose purpose is to prove the harness, not to measure anything. It must be discarded as a
measurement and is not reported.

**Pass conditions — all must hold before the real pilot is submitted:**

| Gate | Check |
|---|---|
| G-PARITY | Both scored prompts, evidence span removed, are byte-identical |
| G-BLIND | Answerer inputs contain zero vision tokens and no `pixel_values` |
| G-ALIGN | Continuation token ids identical under both renderings; per-context logit offsets verified |
| G-EOS | The EOS position is present in the summed KL |
| G-SAMPLED | Answer trajectories are sampled, not greedy; sampling params logged as untruncated |
| G-PARSE | Every row classified MCQ/non-MCQ; zero option strings survive in any stem; unparseable rows dropped and counted |
| G-GRADE | `grade_answer(a,a)` filter runs; retained fraction and rejects dumped |
| G-FINITE | Every `D̂` is finite and non-negative in aggregate (individual per-position KLs are ≥0 by definition — **a negative value proves an implementation bug**) |
| G-SHAPE | Logit tensors match `[batch, T, 151936]`; token counts agree across contexts |
| G-NOOOM | Completes without OOM at `max_pixels=4194304` |

**G-FINITE is the single most valuable gate here:** exact per-position KL is mathematically ≥0, so unlike the
spec's signed log-ratio estimator, *any* negative value is proof of a bug (misalignment, wrong context, or
swapped `p`/`q`). This is a correctness property the Rao-Blackwellized estimator gives us for free, and the
1-sample estimator would not have.

## 8. Cost

All inference on a 2B model. ~3,500 short generations + ~3,000 scoring forwards. The image-context forwards
dominate (vision tokens scale with `max_pixels`). Estimated well under one debug-partition slot, but I will
put a measured estimate in front of you before submitting anything.
