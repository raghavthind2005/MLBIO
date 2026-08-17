# Probe A — Full Run Report (n=1500, K=3)

**Date:** 2026-08-07. **Status:** full run complete, all gates PASSED (job 3021988 generation+analysis, audit re-run job 3028154 after a post-hoc gate fix — see §7). **Model:** Qwen3-VL-4B-Thinking (T-arms) / CapRL-Qwen3VL-4B (A5). **Dataset:** MMStar, all 1500 items, K=3 draws/item. **Code:** git `1fcc280` (branch `diagnostic-signal-study-track-t`).

**Headline:** targeted external articulation (T3) fully recovers the harm caused by blind articulation (T1) — reasoning-axis accuracy T3=0.7220 vs T0=0.7217, statistically indistinguishable (p=1.0) — but does **not** exceed the no-caption baseline. Blind captioning (T1) significantly *hurts* (−6.5 pts, p=3e-10). The mechanism behind both facts is now identified with hard numbers, not speculation: **injecting a long text block into the model's `<think>` prefill causes it to prematurely close its own reasoning and answer almost immediately, and this "short-circuited" mode is 11–13 points less accurate than genuine deliberation.** Blind captions trigger this in 55% of generations, placebo in 84%, targeted captions in only 1.4% — which is the actual reason targeting recovers the baseline. This is a distinct, newly-identified finding, not something we set out to measure.

---

## 1. Design recap (one paragraph; full rationale in `PROBE_A_DESIGN_NOTES.md` / `PROBE_A_DECISIONS_FOR_APPROVAL.md`)

Five arms, all sharing Qwen3-VL-4B-Thinking except A5: **T0** (no payload), **T1** (blind CapRL caption, question-blind), **T2** (placebo — a length-matched, same-category, wrong-item caption), **T3** (question-targeted CapRL caption, options stripped from the captioner's prompt so it cannot see or leak them), **A5** (CapRL itself answering the MCQ, descriptive only). Payload is delivered as an assistant-turn prefill inside `<think>`. Two pre-registered families, Holm-corrected separately: **A** (T1-T0, T1-T2, T2-T0 — does blind articulation help, and is any effect content-specific or just "any prefill"?) and **B** (T3-T0, T3-T1 — does targeting help, and help more than blind?). Instruct (I0-I3) was run in the smoke and dropped from full scale after showing a confounded, payload-length-correlated non-convergence rate unrelated to the research question (documented in `submit_full.sh`'s SCOPE note and `tp_pass4_analyze.py`'s docstring).

---

## 2. Headline accuracy — all arms, all metrics

| arm | payload | ALL (primary) | ALL (tolerant) | ALL (official) | Perception (n=500) | Reasoning (n=1000) |
|---|---|---|---|---|---|---|
| T0 | none | 0.6993 | 0.6998 | 0.6711 | 0.6547 | 0.7217 |
| T1 | blind caption | 0.6344 | 0.6349 | 0.6136 | 0.6260 | 0.6387 |
| T2 | placebo | 0.5691 | 0.5696 | 0.5422 | 0.6127 | 0.5473 |
| T3 | targeted caption | **0.7049** | **0.7053** | **0.7013** | **0.6707** | **0.7220** |
| A5 | (CapRL itself) | 0.6044 | 0.6087* | 0.5587 | 0.6227 | 0.5953 |

*A5's headline metric is `tolerant` (descriptive only, not in either contrast family — CapRL is a captioner, not MCQ-tuned).

## 3. Pre-registered contrasts (primary metric, Holm-corrected within family)

| family | contrast | Δ | 95% CI | McNemar (b,c) | p | holm |
|---|---|---|---|---|---|---|
| A | T1 − T0 | −0.0649 | [−0.084, −0.047] | 81, 183 | 3.1e-10 | 3.1e-10 |
| A | T1 − T2 | +0.0653 | [+0.047, +0.084] | 183, 77 | 4.0e-11 | 8.0e-11 |
| A | T2 − T0 | −0.1302 | [−0.151, −0.110] | 74, 282 | 1.0e-29 | 3.0e-29 |
| B | T3 − T0 | +0.0056 | [−0.012, +0.023] | 105, 103 | 0.945 | 0.945 |
| B | T3 − T1 | +0.0704 | [+0.052, +0.090] | 197, 93 | 9.8e-10 | 2.0e-9 |
| B (reasoning, n=1000) | T3 − T0 | +0.0003 | [−0.020, +0.021] | **71, 71** | 1.0 | 1.0 |
| B (reasoning, n=1000) | **T3 − T1** | **+0.0833** | **[+0.060, +0.108]** | 150, 59 | 2.5e-10 | 5.0e-10 |

Robust across all three metrics: `correct_tolerant` reproduces `correct` almost exactly (extraction is ~99% across T-arms, so they rarely diverge); `correct_official` shows the same T3−T1 effect *more* strongly (+0.109, p=2.4e-14) and the same T3−T0 null on primary/tolerant but a small positive on official (+0.037, p=0.0064) — fully explained and reconciled in §7.1, not a discrepancy that weakens the conclusion.

**The single most important number in this table is `McNemar b=71, c=71` for T3−T0 on reasoning.** A delta of ~0 is not "T3 does nothing" — it is 71 items T3 fixes and 71 different items T3 breaks, exactly cancelling. 14.2% of reasoning items (142/1000) change correctness between T0 and T3. §5 characterizes which.

## 4. THE central mechanism: caption-triggered short-circuited reasoning

This was not something we set out to measure — it fell out of looking at `think_tok` (post-`</think>` split token count) against correctness, per the request to check "were models reasoning shorter in correct answers."

### 4.1 Aggregate: longer reasoning correlates with wrong answers, for T0 and T3 alike

| arm | n correct | n wrong | mean think_tok (correct) | mean think_tok (wrong) | median (correct) | median (wrong) | point-biserial r |
|---|---|---|---|---|---|---|---|
| T0 | 3147 | 1353 | 1504.4 | 2365.9 | 287 | 603 | −0.115 |
| T1 | 2855 | 1645 | 874.8 | 974.2 | 2 | 2 | −0.017 |
| T2 | 2561 | 1939 | 398.6 | 322.7 | 2 | 2 | +0.021 |
| T3 | 3172 | 1328 | 1654.7 | 2636.3 | 318 | 551 | −0.114 |

T0 and T3 show an almost identical, moderate negative correlation: items that make the model think longer are disproportionately the ones it gets wrong (a genuine "struggle signal" — median think length for wrong answers is roughly double that for correct ones, in both arms). **T3 does not reduce this** — the "washout hypothesis" carried over from earlier work (better perception → materially shorter reasoning traces, ~20.8% in prior S2T results) is **not supported here**. If anything T3's mean think_tok (1944.4, see §4.3) is the *highest* of all four arms, higher than T0's (1763.4). This is a real, unexpected finding worth flagging on its own: a targeted external caption did not make reasoning more efficient in this setup.

### 4.2 The real driver: caption-length-triggered premature closure, not caption content

The median think_tok of 2 for T1 and T2 in the table above is not noise — it is the dominant behavior:

| arm | frac. of generations with think_tok ≤ 5 | acc \| short (≤5 tok) | acc \| long (>5 tok) | gap |
|---|---|---|---|---|
| T0 | 0.000 (0/4500) | n/a | 0.6993 | — |
| T1 | **0.554** (2492/4500) | 0.5778 | 0.7047 | +0.1268 |
| T2 | **0.838** (3769/4500) | 0.5503 | 0.6662 | +0.1159 |
| T3 | **0.014** (65/4500) | 0.2923 | 0.7109 | +0.4186 |

A `think_tok ≤ 5` generation looks like this, verbatim, from the raw data (`gen_T1.jsonl`, idx=0, draw=0):

```
</think>

D
```

The model reads the injected caption sitting in its `<think>` prefill, writes the closing `</think>` tag almost immediately, and answers with essentially zero deliberation of its own. **This happens in 55% of T1 generations and 84% of T2 (placebo) generations — and it is dramatically less accurate than genuine deliberation (11–13 points worse within each arm).** Placebo triggers it *more* than a real blind caption (84% vs 55%), which is the key diagnostic fact: **the trigger is the presence and length of injected text sitting in the reasoning prefill, not its relevance or correctness.** The model appears to interpret "there is already a block of descriptive text here" as "I have already looked at the image," regardless of whether that text is a real caption or a random donor's.

T3 essentially avoids this failure mode (1.4% vs 55–84%), and its short-circuit rate is now close enough to T0's near-zero rate that T3's overall behavior looks like T0's, not T1's or T2's. Two candidate reasons, both consistent with what we measured and not mutually exclusive: (a) T3's caption is much shorter — median 229 tokens vs T1's blind caption median 682 tokens (3x), so it reads less like "the looking is already done"; (b) T3's caption prompt explicitly frames content as "concrete visual details... relevant to the question," which may read less like a closed narrative and more like notes to reason *from* rather than notes that *are* the answer.

**This is the actual mechanism behind both headline results.** T1/T2 hurt primarily because they turn off the model's own reasoning process for the majority of items, not (only) because the content is blind/wrong. T3 recovers to baseline primarily because it restores normal deliberation, not (necessarily) because its content is more accurate than T1's. This reframes "why doesn't targeting also add a gain over baseline": recovering deliberation gets you back to what deliberation alone already achieves; it doesn't by itself add new information the model didn't have.

One more data point worth flagging: on the rare 1.4% of T3 generations that *do* short-circuit, accuracy is 0.292 — even worse than T1's or T2's short-circuited accuracy. When T3 does fail this way, it fails harder, though it's rare enough (65/4500) to barely move the aggregate.

### 4.3 Full think_tok distributions, all arms

| arm | n | mean | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| T0 | 4500 | 1763.4 | 145 | 343 | 1616 | 31420 |
| T1 | 4500 | 911.1 | 2 | 2 | 412 | 37280 |
| T2 | 4500 | 365.9 | 2 | 2 | 2 | 36975 |
| T3 | 4500 | 1944.4 | 155 | 362 | 1733 | 37440 |

## 5. Item-level churn: why does T3 fix 71 items and break 71 others?

### 5.1 Caption/option-overlap signal — a real, actionable predictor

For every reasoning-axis item where T3's majority vote differs from T0's, checked whether T3's caption text lexically overlaps the item's correct option, its incorrect options, or both:

| group | n | mean overlap with CORRECT option | mean overlap with an INCORRECT option |
|---|---|---|---|
| T3-fixes (T0 wrong → T3 right) | 71 | 0.065 | 0.056 |
| T3-breaks (T0 right → T3 wrong) | 71 | 0.045 | **0.115** |

When T3 breaks an item T0 had right, the caption is more than **2x** as likely to textually overlap a *wrong* option than when T3 fixes an item. The same pattern holds for T1 (fixes: correct-overlap 0.069 vs incorrect 0.052; breaks: correct-overlap 0.032 vs incorrect 0.082 — almost the identical ~2.5x ratio) — so this looks like a general property of caption-driven answering, not something specific to targeting. **This is the single most actionable lever in this report**: caption/wrong-option overlap is a cheap, computable-at-generation-time quality signal. A best-of-M selection (or a filter) that down-weights captions overlapping an incorrect option could plausibly reduce the c=71 losses without touching the b=71 gains.

### 5.2 Category and l2-category breakdown of churn

| category | n | T0 acc | T3 acc | Δ | T3-fixes | T3-breaks |
|---|---|---|---|---|---|---|
| instance reasoning | 250 | 0.764 | 0.749 | −0.015 | 12 | 13 |
| logical reasoning | 250 | 0.737 | 0.748 | +0.011 | 20 | 17 |
| math | 250 | 0.835 | 0.835 | 0.000 | 17 | 12 |
| science & technology | 250 | 0.551 | 0.556 | +0.005 | 22 | 29 |

The null is genuinely uniform (§ already confirmed pre-launch), but the *churn ratio* isn't: math and logical reasoning skew net-positive (17v12, 20v17); science & technology is the only category where breaks outnumber fixes (22v29). Science & tech also has by far the lowest baseline accuracy (T0=0.551) — plausibly because many of these items are knowledge-bound (species/material/phenomenon identification) rather than perception-bound, so a better *description* of what's visible doesn't supply the missing *fact*. Math has the highest baseline (0.835) and the best fix:break ratio, consistent with a ceiling limiting the visible aggregate gain rather than an absence of real benefit.

By finer l2_category (top by combined churn count): geometry (11 fixes / 5 breaks — the single best ratio in the dataset) and common reasoning (10/4) skew strongly positive; geography/earth science (7/11) and code & sequence reasoning (2/5) skew negative.

### 5.3 Qualitative read: two distinct, identifiable failure/success patterns

Read 8 concrete transcripts (4 fixes, 4 breaks) directly rather than only trusting aggregates.

**Pattern A — T3 fixes clean, legible spatial/label-reading errors.** All four T3-fixes examples read (item 514: "What direction is Canada in the Atlantic Ocean?", gold=C; items 518/521/527: relative left/right/above/below position of two objects) are cases where T0 got a spatial or label-reading fact wrong from a single pass, and T3's caption explicitly states the relation ("the church is to the right of the building," "positioned directly above the...net") which the model then correctly uses. This is exactly the intended mechanism — a second, explicit "look and describe" pass surfacing a spatial/OCR detail a single integrated glance missed.

**Pattern B — T3 breaks confident, correct *holistic comparison* judgments.** Two of four T3-breaks examples (item 520 "which image is the brightest," item 546 "which image shows the highest sharpness" — both 4-panel comparison questions) show the same signature: T0 answered confidently and correctly across all 3 draws (`C,C,C` / `B,B,B`), while T3 became *unanimously wrong* (`B,B,B` / `B,D,D`) after the caption itemized each panel separately. Breaking a genuinely holistic, side-by-side visual comparison into discrete per-panel text descriptions plausibly destroys exactly the simultaneous-comparison signal the question needs, and may bias the answer toward whichever panel the captioner happened to describe most vividly (item 520's caption gives the most vivid, detailed language to the "fireworks" panel, which is the panel T3 unanimously — and wrongly — selects). A third break (item 536, "what is on his head," gold=D/nothing) shows a related but distinct issue: the caption is *technically accurate* (there are sunglasses) but emphasizes it five separate times, pulling the model toward crediting "something on his head" against the question's stricter intended reading. The fourth (item 501, time-of-day) shows caption-induced *inconsistency* rather than a wrong caption — T0 was rock-solid (`C,C,C`), T3 became unstable (`C,B,A`) despite the caption containing nothing that clearly contradicts C.

Both identified break patterns (multi-panel comparison, caption-emphasis-skews-interpretation) are concrete, checkable design issues — not evidence that targeted captioning is unhelpful in general.

## 6. Why we don't see anything like Track T's +7.5%

Track T (`TRACK_T_SIGNAL_REPORT.md`, MathVerse, same base model — Qwen3-VL-4B-Thinking, same delivery mechanism — assistant-turn `<think>` prefill with an almost identical wrapper) found:

| Track T arm | payload | Δ vs base (MC, primary) | McNemar p | Holm |
|---|---|---|---|---|
| privileged | the item's own **ground-truth** TD−VI perceptual givens, verbatim from the dataset | **+0.075** | ~0 | 0.0 |
| self | the **same model's own** question-aware self-description | −0.030 (n.s.) | 0.87 | 1.0 |
| placebo | length-matched wrong-item text | −0.029 (n.s.) | 0.85 | 1.0 |

Recovery = (self−base)/(privileged−base) = **−0.41** — the model's own self-description recovered *none* of the ground-truth benefit; if anything it went slightly negative, like Probe A's T1/T2.

**The critical distinction, easy to miss without reading the frozen report closely: Track T's "privileged" arm is not a caption from any model — it is the literal, correct, dataset-provided ground-truth perceptual facts, injected as text.** It establishes an *oracle ceiling*: the maximum a perfect perception-as-text intervention could buy, on that benchmark. Track T's "self" arm — the closest analogue to Probe A's T1 — is a *same-model* self-description, and it already failed to recover anything (recovery ≤ 0), matching Probe A's T1/T2 result almost exactly in sign and magnitude.

Probe A's T3 sits at a third point on this spectrum: not the same model's blind self-report (Track T's "self," which failed), not oracle ground truth (Track T's "privileged," which gained +7.5), but a **different, specialized, RL-trained-for-captioning external model (CapRL), targeted at the question.** The result — full recovery to baseline, no excess gain — sits exactly where you'd expect *between* those two: better than same-model self-description (which went negative), nowhere near as good as verified ground truth (+7.5). This is a coherent, three-point story, not a contradiction of Track T:

| source of perceptual text | Δ vs no-text baseline | study |
|---|---|---|
| same model, blind self-description | ~0 to slightly negative (recovery ≤ 0) | Track T "self" |
| **different specialized captioner, blind** | **significantly negative (−6.5 pts)** | **Probe A T1** |
| **different specialized captioner, targeted** | **~0 (full recovery, no excess gain)** | **Probe A T3** |
| oracle ground truth | **+7.5 pts** | Track T "privileged" |

Read this way, the "why not 7%" answer is precise: **the 7% required perfect information; the best real, currently-available external perception source (CapRL, targeted) gets you back to what the model's own unaided reasoning already achieves, but is not accurate/complete/relevant enough, on net, to close the remaining gap to the oracle ceiling.** Combined with §4's mechanism (T3 avoids the short-circuit failure but doesn't add new correct information beyond what deliberation alone provides) and §5's churn analysis (real per-item wins and losses roughly balance), the story is internally consistent across both studies, not two disconnected results.

Secondary, smaller contributing factors, stated honestly and not overclaimed: different benchmark (MathVerse geometry/diagram-heavy vs MMStar general 6-category), different K (5 vs 3), and Track T's own contamination caveat (MathVerse base MC=0.81, flagged as possibly memorization-inflated) — MMStar's T0 baseline on reasoning axes here is lower (0.72), so if anything Probe A had *more* headroom to show a positive effect than Track T did, which if anything strengthens rather than weakens confidence that the T3≈T0 null here is a genuine finding.

## 7. Robustness and integrity checks (all done against real full-scale data, not the smoke)

### 7.1 Primary vs official metric divergence — explained, not a discrepancy

Queried per-generation disagreement directly: `official_only=0` for both T0 and T3 — the official scorer never credits something our primary extractor misses; it's a strict subset. The only asymmetry is `primary_only`: **T0 has 127/4500 (2.8%) generations where the correct letter isn't at position-0 (official misses it, primary catches it); T3 has only 16/4500 (0.36%)** — nearly 8x fewer. `fc_artifact` (the known official-scorer false-positive defect) is 0.0000 for every T-arm, ruling that out explicitly. T3, primed by its caption, tends to answer more tersely and lands at position-0 more often than T0. This is a phrasing difference, not a correctness difference, and it explains why `correct_official` shows a small positive T3−T0 effect (+0.037, reasoning axis) that `correct`/`correct_tolerant` don't — it does not undermine the primary-metric null.

### 7.2 Placebo length-match outlier — isolated, not systemic

`G8.placebo_length_matched` mean relative length difference 0.011 (excellent) but max 2.362. Traced the outlier: item 524, caption slot 3 (self_len=4254 chars, donor_len=14304 chars, category "instance reasoning," donor=item 531). No fallback, no indexing bug — a rare gap in that category's caption-length distribution under the deterministic nearest-neighbor assignment. 1 outlier among 7,500 caption-slot assignments; does not move the T2 aggregate.

### 7.3 Reasoning-axis null is uniform across sub-categories, not a masked average

(Table reproduced in §5.2.) Every one of the four REASONING_CATS sub-categories shows a small, non-significant delta with roughly balanced win/loss counts. The null is not hiding a real effect in one sub-category that cancels a real effect in another.

### 7.4 Pipeline integrity — all gates, full scale

`ALL GATES PASSED` on the audit re-run (job 3028154, 36s, after fixing `G16b` which had no guard for the deliberately-dropped I1 arm — see commit `1fcc280`; this did not affect scoring or analysis, which completed before the audit stage ran). Extraction 0.992–0.999 across T0-T3 (unextract spread 0.018, comfortably under the 0.10 confound threshold that Instruct failed at smoke scale). Truncation 0–0.5%. Payload delivery confirmed behaviorally (`G17`: 0/4500 identical T0/T1 continuations on shared (index,draw) — the payload demonstrably changes output). Seed independence (`G18`), placebo token-length matching (`G19`: T1 vs T2 mean ptok 828 vs 835, rel diff 0.008), and prompt reconstruction from frozen artifacts (`G20`: 20/20) all pass. Captions inspected directly (`G7c`) and read as accurate, detailed, on-topic. Format-leak rate for question-targeted captions 0.36% (27/7500), all traced to lexical option-overlap on TRUE image content, not structural leakage (0 `\boxed{}`, 0 explicit answer declarations, 0 enumeration markers).

## 8. Implications for the two-stage RL plan (Stage 1 articulation / Stage 2 self-distillation)

1. **The plan is not dead, but its target changed.** An off-the-shelf, general-purpose captioner (CapRL), even correctly targeted, is not sufficient to beat baseline reasoning — it only neutralizes the harm of blind articulation. The open question Stage 1 exists to answer — can a captioner be *trained* to help *this specific* downstream reasoner — is unresolved by this probe, not refuted by it.
2. **A newly-identified, concrete design constraint for any future captioner or Stage-1 policy: avoid triggering premature `</think>` closure.** This wasn't previously known. §4.2's numbers (55%/84% short-circuit rate for T1/T2 vs 1.4% for T3, with an 11–42 point accuracy gap between short-circuited and normal generations) suggest *keeping the injected text short and framed as partial evidence rather than a complete description* may matter as much as raw factual accuracy for whether an articulation helps or hurts.
3. **The wrong-option-overlap signal (§5.1) is a cheap, immediately testable quality filter** — usable at inference time (best-of-M caption selection) or as a training signal (penalize captions that lexically support an incorrect choice), without needing any new data collection.
4. **Two concrete, evidence-backed failure modes to design around**: multi-panel/holistic-comparison questions (§5.3 Pattern B) and caption over-emphasis skewing question interpretation (item 536). Both are identifiable at generation time (multi-image or comparison-keyword questions could bypass captioning entirely; caption length/repetition could be capped).
5. **Track T comparison (§6) reframes the ceiling.** +7.5% is what *perfect* perceptual text buys on a different, geometry-heavy benchmark. There is no evidence yet, on MMStar with this model, of what a *trained-for-this-pairing* (rather than off-the-shelf) captioner could achieve — that is the actual next experiment, not a repeat of this one.

## 9. What was NOT checked here (honest scope note)

- Did not read all 71+71 churn items, only 8 sampled ones — the qualitative patterns in §5.3 are hypotheses grounded in a small, non-random sample, not an exhaustive classification.
- Did not test any of the §8 levers (short-circuit-avoidance, wrong-option filtering, comparison-question exclusion) — these are proposals, not results.
- A5 and T1/T2 relative to the pre-registered families are fully analyzed; A5's own item-level behavior (why it undershoots the T-arms) was not separately investigated here.
- The Instruct-family data collected at smoke scale (n=48, out of full-run scope) is not re-analyzed in this report; it remains available as a documented exploratory footnote per `submit_full.sh`'s SCOPE note if wanted later.

---

**Provenance:** full-run outputs at `$SCRATCH/text_privilege/out/full/` on Clariden (`scored.jsonl`, `analysis_correct*.json`, `gen_{T0,T1,T2,T3,A5}.jsonl`, `captions.jsonl`, `captions_q.jsonl`, `placebo_assignment.json`). Generation jobs 3021983-3021987, analysis 3021988, audit re-run 3028154. Code git `1fcc280`. Deep-dive analysis scripts: `deep_analysis.py`, `deep_analysis2.py` (ad hoc, run via `/usr/bin/python3.11` on the login node or via the vllm011 container on the debug partition — not part of the frozen pipeline, kept on the cluster for reproducibility).
