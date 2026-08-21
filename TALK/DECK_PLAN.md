# Programme Talk — Deck Plan

> **Companions:** `CORRECTIONS.md` (full-repo sweep, 2026-08-19 — one retracted claim, one figure that
> must not be drawn as planned, several numbers corrected) · `BIBLIOGRAPHY.md` (12 verified, 15 pending;
> nothing `[PENDING]` may go on a slide).

**Status: WORKING PLAN, not frozen.** Numbers marked ⚠️ need re-verification against their frozen
source before they go on a slide. Figures marked **[HAVE]** exist on disk at the given path; **[MAKE]**
do not exist yet.

**Attribution convention** (inherited from `RL_SeeingToThinking/runs/analysis/SLIDES.md`, keep it —
a knowledgeable audience will ask "whose result is this?"):
`[PAPER]` = published prior work · `[SENIOR]` = supervisor's prior analysis · `[OURS]` = first-party ·
`[3P]` = third-party report (GitHub issue / independent repro).

**Spine in one sentence:** the field explains VLM perception failure by attention decay or language
priors; we falsify the attention account, mechanistically localize the deficit to LLM **read-out** of
already-adequate visual tokens, show every image-side delivery mechanism fails where encoding was not
the limit, and propose the missing objective — text sufficiency — as the fix.

---

# ACT 1 — Motivation (3 slides)

### 1.1 Title
Perception, not reasoning: where VLM visual failure actually lives — and what to do about it.

### 1.2 The problem is perception, and thinking does not fix it
- `[PAPER]` UCSC-VLAA *From Seeing to Thinking* (arXiv 2605.20177): **86.9%** of VLM errors are
  perception errors more reasoning cannot fix. This is the origin of the whole programme.
- **[MAKE]** simple stat callout slide.

### 1.3 The field's three explanations — and this talk's claim
- **(a) Attention decay.** `[PAPER]` *More Thinking, Less Seeing?* (NeurIPS 2025, 2505.21523):
  longer chains → attention to visual tokens falls, attention to instruction tokens rises. RH-AUC/RH-Bench.
- **(b) Language priors.** `[PAPER]` Visual-Counterfact "Vision-Default, Prior-Override" (2606.28273);
  CounterCount (2605.17826). ⚠️ both `[PENDING]` verification.
- **(c) ViT/LLM misalignment.** The vision tower and the language model are trained largely separately and
  never jointly aligned, so the LLM cannot read what the encoder encoded. Two *opposing* readings exist
  inside this account, and the distinction is the one this talk resolves:
  - **encoder-side:** the information is genuinely absent from the visual tokens — `[PAPER]` *Eyes Wide
    Shut?* (CVPR 2024), CLIP-blind pairs, fixed by adding DINOv2 features (MoF).
  - **read-out-side:** the information is present but the LLM fails to extract it — `[PAPER]`
    *Diagnosing Bottlenecks in Data Visualization Understanding* (2510.21740), linear probes on frozen
    encoders show **extraction failure dominates**.
  > ⚠️ **Honesty constraint:** we have **no citation for the training-procedure claim itself** (that the
  > towers are optimized separately and never aligned). Present (c) as a live hypothesis in the field with
  > the evidence that bears on it — do **not** assert it as established. See `BIBLIOGRAPHY.md` §D.2.
- **Our claim:** (a) is correlational, not causal; and within (c) we come down causally on the
  **read-out** side — with the vision tower bit-frozen.
- **[MAKE]** three-box "competing explanations" diagram → our verdict column.

---

# ACT 2 — Problem exploration: we replicate the phenomenon (3 slides)

### 2.1 Attention really does decay
- `[OURS]` image_toolCalling: visual attention falls **−38%** over the chain; correlation with chain
  length r ≈ −0.5 / −0.6. ⚠️ re-verify exact figures from `image_toolCalling/`.
- **[HAVE]** `image_toolCalling/plots_attention/A_visual_decay_over_position.png`
- **[HAVE]** `image_toolCalling/plots_attention/C_visual_attention_vs_length.png` ("Think longer, see less")

### 2.2 And it correlates with being wrong
- `[OURS]` babyVision: correct answers hold ~2× the visual attention of wrong ones.
- **[HAVE]** `babyVision/plots/attn_correct_wrong.png`, `babyVision/plots/attention_decay.png`
- **[HAVE]** `image_toolCalling/plots_attention/B_visual_thirds_correct_wrong.png`

### 2.3 Accuracy collapses with reasoning length ✅ verified
- `[OURS]` Gemma-4-31B-it, HallusionBench stratified 30% subset (~273 items), **edited (vi=2)** images,
  by reasoning-length quartile: **94.4 → 75.0 → 69.4 → 52.8%**. Binary questions ⇒ Q4 ≈ chance.
- Also: standard accuracy 83.3%; prior reliance vi=1 87.1% vs vi=2 79.9%; **voluntary re-examination rate
  only 2.4%** — the model almost never chooses to look again.
- `[OURS]` replicated in a third setting: Probe A (Qwen3-VL-4B-Thinking, MMStar) — point-biserial
  r = **−0.115** (T0) / −0.114 (T3); median think_tok 603 when wrong vs 287 when correct.
- **[MAKE]** quartile bar chart (source: `image_toolCalling/SUMMARY.md`).

### 2.4 We also ran the field's own benchmark ✅ *(newly surfaced — was missing)*
- `[OURS]` **RH-Bench** (the benchmark from 2505.21523), Qwen3-VL-4B-Thinking, 900 items:
  **reasoning 69.3% vs perception 64.4%** — a 4.9pp gap in the paper's predicted direction.
- **Declare the deviations:** judge = Qwen3-32B not GPT-4o; 900 vs 1000 items; **not comparable to
  published scores**; 57/900 (6.3%) produced no answer (thinking exhausted the 16384 budget), scored
  incorrect. Single budget ⇒ **RH-AUC cannot be reported.**
- Use as "we engaged the field's instrument," not as a headline number.
- Source: `RH-Bench/Qwen3-VL-4B-Thinking_Results.md`.

> **Transition line:** "Everything so far replicates the field's story. Now we test whether it is causal."

---

# ACT 3 — Falsification: attention is not the bottleneck (4 slides) ★ first real result

### 3.1 We restored the attention. The answers did not move.
- `[OURS]` forced re-injection: attention to the image re-engages **+91%**; **96% of answers unchanged**.
- **[HAVE]** `image_toolCalling/plots_forced/F1_attention_reengage.png` (the intervention works)
- **[HAVE]** `image_toolCalling/plots_forced/F3_answer_stickiness.png` (…and nothing happens)
- *These two side by side are the strongest single slide in the deck.*

### 3.2 Where it does help, it is perception-limited items only
- **[HAVE]** `image_toolCalling/plots_forced/F2_gain_by_subcategory.png`
  ("perception-limited tasks gain; prior-dominated / saturated don't")
- This is the first appearance of the **information-gain condition** — plant it here, pay it off in Act 5.

### 3.3 Isolating the two things re-examination does ✅ CORRECTED — much stronger than I had
- `[OURS]` babyVision (Gemma-4-31B-it, 388 items), B1′ vs standard-majority-of-3:

  | Task family | standard | B1′ | Δ | p |
  |---|---|---|---|---|
  | perception (counting, search, tracing) | 15.7% | **22.5%** | **+6.8** | **0.015** |
  | reasoning (rotation, folding, overlay) | 39.6% | 33.0% | −6.6 | 0.12 (trend) |

- ★ **The load-bearing control:** B2′ — identical, but the image is **not** re-shown — gives the same
  split (perception **+5.2**, reasoning **−7.6**). B1′ vs B2′ directly: **only 9 of 388 items change.**
  > **"The driver is the reasoning, not the image."**
  This is first-party support for the text channel that does **not** depend on the retracted D6 (see 3.4).
- Say the limitation: only the perception gain is significant; the reasoning drop is a consistent
  direction, not a proven effect. B1′ also changed turn-2 wording, so it is not a perfectly clean
  one-variable swap.
- **[HAVE]** `babyVision/plots/dissociation.png`

### 3.4 There is no stored percept to decay ⚠️ ONE CLAIM RETRACTED — read `CORRECTIONS.md` C1
- `[OURS]` Set 2 (CLEVR, Qwen3-VL-Thinking): linear probe over 15 attribute-marginals — **image-token
  positions bal_acc 0.918** (the probe works) vs **text/reasoning positions 0.504 / 0.511 / 0.517 =
  chance**, across all PCA-k, all 6 layers, all positions (n=152).
- Apparent "drift" did not replicate: RIPE flips 2/39 while **controls flip more (4/39)**; `count0` items
  carry every flip. **`flipAns` was a construction confound** — RIPE `wrong_target = model_ans` forces the
  sign, so controls *cannot* score it. Excellent cautionary material; show it.
- Verdict: **the answer is determined at emission, not drifted.**
- ★ **The control that legitimizes every pixel null in this talk:** a **different** image injected at
  `think` **moves answers**; the **same** image is **attended but inert**. So "re-injecting the same image
  doesn't help" is a real result, not a broken instrument. Put this on the slide.
- **[MAKE]** probe-vs-control bar with chance line.
- ⛔ **DO NOT present "text potent, pixels inert."** Set 3 **overturned D6**: on the robust pool,
  V_text (+0.067) and V1 (+0.060) have similar small effects — **no text ≫ pixel asymmetry**; the Set-2
  effect is attributed to flaky-item contamination (~58% of "errors" removed by the robustness filter).
  Correct statement: *the asymmetry does not replicate* — not "pixels are potent."
- ⛔ Consequently do **not** cite 2510.23482 as replicating that dissociation. Cite it in Act 6.2 only.

> **Verdict slide:** attention decay is real, correlational, and **not** the causal bottleneck.
> `[PAPER]` corroboration: *MLLMs Know Where to Look* (ICLR 2025, 2502.17422) — attention ratio > 1
> in most layers **even when the answer is wrong**. The model looks in the right place and still fails.

---

# ACT 4 — Where perception actually lives (6 slides) ★ the crown jewel

> Source deck: `RL_SeeingToThinking/runs/analysis/SLIDES.md` (16 slides, already built, numbers verified
> against `FINDINGS.md`). **Compress to 6.** Keep the `[SENIOR]` S1–S5 framing — testing and *correcting*
> a supervisor's hypotheses is a credibility asset, not a liability.

### 4.1 Setup and headline
- Qwen3-VL-4B-Instruct; Stage-1 RLVR (GRPO) on 3360 DOCCI perception MCQs.
- **Perception accuracy 0.365 → 0.746** (16 epochs / 96 steps).
- **[HAVE]** `RL_SeeingToThinking/runs/analysis/figures/fig_training.png`

### 4.2 The fix is LLM-internal — and we proved the freeze
- `[OURS]` `llm_only 0.749 ≈ full 0.746 ≫ vit_only 0.443 > base 0.365` (reward);
  DOCCI probe `0.593 / 0.657 / 0.423 / 0.377`.
- **Freeze proof:** `vision rel_fro = 0.000e+00` across **315 tensors**; `llm rel_fro = 0.000` across 397.
- **The line to say out loud:** *perception nearly doubled without altering one vision-encoder weight.*
- **[HAVE]** `RL_SeeingToThinking/runs/analysis/figures/fig_ablation.png`

### 4.3 It is a tiny edit
- `[OURS]` mean relative Frobenius change ≈ **5e-4 (0.05%)**; MLP/attn ratio 1.59 / 1.40 / 1.37
  (early / mid / late) — MLP-biased but **not** late-concentrated. `[SENIOR]` **S2 not confirmed.**
- **[MAKE]** rel_fro-by-layer plot with MLP/attn split (data exists in `analysis/`; script `weight_delta.py`).

### 4.4 Functionally late…
- `[OURS]` depth probe: base and trained are identical layers 0–23, diverge sharply at **L24–25**.
- **[HAVE]** `RL_SeeingToThinking/runs/analysis/figures/fig_depth.png`
- Note honestly: `[SENIOR]` **S4** ("computed mid-stack then lost") is **not** supported — the L19–23
  dip is a logit-lens artifact present in both models.

### 4.5 …but causally distributed
- `[OURS]` module graft: mlp **0.553 (63%)**, attn 0.460 (30%), early_mlp 0.430 (19%),
  **late_mlp 0.387 (3.6%)**. A stated prediction, corrected by the data.
- **[HAVE]** `RL_SeeingToThinking/runs/analysis/figures/fig_graft.png`

### 4.6 Capstone: the answer is written into the residual stream
- `[OURS]` activation patch at **L24 recovers 82%** (from 11% at L20); a **fixed steering vector tops
  out ~40%** → the representation is **portable but input-specific**.
- **[HAVE]** `RL_SeeingToThinking/runs/analysis/figures/fig_patch.png` *(verified — publication quality)*
- **Design constraint this buys us:** a canned fix cannot work. Perception must be **re-derived per item**.

> **Act verdict:** the visual tokens already carry the information. The deficit is the LLM's **read-out**.
> `[PAPER]` convergent: *Diagnosing Bottlenecks in Data Visualization Understanding* (2510.21740) —
> linear probes on frozen encoders show **extraction failure dominates**. Ours is the stronger form: a
> probe shows decodability, our ablation shows the model can be *made to use it* with the encoder frozen.
> `[PAPER]` competing hypothesis to name and rule out: *Eyes Wide Shut* (CVPR 2024) argues the **encoder**
> is deficient (CLIP-blind pairs, MoF fix). Our freeze ablation is the evidence against it on our data.

---

# ACT 5 — Methodology exploration: every delivery mechanism (7 slides)

> **One repeated schema per slide: what we thought → why → what happened → what it taught.**
> This is the merged old-§5+§6. Do not split diagnosis from external methods; they answer the same question.

### 5.1 The schema slide (show the whole table, then walk it)
| Method | Why it should work | Result | Lesson |
|---|---|---|---|
| Re-inject the image (Set 3 V1) | restore decayed attention | **null** | info, not attention, is the currency |
| Reward pixel-sensitivity (**PAPO**) | force reliance on the image | **KL ↑↑, acc flat** | rewards a *correlate*; repulsive KL has no optimum |
| Tool-call re-perception (**DeepEyes**) | let the model re-look where it chooses | +1.6 / −0.3 / **+11.8** | works only where re-looking adds resolvable detail |
| Oracle facts as text (Track T) | is the text channel live? | **+0.075** | **yes** |
| Its own description (Track T) | can it drive that channel? | **−0.41 recovery** | **no** |
| External captioner (text_privilege) | buy articulation off the shelf | +0.006, p=0.945 | not purchasable |
- **[MAKE]** this table as a build-animated slide; it is the act's backbone.

### 5.2 Re-injecting the identical image — the informationally-vacuous null
- `[OURS]` Set 3 V1: re-inject the *same* image mid-chain → null. Predicted by any theory; include it as
  the control that calibrates the others, not as a finding.
- `[OURS]` Set 3 confirmatory **H1 FAIL** (Pool-S, f0.25, n=149): V0 0.060 → V_self 0.067,
  **Δ = +0.0067**, 95%CI (−0.047, +0.060), McNemar p=1.0 (9 fixed / 8 broke). Concluded-only (n=120):
  Δ=+0.025, p=0.61. **The null is robust to truncation.** Audit verdict: **NO FAULT FOUND**;
  injected payload was GT-perfect (multiset match 149/149).
- Exploratory, multiplicity-uncorrected: V1 +0.060 (p=0.035) · V_text +0.067 (p=0.052) ·
  V_scaffold **+0.087** (p=0.007) · V_restart +0.087 (p=0.015) · **V_self_pre +0.114** (p=0.006);
  placebos flat (V_scr +0.013, V_text_wrong +0.007). Restart decomposition localizes the gain to the
  **re-presented image** (+0.074) not the instruction (+0.013).
- ★ **Across the 50-test block NOTHING survives Holm**: V_self_pre 0.006→**0.299**,
  V_scaffold 0.007→**0.353**, V_restart 0.015→**0.702**, V1 0.035→**1.000**. Say this out loud.
- ★ **Why the text arm failed — the caveat that bounds our own text nulls:** both payloads enumerated
  **objects but not spatial relations**; CLEVR is relational, so objects-as-text cannot restore the layout
  the reasoning needs. **We tested impoverished text, not text per se.** This matters directly for Act 7:
  a free-form caption *can* carry relations. Do not let the audience read Set 3 as "text doesn't work."
- Continuation reads: **7/10 V_self-null items engage the injected list and are still wrong** → real null,
  not a delivery failure.
- **[MAKE]** forest plot: effect sizes + CIs, raw p vs Holm-adjusted p side by side.

### 5.3 PAPO — and the reframing that explains it ★ new, and the audience will like it
- **PAPO's perception term is per-token on-policy distillation against a frozen, deliberately-blinded
  copy of itself, with the divergence maximized.** Verified in code:
  - `dp_actor.py:184-190` — masked branch scores **the same rollout tokens**; only `multi_modal_inputs` swaps.
  - `papo_utils.py:17-30` — `random_patch_blackening(patch_size=14, black_prob=0.6)`.
  - `dp_actor.py:412` — `pg_loss = pg_loss - kl_prcp_loss * kl_prcp_coef` → **maximized**.
  - `dp_actor.py:329-335` — with the authors' default `RECOMPUTE=False`, the masked branch is **detached**
    = literally a frozen teacher. This is the paper-faithful C-pure config.
- **Why it must fail:** an *attractive* KL to a good teacher has a unique optimum and a ceiling. A
  *repulsive* KL to a bad teacher has **neither** — it names a direction to flee, not a destination, and
  is satisfied by any change that increases discrepancy, including changes orthogonal to accuracy.
- **The authors concede it:** the Double Entropy Loss exists to "regularize the new KL objective" —
  i.e. to stop degenerate maximization.
- `[OURS]` both horns measured: **C-pure** (unregularized) → perception-KL rises, val flat vs GRPO;
  **Arm B** (regularized) → **0.466 vs Arm A 0.540**.
- `[PAPER]` lineage worth naming: this is **Visual Contrastive Decoding** (CVPR 2024) moved from decoding
  into the loss, with a distillation term's sign flipped.
- **[MAKE]** ★ **the money figure**: two panels sharing an x-axis (training step) — **C-pure's own**
  perception-KL rising while **C-pure's own** validation accuracy stays flat.
- ⛔ **Draw it as a WITHIN-ARM dissociation only.** `PAPO_probe/PROBE_DESIGN.md` §1 states the wandb
  `actor/kl_prcp_loss` is **contaminated for cross-arm comparison**: on-policy on a moving distribution;
  fresh unseeded mask each step; **Arm A's value is a bug artifact** (the term was never trained); and it
  is measured on the **training set**, not held out. Plotting A/B/C KL curves together would be an error
  a knowledgeable audience can catch. The fair cross-arm number needs the **offline perception-KL probe —
  designed, NOT built.**
- ⚠️ **BLOCKING:** C-pure numbers are checkpoint-verified + user-reported, **not yet extracted from wandb**
  (`DESIGN_NOTES.md` §10). This is the only first-party *causal* measurement in the thesis. Extract first.
- Caveats to state: 60 steps vs paper's ~200; 2B; A-vs-B differs by two variables.

### 5.4 DeepEyes — form without function, except where it isn't
- `[OURS]` reproduced **88.48** on V\* (projected; inside community envelope 87.43–91.10 vs paper 90.1).
- `[PAPER]` their own Table 9 ablation: interleaving is worth **+1.6 V\***, **−0.3 HR-4K**, **+11.8 HR-8K**.
- `[OURS]` verified V\* images are ≤8.29M px vs the 12.85M limit → **no downscaling** → a crop adds
  magnification, not pixels. That alone explains +1.6 vs +11.8.
- `[3P]` issue #60: withholding the crops entirely → **no drop**.
- **[MAKE]** three-bar chart (+1.6 / −0.3 / +11.8) annotated with "does re-looking add resolvable detail?"
- **This is the slide that proves the information-gain condition**, and it is what makes the thesis
  survive the obvious objection. Do not bury it.

### 5.5 DeepEyes — evaluation validity (short, but it earns trust)
- `[OURS]` correct answer is **always option A** (`judge_result.py:140`); `[3P]` issue #66: shuffling
  costs **~5 pts**; baseline irreproducible across three attempts: **63.35 / 76.44 / 79.06** vs claimed 71.2.
- `[OURS]` clean demo, item `sa_10204`, same model same image: tool prompt → *"there is no van visible"* →
  **wrong**; plain-MC prompt → `"A."` → **correct**. Identical failed perception, opposite score.
- **[MAKE]** the `sa_10204` side-by-side. Qualitative, memorable, damning.

### 5.6 "Fake thinking" — the decoupling, shown not asserted
- `[3P]` issue #80: model zooms in, verbalizes **"no helmets visible,"** answers **"red."**
- Author's own diagnosis: *"the model actually 'knows' its answer in the first place, but it pretends to think."*
- **[HAVE]** candidate qualitative material: `image_toolCalling/examples_illustration/Q4_original.png`
  + `Q4_edited.png`, `Q3_math_original.png` + `Q3_math_edited.png`.
- ⚠️ **Cheapest high-value experiment we have not run:** on the **191 saved DeepEyes trajectories +
  355 self-chosen bboxes**, measure how often the pre-tool-call `<think>` already states the final
  answer. **Zero new inference.** This would convert a third-party anecdote into a first-party rate.
  *Strongly recommend running before the talk.*

### 5.7 The text channel: live, but the model cannot drive it ✅ verified, and richer
- `[OURS]` Track T (MathVerse, MC primary n=373, Qwen3-VL-4B-Thinking, avg@5, K=5, audit **13/13 PASS**):

  | contrast | Δ | 95% CI | McNemar (b,c) | Holm |
  |---|---|---|---|---|
  | privileged − base | **+0.075** | [+0.046, +0.105] | 42, 3 | **0.0** |
  | **privileged − placebo** | **+0.105** | [+0.075, +0.134] | 42, 5 | **0.0** |
  | self − base | −0.030 | [−0.056, −0.004] | 17, 19 | 1.0 |
  | placebo − base | −0.029 | [−0.054, −0.005] | 15, 13 | 1.0 |

  **Recovery (self−base)/(privileged−base) = −0.41** [−0.91, −0.11].
- ★ **The placebo arm is the point** — I had omitted it. privileged ≫ placebo ≈ base proves the benefit is
  specific to **correct content**, not to "any text in the prefill." Without it the +0.075 is unconvincing.
- Robustness: free-form (no 25% guessing floor) privileged **+0.106** [+0.042, +0.171], p=0.019.
- Declare: MathVerse (2024) contamination; base MC 0.81 may be inflated. The finding rests on *gaps*,
  which contamination affects symmetrically.
- ★ **Rigor story worth 30 seconds:** the self arm originally reused one description across all K draws.
  Corrected to K independent descriptions before the confirmatory run. **The buggy version would have
  reported spurious positive recovery.**

### 5.8 Buying the articulation externally also fails — and reveals a mechanism ✅ verified
- `[OURS]` Probe A (MMStar, n=1500, K=3, CapRL captioner): T0 0.6993 · T1 0.6344 · T2 0.5691 · **T3 0.7049**
- **T3−T0 = +0.0056** [−0.012, +0.023], **p=0.945** · T3−T1 +0.0704 (p=9.8e-10) · T1−T0 −0.0649 ·
  T2−T0 −0.1302 (p=1.0e-29)
- ★ **Reasoning axis, T3−T0: McNemar b=71, c=71, p=1.0.** *"A delta of ~0 is not 'T3 does nothing' — it is
  71 items T3 fixes and 71 different items T3 breaks, exactly cancelling."* 14.2% of items change.
  **[MAKE]** the b=71/c=71 churn visual — memorable, and it reframes what a null means.
- ★ **Premature closure, and the diagnostic that identifies the trigger:** fraction of generations with
  think_tok ≤ 5 — T0 **0.000** · T1 **0.554** · T2 **0.838** · T3 **0.014**. **Placebo triggers it MORE than
  a real caption**, so the trigger is the *presence and length* of injected text, not its content.
  T3's caption median 229 tok vs T1's 682 (3×). Short-circuited generations are 11–13 pts worse.
- Unexpected and worth flagging: the **washout hypothesis is not supported** — T3's mean think_tok
  (1944.4) is the *highest* of all arms, above T0 (1763.4).
- Actionable: when T3 *breaks* an item, its caption overlaps a **wrong** option 2× more often
  (0.115 vs 0.056) — a cheap generation-time quality signal for best-of-M selection.
- **[MAKE]** ★ **the pivot figure**: base / +oracle text / +own text / +external caption, four bars with
  recovery annotated. The single most important figure for the proposal.

---

# ACT 6 — The gap, stated precisely (2 slides)

### 6.1 The gap — stated at exactly the strength the evidence supports
> **The deficit is read-out, not encoding.** The vision tower, bit-frozen, already carries what is needed
> (Act 4.2). **Serialized text is a demonstrated delivery channel into that read-out** — correct
> perceptual facts as text give +0.075, and the placebo arm shows the benefit is specific to correct
> *content* (+0.105 over placebo). **But the model cannot produce that text itself** (recovery −0.41),
> and it cannot be bought externally (T3−T0 = +0.006, p=0.945).

- ⛔ **We do NOT claim text beats pixels.** Set 3 found similar small effects for both arms and **neither
  survives Holm**; D6 is overturned (`CORRECTIONS.md` C1). Anyone claiming a text≫pixel asymmetry from our
  data would be citing a result we withdrew.
- ⛔ Equally, do **not** claim "image methods don't work." *MLLMs Know Where to Look* (cropping works) and
  DeepEyes HR-8K **+11.8** falsify it, and someone will have both ready.
- ✅ **What we DO claim, and can defend:** image-side interventions act on the **encoding** stage and pay
  only when encoding was genuinely information-limited (small objects, downscaled megapixels) — this is
  the *information-gain condition*, and Act 5.4's +1.6 / −0.3 / +11.8 pattern is its direct evidence.
  Where encoding was not the limit, they have nothing to act on. The read-out stage is where the residual
  deficit lives, and self-serialization is the specific missing capability.
- **Bound our own text nulls honestly:** Set 3's text payloads carried objects but **no spatial relations**
  on a relational benchmark. We tested impoverished text, not text per se (Act 5.2).
- **[MAKE]** ★ **the unifying pipeline diagram**: image → ViT → tokens → LLM read-out → answer, with each
  method drawn at the stage it acts on, colour-coded by result. *This is the diagram the talk is built around.*

### 6.2 Why can't it articulate — even where it answers correctly? ★ anticipate the first question
The premise assumes right answer ⟹ fact used ⟹ fact stateable. The second arrow is measurably false:
- `[OURS]` Track T self arm: recovery −0.41 on a pool it often answers correctly.
- `[3P]` DeepEyes #80: verbalizes one thing, answers another.
- `[PAPER]` 2510.23482: models "incorporate visual information inaccurately, **yet still produce correct answers**."
- `[PAPER]` 2505.05410: reasoning models verbalize hints they demonstrably used only **~25%** (Claude 3.7
  Sonnet) / **~39%** (R1) of the time; outcome-based RL improves faithfulness then **plateaus**.

**The mechanism — and this is the load-bearing argument:**
- Captioning/instruction tuning optimizes *plausible generic description*.
- RLVR — ours, PAPO's, DeepEyes' (`R_acc + R_format + 𝟙[R_acc>0]·R_tool`) — scores the **final answer**.
- **No objective anywhere in the standard VLM pipeline has "text sufficient to answer the question" as
  its optimum.** Articulation was never optimized. That is why it fails on items answered correctly.
- Consistent with `[OURS]` Part 11: the answer-determining content is an input-specific distributed code
  (fixed vector ~40%), not a proposition in a slot. Articulating is **re-encoding, not lookup.**
  *(Flag as hypothesis: patching measured answer decodability, not articulation.)*

---

# ACT 7 — Proposal: the two-stage method (3 slides)

### 7.1 Stage 1 — the missing objective
- `D(c) = KL( π(·|c,x) ‖ π(·|I,x) )`, **minimized**; gradient reaches **caption tokens only**.
- *Write text that makes your blind self behave like your sighted self.* The caption is forced to be a
  **sufficient statistic** of the image for the model's own answer distribution.
- **Present it as PAPO with all three differences reversed:**

| | PAPO | Stage 1 |
|---|---|---|
| teacher | self, blinded (**bad**) | self, **with the image** (good) |
| sign | maximize | **minimize** |
| space | pixel perturbation | **text channel** |
| optimum | none — unbounded | **unique: parity** |

- **[MAKE]** this comparison table + the two-role pipeline diagram (captioner / blind answerer / reference).
- `[OURS]` status: pool **complete and gated** — 27,326 eligible of 38,870; 200 drawn (76 letter /
  124 numeric); manifest `1b28495b…`; **11/11 gates pass**. **Training loop not yet built.**

### 7.2 Why correct-answer items are the *only* well-posed ones ★ answers the obvious objection
- On items the sighted model gets **wrong**, the target behaviour is a wrong answer → we would train
  captions to faithfully reproduce a perceptual error (their own `METHOD.md` §5.2) → **PAPO's bad-teacher
  problem re-entering through a different door.**
- On **correct** items the target is known-achievable by this model on this item.
- ⇒ open decision **D11** (filter to image-correct items) is **constitutive, not a tuning knob.**

### 7.3 Stage 2 — internalize by OPD, with derivable privileged information
- Teacher conditioned on (image + Stage-1 caption); student on image only.
- **The caveat, and why our design satisfies it:**
  - `[PAPER]` **imitation gap** (Weihs et al., NeurIPS 2021): when the expert has privileged information
    the imitator lacks, marginalizing it out can yield a **"sub-optimal, even uniformly random"** student.
  - `[PAPER]` **Gekhman et al.** (EMNLP 2024): fine-tuning on knowledge outside the model's store is
    learned slowly and **linearly increases hallucination**.
  - ⇒ Distilling Track T's *oracle* facts directly would hit exactly this. **Our privileged signal is a
    caption the same weights produced from the same image — inside the student's reachable set by
    construction. Imitation gap zero, not small.**
- **[MAKE]** two-stage method diagram with the "derivable vs oracle" contrast called out.

---

# ACT 8 — Status, risks, next (2 slides)

### 8.1 Honest status board
| Line | Status |
|---|---|
| Attention falsification | **done**, figures exist |
| Mechanistic localization | **done**, 5 figures exist |
| Set 2 / Set 3 nulls | **done**, audited NO FAULT |
| Track T extraction deficit | **done** |
| PAPO A / B | **done**; **C-pure numbers not extracted** ⚠️ |
| DeepEyes repro + baselines | **done** |
| Stage 1 pool | **done, gated** |
| Stage 1 training loop | **not built** |
| Stage 2 | design only |

### 8.2 Declared limitations — say them before the audience does
- Stage 1's optimum is **parity with the image, never better** (`METHOD.md` §5.1). Stage 1 delivers a
  *capability*; Stage 2 is the only place numbers can move.
- Estimand mismatch (§5.3): Stage 1 judges **short-answer Instruct** behaviour; the deficit was measured
  in a **Thinking** chain. Becomes load-bearing once Stage 2 is a Thinking model.
- Success criteria (OPEN-8) **not frozen** → blocks a full run under our own rigor protocol.
- V\* absolute numbers are position-bias inflated; only deltas and flip patterns are trustworthy.

---

# BLOCKERS before this is presentable

1. **Extract C-pure numbers from wandb.** The only first-party causal measurement. Blocking.
2. **Re-verify from frozen sources:** HallusionBench 94.4→52.8; attention −38% and r values;
   text_privilege premature-closure 55/84/1.4.
3. **Read before claiming novelty:** Vision-SR1 (2508.19652, ICLR 2026) — closest to Stage 1;
   *Anchored Residual Guidance for Privileged OPD* (2606.10385) — privileged OPD is active;
   *From Drop-off to Recovery* (2603.17228) — may independently corroborate the Part 11 capstone.
4. **Design question that changes what Stage 1 runs:** `freeze_vision_tower=false` (D12) contradicts the
   claim "improve extraction *from* ViT tokens." Our own Part 12 says `llm_only ≈ full`. **Freeze it, or
   run both arms** — otherwise a gain is uninterpretable between re-encoding and better read-out.

# Figures to build (priority order)

1. ★ PAPO dissociation two-panel (KL ↑ / accuracy flat) — carries Act 5
2. ★ Track T three-bar + recovery fraction — carries Act 6/7
3. ★ Unifying pipeline diagram (where each method acts) — carries the whole talk
4. DeepEyes +1.6 / −0.3 / +11.8 three-bar
5. Set 2 probe-vs-control with chance line
6. Set 3 forest plot with Holm threshold
7. HallusionBench quartile collapse
8. Two-stage method diagram
9. rel_fro-by-layer with MLP/attn split
10. `sa_10204` side-by-side (qualitative)
