# Re-Perception Causality — WORKING DESIGN NOTES

**STATUS: WORKING NOTES — NOT FROZEN, NOT PRE-REGISTERED.**
Nothing here is a committed bar or definition. Started 2026-08-13, out of the DeepEyes reproduction
(`DeepEyes_clone/DEEPEYES_RECORD.md`) and the PAPO C-pure result.

---

## 1. One-paragraph statement (candidate abstract core)

> Vision-language models are increasingly trained to "think with images," re-examining visual evidence
> mid-reasoning through cropping, zooming, or perception-targeted rewards. We show that these
> interventions reliably change behaviour far more than they change answers, and that the benchmarks
> used to justify them cannot distinguish the two. Across five independent interventions spanning
> prompting, privileged text, perception-loss RL, and tool-calling RL, accuracy moves only when the
> intervention supplies visual information the model had not already encoded; where that condition is
> unmet, models produce fluent, accurately-grounded re-perception with no measurable effect on their
> outputs. Supporting evidence is threefold: the tool-calling method's own ablation attributes only
> +1.6 points of its +18.9 headline gain on V\* to image interleaving (and −0.3 on HR-Bench-4K), with a
> substantial gain appearing only at 8K resolution where the original encoding genuinely loses detail;
> an independent reproduction finds no accuracy drop when the returned crops are withheld entirely; and
> our own perception-loss RL run raises the perception objective substantially while validation accuracy
> remains flat. We further find the standard V\* protocol places the correct answer at a fixed option
> position, inflating reported scores by ~5 points and rendering the baseline irreproducible across three
> good-faith attempts (63.4–79.1 against a claimed 71.2). We therefore introduce a causal test that
> intervenes inside the region a model itself selected for re-examination, on items constructed so that
> the edited detail is unresolvable at full resolution but resolvable in the crop, isolating whether
> models fail to *acquire* visual information or fail to *use* it.

---

## 2. RETRACTED: the stronger thesis, and why it died

An earlier framing was proposed and **falsified within the same session**. Recording it so it is not
re-proposed:

> ~~"The reasoning channel is text-fed. Perceptual evidence moves the answer only insofar as it has been
> serialized into text."~~

**Killed by three objections:**

1. **Direct counterexample in existing evidence.** DeepEyes Table 9: iMCoT beats text-only CoT by
   **+11.8 on HR-Bench-8K**. If pixels never fed the answer pathway this is impossible. Thesis is
   falsified by data already in hand.
2. **Two of the supporting "pixel nulls" are informationally vacuous.** Set 3's V1 re-injects the
   *identical* image already in context (zero new information — null predicted by any theory). And on
   V\* every image is ≤8.29M px vs the 12.85M `max_pixels` limit (**verified**), so no downscaling
   occurs and the model already encoded full resolution; a crop adds magnification, not new pixels.
   That alone explains +1.6 (V\*) vs +11.8 (HR-8K) with no channel theory.
3. **Track T undercuts the text framing.** Privileged text (+7.5) is *oracle* text; self-description
   (~0) is also text. The operative distinction is **correct-new-information vs none**, not
   text vs pixels.

**Lesson:** the surviving account must explain the HR-8K positive, not just the nulls.

---

## 3. Surviving thesis (two parts, more parsimonious)

**(a) Information-gain condition.** Re-perception changes answers only when it supplies information the
model had not already encoded. On standard benchmarks that condition is rarely met, so these methods
yield large behavioural change with negligible accuracy change.

**(b) Extraction/usage bottleneck.** Even when the information *is* newly available, the model often
fails to carry it into the reasoning chain (this is Track T's extraction deficit, localized to
re-perception).

Existing evaluations **conflate (a) and (b)**. That conflation is the gap.

---

## 4. Evidence ledger

| # | Intervention | Channel | Info actually added? | Δ accuracy | Source |
|---|---|---|---|---|---|
| 1 | Set 3 V1 — re-inject identical image mid-chain | pixels | **no** (same image) | null | ours, Set 3 |
| 2 | DeepEyes crop re-injection, V\* | pixels | little (no downscale) | +1.6 | DeepEyes Table 9 |
| 3 | DeepEyes crop **withheld** entirely, V\* | pixels | — | **no drop** | issue #60 (independent) |
| 4 | DeepEyes crop re-injection, HR-8K | pixels | **yes** (8K downscaled) | **+11.8** | DeepEyes Table 9 |
| 5 | PAPO C-pure — reward pixel-sensitivity | pixels | no | **null** (KL ↑↑, val flat) | **ours, first-party** |
| 6 | Track T — oracle perceptual fact as text | text | **yes** (oracle) | **+7.5** | ours, Track T |
| 7 | Track T — model's own self-description | text | no (model-limited) | null (recovery −0.41) | ours, Track T |
| 8 | text_privilege — external captioner | text | partial | null (premature closure) | ours, Probe A |

Rows 2/3/4 are the crux: the *same mechanism* is null where no information is added and strongly
positive where it is.

**Caveat to state plainly in any writeup:** only rows 5–8 are first-party; only row 5 is a first-party
*causal* measurement. Rows 2–4 are inherited from the paper's own ablation; row 3 is a third-party
report. This is currently a thesis built largely on secondary evidence.

---

## 5. Evaluation-validity findings (own audit, V\*)

- Correct answer is **always option A** (`judge_result.py:140` hardcodes `answer = 'A. ' + answer`).
  Accuracy tracks %-answered-A in every cell.
- Independent measurement (issue #66): shuffling options costs **~5 points** (89.0 → 84.3).
  Maintainer's defence ("we follow the official V\* setting") was rebutted and never answered — the
  official V\* eval is **likelihood-based**, hence position-invariant.
- Baseline **not reproducible**: 63.35 (issue #91) / 76.44 (ours, tool prompt) / 79.06 (ours, paper
  protocol) vs claimed 71.2. The headline +18.9 therefore ranges +9.4 to +25.1 depending on denominator.
- Clean demo of the confound (`sa_10204`, same model, same image): tool prompt → *"there is no van
  visible"* → **wrong**; plain-MC prompt → `"A."` → **correct**. Identical failed perception, opposite score.

---

## 6. The decisive experiment (sketch, NOT frozen)

**Construct items where the edited detail is unresolvable at full resolution but resolvable in the
crop.** Information gain is then guaranteed by construction, which immunizes the design against the
mundane explanation that killed the earlier thesis.

Intervene **inside the region the model itself selected** (its own `bbox_2d`) — not a
designer-chosen object, which is what all existing counterfactual benchmarks do.

Measure two things separately:

| verbalized description changes? | answer changes? | conclusion |
|---|---|---|
| ✗ | ✗ | **acquisition/extraction failure** — pixels never reach text |
| ✓ | ✗ | **usage failure** — text produced, answer ignores it |
| ✓ | ✓ | re-perception **is** causal → surviving thesis is wrong |

Every branch is informative; one falsifies outright.

**Required controls (not yet designed):** perceptibility check (is the edit resolvable in the crop at
all, by any model/human?); placebo edit outside the attended region; unedited pair as baseline;
item-difficulty matching. Set 2's `flipAns` circularity failure is the cautionary precedent — the
metric must not be true by construction.

---

## 7. Prior-work landscape (crowded — checked 2026-08-13)

**Methods claiming to induce visual grounding (all reward a *correlate*, none tests counterfactual necessity):**
- PAPO — perception-KL on masked image (ours)
- **Vision-SR1** (arXiv 2508.19652, **ICLR 2026**) — self-contained visual description, blind re-prompt
  for sufficiency. **Confirmed: never tests counterfactual dependence.**
- Attend to Evidence (2605.30912) — evidence-anchored spatial attention supervision for RLVR
- CoSo — counterfactual token-level causal influence
- CSS/CSST (2003.06576) — counterfactual sample synthesis, VQA-era

**Counterfactual diagnostics (all test *passive, single-pass* perception on a designer-chosen object):**
- CounterCount (2605.17826) — paired factual/counterfactual, edited attributes; finds VLMs degrade under
  counterfactual edits, i.e. prior-reliance
- Visual-Counterfact (2606.28273) — 469 recolored objects; "Vision-Default, Prior-Override"
- HalluSegBench (CVPR 2026) — controlled visual counterfactuals for segmentation

**Claimed remaining gap:** no work tests causal dependence on **self-directed** re-perception — the
region the model chose. Requires a proper systematic novelty check before any submission; the searches
behind this list were not exhaustive.

> ⚠️ **Separate scoop warning for `text_privilege`:** Vision-SR1 (ICLR 2026) is close to the planned
> Stage-1 articulation → Stage-2 self-distillation design. Read before further investment there.

---

## 8. Open decisions (owner: user)

- Slot A (central claim): user indicated **A2 (dissociation diagnosis)** and **A3 (method)** both
  attractive; A3 is gated on A2 and on the OPD↔agentic-verl merge. Not settled.
- Eval substrate: V\* (confounded, low headroom) vs HR-Bench-8K (where the effect is real; **note
  issue #141 bbox coordinate-space bug DOES affect HR-Bench**) vs purpose-built item set.
- Metric: counterfactual edit (behavioural, judge-free) vs perception-KL (differentiable, reusable as
  reward) vs both cross-validated.
- Whether the paper carries a method at all, or stops at diagnosis + metric.

## 9. Assets in hand

DeepEyes-7B reproduced & audited (88.48 V\*, projected) · base baselines 76.44 / 79.06 · **191
trajectories + 355 self-chosen bboxes on disk** · PAPO C-pure trained 60 steps (dissociation) ·
tokenizer compat verified for OPD (DeepEyes↔Qwen2.5-VL-7B: 0 mismatches; **Qwen3-VL excluded**) ·
V\* data + full eval harness · Vision-SR1 open-source (third method family testable).

## 10. Not done

Causal metric undefined · no counterfactual items constructed · teacher-function probe not run ·
pre-commitment analysis on the 191 saved trajectories not run (zero-inference, cheapest next step) ·
C-pure numbers not yet extracted precisely from wandb.
