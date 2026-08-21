# Corrections found in the 2026-08-19 full-repo sweep

Every item below changes something in `DECK_PLAN.md`. Ordered by severity.

---

## C1 — ★ RETRACTED CLAIM: "text potent, pixels inert" must NOT be presented

`DECK_PLAN` Act 3.4 carried Set-2's causal dissociation as a finding. **Set 3 overturned it.**

> `SET3_EXPERIMENT_RECORD.md`: "**(a) D6 OVERTURNED.** Set-2's *text-potent / pixels-inert dissociation*
> does **not** survive the Set-3 robust pool: in Pool-S, GT-text and pixel re-injection have **similar
> small effects** (V_text +0.067 raw p=0.052; V1 +0.060 raw p=0.035) — **no text≫pixel asymmetry.**
> Set-2's apparent V_text potency is attributed to the flaky-item contamination documented in D3 (the
> robustness filter removed ~58% of flaky 'errors'; the dissociation vanishes on the clean pool)."

Honesty rider from the same record: after Holm neither survives, so the correct statement is
"**the asymmetry does not replicate**," *not* "pixels are now proven potent."

**Consequence:** the talk cannot claim text > image on Set-2 evidence, and I must remove the framing that
*On the Faithfulness of Visual Thinking* (2510.23482) "independently replicates" it — we withdrew the
finding it would be replicating. Cite 2510.23482 for the articulation/faithfulness claim only.

**What still supports the text channel** (unaffected by D6): Track T privileged (+0.075, content-specific
vs placebo +0.105, Holm 0); babyVision B1′/B2′ (driver is the reasoning text, not the image); Part 12
freeze ablation (read-out is LLM-internal). These are three independent legs and they hold.

**Also A8 is VOID** as written (presupposed the dissociation).

---

## C2 — ★ The PAPO "money figure" cannot be a cross-arm KL comparison

`PAPO_probe/PROBE_DESIGN.md` §1 states the wandb `actor/kl_prcp_loss` is **scientifically contaminated
for cross-arm comparison**, for four reasons: on-policy on a moving distribution; fresh unseeded mask each
step; Arm A's value is a bug artifact (term never trained); measured on the **training set**, not held-out.

**Consequence:** the two-panel figure is valid only as a **within-arm** dissociation (C-pure's own
objective rises while its own validation accuracy does not). It must **not** be drawn as A-vs-B-vs-C KL
curves. The fair cross-arm grounding number requires the **offline perception-KL probe — which is
DESIGN ONLY, not built.**

---

## C3 — babyVision numbers were wrong, and the real ones are much stronger

I had "re-seeing +1.17 vs re-thinking −1.83." **Wrong.** From `babyVision/RESULTS.md` (Gemma-4-31B-it, 388 items):

| Task family | standard | B1′ | Δ | p |
|---|---|---|---|---|
| perception (counting, search, tracing) | 15.7% | **22.5%** | **+6.8** | **0.015** |
| reasoning (rotation, folding, overlay) | 39.6% | 33.0% | −6.6 | 0.12 (trend) |

**And the load-bearing part I had missed entirely:** B2′ (same, image *not* re-shown) gives the same
split — perception **+5.2**, reasoning **−7.6**. Comparing B1′ vs B2′ directly, where the only difference
is whether the image is re-shown, **only 9 of 388 items change.**

> "**The driver is the reasoning, not the image.**"

This is first-party evidence for the text channel that does **not** depend on the retracted D6.

---

## C4 — Three babyVision findings absent from the plan

- **Finding 2:** switching reasoning regime flips an item right↔wrong **25.5%** of the time; re-running
  *standard* with a new seed flips it **25.3%** — identical. "Reasoning re-samples the answer around a
  fixed perception; it doesn't steer it toward the right one." Item-level: 7% always right, 35% always
  wrong, 58% mixed.
- **Finding 3:** serial / step-by-step subtypes **18.8%** mean correct vs single-glance **43.4%**.
- **Finding 4:** on unsure items, neither confidence, length, nor chosen answer predicts the flip — an
  uncalibrated coin-flip at the item's own success rate.

---

## C5 — `image_toolCalling/SUMMARY.md` contains a SUPERSEDED number

SUMMARY.md finding **D** says the model attends **−28% less** to the re-injected image (0.081 → 0.058).
`forced_reexamination.md` explicitly rejects that comparison:

> "A naive turn-0-vs-turn-1 attention mean is **misleading**: turn-0 reasoning tokens come *before* the
> re-injected image, so by causal masking they attend exactly 0 to it."

Corrected, measured **within** each segment: turn-1 → re-injected = **0.120**, *higher* than turn-0 ever
attended to the original (0.097); total visual **0.097 → 0.184 = +91%**.

**Do not use `D_forced_turn0_vs_turn1.png` with the −28% framing.** My +91% / 96% figures were correct.

Note a genuine cross-dataset difference to state honestly: on **babyVision**, ~8 of 10 items attend *more*
to the original copy than the fresh one. HallusionBench-forced shows the opposite. Different models
(Gemma-4-31B both) and different injection protocols — report it, don't hide it.

---

## C6 — Numbers now verified against frozen sources (were ⚠️ in the plan)

- **HallusionBench collapse — CONFIRMED.** Gemma-4-31B-it, stratified 30% subset (~273 items), **edited
  (vi=2)** images, by reasoning-length quartile: **94.4 → 75.0 → 69.4 → 52.8%** (binary ⇒ Q4 ≈ chance).
- **Attention decay — CONFIRMED.** 0.104 → 0.065 = **−38%** (standard), −34% (tool).
- **Correlations — CONFIRMED.** r = **−0.52** (standard), **−0.58** (tool), **−0.60** (forced).
- **Premature closure — CONFIRMED.** think_tok ≤5 fraction: T0 0.000, T1 **0.554**, T2 **0.838**, T3 **0.014**.
- **Forced re-examination — CONFIRMED.** +91% attention; 254/265 = **96%** unchanged; 80.4 → 82.3%,
  Δ+1.9pp, McNemar χ²=1.45, **not significant**; 8 wrong→right, 3 right→wrong.
- **Track T — CONFIRMED and richer** (see C7).
- **86.9% perception-error claim — NOT CONFIRMED.** Verified the source paper's title/authors
  (*From Seeing to Thinking*, arXiv 2605.20177, Wu, Chen, Tu, Tang, Shi, Liu, Lu, Xie, Zhou) but the
  abstract does **not** contain this statistic. **Must be located in the paper body before it goes on
  slide 1.2**, or replaced with the abstract's actual claim.

---

## C7 — Evidence found that strengthens the talk (was missing)

**Track T** (`TRACK_T_SIGNAL_REPORT.md`, MathVerse, Qwen3-VL-4B-Thinking, avg@5, K=5, seed 0, audit 13/13 PASS):
- MC primary n=373: base 0.808 → **privileged 0.883**; Δ **+0.075** [+0.046, +0.105], McNemar (42,3), Holm **0.0**
- **privileged − placebo = +0.105** [+0.075, +0.134], Holm **0.0** ← *content-specificity gate; I had omitted the placebo arm entirely*
- self − base −0.030 [−0.056, −0.004], p=0.87, Holm 1.0; placebo − base −0.029, p=0.85
- recovery **−0.41** [−0.91, −0.11]; FF robustness: privileged **+0.106** [+0.042, +0.171], p=0.019
- **Rigor story worth telling:** the self arm originally reused one description across all K draws; corrected
  to K independent descriptions before the confirmatory run. *The buggy version would have reported
  spurious positive recovery.*
- Declared caveat: MathVerse (2024) contamination; base MC 0.81 possibly inflated. The finding rests on
  gaps, which contamination affects symmetrically.

**Probe A** (`PROBE_A_FULL_RUN_REPORT.md`, MMStar n=1500, K=3, Qwen3-VL-4B-Thinking):
- T0 0.6993 · T1 0.6344 · T2 0.5691 · T3 **0.7049**
- T3−T0 **+0.0056** [−0.012, +0.023] p=0.945 · T3−T1 +0.0704 p=9.8e-10 · T1−T0 −0.0649 p=3.1e-10 ·
  T2−T0 −0.1302 p=1.0e-29
- ★ **Reasoning axis T3−T0: McNemar b=71, c=71, p=1.0** — "a delta of ~0 is not 'T3 does nothing' — it is
  71 items T3 fixes and 71 different items T3 breaks, exactly cancelling." 14.2% of items change.
- ★ **The trigger is text presence/length, not content**: placebo short-circuits **more** than a real
  caption (84% vs 55%). T3's caption median 229 tok vs T1's 682 (3×).
- Short-circuited accuracy is 11–13 pts worse within arm; T3's rare short-circuits are worst of all (0.292).
- **Washout hypothesis NOT supported**: T3 mean think_tok 1944.4 is the *highest* of all arms (T0 1763.4).
- §5.1 actionable: when T3 *breaks* an item, its caption overlaps a **wrong** option 2× more often
  (0.115 vs 0.056). Cheap generation-time quality signal for best-of-M.

**Set 2** (`NEGATIVE_RESULTS.md`):
- N1: image-token positions **bal_acc 0.918** (probe works); text/reasoning positions **0.504 / 0.511 /
  0.517** = chance, across all k, all 6 layers, all positions (n=152).
- N3: the flip does **not** replicate — RIPE 2/39, **controls flip MORE (4/39)**; `count0` carries all
  flips. **`flipAns` is a construction confound** (RIPE `wrong_target = model_ans` forces the sign) —
  excellent cautionary slide.
- N4: "the answer is determined at **emission**, not drifted."
- ★ **The control that legitimizes the null:** "A DIFFERENT image at `think` **moves answers** (causal
  leverage). SAME image = **attended but inert**." So "re-injecting the same image doesn't help" is a real
  result, not a broken instrument. *This must be on the slide whenever a pixel null is shown.*
- Methodological: batched HF generation is **VOID** for VLMs (M-RoPE + left-padding bug).

**Set 3** (`SET3_EXPERIMENT_RECORD.md`):
- **H1 FAIL**: V_self − V0, Pool-S, f0.25, n=149 — V0 0.060, V_self 0.067, **Δ=+0.0067**, 95%CI
  (−0.047, +0.060), McNemar p=1.0 (9 fixed / 8 broke). Concluded-only n=120: Δ=+0.025, p=0.61. Robust.
- Exploratory: V1 +0.060 (p=0.035) · V_text +0.067 (p=0.052) · V_scaffold **+0.087** (p=0.007) ·
  V_restart +0.087 (p=0.015) · **V_self_pre +0.114** (CI 0.040–0.195, p=0.006); placebos flat
  (V_scr +0.013, V_text_wrong +0.007).
- **Across the 50-test block NOTHING survives Holm**: V_self_pre 0.006→**0.299**, V_scaffold 0.007→**0.353**,
  V_restart 0.015→**0.702**, V1 0.035→**1.000**.
- Restart decomposition: re-presented image **+0.074**; instruction alone +0.013.
- Audit: **NO FAULT FOUND**. V_self payload = GT-perfect (multiset match 149/149).
- ★ **Why the text arm failed, mechanistically:** "both text payloads enumerate **objects but not spatial
  relations**; CLEVR is relational, so re-serializing objects-as-text cannot restore the spatial layout the
  reasoning needs." **This bounds what our text nulls mean** — we tested impoverished text, not text per se.
  Directly relevant to caption_stage1, where the caption is free-form and *can* carry relations.
- Continuation reads: **7/10 V_self-null items engage the injected list and are still wrong** → real null.

**RH-Bench** (`RH-Bench/Qwen3-VL-4B-Thinking_Results.md`) — *entirely absent from the plan*:
- We ran the *More Thinking, Less Seeing?* benchmark ourselves. Qwen3-VL-4B-Thinking, 900 items:
  **reasoning 69.3%, perception 64.4%** — a 4.9pp gap in the paper's predicted direction.
- Deviations to declare: judge = Qwen3-32B not GPT-4o; 900 vs 1000 items; **not comparable to published
  scores**. 57/900 (6.3%) produced no answer (thinking exhausted 16384 budget), scored incorrect.
- Single point only — **RH-AUC requires multiple thinking budgets**, so we cannot report RH-AUC.

---

## C8 — Assets found that were not in the plan

- `image_toolCalling/forced_traces.html` — browsable turn-0 vs turn-1 traces for **every** forced sample,
  flips colour-coded. Live demo material.
- Per-subcategory forced gains: math **+6.7**, ocr **+4.8**, figure **+4.2**; table/chart 0.0 (ceiling);
  video 0.0 (interpretation); map −5.0 (n=20, noise). Framing: *"re-examination repairs perception, not
  commitment."*
- `babyVision/RESULTS.md` appendix: full per-subtype solvability table (22 subtypes, serial vs single-glance).
- `PAPO_probe/FURTHER_ANALYSES.md` — three further designed analyses (intra-chain decay,
  accuracy-under-masking, attention-to-image mass). Design only.
- `RL_SeeingToThinking/runs/PROJECT_MASTER_DOCUMENT.md` (67 KB) and `PROJECT_PRESENTATION.md` (42 KB)
  remain **unread**. Should be checked before the deck is frozen.
