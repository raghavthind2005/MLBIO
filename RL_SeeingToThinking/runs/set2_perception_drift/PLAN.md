# Set 2 — Perception Drift in Long Visual Reasoning
**Research Plan · v1 · 2026-07-01 · status: APPROVED (design), Step 0 in progress**

---

## 0. One-line thesis

> Set 1 asked *"does RL fix perception?"* (yes; the better percept is reusable — MLP-dominant, readable ~L24).
> **Set 2 asks a different question:** when a model **perceives correctly** but then **reasons for a long time**, does it **corrupt its own correct percept** — and can we **prove it, localize it, and causally reverse it**, without an LLM judge?

The target failure is **not** "the model can't see." It is: *the model **could** see the fact correctly (proven in isolation), yet during a long chain it **used a wrong version** of that fact.* We call this **perception drift**.

---

## 1. Relationship to Set 1 — keep vs. retire

| From Set 1 | Verdict | Why |
|---|---|---|
| Tooling: param-map, logit-lens/depth-probe, activation-patch, weight-delta | **KEEP** (port to new substrate) | validated w/ sanity checks; same Qwen3-VL arch family → ports directly |
| Finding: answer decodable ~L24; MLP > attention; a better reusable percept exists | **KEEP as prior** | tells us *where to read* (E2) and *what to inject* (E3) |
| Config-discipline (byte-identical ablation, freeze proofs = 0.000) | **KEEP as method** | the rigor that made Set-1 mechanism claims hold |
| Direct-answer MC probe (no reasoning) | **RETIRE** | Fallacy #1 — measured a setting we never deploy in |
| Metrics on training-distribution items | **RETIRE** | Fallacy #3 — that's train accuracy, not generalization |
| Off-task/disabled geometry3k val | **RETIRE** | Fallacy #2 — no valid held-out signal |
| babyVision OOD probe | **RETIRE** | confounded (OOD, no clean GT) |

The three Set-1 fallacies are now **design constraints** (see §6), not afterthoughts.

---

## 2. The problem, decomposed into three testable claims

One sentence — *"perceives correctly but reasons wrong about the same object"* — is really three separate claims, proven in order:

- **C1 (Existence):** there is a measurable population of items where atomic perception is *independently verified correct*, yet the long-reasoning answer is wrong — and it **grows with reasoning length**.
- **C2 (Localization):** on those items, the correct visual fact is *present internally early*, then *decays / is overwritten* as the chain proceeds — we can point at the **layer and token** where the belief flips.
- **C3 (Recoverability / methodology bridge):** re-injecting the model's *own early correct percept* at the drift point **flips the answer back to correct** — proving the info was **lost, not absent**, and telling us **where** and **what** to inject.

---

## 3. ⚠️ Corrected measurement logic (the key refinement)

**RIPE alone does NOT prove perception drift.** When a `D=1` item (model *can* perceive) gets the final answer wrong (`A=0`), two failure modes are mixed:

- **(a) Perception drift** — had the correct percept, then *used a wrong version* mid-chain. ← the target.
- **(b) Pure reasoning error** — held the *correct* percept throughout, but botched the *logic/arithmetic*. ← NOT a perception problem.

`RIPE = P(A=0 AND D=1)` counts **both**. It is an **upper envelope**, not the phenomenon. The two modes are **behaviorally indistinguishable** — separating them requires looking *inside* the chain. Hence:

| Level | Measures | Proves |
|---|---|---|
| **E1 (behavioral)** | RIPE envelope = failures despite perceptual capability | the phenomenon's *outer bound* exists (drift **+** reasoning errors), and is length-dependent |
| **E2 (internal)** | splits RIPE: did the internal percept **flip** mid-chain, or **stay correct**? | the *drift* fraction specifically (judge-free, exact-match probe vs GT) |
| **E3 (causal)** | on drift items, **restore** the early percept → does the answer flip back? | drift **caused** the error (not merely correlated) + the injection point |

**Drift is defined and proven at E2/E3; E1 only bounds and finds the candidate set.**

---

## 4. The three experiments (with a running example)

**Running example (CLEVR-style scene, used throughout):**
> Scene (5 objects): large **red** metal cube · small blue rubber sphere · large **red** rubber cylinder · small green metal cube · small **red** metal sphere.
> Question Q: *"There is a large metal cube. How many **other** objects have the same color as it?"*
> GT reasoning (from scene graph): find large metal cube → color = **red** → count other reds → {red cylinder, red sphere} → **answer = 2**.
> Atomic fact **f** we track: *"the large metal cube is red."*
> **Drift window (designed):** "red" is perceived at step 1 but only *used* several steps later during the count — that gap is where a percept can rot.

### E1 — Existence & length-dependence (behavioral, judge-free) → C1
Per item:
1. **Capability check (D):** ask the atomic fact directly (*"What color is the large metal cube?"*). Correct → `D=1`. Exact-match vs scene graph, **no judge**.
2. **Reasoning check (A):** ask full Q, let the Thinking model generate its whole `<think>…</think>`, read `\boxed{}`. Correct → `A=1`. Exact-match, **no judge**.
3. **RIPE candidate** = `D=1 AND A=0`.

**Key plot — RIPE vs. reasoning length:** re-run each item capping the chain at k = 64/128/256/512… tokens. If the failure rate on `D=1` items **climbs with k**, that is direct evidence that *more thinking corrupts a percept the model demonstrably had*.
**Proves:** the envelope is real and length-dependent. **Does NOT prove:** the cause is drift → hand to E2.

### E2 — Localization (internal, mechanistic) → C2
On RIPE candidates, re-run the *exact same generation* capturing hidden states at every **(layer L × reasoning-token t)**. At each cell, logit-lens / trained linear probe reads the *current internal belief about f*, compared to GT (exact-match, **no judge**).

**Output — 2-D percept-decay heatmap** (layer y, reasoning token x, cell = P(correct)):
- **Expected on drift items:** correct ("red") early around L24–26 (matches Neo et al. L≈25 + Set-1 L24), fading to wrong as the chain lengthens; identify the **flip token**.
- **Split:** per item, did f_internal **flip** correct→wrong (drift) or **stay correct** (reasoning error)? → the clean drift fraction.
- **Control:** same probe on **non-RIPE** (correct) items — trajectory should stay correct; if decay is failure-specific, it's not a probing artifact.

### E3 — Causal recoverability (intervention) → C3 + methodology bridge
On **drift** items only: take the model's *own* correct percept vector from the early token/layer where E2 shows it was still "red", **patch it into the residual stream at the drift point**, finish generation. Sweep **(inject-layer × inject-token)**.

- **Signal — flip-to-correct rate.** High → info was *lost, not absent*, and we've localized *where re-access should happen*.
- **Two arms (answers the methodology questions):** (a) inject the model's own **early internal percept** vs (b) re-encoded **raw image tokens**. Whichever recovers more, with a smaller vector, is evidence for what the eventual method should inject and *where*.
- **Sanity:** identity patch = 0 change; patch non-drift items = no harm; patch reasoning-error items → should **not** help (falsification test).

### Illustrative funnel (placeholder numbers — to show the logic)
> 1000 items → **850** perceivable (`D=1`) → **200** fail after reasoning (`A=0`) = **RIPE envelope**.
> E2 → **130 drift** (flip token found) + **70 pure reasoning errors**.
> E3 → re-inject own early percept fixes **~100/130** (77% causal recovery); fixes **~0/70** reasoning-errors (correct — never a percept problem).

That chain — **200 candidates → 130 proven drift → 100 causally reversible at L\*** — is the direct, judge-free evidence, and it hands the methodology a *where* (L\*) and a *what* (early percept vector).

---

## 5. Dataset construction

Two properties must hold at once: **(i)** every atomic fact is known for free (no judge), **(ii)** the question is hard enough to force a long chain that *keeps re-using perception*.

**Ground truth = scene graph + functional program.** CLEVR scenes are *rendered from a symbolic scene graph* (we know every object's shape/color/size/material/position). Questions are generated by a **functional program** — a pipeline of operations over the graph (`filter(large,metal,cube) → query_color → filter(same_color) → count`). Running it yields the **exact answer**, the **exact atomic facts** used at each step (→ what to probe in E2), and the **exact vector to patch** (E3). GQA works the same over real Visual-Genome scene graphs.

**Four dials to force long, perception-grounded chains:**
1. **Compositional depth (D):** # chained ops in the functional program. Deep (D≈10–15) forces many reasoning steps → drives the E1 length curve.
2. **Scene complexity:** more objects + **confusable attributes** (several reds, several cubes) → must hold & re-check facts → drift pressure.
3. **Perceive-early-use-late structure (important):** select/generate questions where a fact is established early and *consumed* many steps later; bigger gap = more room to drift. A *designable* property of the program.
4. **No text-prior shortcut:** CLEVR scenes are *random* → the model **cannot** guess from world knowledge → *forced* to use the image (unlike some GQA questions answerable from priors). This is why CLEVR is the clean primary set.

**Generate vs. use existing:** CLEVR's **generation engine is open-source** → we crank compositional depth ourselves for the length dial. **Recommendation:** CLEVR engine, depth-graded set (primary, indisputable) **+** a GQA slice (real-image robustness).

**Step 0.5 (empirical, before E2/E3):** generate a pilot set, **measure the actual chain lengths** the Thinking model produces, confirm we can push mean length up via depth/complexity. If chains stay short at high depth → that's itself a finding and we adjust substrate/data *before* investing in heavy machinery.

---

## 6. Fallacy guards (explicit — where we win or lose)

| Guard | How |
|---|---|
| **F1: reasoning-inclusive probe** | E2 reads hidden states *during the real `<think>` generation*, at reasoning-token positions — never a direct-answer probe |
| **F2: real held-out validation** | eval sets provably **disjoint** from training data; disjointness audited in Step 0 |
| **F3: no train-set leakage** | every metric on held-out items the model wasn't trained on |
| **F4: isolate the phenomenon** | capability gate `D=1` + E2 internal-flip split → drift separated from perception-inability **and** from reasoning-logic error |
| **F5: correlation ≠ causation** | E2 (probe) must be backed by E3 (causal patch); a readable-but-unused signal proves nothing alone |
| **F6: no judge, no cherry-pick** | exact-match GT throughout; full heatmaps; pre-registered layer band (~L24–26) from Neo et al. + Set-1 |

---

## 7. Design decisions

1. **Substrate → off-the-shelf `Qwen3-VL-4B-Thinking`.** The problem lives in long chains; Instruct produced short chains (why Set-1's decay probe had no signal). Thinking gives realistic long reasoning **immediately, no training**; same architecture → all tools port. Keep Set-1's "better representation" finding as prior for E3.
2. **NO Stage-3 training now — make it a contingency, not a prerequisite.** The diagnosis (E1/E2/E3) needs **zero training**. Decision tree:
   ```
   Run E1 on off-the-shelf Thinking (cheap, no training)
     ├─ phenomenon present (RIPE grows w/ length) → proceed E2/E3 on Thinking. Stage-3 NOT needed. ✅
     └─ phenomenon weak/absent → diagnose:
          • chains too short?  → raise CLEVR depth
          • reasoning ignores image (text-only)? → THEN Stage-3 (visual-reasoning RL) is justified
          • data too easy?     → harder scenes
   ```
   (Also: paper does 1→2→3; **skipping Stage-2** is a deviation to defend later. Defer.)
3. **Data → CLEVR (engine, depth-graded) primary + GQA slice** (§5).
4. **Scope for next week → E1 + E2 core** (diagnosis + mechanism); **E3 stretch** (methodology bridge).

---

## 8. Novelty — honest, repositioned (see `RELATED_WORK.md`)

**The phenomenon is NOT novel.** Visual/perception faithfulness decaying over long reasoning is an active 2025–26 area (Journey Before Destination [2512.12218]; Faithfulness of Visual Thinking [2510.23482]; Hidden Life of Tokens/VISTA [2502.03628]). **We must not claim to have discovered it.**

**What is ours (measurement + mechanism + causality + intervention-design):** a **judge-free, causally-verified, mechanistically-localized** characterization. Every close paper relies on **LLM/VLM judges**, **text-only** step analysis, **aggregate correlational** ranking, and **generic all-layer** steering; **none** causally prove drift on **capability-gated reasoning** items with **verifiable GT**, track a **specific verifiable percept flipping**, or **derive the injection point**. The eventual novelty *headline* must live in the **methodology** (a learned representation that beats raw re-injection at the localized point); the diagnosis is the rigorous **evidence base** that justifies and designs it. Full analysis + the exact claim sentence + reviewer-attack defenses in `RELATED_WORK.md`.

---

## 9. Deliverables & timeline (to next week)

```
runs/set2_perception_drift/
  PLAN.md          ← this
  RELATED_WORK.md  ← Step 0a: novelty lock (adversarial lit pass)   [IN PROGRESS]
  data/            ← CLEVR-engine builder + GQA slice + disjointness audit (Step 0b/0.5)
  e1_ripe/         ← RIPE + RIPE-vs-length
  e2_localize/     ← (layer×token) percept-decay heatmaps + drift split
  e3_patch/        ← causal re-injection sweep (stretch)
  FINDINGS.md      ← running results; verified numbers only
```
**Order:** Step 0a (novelty lock) → Step 0b (data audit + CLEVR pilot, measure chain lengths) → E1 (envelope + length curve) → E2 (localize flip + drift split) → write diagnosis+findings → E3 if time. **Sanity check before every reported number.**

---

## 10. Open risks

- **R1 (substrate):** Thinking may reason long but **text-only** (ignores image) → weak drift. → E1/Step-0.5 tests this; Stage-3 is the contingency.
- **R2 (novelty perceived as incremental):** mitigated by judge-free + causal + localized framing, and by pushing the headline to the methodology (§8).
- **R3 (probe validity):** logit-lens can read info the model doesn't *use* → **F5** (E3 causal) is mandatory, not optional.
- **R4 (CLEVR too easy / synthetic):** GQA real-image slice + depth dial; report both.
