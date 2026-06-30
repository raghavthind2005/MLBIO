# SLIDE DECK — Where & How RL Fixes Perception in a VLM

> Ready-to-build slide content: each slide has **[ON SLIDE]** (text/table/figure to show) and **[SAY]** (speaker
> notes — the "how computed / interpret / attribution" you say out loud). Figures have an ASCII preview + a
> one-line "plot spec" (axes + source CSV) if you want to render them. ~16 slides + appendix. All numbers verified
> against `FINDINGS.md` / the run logs. Attribution is explicit: **[PAPER]** = UCSC-VLAA, **[SENIOR]** = your
> supervisor's prior analysis, **[OURS]** = this project, **[VISION]** = the broader tool-call goal.

---

## Slide 1 — Title
**[ON SLIDE]**
- **Where & How Does RL Fix Perception Inside a Vision-Language Model?**
- A mechanistic study: localizing — and re-using — the perception fix
- *Your name · date · Qwen3-VL-4B · Stage-1 RLVR*

**[SAY]** "I post-trained a vision-language model with RL on perception and it improved a lot. This talk is about *where* inside the model that improvement lives, *what* it is mechanistically, and whether we can *re-use* it — which points at a concrete next method."

---

## Slide 2 — The problem (motivation)
**[ON SLIDE]**
- VLMs make **perception** errors — they mis-*see* (miscount, misread, mislocate).
- **[PAPER]** "From Seeing to Thinking" (UCSC-VLAA): **86.9% of VLM errors are perception errors that more reasoning cannot fix.**
- Fix proposed by the paper: **staged RLVR** — RL with verifiable rewards, trained per capability (Stage 1 = perception).
- **Our question:** if RL fixes perception, *where and how, inside 4.4 billion weights, does the fix live?*

**[SAY]** "The motivating paper shows most VLM errors are *seeing* errors, not *thinking* errors — so thinking longer doesn't help. Their fix is staged RL. I reproduced Stage 1 (perception) and then asked the mechanistic question they didn't: where does the fix actually sit in the network?"

---

## Slide 3 — Background: the senior's observation = our starting hypotheses
**[ON SLIDE]** *(attribute clearly)*
- **[SENIOR]** compared the paper's **base vs. fully-trained** checkpoints and proposed five claims:
  - **S1 — "Not a new representation" (control).** The model's internal *map* of concepts — how image- and text-features sit in vector space, and the gap between them — is **unchanged** before vs. after RL. So RL didn't reshape *how* the model represents the world; the fix is something narrower.
  - **S2 — "A tiny edit, in the MLPs" (where).** The weights barely move, and the little that moves sits in the **MLP / feed-forward blocks** (the feature-readout machinery) of the **upper layers** — **not** in attention (the cross-token routing).
  - **S3 — "The MLP edit is what fixes it" (cause).** Transplant **only the MLP** changes onto the untrained model → perception **improves**; transplant **only attention** → **nothing**. The MLP change is *sufficient*.
  - **S4 — "The model already sees, then forgets" (precondition).** Even the **untrained** model figures out the right percept around its **middle** layers — but it **fades toward the top**, so by the output it's gone. The information is computed mid-stack and lost before it's used.
  - **S5 — "Re-access, not re-learning" (the mechanism, his core hypothesis).** RL's late-MLP edit adds **no new information** — it **rescues** the mid-layer perception the upper layers were discarding, surfacing it at the output. *The model could always see; RL lets it keep seeing.*
- **[OURS]** test these **on our own model**, **across training**, and **under controlled freezes** — reproduce *and* extend.

**[SAY]** "These five claims are my supervisor's — from a two-point (base vs final) analysis on the paper's released 8B model. They're my starting hypotheses. My contribution is to test them prospectively: on our own 4B model, across the whole training trajectory, and with controlled experiments he didn't run."

---

## Slide 4 — What we did (setup + the headline)
**[ON SLIDE]**
- Model: **Qwen3-VL-4B-Instruct** = vision tower (0.42B, 24 blocks) + **LLM (4.0B, 36 layers)**.
- Training: **Stage-1 RLVR (GRPO)** on **3360 DOCCI perception MCQs** (4-option "what do you see?" questions).
- Result: **perception accuracy 0.365 → 0.746** (16 epochs, 96 steps).
- **The fix exists — now: *where* is it?**

**[SAY]** "Concretely: a 4B VLM, RL-trained on 3360 multiple-choice perception questions built from DOCCI photos. Reward is just 'is the boxed answer correct.' Accuracy roughly doubled. So the fix is real and substantial — the rest of the talk dissects it."

*(Backup, if RL-savvy audience: GRPO = no critic; advantage = group-relative z-score over 5 samples; lr=1e-6 + a KL leash to the base model — which is why, as we'll see, the weights barely move.)*

---

## Slide 5 — The experimental lever: 3 freeze conditions  **[OURS]**
**[ON SLIDE]**
| condition | trains | frozen |
|---|---|---|
| **full** | ViT + LLM | — |
| **llm_only** | LLM | vision tower |
| **vit_only** | vision tower | LLM |
- Byte-identical except the freeze flag (same data, seed=1, lr, KL…). **Train once, analyze forever** (16 checkpoints each).

**[SAY]** "My main design choice: run the *same* training three times, changing only which half is frozen. That isolates *which component* carries the fix — a forward, controlled test of the senior's correlational copy-result. Everything else is held identical, so any difference is causal to the freeze."

---

## Slide 6 — Finding 1: RL barely moved the weights — and what moved was the MLPs  (tests S2)
**[ON SLIDE]**
- **Question:** did training *overhaul* the model, or barely touch it — and *where*?
- **How:** for each of the model's **713 weight-matrices**, measure how far it moved during training, **as a % of its own size** (so big and small matrices are comparable).
- **Answer: it barely moved — ~0.05% on average** (most-changed matrix: 0.1%). The model was **nudged, not rewritten** (direction essentially unchanged).
- The little that moved was **MLP-biased**: the **feed-forward ("what it knows") blocks moved ~1.4–1.6× more than attention ("what it looks at")**, at every depth.

| LLM block | early layers | mid | late |
|---|---|---|---|
| how much MLP moved vs attention | **1.59×** | 1.40× | 1.37× |

- **Headline: a 0.05% weight change ≈ *doubled* perception.** → a tiny, targeted edit — not a rebuild.
- *(Honest note for later: the MLP-bias is roughly **uniform** across depth — not concentrated in the late layers as the senior's S2 predicted. We return to why on Slides 9–10.)*
- **[condition]** *full (Cond 1); **replicated in llm_only** — same MLP-bias (1.61/1.41/1.38) with the ViT frozen, so the edit is genuinely LLM-internal. N/A for vit_only (LLM frozen).*

**[SAY]** "First question: did RL overhaul the model, or barely touch it? For every one of the 713 weight-matrices I measured how far it moved — as a fraction of its *own* size, so a giant matrix and a tiny one are on the same scale. The answer is striking: about **0.05%**. The weights are essentially where they started — yet that 0.05% nudge nearly *doubled* perception accuracy. That's the 'surgical edit, not a rebuild' picture, and it's exactly what our setup predicts: a tiny learning rate plus a leash keeping the model close to its starting point. And the little that *did* move was concentrated in the **MLP** — the feed-forward, 'knowledge-readout' part — about 1.5× more than attention. That matches the senior's S2 *in direction*. One honest caveat I'll come back to: the MLP-bias is uniform across depth, not piled up in the late layers like he found — and the resolution of that, on Slides 9–10, turns out to be the most interesting part of the talk."

---

## Slide 7 — Finding 2: it's the **LLM**, not the encoder  (hypothesis H, **OURS**)
**[ON SLIDE]**
| condition | trains | train-reward | direct probe | freeze proof |
|---|---|---|---|---|
| base | — | 0.365 | 0.377 | — |
| **full** | both | **0.746** | **0.657** | both > 0 |
| **llm_only** | LLM | **0.749** | **0.593** | **vision = 0.000** |
| **vit_only** | ViT | **0.443** | **0.423** | **llm = 0.000** |

**What the three measurements mean (define before showing):**
- **train-reward** — the RL training signal itself: the fraction of the model's answers (generated *with* reasoning, in `\boxed{…}`) that are correct.
- **direct probe** — our perception test: show image + a multiple-choice question, do **one forward pass**, read **which option-letter the model scores highest** (no reasoning), and check it against the correct letter → accuracy. *(Mechanism detailed on Slide 8.)*
- **freeze proof** — a **weight-level** check that a "frozen" part truly never trained: we compare its weights before vs. after and find them **bit-identical** (relative change = **exactly 0.000**). `vision = 0.000` → the vision tower never moved; `llm = 0.000` → the LLM never moved.

- **Verdict: `llm_only ≈ full ≫ vit_only > base`** — on *both* the training reward and the direct probe.
- **🖼 FIGURE: `figures/fig_ablation.png`**

**[SAY]** "Freezing the vision tower costs *nothing* — LLM-only matches full. Training only the encoder gets ~⅕ of the way. So the fix is overwhelmingly **LLM-internal — re-reading what's seen, not seeing better.** And the freeze 'proof' is literal: the frozen part's weights moved *exactly* zero, because the optimizer never receives its gradients. Honest nuance: the ViT isn't strictly zero — it can help a little — but the LLM dominates."

---

## Slide 8 — Finding 3: the answer becomes readable **late** (~layer 24)  (tests S4/S5)
**[ON SLIDE]**
- **First, two terms:**
  - **hidden state / residual** — the model's internal vector at a given layer (its running "notes" on the input so far). Information flows *up* through the 36 layers.
  - **logit-lens** — take *any* layer's hidden state and run it through the model's **own** output head, as if it had to answer right there → "**is the correct answer already readable at this depth?**" (y-axis = how often it's right).
- *figure: this "readability" vs. depth, for base vs. trained:*
```
argmax-accuracy (logit-lens) vs. LLM layer
0.66 |                                          ╭─────●─●─● trained
0.55 |                                      ╭──╯
     |   base & trained identical here →   │  ⇦ DIVERGE at L24–25
0.38 |  ●─●─●──●… (both)            ╭──────┴──○─○─○─○ base
0.25 |  ························  (logit-lens dip, both, artifact)
     +----------------------------------------------------------
        0    4    8   12   16   20   24   28   32   36  (layer)
```
- Both curves **identical through layer 23**; **diverge sharply at L24**; trained sustains ~0.62–0.66, base ~0.37.
- Final-layer values (0.377 / 0.657) **exactly match the real probe** → method calibrated.
- **[condition]** *full vs base; **replicated in llm_only** (same L24 divergence, final 0.593). Not run on vit_only.*

**[SAY]** "Now *where* in the 36 layers. I use the 'logit-lens': take each layer's internal state and push it through the model's own output head — 'if forced to answer here, would it be right?' Early and middle layers are *identical* in base and trained — RL changes nothing there. They split hard at layer 24: the trained model reads the answer out in the *late* layers; the base model never recovers it. This is the depth signature of the fix."

**🖼 FIGURE: `figures/fig_depth.png`**  *(plot spec: x=layer, y=argmax_acc, base vs trained, marker at L24)*

> **How can a middle layer "pick" one of the 4 answers at all?** The output head turns *any* layer's internal vector into a score for **every** word in the vocabulary. We just look at the scores it assigns the four option-letters (A/B/C/D) and take the highest — so **one of them always "wins," at every layer.** At middle layers that winner is essentially arbitrary (the answer isn't encoded there yet), so it's right only ~25% of the time (chance); at late layers the winner is the true answer, so it's right ~66%. **The curve is simply how often that forced pick is correct, layer by layer.**

---

## Slide 9 — Finding 4: **MLP causes it — but distributed, not late-localized**  (tests S3; revises a prediction)
**[ON SLIDE]**
- **What a "graft" is:** take the **untrained base model** and copy onto it **only one subset** of the trained weights (e.g. just the MLPs), leaving everything else untrained → measure perception. If that subset alone recovers the gain, it is **sufficient** → a *causal* test (we *built* the model and measured it). `% recovered = (grafted − base) / (full − base)`.
- *figure: % of the +0.28 gain each subset recovers:*
```
% of the +0.28 perception gain recovered (graft = base + only-this-subset)
full      ████████████████████████████████████████ 100%
mlp       █████████████████████████                 63%
attn      ████████████                              30%
early_mlp ███████                                   19%
late_mlp  █                                          3.6%
```
- MLP ≫ attention (63 vs 30%) → **MLP-dominant** (supports S3). But **`late_mlp` 3.6% ≪ `early_mlp` 19%** → **distributed**, not late-localized.
- **[condition]** *full (Cond 1); **replicated in llm_only** — near-identical grafts (mlp 0.55/0.55, attn 0.46/0.46). N/A vit_only (LLM frozen).*

**[SAY]** "To get *causation*, not correlation, I transplant only part of the trained weights onto the base model and measure. MLP recovers about twice what attention does — the fix is MLP-dominant, confirming the senior's S3. But here the data **corrected my prediction**: I expected the *late* MLP to carry it; instead late-MLP recovers almost nothing and the effect is spread across the stack, synergistically. I'll show why that isn't a contradiction."

**🖼 FIGURE: `figures/fig_graft.png`**  *(plot spec: bars of recovery% per graft mode)*

---

## Slide 10 — The reconciliation (the key insight to defend)
**[ON SLIDE]**
- Same layer, two interventions:
  - transplant late **weights** (`late_mlp` graft) → **3.6%**
  - transplant late **representation** (patch trained's L24 state) → **82%** *(next slide)*
- **→ RL's distributed MLP edits across layers 0–24 *write the answer into the residual stream*; by layer 24 it's present and readable, and the *unchanged* late layers read it.**
- This explains all four findings at once.

**[SAY]** "Here's the resolution. Transplanting the late *weights* fails — because base's residual stream entering layer 24 doesn't carry the answer. But transplanting the late *representation* works — 82%. So the late layers don't need new weights; they need the *right input*, which the early/mid MLP edits build up the stack. The fix *manifests* late but is *caused* throughout. That one sentence accounts for the tiny edit, the MLP bias, the L24 divergence, and the distributed graft."

---

## Slide 11 — Capstone: is the better representation **RE-USABLE**?  **[OURS]**
**[ON SLIDE]**
- **Two interventions (define both):**
  - **per-item patch** — run the *trained* model on an image, copy its layer-L hidden state, and **paste it into the *base* model** at layer L; let base finish → does base now answer right? *(needs the trained model as an oracle — proves the representation is portable.)*
  - **steering vector** — instead of per-image copying, compute **one fixed direction** = average(trained − base) and add it to base. *(a static, deployable knob — no trained model needed.)*
- *figure: how much of the gain each recovers, vs. the layer where we inject:*
```
% of gain recovered by injecting trained's residual into base, at layer L
100% |                                    ●─────●─────●  (L28,32,35)
 82% |                         ●  (L24)
     |                        ╱
 11% |                  ●  (L20)
  0% |  ●──●──●──●──────╯
     +-------------------------------------------------------
        8   12  16   20   24    28    32   35   (patch layer)
sanity: self-patch (base→base) = 0.377 = base exactly  ✓ (method valid)
steering vector (single fixed direction): best ≈ 40%
```
- **Per-item patch at L24 → 82% recovery.** The improvement is a **portable representation**.
- **A fixed steering vector caps at ~40%** → the representation is **input-specific**.
- **[condition]** *full only — **not yet replicated** across conditions (planned follow-up: `activation_patch llm_only`). Disclose this.*

**[SAY]** "The capstone, and the bridge to a method. I capture the trained model's layer-24 state and inject it into the *base* model — base recovers 82%. So the 'better understanding' is a representation you can extract and re-inject. But a single *averaged* direction only gets ~40% — meaning the useful representation is **input-specific**, different per image. The sanity check (injecting base into itself reproduces base exactly) confirms the method is sound."

**🖼 FIGURE: `figures/fig_patch.png`**  *(plot spec: recovery% vs patch layer + steering ceiling line)*

---

## Slide 12 — What it means for the methodology  **[VISION]**
**[ON SLIDE]**
- The better representation is **(a) portable** (re-injectable at ≈L24 → ~full recovery) and **(b) input-specific** (static vector ≤40%).
- → the empirical case for an **on-demand, image-derived re-inspection** mechanism, localized to ≈layer 24:
  *mid-reasoning, the model re-derives a better perceptual representation from the image and re-injects it.*
- **Premise validated. Mechanism (how to generate that representation without the trained model) = the next project.**

**[SAY]** "This is exactly the substrate for the tool-call idea: re-inspect the image, produce a better representation, inject it. My experiments show that injection *would* help (the premise) and *where* (≈L24), and that a canned vector won't do — it must be computed per image. What they don't yet show is how to generate that representation *without* using the trained model as an oracle — that's the next, bigger step."

---

## Slide 13 — Honest scope & caveats  (say these — they build credibility)
**[ON SLIDE]**
- **Probe ≠ training protocol:** training answered with reasoning + `\boxed{}`; our probe reads the **direct** next-token letter (no reasoning). So these are **direct-perception** results; they *track* the trained gain (base-probe 0.377 ≈ base-reward 0.365) but the ~9-pt reasoning increment is separate.
- **Contamination:** the probe uses the training distribution → trained numbers are *train-accuracy* (fine for *localization*; base is clean). Generalization is limited (a different set, babyVision, showed no transfer).
- **Scope:** single 4B model, seed 1, Stage-1 only — a **mechanism study**, not a 1:1 reproduction.
- **Patch uses the trained model as an *oracle*** — proves portability, not the generation mechanism.

**[SAY]** "Four honest caveats. The most important: my probe reads a direct answer with no reasoning, whereas training used reasoning — so these are *direct-perception* claims. They track the trained gain, but the reasoning contribution is separate. I'm also probing on the training distribution, so it's a localization study, not a generalization claim — and on a different benchmark the gain didn't transfer."

---

## Slide 14 — Next steps
**[ON SLIDE]**
- **Tier 1 (½ day):** with-reasoning accuracy probe (generate `<think>…\boxed{}`, parse) → confirm the mechanism holds *with reasoning*.
- **Tier 2 (~days):** **mid-reasoning patch** — inject the L24 representation *during reasoning* → directly tests the tool-call premise in the real setting.
- **Tier 3 (project):** the **tool-call prototype** — generate the L24-style representation from the image *without* the trained-model oracle.

**[SAY]** "Three steps. First, confirm everything under reasoning. Second — the headline bridge — inject the good representation *mid-reasoning* and see it help. Third, the real contribution: a module that *produces* that representation from the image on demand. Tiers 1–2 are runs; Tier 3 is the next research project."

---

## Slide 15 — Summary (the one slide to remember)
**[ON SLIDE]**
| angle | method | result |
|---|---|---|
| where weights moved | Frobenius rel-change | tiny (0.05%), MLP-biased |
| which component | 3-condition freeze | LLM, not encoder (frozen Δ=0) |
| where it shows | logit-lens by layer | late, at L24 |
| which weights cause it | counterfactual graft | MLP-dominant, distributed |
| is it re-usable | activation patch | portable@L24 (82%), input-specific |

**Mechanism:** *RL's tiny distributed MLP edits across layers 0–24 write the answer into the residual stream; by layer 24 it's present and readable, and the unchanged late layers read it.*

**[SAY]** "To summarize: five independent angles, one coherent story. The fix is a tiny, LLM-internal, MLP-dominated, depth-distributed edit that writes perception into the residual stream by layer 24 — and that representation is re-usable but must be re-derived per image. That last point is the empirical seed of an image-re-inspection tool-call."

---

## Slide 16 — Thank you / questions
**[ON SLIDE]** "Mechanism located, representation shown re-usable → a concrete path to an on-demand re-inspection tool-call." · *contact / repo*

---

# APPENDIX — grilling backup slides (have these ready, don't present unless asked)

**A1 — RL exactly.** GRPO advantage `A=(score−group_mean)/(group_std+1e-6)`, no critic (`core_algos.py:213`); PPO dual-clip 0.2/0.3/3.0; KL k3 estimator `exp(x)−x−1` added to the loss with coef 1e-2 (`dp_actor.py:281`); reward `0.9·accuracy+0.1·format` (`math.py:85`); lr 1e-6 + KL → tiny edit by design.

**A2 — Why probe vs training mismatch is OK.** Consistent yardstick → the *difference* is valid; base-probe≈base-reward; it deliberately measures *direct* perception (shows the gain survives without reasoning).

**A3 — Why late-MLP graft fails but L24 patch works.** Graft gives late *weights* but base's *own* (unimproved) residual → nothing to read (3.6%); patch gives the trained *representation* → base reads it (82%). Fix is in the residual the early/mid MLPs build.

**A4 — Logit-lens validity.** Final-layer readout reproduces the real probe (calibrated); mid-layer dip is an artifact identical in both models (no signal); tuned-lens would clean it.

**A5 — Freeze proof.** Frozen component `rel_fro = 0.000` (mean & max, all tensors) because optimizer = `filter(p.requires_grad)` (`fsdp_workers.py:343`).

**A6 — Numbers' provenance.** weight_delta→`deltas.csv`; probe/H→`mc_eval`; depth→`depth_*.csv`; graft→`graft_full_96.csv`; patch→`actpatch_full_96.csv`. (Full detail: `METHODS.md`, `FINDINGS.md`.)

---

### Build tips
- **One result per slide; the number/table goes on the slide, the "[SAY]" goes in speaker notes.**
- Slides 8, 9, 11 are the three to render as real plots (specs above) — ask me and I'll generate them from the CSVs (~20 min).
- Flow: Problem (2) → Hypotheses/attribution (3) → Setup+lever (4–5) → Findings (6–9) → Reconciliation+capstone (10–11) → Meaning+caveats+next (12–14) → Summary (15).
