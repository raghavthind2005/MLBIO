# Experiment 2 — Are the better representations RE-USABLE? (activation patching / steering)

> The bridge from the weight-space mechanism (Parts 2–9) to the program's goal: *can we extract the better
> representation RL found and re-inject it at inference to lift accuracy?* If yes → the substrate for a
> re-inspection tool-call exists. See FINDINGS Part 10 / PRESENTATION §9.

---

## 1. Objective & hypothesis
**Objective:** show whether the perception improvement is a **portable representation** (extract + re-inject),
not only a weight change.

**Hypothesis (from Parts 8–9):** injecting the trained model's **late-layer** (≈L24+) answer-position residual
into the **base** model recovers a large share of the gain (base 0.377 → toward trained 0.657). A single fixed
**steering vector** recovers *some* of it (portable, but partial because the cause is distributed/synergistic).

---

## 2. Method (two variants, same machinery)

Notation: residual stream at the **output of LLM layer L**, at the **answer position** (the last prompt token,
whose next-token logit is the answer). Base and trained see identical inputs → identical tokenization/length,
so positions align exactly.

**(A) Per-item activation patching (causal mediation; upper bound).**
For each item: capture the *trained* model's residual `r^trained_L` at the answer position; run the *base*
model but **overwrite** its residual at layer L (answer position) with `r^trained_L`; let base's layers L+1…35
process it; read the answer. Sweep L. → *Does the trained late representation, read by base's remaining layers,
produce the right answer, and from which depth?*

**(B) Portable steering vector (deployable artifact).**
On a calibration split, `v_L = mean_items( r^trained_L − r^base_L )`. On a held-out split, run base with
`r := r + α·v_L` at the answer position, layer L. Sweep L and α. → *Is there a single fixed direction (no
trained model needed at deploy time) that lifts base?* This is the closest analog to what a tool-call injects.

**Metric:** the existing DOCCI MC probe accuracy. Reference points: base (no-op) 0.377, full-trained 0.657.

---

## 3. Implementation → `activation_patch.py`

Reuses `ckpt_model.load_model` (base + trained), `docci_data`/`probe_loader` (items), `mc_eval` letter-readout,
and **forward hooks** on `model.model.language_model.layers[L]` (the decoder layers).

**Hook mechanics.** A Qwen3 decoder layer returns `(hidden_states, ...)`. A forward hook can read/replace
`output[0][:, pos, :]`. Capture hook stores it; patch hook overwrites it; steer hook adds `α·v_L`.

**Efficient phased run (avoids holding two models + re-running needlessly):**
1. **Phase 1 (trained):** load trained, run all items once with capture hooks on *all* layers → cache
   `r^trained[item][L]` (answer position only; 300×37×2560 ≈ 28M floats ≈ tiny). Free trained.
2. **Phase 2 (base baseline):** load base, run all items once with capture hooks → cache `r^base[item][L]` and
   record **base accuracy** (sanity 0.377). Compute `v_L` from the two caches.
3. **Phase 3 (patch sweep):** for each L in the sweep, for each item, base forward with a **patch hook@L**
   (overwrite answer-pos residual from `r^trained[item][L]`) → accuracy@L.
4. **Phase 4 (steer sweep):** for each (L, α), base forward with **steer hook@L** (`+α·v_L`) → accuracy@L,α.

**Outputs:** CSV `layer, mode(patch|steer), alpha, accuracy` + a text curve. Plus the two reference rows
(base, full-trained).

**CLI sketch:**
```
python activation_patch.py --base <model> --ckpt <full/step96/actor> \
  --dataset docci --jsonl <…> --image-dir <…> --n-sample 300 \
  --layers 16 20 24 28 32 35 --alphas 1 2 4 --out actpatch_c1.csv
```

**Memory:** phased design needs only one model at a time + a tiny residual cache → comfortable on one GH200.

---

## 4. Experiment matrix (v1)
- **Checkpoint:** Condition 1 (full) step 96 first; then Condition 2 (llm_only) to confirm portability is also
  LLM-internal.
- **Patch-layer sweep:** L ∈ {16, 20, 24, 28, 32, 35} (brackets the L24–25 divergence from Part 8).
- **Patch scope (v1):** answer-position only (the answer-bearing text token). (Option: all-positions — stronger
  but less "tool-like"; defer.)
- **Steering α sweep:** {1, 2, 4} (and the per-item-patch as the α→"exact" upper bound).

---

## 5. Decision criteria (what each outcome means)
| outcome | interpretation |
|---|---|
| **Patch@L≥24 recovers most of the gain** (→ ~0.60+) | the late representation **is** the carrier; representation-injection works; tool-call substrate confirmed |
| Patch needs **very late L only** (≈35) | trivial (already the output rep) — re-injection adds little earlier; weaker for the tool-call |
| Patch recovers **partially**, plateaus | consistent with distributed cause (Part 9); representation is *partly* portable — quantify the ceiling |
| **Steering vector lifts base** (even partially) | a **portable, deploy-time** direction exists → strongest signal for the tool-call (no trained model needed) |
| Steering does **nothing** but per-item patch works | the useful info is item-specific, not a single direction → tool-call must *compute* the representation per input (e.g. re-inspect), not apply a fixed vector |

The last row is itself a key finding: it would say the tool-call must **re-derive** the representation from the
image per item (which is exactly "re-inspect"), not apply a canned steering vector.

---

## 6. Risks / caveats
- **Logit-lens vs true readout:** here we don't use the lens — we patch real residuals and read the real output
  head, so this is a *true* causal test (cleaner than Part 8).
- **Distributed cause (Part 9):** single-L injection may underperform; consider **multi-layer patch** (patch a
  band L…35) as a follow-up if single-L is weak.
- **Position choice:** answer-position-only is the minimal, tool-like intervention; all-position patching is the
  upper bound — run it only if answer-position is surprisingly weak.
- **Contamination:** same as Part 7 (train-distribution probe) — fine for a mechanism/portability test.

---

## 7. Sequencing
1. Implement `activation_patch.py` + `run_activation_patch.sh` (1 GPU).
2. Validate on **base vs base** (patch base into base → must reproduce 0.377 exactly: a null-op sanity).
3. Run patch sweep on **Cond 1 / step 96**; read the recovery-vs-layer curve.
4. Run steering sweep; check portability.
5. Repeat on **Cond 2 (llm_only)** to confirm LLM-internal portability.
6. Write up as FINDINGS Part 11 + a PRESENTATION slide ("the representation is [/is not] reusable").

**Dependency:** none new — base + trained checkpoints already on disk; can run now, in parallel with the
Condition-3 wrap-up.
