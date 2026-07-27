# PAPO Mechanistic Study — Master Record

**Status:** living document. Last full audit + rewrite: 2026-07-27.
**Scope:** the perception-degradation mechanistic study built on PAPO (arXiv:2507.06448,
ICLR 2026), Qwen3-VL-2B-Thinking, Clariden GH200. This file is the single organized record of
(1) a definitive paper↔code audit, (2) what we actually ran, (3) the decision/journey history,
(4) the frozen offline perception-KL probe spec, (5) corrected next steps.

> **Provenance rules used here.** Every result-affecting claim is tagged with its source:
> `[paper]` = verbatim from the arXiv HTML full text (downloaded 2026-07-27, byte-exact grep, not a
> summarizer); `[code]` = a file:line in the local clone `PAPO_clone/PAPO@main_qwen3 1263a29`;
> `[readme]` = the repo README; `[run]` = our run scripts; `[obs]` = an observed run number.
> Anything I could **not** verify first-hand is explicitly marked **UNVERIFIED**.

---

## 0. TL;DR of the 2026-07-27 audit (what changed)

1. **The paper's canonical `PAPO_G-2B` (Qwen3-VL) uses NO double entropy and keeps the reference
   KL.** Paper Table 3, verbatim: `PAPO_G-2B: γ=0.01, η₁,η₂ = —, Mask 0.6, β=0.01`. `[paper]`
   This is **identical to the code default** ([`qwen3_vl_2b_grpo_papo.sh`](examples/papo_grpo/qwen3_vl_2b_grpo_papo.sh) + [`config_grpo_papo.yaml`](examples/configs/config_grpo_papo.yaml)). **Paper and code agree — there is no deviation for the 2B recipe.**
2. **CORRECTION to our earlier framing.** Our completed PAPO run added **Double Entropy** and flipped
   **`RECOMPUTE_AUG_LOG_PROBS=True`** ("C+DE"), justified in [`README_FIX.md`](README_FIX.md) as
   "faithful to the *written* Eq. 2." The audit shows this was a **misreading**: Eq. 2 is a template
   whose η terms are set **per-model in Table 3**, and for `PAPO_G-2B` (and even `PAPO_G-3B`) η = —.
   So **C+DE is a deviation *away from* the paper's PAPO_G-2B, not toward it.** The faithful
   reproduction is **C-pure** (perception KL only, ref-KL on, no double entropy, RECOMPUTE=False).
3. **Neither completed arm is the paper's PAPO_G-2B.** Arm A (our "baseline", 0.540) is effectively
   **GRPO + ref-KL** (= paper `GRPO-2B`, faithful). Arm B (0.466) is an **off-spec PAPO_G + DE**
   variant. The clean paper `PAPO_G-2B` (C-pure) is **not yet run** — it is the recommended next run.
4. **The authors ship NO offline perception-KL probe.** `KL_prcp` exists only as a *training* loss;
   the released eval harness (`PAPO-Eval` submodule) is accuracy-only. Our probe is net-new but is
   defined to compute the *exact* training quantity (see §5).

---

## 1. Objective & scope

- **Goal:** a *mechanistic* study of how visual perception/grounding degrades along a VLM reasoning
  chain, and how PAPO's Implicit Perception Loss shifts that, studied via a **checkpoint trajectory**.
  **Not** a benchmark-number reproduction.
- **Method:** PAPO = GRPO/DAPO + **Implicit Perception Loss** (maximize KL between outputs on real vs
  patch-masked image) + optional **Double Entropy Loss** (stability regularizer).
- **Model:** `Qwen3-VL-2B-Thinking` (PAPO's own validated small model; pivoted here from an earlier 4B
  attempt that hit a GH200 memory wall — see §4).
- **Compute:** Clariden CSCS, 4×GH200 (aarch64, 95 GB), enroot/pyxis containers, verl/EasyR1 + Ray + vLLM.

---

## 2. Definitive paper ↔ code audit (grounded)

### 2.1 The PAPO objective

**`PAPO_G` (GRPO variant), Eq. 2** `[paper]`:

```
J_PAPO_G(θ) = E_{ {o_i} ~ π_θold(O|q,I) }  (1/G) Σ_i (1/|o_i|) Σ_t {
      min[ r_{i,t}(θ) Â_{i,t} , clip(r_{i,t}(θ), 1-ε_l, 1+ε_h) Â_{i,t} ]
    − β  D_KL[ π_θ ‖ π_ref ]            (reference-model KL,  β)
    + γ  D_KL[ π_θ ‖ π_θ^mask ]         (Implicit Perception Loss,  γ,  MAXIMIZED)
    − η₁ H[ π_θ ]  − η₂ H[ π_θ^mask ]   (Double Entropy Loss,  penalized)
}
```

**`PAPO_D` (DAPO variant), Eq. 3 (Appendix B)** `[paper]`: same but **the −β D_KL[π_θ‖π_ref] term is
removed** (as in DAPO), token-level normalization, dynamic sampling, ε_h=0.28.

**Perception-KL estimator** (Schulman 2020 k3) `[paper]`: `D_KL[π_θ‖π_θ^mask] = r_prcp − log r_prcp − 1`.
**Code match** `[code]` [`core_algos.py:626-630`](verl/trainer/core_algos.py#L626): for `low_var_kl`,
`kl=(ref_log_probs − log_probs).clamp(±20); kld=exp(kl)−kl−1; clamp(−10,10)`. With
`ref_log_probs = aug_log_probs (masked)` and `log_probs = real`, `exp(kl)=π_θ^mask/π_θ = r_prcp`, so
`kld = r_prcp − log r_prcp − 1`. **Identical.**

**Entropy estimator** `[paper]`: "The entropy values for the Double Entropy Loss are implemented as
`H[π_θ]=log π_θ(o|q,I)`, `H[π_θ^mask]=log π_θ(o|q,I^mask)`." **Code match** `[code]`
[`dp_actor.py:357`](verl/workers/actor/dp_actor.py#L357): `aug_entropy_loss = -VF.masked_mean(aug_log_probs, response_mask)`.

### 2.2 Paper Table 3 — per-model hyperparameters (VERBATIM) `[paper]`

> *Table 3: Hyperparameter configurations for models in Table 1.*

| Base model | Variant | γ (perception) | η₁,η₂ (double entropy) | Mask Ratio | β (ref-KL) | ε_l, ε_h |
|---|---|---|---|---|---|---|
| Qwen2.5-VL | GRPO-3B | — | — | — | 0.01 | 0.2, 0.3 |
| Qwen2.5-VL | GRPO-7B | — | — | — | 0.01 | 0.2, 0.3 |
| **Qwen3-VL** | **GRPO-2B** | — | — | — | **0.01** | 0.2, 0.3 |
| Qwen2.5-VL | PAPO_G-3B | 0.02 | — | 0.6 | 0.01 | 0.2, 0.3 |
| Qwen2.5-VL | PAPO_G-7B | 0.02 | **0.05** | 0.6 | 0.01 | 0.2, 0.3 |
| **Qwen3-VL** | **PAPO_G-2B** | **0.01** | **—** | **0.6** | **0.01** | 0.2, 0.3 |
| Qwen2.5-VL | DAPO-3B | — | — | — | — | 0.2, 0.28 |
| Qwen2.5-VL | DAPO-7B | — | — | — | — | 0.2, 0.28 |
| Qwen2.5-VL | PAPO_D-3B | 0.01 | **0.03** | 0.6 | — | 0.2, 0.28 |
| Qwen2.5-VL | PAPO_D-7B | 0.01 | **0.03** | 0.6 | — | 0.2, 0.28 |

**Reading of the entropy column:** Double Entropy is used **only** on `PAPO_G-7B` (0.05) and
`PAPO_D-3B/7B` (0.03). It is **OFF for `PAPO_G-3B` and `PAPO_G-2B`.**

**Reading of the β column:** ref-KL is ON for all GRPO/PAPO_G (β=0.01), OFF for all DAPO/PAPO_D.
Setup sentence, verbatim `[paper §4.1]`: *"Note that GRPO uses a reference KL penalty, while DAPO
removes it and employs dynamic sampling."* → **README's "Qwen3 by default No Reference KL" is stale
and wrong; ref-KL is ON for PAPO_G-2B.**

### 2.3 The canonical `PAPO_G-2B` recipe (paper == code)

The effective config from [`qwen3_vl_2b_grpo_papo.sh`](examples/papo_grpo/qwen3_vl_2b_grpo_papo.sh)
(labeled *"PAPO-G (Config for Table 1 Results)"*) over [`config_grpo_papo.yaml`](examples/configs/config_grpo_papo.yaml):

| Item | Value | Source | Paper Table 3 |
|---|---|---|---|
| perception KL γ (`kl_prcp_coef`) | 0.01 | script L28 `[run/code]` | γ=0.01 ✓ |
| double entropy | **OFF** (`use_aug/ori_entropy_loss` not overridden → `false`) | [config:49,52](examples/configs/config_grpo_papo.yaml#L49) | η=— ✓ |
| ref-KL (`use_kl_loss`, `kl_coef`) | ON, 0.01 | [config:26-28](examples/configs/config_grpo_papo.yaml#L26) | β=0.01 ✓ |
| mask (`patch_size`,`black_prob`) | 14 px, 0.6 | [config:41-42](examples/configs/config_grpo_papo.yaml#L41) | 0.6 ✓ |
| `RECOMPUTE_AUG_LOG_PROBS` | **False** (module constant) | [dp_actor.py:47](verl/workers/actor/dp_actor.py#L47) | (see §2.4b) |
| perception/ref penalty | `low_var_kl` | [config:27,36](examples/configs/config_grpo_papo.yaml#L27) | Schulman k3 ✓ |
| epochs / lr / rollout bs / n | 2 / 1e-6 / 384 / 5 | script + config | §4.1: "2 epochs … 1e-6 … rollout batchsize 384 … n=5" ✓ |
| clip ε_l,ε_h | 0.2, 0.3 (EasyR1 default) | code | 0.2, 0.3 ✓ |
| max_response_length | **2048** (config default, not overridden) | [config:11](examples/configs/config_grpo_papo.yaml#L11) | **not specified in paper** (§2.4e) |

**Conclusion: the code's 2B PAPO_G recipe reproduces the paper's Table-3 `PAPO_G-2B` exactly.**

Approx. **full-run length:** rollout batch 384 prompts/step × 2 epochs over ViRL39K (~39K prompts,
after `filter_overlong_prompts`) ≈ **~200 steps** (exact count depends on the filtered pool size). We
ran **60**. `[obs]`

### 2.4 The practical items where CODE differs from the *written* Eq. 2 (and their justifications)

**(a) Double Entropy is per-model, not always-on.** The written Eq. 2 shows both η terms, but Table 3
sets them per model; η is nonzero only for high-γ 7B and for the no-ref-KL PAPO_D. Justification is in
the paper itself:
- `[paper §5.2]`, verbatim: *"In settings without a reference KL penalty (including PAPO_D), γ needs
  to be set more conservatively (e.g., 0.01), and Double Entropy Loss is indispensable (see Figure
  14)."*
- `[paper Fig. 14 caption]`: *"Double Entropy Loss is indispensable for stabilizing training in this
  setting"* — "this setting" = **without reference KL**.
- `[paper Table 4 / Appendix F]`: *"For PAPO_G, we add a Double Entropy Loss with a coefficient of
  0.03 …"* — this appears **only** in the controlled **reference-KL-removal** experiment.
- **Implication:** with ref-KL present (canonical PAPO_G-2B), Double Entropy is **not** used. Our
  turning it on is off-spec for the 2B.

**(b) `RECOMPUTE_AUG_LOG_PROBS` — the detach approximation.** `[code]` [dp_actor.py:47](verl/workers/actor/dp_actor.py#L47) default `False`.
- With **False**: the masked-image log-probs (`aug_log_probs`) are computed **once at rollout**, detached
  ([`fsdp_workers.py:682-700 compute_log_probs_aug`](verl/workers/fsdp_workers.py#L682)). Consequences:
  the perception-KL gradient flows **only through the real branch** `π_θ` (not `π_θ^mask`), and the
  η₂ `H[π_θ^mask]` term becomes a **zero-gradient no-op**.
- With **True**: an extra grad-enabled masked forward runs inside the update loop
  ([`_forward_micro_batch_aug`, dp_actor.py:166](verl/workers/actor/dp_actor.py#L166)) → full-gradient
  perception KL through both branches + active η₂.
- **Justification for the default (False)** — **only in the repo README, NOT in the paper**
  `[readme]`: *"In theory … we need to do an additional forward pass on the masked sequence to
  recompute the aug_log_probs. In practice, we find that whether doing this additional forward pass
  does not significantly affect the performance. Thus, by default … we skipped the recomputation,
  which still empirically brings slight improvement over single entropy."* A switch is provided "if one
  requires the explicit impact on the gradients from the aug_log_probs."
- **GitHub issue #20** (README-cited): the opening post (user *zsxm1998*) raises exactly this — that
  `aug_entropy_loss` computed under detached log-probs is ineffective (no gradient). **UNVERIFIED:** I
  could not retrieve the maintainer's reply (no `gh` CLI in this env; the web fetch returned only the
  opening post). The README quote above is the authors' documented position.
- **The paper never discusses recompute/detach** `[paper]`: greps for "recompute / detach /
  stop-grad / without gradient / no_grad" → **NOT FOUND**. The paper's Appendix O only reports the
  *cost* of the (single) "additional forward pass on the rollout sequences with a corrupted visual
  input" (Table 10: 3B PAPO_G +48.8 s/step, 7B +49.7 s/step) — i.e., the paper's reported PAPO_G runs
  used the code default (one masked forward; RECOMPUTE=False).
- **Net for our study:** for a *C-pure* PAPO_G-2B (no double entropy), RECOMPUTE only changes whether
  the perception-KL gradient also flows through the masked branch; the authors report this is not
  significant. The RECOMPUTE=True flip we made mainly mattered for making η₂ actually train — a term
  the 2B config does not even use.

**(c) Loss averaging: token vs sequence.** Eq. 2 is written per-sequence (`(1/G)Σ(1/|o_i|)Σ_t`); the
code default is global **token**-level (`loss_avg_mode="token"`, [average_loss usage
dp_actor.py:388](verl/workers/actor/dp_actor.py#L388)). `[paper]` states for DAPO/PAPO_D:
*"token-level loss averaging enabled."* Both our arms used token; internally consistent and matches the
authors' code default. Minor; not a variable we introduced.

**(d) Reference KL.** README says "Qwen3 by default No Reference KL"; **the code config
([config:26](examples/configs/config_grpo_papo.yaml#L26) `use_kl_loss: true`) and the paper (§4.1,
Table 3 β=0.01) both keep it ON for GRPO/PAPO_G.** Treat the README line as stale.

**(e) Max response length.** `[paper]` does not specify it (grep "response length" → NOT FOUND). Code
default 2048 `[config:11]`. **We deliberately used 8192** for full reasoning chains — a documented
deviation from the *code default*, not from the *paper*.

### 2.5 Correction to the earlier "C+DE = faithful to Eq. 2" decision

[`README_FIX.md`](README_FIX.md) §"FINAL OBJECTIVE DECISION" argued C+DE is faithful to the written
Eq. 2. **The 2026-07-27 audit supersedes that:** the paper instantiates Eq. 2 per-model via Table 3,
and `PAPO_G-2B` = γ0.01 / **η none** / β0.01 / RECOMPUTE-default. Therefore:
- **Faithful PAPO_G-2B = "C-pure"** (perception KL γ=0.01, ref-KL on, **no** double entropy,
  RECOMPUTE=False) — the authors' unmodified `qwen3_vl_2b_grpo_papo.sh`.
- **Our "C+DE" run is a deliberate off-spec variant** (adds double entropy + RECOMPUTE=True). Still a
  valid experiment, but it is **not** the paper's 2B recipe and should not be described as such.

---

### 2.6 Bug B provenance — a branch-specific port regression (verified 2026-07-27)

Bug B (perception/entropy/sft orphaned from `.backward()`) is **NOT** in the authors' primary code —
it exists **only in the `main_qwen3` branch** (Qwen3-VL-2B, our model), introduced by the port.

- **`main` branch (Qwen2.5-VL 3B/7B — the paper's primary models): CORRECT.** `[code, fetched
  2026-07-27]` L318 `pg_loss = pg_loss + kl_loss*coef` → … → L400 `loss = pg_loss /
  gradient_accumulation; loss.backward()`. `loss` is defined **once at the end** from the fully
  accumulated `pg_loss` ⇒ perception/entropy **do** train. The paper's Table-1 3B/7B numbers come from
  this correct code.
- **`main_qwen3` branch (our clone `1263a29`): BUGGY.** `loss` bound early (L347) before the perception
  additions (L399+); L431-432 re-scale and backprop the stale `loss`. **git blame:** L347/351/431 are
  all commit **`961291f2` "add qwen3 training code" (Sofia Stoica, 2026-02-03)**, merged via **PR #25
  "Main qwen3"**. The port moved the `loss =` assignment to the wrong place = a regression.
- **Current upstream `main_qwen3` STILL has the bug** (raw file fetched 2026-07-27: L347 `loss =
  pg_loss + …`, L432 `loss.backward()`). **UNFIXED.**
- **No GitHub issue/PR addresses it** (scanned all 28). Nearest: **#20 "aug_entropy_loss"** is a
  *different, narrower* gap — the masked-branch η₂ no-op from detached `aug_log_probs` (the RECOMPUTE
  topic); maintainer **confirmed** it ("the aug_entropy_loss does not directly affect the gradients")
  and added the `RECOMPUTE_AUG_LOG_PROBS` switch, all in the **Qwen2.5/`main`** context where the
  perception KL itself *does* backprop. **#14 "training reward does not match the paper"** → maintainer
  attributed to a **hyperparameter** (γ=0.02 per Table 5), not the backprop bug.
- **Implication (measured):** the public `main_qwen3` — the only released 2B code — trains **GRPO, not
  PAPO** as-is. Yet the paper reports **PAPO_G-2B 51.36 vs GRPO-2B 46.84 (+4.52)**. So their actual 2B
  runs used a corrected/internal version, or the port introduced this regression after those runs — I
  cannot determine which from here. **Either way the public 2B code is broken and the B-fix is
  required.** Our B-fix restores exactly the authors' own **`main`-branch** behavior (backprop the
  accumulated `pg_loss`), keeping only the qwen3 port's token-normalization — so it is a correction
  toward the authors' correct implementation, not an invention.

## 3. What we actually ran (two completed arms)

Both arms: `config_grpo_papo.yaml`, `max_response_length=8192`, `max_steps=60`, `kl_prcp_coef=0.01`,
`save_freq=10` (checkpoints 10–60 present on disk for both), val = MMK12 test, n=8 @ temp 1.0.

### 3.1 Arm A — "baseline" = effectively GRPO + ref-KL  ·  final val **0.540**
- Script [`PAPO_clone/runs/papo_2b_8k_run.sh`](../PAPO_clone/runs/papo_2b_8k_run.sh); code dir =
  **PAPO_clone (buggy `dp_actor`)**; `RUN_DIR=papo_2b_8k_run`; `experiment_name=papo_qwen3vl2b_8k_de_run`.
- Nominal config had `use_kl_prcp=true`, `use_aug/ori_entropy_loss=true`, **but bug B orphaned every
  PAPO term** (perception KL, both entropies, sft never backpropped — see §4). **Effective objective =
  GRPO + ref-KL only.** This makes it a clean control **and** a faithful reproduction of the paper's
  `GRPO-2B` (γ—, η—, β0.01). `[obs]` base val 0.253 → final val **0.540**.

### 3.2 Arm B — PAPO "C+DE" (fixed)  ·  final val **0.466**
- Script [`runs/papo_2b_8k_papofix_run.sh`](runs/papo_2b_8k_papofix_run.sh); code dir =
  **PAPO_fixed** (B+C fixes + `RECOMPUTE_AUG_LOG_PROBS=True`); `RUN_DIR=papo_2b_8k_papofix_run`.
- Effective objective = GRPO(token) + ref-KL(0.01) + **perception KL γ=0.01 (full-gradient)** +
  **Double Entropy η₁=η₂=0.03 (active)** + mask 14/0.6. `[obs]` base val 0.254 → final val **0.466**.
- `KL_prcp` trajectory `[obs]`: peaked ~0.054 (step 2) → plateau ~0.033–0.037 (steps 20–60), did **not**
  collapse to 0; masked-branch entropy rose 0.50→0.61 (penalty outcompeted by reward).

### 3.3 Result read (honest)
- **PAPO C+DE (0.466) < GRPO baseline (0.540)** by ~0.074 on clean val — the *opposite* of the paper's
  headline (PAPO > GRPO). Train acc was comparable (~0.6 both); PAPO generalized slightly worse.
- **Most likely causes** (not a code bug — code verified): (i) **C+DE is over-regularized for a 2B**
  — double entropy is off-spec here per Table 3; (ii) **short training** (60 vs ~200 steps); (iii)
  RECOMPUTE=True adds an off-spec grad path (η₂) the 2B recipe doesn't call for.
- **Caveat:** val is a single point at step 60 (n=8 × ~2000 prompts); the ~0.074 gap is likely real but
  not multi-seed. Neither arm is the paper's canonical PAPO_G-2B, so this is **not** a paper-repro
  verdict — it is "GRPO baseline vs an off-spec PAPO+DE variant."

---

## 4. Journey / decision history (condensed, for context)

1. **4B attempt → memory wall.** Qwen3-VL-4B + long chains hit vLLM sleep/wake OOM
   (`cumem_allocator.cpp:112`) and a 32.6 GiB logits OOM on GH200. **Pivoted to 2B** (PAPO's own
   validated model). Resolved the memory class of problems.
2. **Full chains.** Set `max_response_length=8192` (vs code default 2048), `micro_batch=1/4`,
   `max_num_batched_tokens=16384`, `gpu_memory_utilization=0.40`. Chose a **60-step** trajectory
   (constant LR ⇒ resuming/extending is exact; 60 = signal-check length, extendable).
3. **Bug B (critical, training).** In `update_policy`, `loss` was bound at the ref-KL line
   (`loss = pg_loss + kl_loss*kl_coef`) **before** perception/entropy/sft were added to `pg_loss`;
   `.backward()` ran on `loss` ⇒ all PAPO terms orphaned. ⇒ every pre-fix run trained plain GRPO+refKL.
   Fixed in `PAPO_fixed/dp_actor.py` (fold everything into `pg_loss`, backprop that).
4. **Bug C (logging only).** Four aux-loss metrics were **assigned** (last micro-batch) not appended;
   noisy curves, **no training effect**. Fixed to `append_to_dict`.
5. **RECOMPUTE flip** `False→True` to activate η₂ / full-gradient perception KL (see §2.4b).
6. **"C+DE" decision** to run full Eq. 2 with double entropy — **now known to be off-spec for 2B (§2.5)**.
7. Ran Arm A (baseline, buggy = GRPO+refKL) and Arm B (fixed C+DE). Both complete, checkpoints 10–60.

Discipline notes honored throughout: sign-off before result-affecting runs; cross-check code 2–3×;
prove numbers with smokes/diagnostics; scripts as committed files; cluster transfer via git push→pull.

---

## 5. Offline perception-KL probe — FROZEN SPEC

**Why net-new:** the authors ship **no** perception-KL evaluator (§0.4). `KL_prcp` lives only in the
training loss ([dp_actor.py:360-388](verl/workers/actor/dp_actor.py#L360)); the released `PAPO-Eval`
submodule ([.gitmodules](../PAPO_clone/PAPO/.gitmodules)) is **accuracy-only** ("run_infer.sh /
run_eval.sh", avg acc@8 temp 1.0). The paper's only `KL_prcp` analysis is a **training-time collapse
diagnostic** (Fig. 4 / §5.3: per-token `KL_prcp` variance + GPT-4.1-mini relatedness). We therefore
build the probe but define it to compute the **exact training quantity**.

**Design decisions (user-approved 2026-07-27):** fixed seeded mask bank; on-policy primary + fixed-token control.

| Element | Frozen choice | Mirrors |
|---|---|---|
| Models (13) | shared base (Qwen3-VL-2B-Thinking) + steps {10,20,30,40,50,60} × {Arm A, Arm B}; merged via [`model_merger.py`](scripts/model_merger.py) | — |
| Eval set | fixed seeded subset of `PAPO_MMK12_test`, same indices all models (smoke 16 / full 200) | authors' val set ⇒ acc(real) sanity-checks vs 0.540/0.466 |
| Masks | fixed seeded bank, K masks/image via [`random_patch_blackening(14,0.6)`](verl/trainer/papo_utils.py#L17); reused identically across all models | training mask, but seeded (nuisance-var control) |
| **KL_prcp on-policy (PRIMARY)** | each model samples its own real-image response aₜ; per-token `low_var_kl(logπ(aₜ\|real), logπ(aₜ\|mask_k))`; token-mean; avg over K masks & n | [dp_actor.py:360-388](verl/workers/actor/dp_actor.py#L360) + [core_algos.py:626](verl/trainer/core_algos.py#L626) |
| KL_prcp per-token variance | variance of per-token low_var_kl across response tokens | paper collapse diagnostic (Fig. 4/§5.3) |
| KL_prcp fixed-token (CONTROL) | one frozen base-model greedy reference response/prompt, teacher-forced through every model; same low_var_kl | isolates grounding from token-choice |
| acc(real) | grade on-policy real responses with [`qwen3_vl_think.py:compute_score`](examples/reward_function/qwen3_vl_think.py) | — |
| Generation | temp 1.0, n (smoke 2 / full 8), max_new_tokens 8192, vLLM | paper eval protocol |
| Architecture | two-pass (generate→save, then score log-probs) so vLLM & the HF scoring forward never co-reside | avoids the sleep/wake OOM |

**Scope defaults (recommended):** N=200 (full) / 16 (smoke); n=8 / 2; K=4; fixed-token ref = base greedy.
**Decisive output:** `KL_prcp` on-policy vs step, Arm A vs Arm B overlaid. Higher-and-better-maintained
KL_prcp for PAPO = retained grounding; PAPO ≤ baseline on both KL_prcp and acc = C+DE bought no grounding.
**Status:** spec frozen; script not yet written (smoke before any full run).

---

## 6. Corrected recommendations & next steps

1. **Run C-pure `PAPO_G-2B` (the faithful paper reproduction) — ✅ BUILT 2026-07-27, `PAPO_cpure/`,
   smoke pending.** Perception KL γ=0.01, ref-KL on, **no double entropy**, **RECOMPUTE=False**. Code =
   `PAPO_fixed`'s dp_actor with RECOMPUTE reverted to False; functional diff vs released `PAPO_clone`
   = **exactly the B+C fixes** (verified `diff` + `py_compile`). The B backprop fix is **required**:
   the released code orphans the perception term (bug B), so unmodified authors' code + entropy-off =
   Arm A (GRPO), not PAPO. Scripts: `PAPO_cpure/runs/papo_2b_8k_cpure_{smoke,run}.{sh,sbatch}` (8192,
   60 steps, util 0.55, matches Arm A's memory profile). See [`PAPO_cpure/README_CPURE.md`](../PAPO_cpure/README_CPURE.md).
   **Gate: smoke (no OOM + kl_prcp MAXIMIZED + entropy-off config dump) before the full run.**
2. **Build + smoke the offline perception-KL probe (§5)** on the two arms we already have — the
   decisive mechanistic question (does PAPO hold higher grounding than GRPO across the trajectory?).
3. Consider longer training (toward ~200 steps) if the 60-step signal warrants it.

---

## Appendix A — paper quote bank (verbatim, arXiv:2507.06448 HTML, fetched 2026-07-27)

- **§4.1 Setup:** *"We train all models on ViRL39K for 2 epochs using a learning rate of 1e-6. We
  perform direct RL training from Qwen2.5-VL-3B, 7B and Qwen3-VL-2B, comparing the standard GRPO and
  DAPO baselines with our proposed variants, PAPO_G and PAPO_D. Note that GRPO uses a reference KL
  penalty, while DAPO removes it and employs dynamic sampling."*
- **Table 3 caption:** *"Hyperparameter configurations for models in Table 1."* (values in §2.2)
- **§5.2:** *"In settings without a reference KL penalty (including PAPO_D), γ needs to be set more
  conservatively (e.g., 0.01), and Double Entropy Loss is indispensable (see Figure 14)."*
- **Fig. 14 caption:** *"Impact of KL_prcp weighting (γ) under settings without reference KL. Double
  Entropy Loss is indispensable for stabilizing training in this setting."*
- **Table 4 (Appendix F):** *"Controlled experiments on reference KL removal. For PAPO_G, we add a
  Double Entropy Loss with a coefficient of 0.03 for both 3B and 7B models."*
- **Entropy estimator:** *"The entropy values for the Double Entropy Loss are implemented as
  H[π_θ]=log π_θ(o|q,I), H[π_θ^mask]=log π_θ(o|q,I^mask)."*
- **Appendix O (Table 10):** additional forward-pass cost — *"3B GRPO 360.9 / PAPO_G 428.1 (+48.8);
  7B GRPO 258.5 / PAPO_G 367.1 (+49.7)"* seconds/step.
- **Recompute/detach:** **NOT FOUND** in the paper.
- **Max response length:** **NOT FOUND** in the paper.

## Appendix B — key code references (PAPO_clone/PAPO @ main_qwen3 1263a29)

- Perception-KL loss + entropy + sft composition: [`dp_actor.py:340-429`](verl/workers/actor/dp_actor.py#L340)
- `RECOMPUTE_AUG_LOG_PROBS`: [`dp_actor.py:47`](verl/workers/actor/dp_actor.py#L47);
  masked grad-forward [`dp_actor.py:166`](verl/workers/actor/dp_actor.py#L166)
- KL estimators (`low_var_kl` k3): [`core_algos.py:615-635`](verl/trainer/core_algos.py#L615)
- Masking `random_patch_blackening`: [`papo_utils.py:17-30`](verl/trainer/papo_utils.py#L17)
- Rollout masked-image generation: [`ray_trainer.py:608-620`](verl/trainer/ray_trainer.py#L608)
- Detached masked log-probs at rollout: [`fsdp_workers.py:682-700`](verl/workers/fsdp_workers.py#L682)
- 2B PAPO_G Table-1 script: [`examples/papo_grpo/qwen3_vl_2b_grpo_papo.sh`](examples/papo_grpo/qwen3_vl_2b_grpo_papo.sh)
- PAPO config: [`examples/configs/config_grpo_papo.yaml`](examples/configs/config_grpo_papo.yaml)
- GRPO baseline config: [`examples/configs/config_grpo.yaml`](examples/configs/config_grpo.yaml)
- Reward: [`examples/reward_function/qwen3_vl_think.py`](examples/reward_function/qwen3_vl_think.py)
