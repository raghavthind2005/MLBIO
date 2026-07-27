# PAPO_cpure — the FAITHFUL paper `PAPO_G-2B` reproduction ("C-pure")

**Purpose:** run the paper's *canonical* PAPO_G-2B objective — the third, correct arm. This folder is
**separate** from and leaves untouched:
- `PAPO_clone/` = the released (buggy) code = **Arm A** control (GRPO + ref-KL, val 0.540).
- `PAPO_fixed/` = B+C fixes **+ RECOMPUTE=True + double entropy on** = **Arm B** "C+DE" (val 0.466, OFF-SPEC).

This folder holds **ONLY** the one changed file `verl/workers/actor/dp_actor.py`. Everything else is
inherited from `PAPO_clone` (same `main_qwen3@1263a29` + the 2 container-compat patches).

## What "C-pure" is (grounded)

Paper Table 3 (arXiv:2507.06448), verbatim row: **`Qwen3-VL PAPO_G-2B: γ=0.01, η₁,η₂ = —, Mask 0.6,
β=0.01, ε 0.2/0.3`.** i.e. perception KL only (γ=0.01), **NO double entropy**, **ref-KL kept**. The
authors' released runs used **`RECOMPUTE_AUG_LOG_PROBS=False`** (their default). C-pure reproduces
exactly this — see [`PAPO_MASTER_RECORD.md`](../PAPO_fixed/PAPO_MASTER_RECORD.md) §2.3/§2.5.

| Term | C-pure setting | source |
|---|---|---|
| perception KL γ (`kl_prcp_coef`) | **0.01**, maximized | run script; paper Table 3 γ=0.01 |
| double entropy η₁,η₂ | **OFF** (`use_aug/ori_entropy_loss` NOT overridden → `false`) | config default; paper Table 3 η=— |
| ref-KL β (`use_kl_loss`,`kl_coef`) | **ON, 0.01** | config default; paper Table 3 β=0.01 |
| mask (patch,black_prob) | 14 px, 0.6 | config; paper Mask 0.6 |
| `RECOMPUTE_AUG_LOG_PROBS` | **False** (authors' default; real-branch perception grad) | dp_actor.py:61 |
| penalty (ref & prcp) | `low_var_kl` (Schulman k3) | config |

## The ONE code file: `verl/workers/actor/dp_actor.py`

**Functional diff vs the released `PAPO_clone` = exactly the B (backprop) + C (logging) fixes, and
NOTHING else** (`RECOMPUTE` value is `False` in both). Equivalently: **identical to `PAPO_fixed`
EXCEPT `RECOMPUTE_AUG_LOG_PROBS` reverted `True → False`.** Verified by `diff` + `py_compile`.

**Why the B fix is required even for the "authors' faithful config":** the released code has **bug B**
— `loss` is bound at the ref-KL line (`dp_actor.py:347/351`) *before* the perception term is added to
`pg_loss` (399), and `.backward()` runs on `loss` (432), so **the perception KL is orphaned (zero
gradient)**. Confirmed first-hand (`PAPO_clone/.../dp_actor.py:347-432`). Therefore the *literally
unmodified* authors' code with entropy off trains as **plain GRPO + ref-KL = our Arm A**, NOT PAPO.
C-pure = **B/C fixes + RECOMPUTE=False + double entropy off** ⇒ the perception KL **actually trains**,
through the real branch, with **no** extra masked grad-forward (so memory profile == Arm A).

**Bug B is a `main_qwen3`-only port regression** (commit `961291f2` / PR #25, git-blame-verified): the
authors' **`main` branch (Qwen2.5) is correct** (`loss = pg_loss / gradient_accumulation` at the end,
after all terms), and our B-fix **restores exactly that main-branch behavior**. Upstream `main_qwen3`
remains **unfixed**; **no GitHub issue addresses it** (issue #20 is a *distinct* aug-entropy/RECOMPUTE
gradient gap). See [`PAPO_MASTER_RECORD.md`](../PAPO_fixed/PAPO_MASTER_RECORD.md) §2.6.

Trace (C-pure, per micro-batch), all verified in `PAPO_cpure/.../dp_actor.py`:
- L324 `aug_log_probs = model_inputs.get("aug_log_probs", None)` — detached rollout tensor (RECOMPUTE=False).
- L360 `pg_loss = pg_loss + kl_loss*kl_coef` — ref-KL folded in (B-fix).
- L415 `pg_loss = pg_loss - kl_prcp_loss*kl_prcp_coef` — perception **maximized**, grad via real branch.
- L418 `if use_aug_entropy_loss` / L427 `if use_ori_entropy_loss` — **both False → skipped** (no double entropy).
- L447 `loss = pg_loss * sum(mask)/total_tokens; loss.backward()` — backprops the full `pg_loss` (B-fix).

## The 3-arm comparison (all: 8192 resp, 60 steps, rb384/gb128, MMK12 val n8@t1.0)

| Arm | Code | Objective | Faithful to | Final val |
|---|---|---|---|---|
| **A** baseline | `PAPO_clone` (buggy) | GRPO + ref-KL (PAPO terms orphaned) | paper **GRPO-2B** | 0.540 |
| **B** C+DE | `PAPO_fixed` | GRPO+refKL + perceptionKL + **double entropy** (RECOMPUTE=True) | off-spec (paper Eq.2 template, not 2B row) | 0.466 |
| **C** C-pure | `PAPO_cpure` | GRPO+refKL + perceptionKL (no entropy, RECOMPUTE=False) | paper **PAPO_G-2B** (Table 3) | *pending* |

## Deploy on the cluster (baseline + fixed stay intact)
```bash
cp -r $SCRATCH/code/PAPO_clone $SCRATCH/code/PAPO_cpure           # full copy incl. the 2 patches
cp <synced>/PAPO_cpure/verl/workers/actor/dp_actor.py \
   $SCRATCH/code/PAPO_cpure/verl/workers/actor/dp_actor.py        # overlay the C-pure file
```
Run scripts point `PAPO_DIR=$SCRATCH/code/PAPO_cpure` and a fresh `RUN_DIR=$SCRATCH/runs/papo_2b_8k_cpure_run`.

## Smoke FIRST (debug, ~1.5 h) — success criteria
`sbatch $SCRATCH/code/runs/papo_2b_8k_cpure_smoke.sbatch`. Must show, before any full run:
1. **No OOM** at `gpu_memory_utilization=0.55`, 8192 (peak reserved < 95 GB).
2. **Perception KL is MAXIMIZED:** `actor/kl_prcp_loss` magnitude **rises / stays high** across the 2
   steps (proof perception trains — the opposite of the bug's decay-to-0). Mirrors the C+DE smoke,
   which rose 0.047→0.053.
3. **Double entropy OFF:** config dump shows `use_aug_entropy_loss=false`, `use_ori_entropy_loss=false`,
   `kl_prcp_coef=0.01`, `use_kl_prcp=true`; the `actor/ori_entropy_loss` metric does **not** appear.

## Deliberate deviations (shared by ALL three arms, for comparability — not paper-faithful, documented)
- `max_response_length=8192` (paper unspecified; code default 2048) — for full reasoning chains.
- `max_steps=60` (~0.3 epoch; paper ~200 steps / 2 epochs) — mechanistic-trajectory length; per-step
  objective is byte-identical, resume-exact (constant LR, fixed kl_prcp schedule).
