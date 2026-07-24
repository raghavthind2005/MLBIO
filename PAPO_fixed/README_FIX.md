# PAPO_fixed — corrected loss composition (B + C)

**Purpose:** the *real* PAPO code. The baseline (`PAPO_clone`) is left **untouched** — it is the
GRPO+ref-KL control arm whose checkpoints (`$SCRATCH/runs/papo_2b_8k_run`) match the buggy code.
This folder holds ONLY the changed file: `verl/workers/actor/dp_actor.py`. Everything else is
identical to `PAPO_clone` (same `main_qwen3@1263a29` + the 2 container-compat patches).

## The two bugs (found 2026-07-24, full loss-path audit)

**B — TRAINING bug (critical): perception/entropy/sft losses never backpropagated.**
In `update_policy`'s micro-batch loop, `loss` was bound at the ref-KL line
(`loss = pg_loss + kl_loss*kl_coef`) *before* the Implicit Perception Loss, the aug/ori entropy
losses, and the sft loss were added to `pg_loss`. `.backward()` ran on `loss`, so every term added
to `pg_loss` afterward was orphaned. ⇒ all prior runs trained plain **GRPO + ref-KL**, not PAPO.
(Introduced by the `main_qwen3` port PR#25; the commented line `# pg_loss = pg_loss + kl_loss*...`
shows the original, correct intent.)

**C — LOGGING bug (no effect on training): last-sample instead of mean.**
`metrics` is a `defaultdict(list)`, but `kl_loss / kl_prcp_loss / aug_entropy_loss /
ori_entropy_loss` were **assigned** (`metrics[k] = x.detach().item()`) instead of appended, while
`pg_metrics` go through `append_to_dict`. ⇒ those four logged only the *last* micro-batch (noisy at
micro_batch=1). Training was unaffected (they are detached `.item()` copies that never re-enter the
graph); only the displayed curves were corrupted.

## The fix (7 lines, all in dp_actor.py — see `git log` / diff vs PAPO_clone)

- **B:** ref-KL folds into `pg_loss` (`pg_loss = pg_loss + kl_loss*kl_coef`); removed the stray
  `else: loss = pg_loss`; backprop line changed to `loss = pg_loss * torch.sum(response_mask)/total_response_tokens`.
  Now `pg_loss` accumulates ppo + refKL + (−kl_prcp) + aug_ent + ori_ent + sft, and *that* is
  backpropped. Signs verified: perception KL is **maximized** (−kl_prcp term), entropy penalized,
  ref-KL minimized.
- **C:** the 4 aux-loss metrics use `append_to_dict(...)` → proper micro-batch means.

`python3 -m py_compile` passes.

## Fidelity flip: `RECOMPUTE_AUG_LOG_PROBS = False -> True` (dp_actor.py:47)

**Why:** to make the code **dot-identical to the paper's written objective (Eq. 2)**. The paper's
`+γ D_KL[π_θ‖π_θ^mask]` and `−η₂ H[π_θ^mask]` both have θ in the masked branch. The authors'
**default `False`** uses the precomputed/detached `aug_log_probs`, which makes:
  1. the Double-Entropy **η₂ term a no-op** (zero gradient), and
  2. the perception KL backprop **only through the real branch**.

The authors **document this gap themselves** (repo `README.md` L211-213): *"In theory ... we need to
do an additional forward pass on the masked sequence to recompute the `aug_log_probs`. In practice ...
does not significantly affect the performance,"* and they provide the `RECOMPUTE_AUG_LOG_PROBS` switch
*"if one requires the explicit impact on the gradients from the `aug_log_probs`."*

**Decision (user, 2026-07-24): flip to `True`** — the update loop runs the extra masked grad-forward
(`_forward_micro_batch_aug`), giving the full-gradient perception KL **and** an active η₂ = paper Eq. 2.
- Fidelity target: the paper's **written equation** (not necessarily the authors' *reported runs*,
  which used the `False` default and which they claim is empirically similar).
- **Cost:** +1 masked forward per micro-batch (~1.5-2x compute + more memory) -> memory must be
  re-smoked; expect to keep `micro_batch=1/4` and possibly lower `gpu_memory_utilization`.

Other paper-vs-code items all verified faithful (after B+C): GRPO clip 0.2/0.3, ref-KL β=0.01
(PAPO_G), perception KL direction/estimator/sign, masking random-patch 14px/0.6, no annealing,
uniform KL weighting (`apply_mode=all`), no kl_prcp clipping/token-mask. γ=0.01 & η=0.03 match
PAPO_D (paper PAPO_G-3B uses γ=0.02 — a documented alternative, not used here).

## Apply on the cluster (baseline stays intact)
```bash
cp -r $SCRATCH/code/PAPO_clone $SCRATCH/code/PAPO_fixed          # full copy incl. the 2 patches
cp <synced>/PAPO_fixed/verl/workers/actor/dp_actor.py \
   $SCRATCH/code/PAPO_fixed/verl/workers/actor/dp_actor.py       # overlay the fixed file
```
Then the real-PAPO run script points `PAPO_DIR=$SCRATCH/code/PAPO_fixed` and a fresh `RUN_DIR`.

## Decisive verification after the fix (smoke)
Run a 2-step smoke and confirm the perception KL is now being **maximized**: the *actual*
`kl_prcp` (magnitude of the logged `actor/kl_prcp_loss`, now a proper mean) should **rise / stay
high across steps**, not decay toward 0. That is the direct proof PAPO is finally training —
the opposite of the collapse we saw under the bug.
