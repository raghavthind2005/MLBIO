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
