"""The trainer fork: caption generation, S11 scoring, and advantage composition.

THE CENTRAL IMPLEMENTATION IDEA. The caption term is a policy gradient on
``log pi(c | I, x, q_cap)`` -- a *different sequence* from the answer rollout ``y``. Rather
than modify verl's loss, we append the caption sequences to the same training batch with
their own advantages. ``update_policy`` (``dp_actor.py:219-229``) reads ``advantages``,
``old_log_probs`` and ``response_mask`` and does not care which rows are answers and which
are captions. So ``J = J_success + lambda*J_cap`` is expressed entirely as **batch
composition**, and verl's clipping, KL-to-reference and optimiser are reused byte-for-byte
instead of reimplemented -- so they cannot be reimplemented wrong. Nothing in verl is
modified. A useful consequence: ``lambda = 0`` recovers Arm A exactly, through no separate
code path (tested).

WHY fit() IS FORKED RATHER THAN HOOKED. The caption block must run between
``compute_advantage`` and ``update_actor`` (``ray_trainer.py:640-663``).
``compute_advantage`` is a module-level function, not a method, so there is nothing to
override and no hook at that point. Forking carries drift risk, so
``assert_upstream_fit_unchanged`` pins the upstream source and fails loudly if verl's
``fit`` is edited underneath us -- the same discipline as the container gate.

FACTS READ FROM SOURCE THAT THIS FILE DEPENDS ON. Each was checked, not assumed; each would
have produced a running-but-wrong pipeline if guessed:

  * ``uid`` is the group key and survives ``_balance_batch``'s reordering, which the source
    itself warns about at ``ray_trainer.py:597``. Rows are grouped by uid, never by position.
  * ``union`` merges ``meta_info`` (``protocol.py:501``) -- that is how ``temperature``
    reaches ``update_actor``, which never sets it itself.
  * ``concat`` keeps only ``data[0].meta_info`` (``protocol.py:606``), so
    ``global_token_num`` goes stale on concatenation and is recomputed below.
  * ``multi_modal_inputs`` is DERIVED inside the worker from ``multi_modal_data`` + ``uid``
    (``fsdp_workers.py:517-562``); caption rows carry those two and nothing more.
  * The sighted context for scoring is **already in the batch**: row ``i`` is
    ``[sighted prompt][y_i]``, exactly the sequence S11 needs. Rebuilding it would risk
    differing from the sequence the rollout actually came from.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any

CAPTION_UID = "uid"

#: The only keys update_policy consumes (dp_actor.py:223-225), plus what the worker needs to
#: derive multi_modal_inputs. Both halves of the training batch are narrowed to exactly this
#: set before concatenation: DataProto.concat requires identical keys, and carrying an
#: answer-only field like token_level_scores onto caption rows would invent a reward for them.
_TRAIN_TENSOR_KEYS = ("input_ids", "attention_mask", "position_ids",
                      "responses", "response_mask", "advantages")
_TRAIN_NONTENSOR_KEYS = ("uid", "multi_modal_data")


def assert_upstream_fit_unchanged(trainer_cls, expected_sha256: str | None):
    """Fail loudly if verl's fit() has drifted from the version this fork was derived from.

    A silent upstream change is the failure mode this project keeps hitting: the code keeps
    running and produces plausible numbers computed from the wrong flow.
    """
    src = inspect.getsource(trainer_cls.fit)
    got = hashlib.sha256(src.encode()).hexdigest()
    if expected_sha256 is None:
        return got  # first run records it
    if got != expected_sha256:
        raise AssertionError(
            f"verl's RayPPOTrainer.fit has changed.\n  expected: {expected_sha256}\n"
            f"  got:      {got}\nThis fork was derived from the pinned version. Re-read the "
            f"upstream diff and re-derive ca21_trainer before training on it.")
    return got


def assert_uid_grouping(uids, n_per_group: int, label: str):
    """Every group must be complete and every row must carry a uid.

    ``ray_trainer.py:597`` warns in-source that ``_balance_batch`` BREAKS ROW ORDER. GRPO
    survives because it groups by ``non_tensor_batch['uid']`` (:486-488, repeated at :513),
    never by position. S12 must do the same. A positional implementation would run clean and
    normalise each caption against the wrong group -- no exception, no wrong-looking log
    line, just a quietly meaningless advantage.
    """
    from collections import Counter

    if any(u is None for u in uids):
        raise AssertionError(f"[{label}] some rows carry no uid; grouping is undefined")
    counts = Counter(uids)
    bad = {u: c for u, c in counts.items() if c != n_per_group}
    if bad:
        raise AssertionError(
            f"[{label}] {len(bad)} uid groups do not have exactly {n_per_group} members, "
            f"e.g. {list(bad.items())[:3]}. Group-relative normalisation over incomplete "
            f"groups is not the quantity S12 defines.")
    return len(counts)


def caption_advantage(distortion, uids, lam: float, eps: float = 1e-6):
    """S12 plus the sign convention, in one place so the sign cannot drift.

    ``J_cap`` MINIMISES distortion, so a caption whose ``D-hat`` is BELOW its group mean must
    receive a POSITIVE advantage. Hence the negation. Backwards, this trains the model to
    make captions maximally UNLIKE the sighted policy while every gate still passes and
    ``D-hat`` climbs smoothly -- which is why a rising ``D-hat`` is a bug hypothesis first.
    """
    from ca21_estimator import group_normalise

    z = group_normalise(distortion, uids, eps=eps)
    return -lam * z


def narrow_for_training(batch, use_ref: bool):
    """Reduce a batch to exactly the keys update_policy consumes."""
    keys = list(_TRAIN_TENSOR_KEYS) + ["old_log_probs"] + (["ref_log_probs"] if use_ref else [])
    missing = [k for k in keys if k not in batch.batch]
    if missing:
        raise KeyError(
            f"rows are missing {missing}; update_policy (dp_actor.py:223-224) requires them "
            f"and would fail far from the cause")
    return batch.select(keys, list(_TRAIN_NONTENSOR_KEYS))


def compose_batch_advantages(answer_batch, caption_batch, use_ref: bool):
    """Concatenate answer and caption rows into ONE policy-gradient batch.

    Caption rows arrive with advantages already signed and scaled by ``caption_advantage``,
    so after this ``update_policy`` needs no knowledge of the caption term at all.
    """
    import torch
    from verl.protocol import DataProto

    a = narrow_for_training(answer_batch, use_ref)
    c = narrow_for_training(caption_batch, use_ref)
    if a.batch["input_ids"].shape[-1] != c.batch["input_ids"].shape[-1]:
        raise AssertionError(
            f"answer sequences are {a.batch['input_ids'].shape[-1]} long and caption "
            f"sequences {c.batch['input_ids'].shape[-1]}; torch.cat would fail. Both must "
            f"use the same max_prompt_length + max_response_length.")

    out = DataProto.concat([a, c])
    # concat keeps data[0].meta_info (protocol.py:606), so global_token_num still describes
    # the answer half only. update_actor reads it for the FLOPs metric (fsdp_workers.py:583).
    out.meta_info = dict(a.meta_info)
    out.meta_info["global_token_num"] = torch.sum(
        out.batch["attention_mask"], dim=-1).tolist()
    return out


def make_ca21_trainer(ray_ppo_trainer_cls, *, expected_fit_sha256: str | None = None):
    """Build the trainer subclass. Factored so this module imports without verl present."""

    class CA21Trainer(ray_ppo_trainer_cls):
        """RayPPOTrainer + the caption term.

        ``config.ca21``: ``lam`` (O6 D3 = 1.0), ``g_c`` (D4 = 8), ``m`` (S13 trajectory cap),
        ``correctness_gate`` (S13/O4 switch).
        """

        # -- config -------------------------------------------------------
        def _ca21_config(self):
            c = getattr(self.config, "ca21", None)
            if c is None:
                raise AssertionError(
                    "config.ca21 is absent. The caption term is the whole method; running "
                    "without it would be Arm A wearing Arm B's name.")
            for k in ("lam", "g_c", "m"):
                if getattr(c, k, None) is None:
                    raise AssertionError(f"config.ca21.{k} is unset (O6)")
            if c.g_c < 2:
                raise AssertionError(
                    f"g_c={c.g_c}: a group of one has no within-group comparison, so S12's "
                    f"advantage is identically zero and the caption term does nothing.")
            return c

        # -- S13: which trajectories score the captions --------------------
        def _select_trajectories(self, batch, m: int, correctness_gate: bool):
            """Return ``(uids_in_order, row_indices)`` -- the shared ``y`` set per prompt.

            Trajectories are FREE: ``J_success`` generated them regardless, so only the
            scoring passes cost. S13 takes the CORRECT subset (mean 2.62 of 8, 4.11),
            capped at ``m`` to bound cost.

            NOTE, and it corrects an earlier claim of mine: under this gate the caption term
            does NOT cover the prompts where ``J_success`` is dead. Dead answer groups are
            all-8-wrong (20.3%) plus all-8-correct (4.7%); the all-wrong ones have no correct
            trajectory, so they drop out here too. The caption term reaches the 4.7%, not
            the 25%.
            """
            scores = batch.batch["token_level_scores"].sum(-1)
            uids = batch.non_tensor_batch[CAPTION_UID]

            groups: dict[Any, list[int]] = {}
            for i, u in enumerate(uids):
                groups.setdefault(u, []).append(i)

            kept_uids, rows, n_dropped = [], [], 0
            for u in sorted(groups, key=str):        # deterministic, order-independent
                idxs = groups[u]
                cand = [i for i in idxs if scores[i] > 0] if correctness_gate else idxs
                if not cand:
                    n_dropped += 1
                    continue
                kept_uids.append(u)
                rows.append(cand[:m])
            return kept_uids, rows, n_dropped

        def fit(self):
            assert_upstream_fit_unchanged(ray_ppo_trainer_cls, expected_fit_sha256)
            raise NotImplementedError(
                "fit() body is derived in the T0 smoke against a running step.")

    return CA21Trainer
