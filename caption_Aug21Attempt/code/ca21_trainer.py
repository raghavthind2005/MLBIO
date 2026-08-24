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
    (``fsdp_workers.py:503-548``); caption rows carry those two and nothing more.
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
    # the answer half only. update_actor reads it for the FLOPs metric (fsdp_workers.py:569).
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


        # -- caption sampling ---------------------------------------------
        def _generate_captions(self, batch, kept_uids, first_row, g_c):
            """Sample ``g_c`` captions per kept prompt from pi(. | I, x, q_cap).

            Captions are the ACTIONS the caption term optimises, so they must come from the
            same rollout engine as the answers -- a caption produced by any other decoding
            path is not a sample from the policy being differentiated.

            Output rows are caption-interleaved: caption ``j`` of prompt ``pi`` is at
            ``pi * g_c + j`` (vllm_rollout_spmd.py:235-241 repeat-interleaves by n).
            """
            import numpy as np
            import torch
            from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto

            import ca21_prompts as P
            from ca21_contexts import build_prompt_row
            from ca21_dataset import CA21_PROBLEM_KEY

            problems = batch.non_tensor_batch[CA21_PROBLEM_KEY]
            mmdata = batch.non_tensor_batch["multi_modal_data"]
            dcfg = self.config.data

            stacks = {"input_ids": [], "attention_mask": [], "position_ids": []}
            raw_ids, mm_out = [], []
            for u in kept_uids:
                i = first_row[u]
                row = build_prompt_row(
                    self.processor, self.tokenizer,
                    P.build_captioner_messages(str(problems[i])),
                    mmdata[i]["images"], dcfg.max_prompt_length,
                    dcfg.min_pixels, dcfg.max_pixels)
                for k in stacks:
                    stacks[k].append(row[k])
                raw_ids.append(row["raw_prompt_ids"])
                mm_out.append(mmdata[i])

            gen = DataProto.from_dict(
                tensors={k: torch.stack(v, dim=0) for k, v in stacks.items()},
                non_tensors={
                    "raw_prompt_ids": np.array(raw_ids, dtype=object),
                    "multi_modal_data": np.array(mm_out, dtype=object),
                },
                meta_info={"min_pixels": dcfg.min_pixels, "max_pixels": dcfg.max_pixels,
                           "video_fps": dcfg.video_fps, "n": g_c},
            )

            # len(kept_uids) is whatever survived the correctness gate -- NOT a multiple of
            # world_size. _make_batch_data gets away without this only because
            # rollout_batch_size happens to divide evenly. _validate does exactly this
            # (ray_trainer.py:412-414), including scaling pad_size by the repeat factor.
            gen, pad = pad_dataproto_to_divisor(gen, self.actor_rollout_ref_wg.world_size)
            out = self.actor_rollout_ref_wg.generate_sequences(gen)
            out = unpad_dataproto(out, pad_size=pad * g_c)

            want = len(kept_uids) * g_c
            if len(out) != want:
                raise AssertionError(
                    f"caption generation returned {len(out)} rows, expected "
                    f"{len(kept_uids)} prompts x g_c {g_c} = {want}. The caption-major "
                    f"indexing below would silently pair captions with the wrong prompts.")
            return out

        # -- the caption block --------------------------------------------
        def _ca21_step(self, batch, metrics):
            """Everything between compute_advantage and update_actor.

            Returns ``(caption_batch, ca21_metrics)`` where caption_batch already carries
            signed, lambda-scaled advantages -- or ``(None, metrics)`` if no prompt survived
            the correctness gate.
            """
            import numpy as np
            import torch
            from verl.protocol import DataProto

            import ca21_prompts as P
            from ca21_contexts import append_responses, build_prompt_row
            from ca21_dataset import CA21_PROBLEM_KEY, assert_problems_present
            from ca21_logging import (advantage_component_metrics, distortion_metrics,
                                      variance_decomposition)

            cfg = self._ca21_config()
            g_c, lam = int(cfg.g_c), float(cfg.lam)
            gate = bool(getattr(cfg, "correctness_gate", True))
            assert_problems_present(batch.non_tensor_batch, len(batch))

            uids = batch.non_tensor_batch["uid"]
            kept_uids, rows, n_dropped = self._select_trajectories(batch, int(cfg.m), gate)
            m_all = {"ca21/prompts_kept": len(kept_uids),
                     "ca21/prompts_dropped_no_correct_traj": n_dropped}
            if not kept_uids:
                return None, m_all

            first_row = {}
            for i, u in enumerate(uids):
                first_row.setdefault(u, i)

            cap_out = self._generate_captions(batch, kept_uids, first_row, g_c)
            cap_txt = self.tokenizer.batch_decode(
                cap_out.batch["responses"], skip_special_tokens=True)

            problems = batch.non_tensor_batch[CA21_PROBLEM_KEY]
            dcfg = self.config.data
            T = batch.batch["responses"].shape[-1]

            # flat (prompt, trajectory) index -- p-major, matching the sighted rows below
            flat = [(pi, ri) for pi, rs in enumerate(rows) for ri in rs]
            N = len(flat)
            sig_idx = torch.tensor([ri for _, ri in flat], dtype=torch.long)

            # The SIGHTED context is already in the batch: row i is [prompt][y_i], exactly
            # the sequence S11 scores. Rebuilding it could differ from the sequence the
            # rollout actually came from.
            sighted = {k: batch.batch[k][sig_idx] for k in
                       ("input_ids", "attention_mask", "position_ids")}
            responses = batch.batch["responses"][sig_idx]
            resp_mask = batch.batch["response_mask"][sig_idx]

            # BLIND, caption-major: caption j occupies blind[j*N:(j+1)*N], row-aligned with
            # the N sighted rows. Tokenise the blind PROMPT once per (prompt, caption) --
            # it does not depend on the trajectory -- then append each trajectory's y.
            b_stack = {"input_ids": [], "attention_mask": [], "position_ids": []}
            for j in range(g_c):
                prompt_rows = []
                for pi, u in enumerate(kept_uids):
                    cap = cap_txt[pi * g_c + j]
                    prompt_rows.append(build_prompt_row(
                        self.processor, self.tokenizer,
                        P.build_answerer_messages(cap, str(problems[first_row[u]])),
                        None, dcfg.max_prompt_length,          # G-BLIND, structurally
                        dcfg.min_pixels, dcfg.max_pixels))
                for k, (pi, _ri) in enumerate(flat):
                    r = prompt_rows[pi]
                    ids, am, pos = append_responses(
                        r["input_ids"].unsqueeze(0), r["attention_mask"].unsqueeze(0),
                        r["position_ids"].unsqueeze(0),
                        responses[k].unsqueeze(0), resp_mask[k].unsqueeze(0))
                    b_stack["input_ids"].append(ids[0])
                    b_stack["attention_mask"].append(am[0])
                    b_stack["position_ids"].append(pos[0])

            score_in = DataProto.from_dict(
                tensors={
                    "sighted_input_ids": sighted["input_ids"],
                    "sighted_attention_mask": sighted["attention_mask"],
                    "sighted_position_ids": sighted["position_ids"],
                    "blind_input_ids": torch.stack(b_stack["input_ids"], dim=0),
                    "blind_attention_mask": torch.stack(b_stack["attention_mask"], dim=0),
                    "blind_position_ids": torch.stack(b_stack["position_ids"], dim=0),
                    "responses": responses,
                    "response_mask": resp_mask,
                },
                non_tensors={
                    "uid": np.array([kept_uids[pi] for pi, _ in flat], dtype=object),
                    "multi_modal_data": batch.non_tensor_batch["multi_modal_data"][
                        sig_idx.numpy()],
                },
                meta_info={"ca21_g_c": g_c, "min_pixels": dcfg.min_pixels,
                           "max_pixels": dcfg.max_pixels, "video_fps": dcfg.video_fps},
            )
            scored = self.actor_rollout_ref_wg.compute_caption_distortion(score_in)
            D = scored.batch["caption_distortion"]                    # [N, g_c]

            # S13: average each caption over ITS prompt's shared trajectories.
            per_caption = torch.zeros(len(kept_uids), g_c)
            counts = torch.zeros(len(kept_uids), 1)
            for k, (pi, _ri) in enumerate(flat):
                per_caption[pi] += D[k]
                counts[pi] += 1
            per_caption = per_caption / counts

            cap_uids = np.array([u for u in kept_uids for _ in range(g_c)], dtype=object)
            assert_uid_grouping(list(cap_uids), g_c, "caption groups")
            adv = caption_advantage(per_caption.reshape(-1), list(cap_uids), lam)

            # ---- durable records BEFORE anything can fail downstream ----
            recs = []
            for k, (pi, _ri) in enumerate(flat):
                for j in range(g_c):
                    recs.append({"uid": str(kept_uids[pi]), "caption_idx": j,
                                 "traj_idx": k, "kl": float(D[k, j])})
            m_all.update({f"ca21/{k}": v for k, v in
                          variance_decomposition(recs).items()})
            m_all.update({f"ca21/{k}": v for k, v in distortion_metrics(
                D.reshape(-1), scored.batch["entropy_sighted"].reshape(-1),
                scored.batch["entropy_blind"].reshape(-1),
                scored.batch["distortion_n_positions"].reshape(-1)).items()})
            m_all["ca21/entropy_sighted_spread_max"] = float(
                scored.batch["entropy_sighted_spread"].max())
            m_all["ca21/kl_min_position"] = scored.meta_info.get("ca21_kl_min_position")
            m_all["ca21/caption_chars_mean"] = float(
                np.mean([len(c) for c in cap_txt]))
            self._ca21_records = recs

            # ---- caption rows become training rows ----
            cap_out.batch["advantages"] = (
                adv.unsqueeze(-1) * cap_out.batch["response_mask"]).to(
                    batch.batch["advantages"].dtype)
            cap_out.non_tensor_batch["uid"] = cap_uids
            cap_out = cap_out.union(self.actor_rollout_ref_wg.compute_log_probs(cap_out))
            if self.use_reference_policy:
                cap_out = cap_out.union(
                    self.actor_rollout_ref_wg.compute_ref_log_probs(cap_out))

            m_all.update({f"ca21/{k}": v for k, v in advantage_component_metrics(
                batch.batch["advantages"][:, 0].tolist(), adv.tolist()).items()})
            return cap_out, m_all

        def fit(self):
            assert_upstream_fit_unchanged(ray_ppo_trainer_cls, expected_fit_sha256)
            raise NotImplementedError(
                "fit() loop transcription is the last piece; _ca21_step above is complete.")

    return CA21Trainer
