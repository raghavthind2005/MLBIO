"""The trainer fork: caption generation, S11 scoring, and advantage composition.

THE CENTRAL IMPLEMENTATION IDEA. The caption term is a policy gradient on
``log pi(c | I, x, q_cap)`` -- a *different sequence* from the answer rollout ``y``. Rather
than modify verl's loss, we append the caption sequences to the same training batch with
their own advantages. ``update_policy`` (``dp_actor.py:219``) then treats them uniformly:
it reads ``advantages``, ``old_log_probs``, ``response_mask`` and does not care which rows
are answers and which are captions. So ``J = J_success + lambda*J_cap`` is expressed
entirely as **batch composition**, and verl's optimiser, clipping and KL-to-reference are
reused byte-for-byte instead of reimplemented. Nothing in verl is modified.

WHY fit() IS FORKED RATHER THAN HOOKED. The caption block must run between
``compute_advantage`` and ``update_actor`` (``ray_trainer.py:640-663``). ``compute_advantage``
is a module-level function, not a method, so there is nothing to override; there is no hook
at that point. Forking is the honest option -- and it carries drift risk, so
``assert_upstream_fit_unchanged`` below pins the upstream source and fails loudly if verl's
``fit`` is ever edited underneath us. Same discipline as the container gate (G-VITATTN).

THE SIGHTED PASS IS COMPUTED ONCE PER PROMPT, NOT ONCE PER CAPTION. This is not only an
optimisation, it is required for S13's exactness claim. All ``G_c`` captions of a prompt are
scored against the SAME sighted trajectories, so the ``-H(sighted)`` term is *numerically*
identical across the group and S12's centring removes it EXACTLY. Recomputing the sighted
forward per caption would leave floating-point differences that silently weaken a
cancellation the design treats as exact.

    cost per step:  sighted forwards = P * m           (shared -- the CRN saving)
                    blind   forwards = P * G_c * m     (each caption needs its own)
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any

# Pinned so an upstream edit to fit() cannot silently change what we forked from.
# Regenerate deliberately, never to make a failure go away: the point of the pin is that
# somebody re-reads the diff.
UPSTREAM_FIT_SHA256 = None  # set by pin_upstream_fit() on first run; see runs/ notes

CAPTION_ROLE = "ca21_role"      # "answer" | "caption", carried per row
CAPTION_UID = "uid"             # verl's own group key -- see assert_uid_grouping


def assert_upstream_fit_unchanged(trainer_cls, expected_sha256: str | None):
    """Fail loudly if verl's fit() has drifted from the version this fork was derived from.

    A silent upstream change is the failure mode this project keeps hitting (4.6): the code
    keeps running and produces plausible numbers computed from the wrong flow.
    """
    src = inspect.getsource(trainer_cls.fit)
    got = hashlib.sha256(src.encode()).hexdigest()
    if expected_sha256 is None:
        return got  # first run: caller records it
    if got != expected_sha256:
        raise AssertionError(
            f"verl's RayPPOTrainer.fit has changed.\n  expected: {expected_sha256}\n"
            f"  got:      {got}\nThis fork was derived from the pinned version. Re-read the "
            f"upstream diff and re-derive ca21_trainer before training on it.")
    return got


def assert_uid_grouping(uids, n_per_group: int, label: str):
    """Every group must be complete and every row must carry a uid.

    ray_trainer.py:597 warns in-source that ``_balance_batch`` BREAKS ROW ORDER. GRPO
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
        sample = list(bad.items())[:3]
        raise AssertionError(
            f"[{label}] {len(bad)} uid groups do not have exactly {n_per_group} members, "
            f"e.g. {sample}. Group-relative normalisation over incomplete groups is not "
            f"the quantity S12 defines.")
    return len(counts)


def caption_advantage(distortion, uids, lam: float, eps: float = 1e-6):
    """S12 + the sign convention, in one place so the sign cannot drift.

    ``J_cap`` MINIMISES distortion, so a caption whose ``D-hat`` is BELOW its group mean must
    receive a POSITIVE advantage. Hence the negation. Getting this backwards would train the
    model to make captions maximally unlike the sighted policy while every gate still passed
    and ``D-hat`` rose smoothly -- which 8's exit conditions name as a bug hypothesis, not a
    finding, for exactly this reason.
    """
    from ca21_estimator import group_normalise

    z = group_normalise(distortion, uids, eps=eps)
    return -lam * z


def compose_batch_advantages(answer_batch, caption_batch, lam: float):
    """Concatenate answer and caption rows into ONE policy-gradient batch.

    The caption rows arrive with their advantages already signed and scaled by
    ``caption_advantage``. After this, ``update_policy`` needs no knowledge of the caption
    term at all -- which is the point: verl's clipping, KL-to-reference and optimiser are
    reused rather than reimplemented, so they cannot be reimplemented WRONG.
    """
    from verl.protocol import DataProto

    for name, b in (("answer", answer_batch), ("caption", caption_batch)):
        for k in ("advantages", "old_log_probs", "response_mask", "responses"):
            if k not in b.batch:
                raise KeyError(
                    f"[{name}] rows are missing '{k}'; update_policy (dp_actor.py:223-224) "
                    f"requires it and would fail far from the cause")

    return DataProto.concat([answer_batch, caption_batch])


def make_ca21_trainer(ray_ppo_trainer_cls, *, expected_fit_sha256: str | None = None):
    """Build the trainer subclass. Factored so this module imports without verl present.

    Mirrors ``make_ca21_worker``: the training entry point supplies the upstream class, so
    this file stays importable off-cluster and its pure parts stay unit-testable.
    """

    class CA21Trainer(ray_ppo_trainer_cls):
        """RayPPOTrainer + the caption term.

        Config read from ``self.config.ca21``:
            ``lam``            -- lambda in J = J_success + lambda*J_cap  (O6 D3: 1.0)
            ``g_c``            -- captions sampled per prompt             (O6: OPEN)
            ``m``              -- trajectories scored per caption         (S13: correct subset)
            ``correctness_gate`` -- restrict trajectories to R(y)=1       (S13/O4 switch)
        """

        def _ca21_config(self):
            c = getattr(self.config, "ca21", None)
            if c is None:
                raise AssertionError(
                    "config.ca21 is absent. The caption term is the whole method; running "
                    "without it would be Arm A wearing Arm B's name.")
            for k in ("lam", "g_c", "m"):
                if getattr(c, k, None) is None:
                    raise AssertionError(f"config.ca21.{k} is unset (O6)")
            return c

        def _select_trajectories(self, batch, m: int, correctness_gate: bool):
            """S13: choose the shared ``y`` set, once per prompt, for the whole caption group.

            Returns row indices into ``batch``. Trajectories are FREE -- ``J_success``
            generated them regardless -- so only the scoring passes cost, which is why S13
            picks ``m`` from the data (mean 2.62 correct of 8, 4.11) rather than using all 8.
            """
            raise NotImplementedError("wired in the T0 smoke, against a real DataProto")

        def _generate_captions(self, batch, g_c: int):
            """Sample ``g_c`` captions per prompt from pi(.|I, x, q_cap).

            Uses ``build_captioner_messages`` (S2: the captioner sees the FULL question,
            options included). The sampled captions are the ACTIONS the caption term
            optimises, so this must go through the same rollout engine as the answers --
            a caption produced by any other decoding path is not a sample from the policy
            being differentiated.
            """
            raise NotImplementedError("wired in the T0 smoke, against a real rollout engine")

        def _score_captions(self, batch, captions, m_idx):
            """S11 for every caption, reusing ONE sighted pass per prompt.

            Delegates to ``CA21Worker.compute_caption_distortion``; only ``[B]`` scalars
            cross the Ray boundary (see ca21_worker's module docstring).
            """
            raise NotImplementedError("wired in the T0 smoke, against a live worker group")

        def fit(self):
            assert_upstream_fit_unchanged(ray_ppo_trainer_cls, expected_fit_sha256)
            raise NotImplementedError(
                "fit() is forked from the PINNED upstream version. Derive it in the T0 "
                "smoke, where each inserted line can be checked against a running step, "
                "rather than transcribing 130 lines blind.")

    return CA21Trainer
