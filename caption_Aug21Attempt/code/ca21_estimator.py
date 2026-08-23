"""S11: the caption-distortion estimator.

    D(c) = KL( pi(.|I,x) || pi(.|c,x) )        [S8: FORWARD]

evaluated per position along trajectories drawn from the SIGHTED policy, then averaged
over the m shared trajectories of S13.

WHAT THIS MODULE IS AND IS NOT. It is pure tensor math over two logit tensors. It knows
nothing about verl, Ray, FSDP, or how the logits were produced. That separation is
deliberate: this is the one place in the pipeline where a wrong sign, a wrong axis, or a
wrong normalisation would be *invisible* -- training would proceed, every metric would
move, and the numbers would be meaningless. So the maths lives somewhere it can be tested
against closed forms and an independent implementation, without a GPU.

WHERE THE REDUCTION HAPPENS, AND WHY IT MATTERS. `distortion_from_logits` is designed to
be called INSIDE the worker that produced the logits. Full logits are `[B, T, 151936]` --
hundreds of MB per micro-batch -- and must never cross the Ray boundary. Both forward
passes happen in one worker method and only the reduced per-sequence scalars come back.

WHY THE EXACT FORM RATHER THAN THE SPEC'S ONE-SAMPLE SUM (spec line 86):
  - Rao-Blackwell: same estimand, variance never worse.
  - It measures what S8 selected for. Forward KL was chosen for mass-covering; the
    one-sample form sees coverage through a single sampled token, so when sighted is torn
    between two continuations it penalises a caption supporting whichever was NOT drawn.
  - `kl >= 0` per position, by construction. A negative value is a PROOF of a bug. The
    one-sample form has no such check -- its per-position terms are legitimately negative
    and only the expectation is non-negative. Given that three of four instrumentation
    failures in this project printed healthy output right up to the moment they failed
    (DECISION_LOG 4.6), a free arithmetic oracle is worth more than the compute it costs.

Both forms are returned. They estimate the same quantity and must agree in expectation;
divergence beyond Monte-Carlo error is a bug signal, not a curiosity.
"""

from __future__ import annotations

from typing import Any

#: Per-position KL below this is treated as a bug rather than fp noise. Reductions run
#: in float32; observed drift on well-conditioned inputs is ~1e-7.
KL_NEGATIVE_TOL = -1e-3


def _lse_chunked(logits, chunk: int):
    """Yield (slice, log_softmax) over the time axis to bound peak memory."""
    import torch  # noqa: F401

    T = logits.shape[1]
    for s in range(0, T, chunk):
        sl = slice(s, min(s + chunk, T))
        yield sl, logits[:, sl].float().log_softmax(dim=-1)


def distortion_from_logits(
    logits_sighted,
    logits_blind,
    response_mask,
    labels=None,
    temperature: float = 1.0,
    chunk: int = 128,
    check_oracle: bool = True,
) -> dict[str, Any]:
    """Exact per-position forward KL, plus diagnostics.

    Args:
        logits_sighted: ``[B, T, V]`` from ``pi(.|I, x)`` -- the reference `p`.
        logits_blind:   ``[B, T, V]`` from ``pi(.|c, x)`` -- the caption-conditioned `q`.
        response_mask:  ``[B, T]``, 1 on positions that count. Prompt and padding are 0.
        labels:         ``[B, T]`` sampled token ids. If given, the spec's one-sample
                        estimator is computed alongside as a cross-check.
        temperature:    applied to BOTH sets of logits, matching the sampling
                        distribution. verl divides logits by temperature before taking
                        log-probs; we mirror that so `D` describes the policy that
                        actually generated `y`, not a sharpened cousin of it.

    Returns per-sequence tensors of shape ``[B]`` (masked means over T):
        kl            the estimator, D-hat
        entropy_p     H(sighted)      -- failure-mode 2 instrument
        entropy_q     H(blind)        -- failure-mode 2 instrument
        cross_pq      H(sighted, blind); kl == cross_pq - entropy_p identically
        one_sample    the spec's estimator, if `labels` given
    and scalars: ``kl_min_position`` (the oracle's witness) and ``n_positions``.

    Raises:
        AssertionError: if any per-position KL is negative beyond tolerance, or if any
            output is non-finite. Both are proofs of a bug, not degraded estimates.
    """
    import torch

    if logits_sighted.shape != logits_blind.shape:
        raise AssertionError(
            f"logit shapes differ: sighted {tuple(logits_sighted.shape)} vs blind "
            f"{tuple(logits_blind.shape)}. Both must be scored on the SAME trajectory at "
            f"the same positions -- differing shapes mean the contexts were not aligned.")
    B, T, _ = logits_sighted.shape
    if tuple(response_mask.shape) != (B, T):
        raise AssertionError(
            f"response_mask {tuple(response_mask.shape)} does not match [B,T] = {(B, T)}")

    mask = response_mask.float()
    dev = logits_sighted.device
    acc = {k: torch.zeros(B, device=dev, dtype=torch.float32)
           for k in ("kl", "entropy_p", "entropy_q", "cross_pq", "one_sample")}
    kl_min = torch.tensor(float("inf"), device=dev)

    inv_t = 1.0 / temperature
    for sl, log_p in _lse_chunked(logits_sighted * inv_t, chunk):
        log_q = (logits_blind[:, sl] * inv_t).float().log_softmax(dim=-1)
        m = mask[:, sl]

        p = log_p.exp()
        # Written as H(p,q) - H(p) rather than sum p*(log_p - log_q): algebraically
        # identical, but it yields both entropies for free, and they are exactly the
        # quantities failure mode 2 (over-dispersion under forward KL) needs.
        h_p = -(p * log_p).sum(-1)
        h_pq = -(p * log_q).sum(-1)
        kl_pos = h_pq - h_p

        if check_oracle and m.any():
            observed = kl_pos.masked_fill(m == 0, float("inf")).min()
            kl_min = torch.minimum(kl_min, observed)

        acc["kl"] += (kl_pos * m).sum(-1)
        acc["entropy_p"] += (h_p * m).sum(-1)
        acc["entropy_q"] += (-(log_q.exp() * log_q).sum(-1) * m).sum(-1)
        acc["cross_pq"] += (h_pq * m).sum(-1)

        if labels is not None:
            lab = labels[:, sl].unsqueeze(-1).clamp_min(0)
            lp = log_p.gather(-1, lab).squeeze(-1)
            lq = log_q.gather(-1, lab).squeeze(-1)
            acc["one_sample"] += ((lp - lq) * m).sum(-1)

    n = mask.sum(-1)
    if (n == 0).any():
        raise AssertionError(
            f"{int((n == 0).sum())} sequence(s) have an all-zero response_mask; a "
            f"distortion over zero positions is undefined, not zero.")

    # THE ORACLE. Per-position forward KL is >= 0 by construction, so a negative value
    # cannot be a bad estimate -- it can only be a bug (swapped arguments, a mask
    # misalignment, log-probs that are not normalised). Checked, never assumed.
    if check_oracle and torch.isfinite(kl_min) and kl_min < KL_NEGATIVE_TOL:
        raise AssertionError(
            f"KL ORACLE VIOLATED: min per-position KL = {kl_min.item():.6g} < "
            f"{KL_NEGATIVE_TOL}. Forward KL is non-negative by construction, so this is "
            f"a bug, not noise. Check argument order (sighted first, blind second), that "
            f"response_mask lines up with the response positions, and that both logit "
            f"tensors describe the SAME trajectory.")

    out = {k: v / n for k, v in acc.items()}
    if labels is None:
        out.pop("one_sample")
    for k, v in out.items():
        if not torch.isfinite(v).all():
            raise AssertionError(f"non-finite values in {k!r}: {v}")

    out["kl_min_position"] = kl_min
    out["n_positions"] = n
    return out


def average_over_trajectories(per_traj: list, weights: list | None = None):
    """S13: average m shared trajectories into one score per caption.

    ``per_traj[k]`` is the ``[G]`` distortion of every caption in the group against
    trajectory k. With `weights` (e.g. 1 for correct trajectories, 0 otherwise) this is
    O4's correctness gate -- which CHANGES THE ESTIMAND to E[D(c) | R(y)=1], recorded as
    a deliberate deviation in DECISION_LOG S13.
    """
    import torch

    stack = torch.stack(per_traj, dim=0)                       # [m, G]
    if weights is None:
        return stack.mean(dim=0)

    w = torch.as_tensor(weights, dtype=stack.dtype, device=stack.device).view(-1, 1)
    if w.shape[0] != stack.shape[0]:
        raise AssertionError(
            f"{w.shape[0]} weights for {stack.shape[0]} trajectories")
    total = w.sum()
    if total <= 0:
        raise AssertionError(
            "all trajectory weights are zero -- the caller must skip this item rather "
            "than divide by zero. With O4 gating this means no sighted rollout was "
            "correct, which DECISION_LOG 4.8 measures at 26.0% of items.")
    return (stack * w).sum(dim=0) / total


def group_normalise(scores, group_index, eps: float = 1e-6):
    """S12: z-score within group, the baseline the spec does not specify.

    Deviates from spec line 113 (raw `D-hat` as the REINFORCE coefficient). Plain
    REINFORCE with a large non-zero-mean coefficient is the variance failure baselines
    exist to fix, and `D-hat` is exactly such a coefficient.

    This is also what makes S8's cancellation real: under S13's shared trajectories the
    `-H(sighted)` term is identical across a group, so centring removes it EXACTLY, and
    the advantage depends only on each caption's cross-entropy against sighted.
    """
    import torch

    if scores.shape[0] != len(group_index):
        raise AssertionError(
            f"{scores.shape[0]} scores but {len(group_index)} group ids")

    out = torch.zeros_like(scores)
    buckets: dict[Any, list[int]] = {}
    for i, g in enumerate(group_index):
        buckets.setdefault(g, []).append(i)

    for idx in buckets.values():
        sel = torch.tensor(idx, device=scores.device)
        vals = scores[sel]
        # A single-member group has no within-group comparison to make. Its advantage is
        # zero -- not "the score itself", which would leak the raw scale back in and
        # silently reintroduce exactly what S12 removes.
        out[sel] = (vals - vals.mean()) / (vals.std() + eps) if len(idx) > 1 else 0.0
    return out
