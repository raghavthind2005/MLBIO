"""Pulling response-position logits out of verl's packed (padding-free) layout.

THE PROBLEM. S11 needs full next-token distributions from two DIFFERENT contexts --
`pi(.|I,x)` and `pi(.|c,x)` -- compared position by position. verl's actor runs
padding-free by default (`dp_actor.py:_forward_micro_batch`), so logits arrive as
``[total_nnz, V]`` with every sequence's real tokens concatenated and padding removed.
The two contexts have DIFFERENT prompt lengths, so their packed layouts differ and cannot
be compared row-for-row.

WHY NOT JUST PAD BACK. verl pads log-probs back with `pad_input` because they are one
scalar per position. Doing that at full vocab width would materialise
``[B, seqlen, 151936]`` -- at seqlen 16,896 that is ~5 GB per sequence in bf16. The whole
reason the padding-free path exists is to avoid tensors of that shape.

WHAT THIS DOES INSTEAD. Scatter only the RESPONSE positions into ``[B, T, V]``, which is
the smallest tensor that can express the comparison. At T~1,000 that is ~300 MB per
context rather than ~5 GB, and T is bounded by `max_response_length`, not by the prompt.

RAGGED RESPONSES ARE HANDLED, NOT ASSUMED AWAY. A sequence whose response ended early has
fewer than T real positions; those slots stay zero and are reported as invalid in the
returned mask. That mask is exactly the `response_mask` the estimator needs, so validity
is derived from the packing rather than supplied alongside it and trusted to agree.

Every claim here about verl's index arithmetic is checked against a naive
padded-then-sliced reference in the tests -- deliberately, because index bookkeeping is
the kind of thing that produces a plausible tensor of the right shape containing the
wrong rows, and nothing downstream would notice.
"""

from __future__ import annotations


def response_slot_indices(indices, batch_size: int, seqlen: int, response_length: int):
    """Map packed positions to their ``(sequence, response-slot)`` coordinates.

    Args:
        indices: ``[total_nnz]`` flat indices into ``batch * seqlen``, as returned by
            flash-attn's ``unpad_input`` and used by verl's ``pad_input`` round trip.
        response_length: ``T``.

    Returns ``(sel, b_idx, t_idx)`` where ``sel`` is a boolean mask over packed
    positions and ``b_idx``/``t_idx`` are the coordinates of the selected ones.

    The window matches verl exactly. It slices ``[:, -response_length - 1 : -1]`` on the
    padded ``[B, seqlen]`` tensor -- the positions whose next-token prediction IS the
    response -- i.e. ``s`` in ``[seqlen - T - 1, seqlen - 2]`` inclusive. Off-by-one here
    would silently score the wrong tokens.
    """
    b_idx = indices // seqlen
    s_idx = indices % seqlen
    lo = seqlen - response_length - 1
    hi = seqlen - 2
    sel = (s_idx >= lo) & (s_idx <= hi)
    return sel, b_idx[sel], s_idx[sel] - lo


def gather_response_logits(logits_packed, indices, batch_size: int, seqlen: int,
                           response_length: int):
    """Scatter packed logits into ``[B, T, V]``, with a validity mask.

    Returns ``(logits, mask)``: ``[B, T, V]`` and ``[B, T]``. Slots with no corresponding
    packed position -- a response that terminated early -- are left at zero and marked 0
    in the mask. **Use the mask; a zero logit vector is a uniform distribution, not an
    absent one, and would silently contribute a real KL term.**
    """
    import torch

    if logits_packed.dim() != 2:
        raise AssertionError(
            f"expected packed logits [total_nnz, V], got {tuple(logits_packed.shape)}")
    if logits_packed.shape[0] != indices.shape[0]:
        raise AssertionError(
            f"{logits_packed.shape[0]} packed rows but {indices.shape[0]} indices -- "
            f"these must come from the same unpad_input call")

    sel, b_idx, t_idx = response_slot_indices(
        indices, batch_size, seqlen, response_length)

    V = logits_packed.shape[1]
    out = logits_packed.new_zeros((batch_size, response_length, V))
    mask = logits_packed.new_zeros((batch_size, response_length))
    out[b_idx, t_idx] = logits_packed[sel]
    mask[b_idx, t_idx] = 1.0
    return out, mask


def gather_response_logits_reference(logits_padded, attention_mask, response_length: int):
    """Naive reference: pad first, then slice. Only for tests and small tensors.

    This is the implementation the packed version must agree with, written the obvious
    way. It is not used in production precisely because materialising
    ``[B, seqlen, V]`` is what the packed path exists to avoid.
    """
    lo = logits_padded.shape[1] - response_length - 1
    hi = logits_padded.shape[1] - 1
    return logits_padded[:, lo:hi], attention_mask[:, lo:hi].to(logits_padded.dtype)
