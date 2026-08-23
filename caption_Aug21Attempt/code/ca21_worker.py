"""The verl worker method that computes S11's caption distortion.

WIRING. verl builds workers by class reference --
``role_worker_mapping = {Role.ActorRolloutRef: ray.remote(FSDPWorker), ...}`` in
``vision_sr1/main.py``. So we subclass ``FSDPWorker`` and hand Ray our class. **No file in
verl is modified**, which keeps the vendored copy a clean upstream checkout and means an
upstream bump does not silently drop our changes.

WHY THE WHOLE REDUCTION HAPPENS HERE. S11 needs full next-token distributions from two
contexts. Those are ``[B, T, 151936]`` tensors -- hundreds of MB each -- and shipping them
through Ray to the driver would dominate step time and could OOM the object store. Both
forward passes run inside this one method and only ``[B]`` scalars come back.

WHAT IS AND IS NOT TESTED. The maths (`ca21_estimator`) and the packed-layout extraction
(`ca21_packing`) are unit-tested to 53 assertions against closed forms and an independent
reference. **This module is the glue, and glue over FSDP + Ray + flash-attn cannot be
unit-tested offline** -- it is the first component in this project whose correctness rests
on a smoke run rather than on a test. It is written to keep that untested surface as thin
as possible: batch construction is a pure function below, and everything else is a
transcription of `dp_actor._forward_micro_batch` stopped one line earlier.

KNOWN DUPLICATION, ACCEPTED KNOWINGLY. ``forward_packed_logits`` re-implements verl's
unpad/rope/ulysses preamble because ``_forward_micro_batch`` gathers to log-probs
internally and never exposes logits. That is ~40 lines of duplicated upstream logic and it
WILL drift if verl is upgraded. Mitigation: `assert_forward_matches_verl` below recomputes
verl's own log-probs from our logits and compares them to `actor.compute_log_prob` on the
same batch. If the duplication ever drifts, that check fails loudly instead of producing
subtly wrong distributions.
"""

from __future__ import annotations

from typing import Any

CA21_TEMPERATURE_KEY = "ca21_temperature"


def build_distortion_batch(sighted: dict, blind: dict, trajectory: dict) -> dict:
    """Assemble one micro-batch carrying BOTH contexts on a shared trajectory.

    Pure and testable. Keys are namespaced rather than merged so that a missing tensor is
    a KeyError here, not a silent fallback to the wrong context somewhere downstream --
    which would compute KL(sighted || sighted) = 0 and look like a perfectly converged
    caption.

    Args:
        sighted: input_ids / attention_mask / position_ids (+ multi_modal_inputs) for
            ``pi(.|I,x)``. Carries the image.
        blind: the same for ``pi(.|c,x)``. Carries no image -- G-BLIND territory.
        trajectory: ``responses`` (the shared ``y`` of S13) and its ``response_mask``.
    """
    required_ctx = ("input_ids", "attention_mask", "position_ids")
    for name, ctx in (("sighted", sighted), ("blind", blind)):
        missing = [k for k in required_ctx if k not in ctx]
        if missing:
            raise KeyError(f"{name} context is missing {missing}")

    if "responses" not in trajectory:
        raise KeyError("trajectory must carry 'responses' -- the shared y of S13")

    resp = trajectory["responses"]
    for name, ctx in (("sighted", sighted), ("blind", blind)):
        if ctx["input_ids"].shape[0] != resp.shape[0]:
            raise AssertionError(
                f"{name} context has batch {ctx['input_ids'].shape[0]} but the "
                f"trajectory has {resp.shape[0]}; both contexts must be scored on the "
                f"SAME trajectories, in the same order")

    # The one invariant that cannot be recovered later: the blind context must not carry
    # an image. If it does, `D` collapses toward zero for every caption and the caption
    # term silently stops measuring anything.
    if "multi_modal_inputs" in blind and blind["multi_modal_inputs"]:
        raise AssertionError(
            "G-BLIND violated: the blind context carries multi_modal_inputs. The "
            "caption-conditioned pass must not see the image.")

    out: dict[str, Any] = {}
    for k in required_ctx:
        out[f"sighted_{k}"] = sighted[k]
        out[f"blind_{k}"] = blind[k]
    if "multi_modal_inputs" in sighted:
        out["sighted_multi_modal_inputs"] = sighted["multi_modal_inputs"]
    out["responses"] = resp
    if "response_mask" in trajectory:
        out["response_mask"] = trajectory["response_mask"]
    return out


def forward_packed_logits(actor_module, input_ids, attention_mask, position_ids,
                          multi_modal_inputs=None, temperature: float = 1.0,
                          padding_free: bool = True):
    """Run the policy and return PACKED logits, stopping before verl gathers them.

    Transcribed from ``dp_actor._forward_micro_batch``, which computes exactly this and
    then immediately reduces to log-probs. We need the distributions, so we stop a line
    earlier. See the module docstring on drift.

    Returns ``(logits, indices, batch_size, seqlen)``. With ``padding_free`` the logits
    are ``[total_nnz, V]`` and ``indices`` maps them back to flat ``b*seqlen``; without
    it, ``[B, seqlen, V]`` and ``indices`` is None.
    """
    import torch
    from einops import rearrange
    from flash_attn.bert_padding import index_first_axis, unpad_input

    batch_size, seqlen = input_ids.shape
    mm = multi_modal_inputs or {}
    if position_ids.dim() == 3:                       # qwen2vl mrope
        position_ids = position_ids.transpose(0, 1)   # (B,4,S) -> (4,B,S)

    if not padding_free:
        out = actor_module(input_ids=input_ids, attention_mask=attention_mask,
                           position_ids=position_ids, **mm, use_cache=False)
        return out.logits.div_(temperature), None, batch_size, seqlen

    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)          # (1, total_nnz)

    if position_ids.dim() == 3:
        position_ids_rmpad = (
            index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
            .transpose(0, 1).unsqueeze(1)
        )
    else:
        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

    out = actor_module(input_ids=input_ids_rmpad, attention_mask=None,
                       position_ids=position_ids_rmpad, **mm, use_cache=False)
    logits = out.logits.squeeze(0)                              # (total_nnz, V)
    logits.div_(temperature)
    return logits, indices, batch_size, seqlen


def assert_forward_matches_verl(logits_packed, indices, batch_size, seqlen,
                                input_ids, response_length, verl_log_probs,
                                atol: float = 1e-3):
    """Guard against drift between our transcription and verl's own forward.

    Recomputes verl's log-probs from OUR logits and compares. If ``forward_packed_logits``
    ever stops matching ``_forward_micro_batch`` -- an upstream bump, a config we did not
    replicate -- this fails loudly rather than yielding distributions that are subtly
    wrong in a way no downstream metric can reveal.
    """
    import torch
    from flash_attn.bert_padding import pad_input

    rolled = torch.roll(
        pad_input(input_ids.unsqueeze(-1).reshape(-1, 1)[indices], indices,
                  batch_size, seqlen).squeeze(-1), shifts=-1, dims=1)
    lp = logits_packed.log_softmax(dim=-1)
    labels = rolled.reshape(-1)[indices].unsqueeze(-1).clamp_min(0)
    gathered = lp.gather(-1, labels).squeeze(-1)
    full = pad_input(gathered.unsqueeze(-1), indices, batch_size, seqlen).squeeze(-1)
    ours = full[:, -response_length - 1:-1]

    if not torch.allclose(ours, verl_log_probs, atol=atol):
        d = (ours - verl_log_probs).abs().max().item()
        raise AssertionError(
            f"forward_packed_logits has drifted from verl's _forward_micro_batch: max "
            f"log-prob difference {d:.6g} > {atol}. Our transcription no longer matches "
            f"upstream; re-derive it before trusting any distortion computed from it.")
    return True


def make_ca21_worker(fsdp_worker_cls, register, dispatch_mode):
    """Build the worker subclass. Factored so this module imports without verl present.

    Call from the training entry point:

        from verl.workers.fsdp_workers import FSDPWorker
        from verl.single_controller.base.decorator import Dispatch, register
        CA21Worker = make_ca21_worker(FSDPWorker, register, Dispatch.DP_COMPUTE_PROTO)
        role_worker_mapping = {Role.ActorRolloutRef: ray.remote(CA21Worker), ...}
    """
    import torch

    from ca21_estimator import distortion_from_logits
    from ca21_packing import gather_response_logits

    class CA21Worker(fsdp_worker_cls):
        """FSDPWorker + the caption-distortion pass."""

        @register(dispatch_mode=dispatch_mode)
        def compute_caption_distortion(self, data):
            """S11 for one micro-batch. Returns per-sequence scalars, never logits."""
            from verl.protocol import DataProto

            assert self._has_actor
            data = data.to(torch.cuda.current_device())
            temperature = data.meta_info.get(
                CA21_TEMPERATURE_KEY, self.config.rollout.temperature)

            responses = data.batch["responses"]
            T = responses.size(-1)
            module = self.actor.actor_module
            padding_free = getattr(self.actor.config, "padding_free", True)

            with torch.no_grad():
                sighted, idx_s, B, S_s = forward_packed_logits(
                    module,
                    data.batch["sighted_input_ids"],
                    data.batch["sighted_attention_mask"],
                    data.batch["sighted_position_ids"],
                    multi_modal_inputs=data.non_tensor_batch.get(
                        "sighted_multi_modal_inputs"),
                    temperature=temperature, padding_free=padding_free)
                lg_s, m_s = gather_response_logits(sighted, idx_s, B, S_s, T)
                del sighted

                blind, idx_b, _, S_b = forward_packed_logits(
                    module,
                    data.batch["blind_input_ids"],
                    data.batch["blind_attention_mask"],
                    data.batch["blind_position_ids"],
                    multi_modal_inputs=None,          # G-BLIND, structurally
                    temperature=temperature, padding_free=padding_free)
                lg_b, m_b = gather_response_logits(blind, idx_b, B, S_b, T)
                del blind

                # Only positions valid under BOTH contexts are comparable.
                mask = m_s * m_b
                if "response_mask" in data.batch:
                    mask = mask * data.batch["response_mask"].to(mask.dtype)

                res = distortion_from_logits(
                    lg_s, lg_b, mask, labels=responses, temperature=1.0)
                del lg_s, lg_b

            out = DataProto.from_dict(tensors={
                "caption_distortion": res["kl"].cpu(),
                "distortion_one_sample": res["one_sample"].cpu(),
                "entropy_sighted": res["entropy_p"].cpu(),
                "entropy_blind": res["entropy_q"].cpu(),
                "distortion_n_positions": res["n_positions"].cpu(),
            })
            return out.to("cpu")

    return CA21Worker
