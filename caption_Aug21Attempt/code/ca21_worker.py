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


def slice_multi_modal_inputs(mm, lo: int, hi: int, n_rows: int):
    """Slice collated ``multi_modal_inputs`` down to sighted rows ``[lo, hi)``.

    Needed because the caption pass is row-chunked (see the memory note in
    ``compute_caption_distortion``). This is NOT a row slice: ``pixel_values`` is
    ``[total_patches, D]`` with every image's patches concatenated end to end, so selecting
    rows means slicing at the cumulative patch offsets implied by ``image_grid_thw``.

    Getting this wrong is silent: a plain ``mm["pixel_values"][lo:hi]`` would hand the model
    the first few patch rows of the first image instead of the images for this chunk, and the
    forward would still return finite logits.
    """
    import torch

    if mm is None:
        return None
    unexpected = set(mm) - {"pixel_values", "image_grid_thw"}
    if unexpected:
        raise AssertionError(
            f"multi_modal_inputs carries {sorted(unexpected)}, which this slicer does not "
            f"know how to chunk (video inputs would need their own offsets). Passing them "
            f"through unsliced would pair the wrong media with each chunk.")
    grid = mm["image_grid_thw"]
    if grid.shape[0] != n_rows:
        raise AssertionError(
            f"{grid.shape[0]} images for {n_rows} sighted rows. Row-chunking assumes exactly "
            f"ONE image per row; under any other mapping the offsets below are wrong.")
    counts = grid.prod(dim=-1)                       # patch rows contributed per image
    starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    return {"pixel_values": mm["pixel_values"][int(starts[lo]):int(starts[hi])],
            "image_grid_thw": grid[lo:hi]}


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
            """S11 over a CAPTION GROUP. Returns per-pair scalars, never logits.

            LAYOUT, and why it is not the obvious aligned one. Rows are
                sighted_*  : [N, S_s]        N = prompts_in_chunk * m trajectories
                blind_*    : [N * G_c, S_b]  caption-major: caption j occupies
                                             blind[j*N : (j+1)*N], aligned row-for-row
                                             with the N sighted rows
                responses  : [N, T]          the SHARED y of S13
            so the sighted forward runs **once per (prompt, trajectory)** and its
            distribution `p` is reused across all `G_c` captions.

            This is not tidiness, it is feasibility. Forward KL needs `p` and `q` over the
            full vocabulary at every position simultaneously, so there is no summary of `p`
            that could be carried between passes -- the choice is to hold it or to recompute
            it. Recomputing means `G_c`x the sighted forwards, and sighted sequences carry
            the image (up to 12,800 prompt tokens against ~500 for blind). At P=512, G_c=8,
            m=2.6 that is ~10,650 image-bearing forwards per step instead of ~1,331.
            Holding `p` instead costs ~158 MB per prompt, which is nothing on a 96 GB GH200.

            It also makes S13's cancellation exact rather than nearly exact: every caption in
            a group is scored against bit-identical sighted logits, so S12's centring removes
            the `-H(sighted)` term precisely instead of leaving a float-level residual.
            """
            from verl.protocol import DataProto

            # protocol.py:113, the same import dp_actor.py:29 uses.
            from verl.protocol import batch_collate

            assert self._has_actor

            # verl DERIVES multi_modal_inputs from multi_modal_data + uid; it is never
            # passed in. Every other worker entry point calls this first
            # (fsdp_workers.py:554, 636, 672, 701, 723) and so must we -- otherwise the
            # sighted pass runs with NO pixel values, i.e. blind, and D collapses toward
            # zero for every caption while the run looks converged.
            self._process_multi_modal_inputs(data)
            data = data.to(torch.cuda.current_device())
            temperature = data.meta_info.get(
                CA21_TEMPERATURE_KEY, self.config.rollout.temperature)

            responses = data.batch["responses"]
            T = responses.size(-1)
            module = self.actor.actor_module
            padding_free = getattr(self.actor.config, "padding_free", True)

            # forward_packed_logits transcribes dp_actor._forward_micro_batch:68-133, which
            # has no ulysses branch of ours. Vision-SR1 sets ulysses_size: 1 (config.yaml:42)
            # so this never fires -- but if it is ever raised, our transcription would skip
            # ulysses_pad_and_slice_inputs (dp_actor.py:109-115) and silently score a
            # differently-sharded sequence.
            if getattr(self.actor.config, "ulysses_size", 1) > 1:
                raise AssertionError(
                    f"ulysses_size={self.actor.config.ulysses_size} > 1 is not transcribed "
                    f"in forward_packed_logits; the caption distortion would be computed on "
                    f"a differently-sharded sequence than verl's own forward.")

            # dp_actor.py:82-87 -- per-row dicts collated into batched tensors. Passing the
            # raw object array through would not raise; it would feed the model garbage.
            mm_sighted = None
            if "multi_modal_inputs" in data.non_tensor_batch:
                collated = batch_collate(data.non_tensor_batch["multi_modal_inputs"])
                mm_sighted = {k: torch.cat(v, dim=0) for k, v in collated.items()}

            N = data.batch["sighted_input_ids"].shape[0]
            g_c = data.meta_info["ca21_g_c"]
            # blind is [N, G_c, S_b]: the caption axis lives INSIDE the row, so DP chunking
            # moves a row together with all of its captions. A flat [N*G_c] would be chunked
            # independently of sighted[N] and could score caption j of prompt p against a
            # different prompt's `p` -- finite, plausible, and wrong.
            blind_shape = tuple(data.batch["blind_input_ids"].shape)
            if len(blind_shape) != 3 or blind_shape[:2] != (N, g_c):
                raise AssertionError(
                    f"blind_input_ids is {blind_shape}, expected (N={N}, G_c={g_c}, S_b). "
                    f"This layout is what aligns caption j with its own sighted row.")
            if responses.shape[0] != N:
                raise AssertionError(
                    f"{responses.shape[0]} shared trajectories for {N} sighted rows")

            KEYS = ("kl", "one_sample", "entropy_p", "entropy_q", "n_positions")
            per_chunk = {k: [] for k in KEYS}
            kl_min_witness = []

            # ROW CHUNKING, and why the docstring above was not enough. `p` costs ~158 MB per
            # prompt once GATHERED -- but forward_packed_logits materialises the FULL
            # [total_nnz, V] logits before that gather, and V = 152k. At production size,
            # 512 prompts / 4 ranks x m=2 = 256 rows x ~2000 tokens = 512k packed tokens
            # -> 512k x 152064 x 2 B ~= 155 GB, against 95 GiB on a GH200. The blind pass is
            # no better. T0a and T0b never hit it because they scored ONE row at a time.
            # [CC] My own "nothing on a 96 GB GH200" claim measured the wrong tensor.
            #
            # Chunking rows preserves the property that makes this affordable: within a
            # chunk the sighted pass still runs ONCE and is reused by all G_c captions, so
            # S13's cancellation stays bit-exact. It costs one extra sighted forward per
            # chunk boundary, not per caption.
            chunk = int(data.meta_info.get("ca21_row_chunk", 16))
            if chunk < 1:
                raise AssertionError(f"ca21_row_chunk={chunk} must be >= 1")

            with torch.no_grad():
                for lo in range(0, N, chunk):
                    hi = min(lo + chunk, N)
                    resp_c = responses[lo:hi]

                    # ---- ONE sighted pass per chunk, reused by every caption ----
                    sighted, idx_s, B_s, S_s = forward_packed_logits(
                        module,
                        data.batch["sighted_input_ids"][lo:hi],
                        data.batch["sighted_attention_mask"][lo:hi],
                        data.batch["sighted_position_ids"][lo:hi],
                        multi_modal_inputs=slice_multi_modal_inputs(mm_sighted, lo, hi, N),
                        temperature=temperature, padding_free=padding_free)
                    lg_s, m_s = gather_response_logits(sighted, idx_s, B_s, S_s, T)
                    del sighted

                    base_mask = m_s
                    if "response_mask" in data.batch:
                        base_mask = base_mask * data.batch[
                            "response_mask"][lo:hi].to(m_s.dtype)

                    # ---- one blind pass per caption, scored against the SAME p ----
                    # blind is [N, G_c, S_b], so caption j for this chunk is [lo:hi, j].
                    cols = {k: [] for k in KEYS}
                    for j in range(g_c):
                        blind, idx_b, B_b, S_b = forward_packed_logits(
                            module,
                            data.batch["blind_input_ids"][lo:hi, j],
                            data.batch["blind_attention_mask"][lo:hi, j],
                            data.batch["blind_position_ids"][lo:hi, j],
                            multi_modal_inputs=None,      # G-BLIND, structurally
                            temperature=temperature, padding_free=padding_free)
                        lg_b, m_b = gather_response_logits(blind, idx_b, B_b, S_b, T)
                        del blind

                        # Only positions valid under BOTH contexts are comparable.
                        res = distortion_from_logits(
                            lg_s, lg_b, base_mask * m_b, labels=resp_c, temperature=1.0)
                        del lg_b
                        for k in KEYS:
                            cols[k].append(res[k].cpu())
                        kl_min_witness.append(res["kl_min_position"].reshape(1).cpu())

                    del lg_s
                    for k in KEYS:
                        per_chunk[k].append(torch.stack(cols[k], dim=1))   # [chunk, G_c]

            # [N, G_c] -- row = (prompt, trajectory), column = caption.
            stacked = {k: torch.cat(v, dim=0) for k, v in per_chunk.items()}
            if stacked["kl"].shape != (N, g_c):
                raise AssertionError(
                    f"chunk reassembly produced {tuple(stacked['kl'].shape)}, expected "
                    f"{(N, g_c)}. Row order across chunks must match the sighted rows, or "
                    f"S12 would normalise each caption against the wrong group.")

            # FREE ORACLE, available only because `p` is now shared. H(sighted) is computed
            # from bit-identical logits for every caption, so it can differ across columns
            # ONLY through the mask -- i.e. if some caption's blind sequence lost response
            # positions (a long caption pushing `y` past the context, or ragged packing).
            # A non-zero spread means those captions were scored on different position sets
            # and their D-hat values are not comparable, which is precisely the comparison
            # S12 then normalises over. Reported rather than asserted because a small spread
            # has a legitimate cause worth seeing at T0 instead of crashing on.
            es = stacked["entropy_p"]
            entropy_spread = (es.max(dim=1).values - es.min(dim=1).values)

            out = DataProto.from_dict(tensors={
                "caption_distortion": stacked["kl"],
                "distortion_one_sample": stacked["one_sample"],
                "entropy_sighted": stacked["entropy_p"],
                "entropy_blind": stacked["entropy_q"],
                "distortion_n_positions": stacked["n_positions"],
                # Must be ~0: proves the sighted pass really was shared and the masks align.
                "entropy_sighted_spread": entropy_spread,
            })
            out.meta_info["ca21_kl_min_position"] = float(
                torch.cat(kl_min_witness).min().item())
            return out.to("cpu")

    return CA21Worker
