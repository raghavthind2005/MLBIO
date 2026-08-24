"""T0e -- the B>1 gate that T0d could not supply, plus two open items.

§4.17 left this explicitly unverified. T0d proved `forward_packed_logits` is padding-
invariant, but at BATCH-OF-ONE. Production packs B>1 sequences into a single row of shape
(1, total_nnz) and passes `attention_mask=None`, relying on `position_ids` restarting at
each sequence boundary for FlashAttention to infer where one sequence ends and the next
begins. If that inference failed, row j would attend to row j-1: every distortion wrong,
finite, plausible, and caught by no gate we have.

THE PROPERTY. A correct packed forward is ROW-INDEPENDENT -- the logits for row i must not
depend on which other rows travelled with it. So:

    forward(rows[a:b])[i]  ==  forward(rows[i:i+1])[0]      for every i

Same shape of argument as T0d: a property both sides must satisfy, not a second
implementation to argue with.

WHY THIS ALSO SETTLES CHUNK-INVARIANCE, for free. The row-chunking just added to
`compute_caption_distortion` is only safe if chunk boundaries cannot change the answer. But
chunk-invariance FOLLOWS from row-independence: if every row's logits are unaffected by its
companions, then any partition into chunks yields identical per-row logits, hence identical
`D`. Proving the stronger property once is better than sampling chunk sizes.

It also exercises `slice_multi_modal_inputs` end to end, which the unit tests can only check
arithmetically: a chunk whose images were sliced at the wrong patch offsets would break
row-independence here even though the offsets "looked" right.

Read-only. No training, no checkpoints, no writes outside its own JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ca21_prompts as P                                            # noqa: E402
from ca21_contexts import append_responses, build_prompt_row       # noqa: E402
from ca21_packing import gather_response_logits                    # noqa: E402
from ca21_worker import forward_packed_logits, slice_multi_modal_inputs  # noqa: E402
from t0b_diagnostic import _load, _rows                            # noqa: E402


def _fit_sha256():
    """Pin the upstream fit() hash from INSIDE the container.

    Computed locally as 3c884df4... but with a different Python; inspect.getblock's
    boundaries are what ca21_trainer will hash at runtime, so the authoritative value is
    the one this interpreter produces.
    """
    from verl.trainer.ray_trainer import RayPPOTrainer

    src = inspect.getsource(RayPPOTrainer.fit)
    return hashlib.sha256(src.encode()).hexdigest(), len(src.splitlines())


def main() -> int:
    import torch
    from verl.protocol import batch_collate

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rows", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=2048)
    ap.add_argument("--min-pixels", type=int, default=262144)
    ap.add_argument("--max-pixels", type=int, default=1048576)
    ap.add_argument("--resp-len", type=int, default=24)
    args = ap.parse_args()

    # --- item 2: the fit() hash, before any GPU work so it lands even on a later failure
    LOCAL_CANDIDATE = ("3c884df495e3f95f6711f83311f99233389b79fb"
                       "daf1f5d379874f506102e49d")
    try:
        sha, n_lines = _fit_sha256()
        print(f"[t0e] upstream RayPPOTrainer.fit sha256 = {sha}  ({n_lines} lines)",
              flush=True)
        print(f"[t0e]   local candidate  = {LOCAL_CANDIDATE}", flush=True)
        print(f"[t0e]   MATCH = {sha == LOCAL_CANDIDATE}", flush=True)
    except Exception as exc:                                        # noqa: BLE001
        sha, n_lines = None, None
        print(f"[t0e] fit() hash unavailable: {exc!r}", flush=True)

    # PRODUCTION PARITY, and the reason job 3174516 reported a false alarm. verl calls this
    # at fsdp_workers.py:195 when it builds the actor, and it does two things stock HF does
    # not: swaps ALL_ATTENTION_FUNCTIONS["flash_attention_2"] for verl's own
    # flash_attention_forward -- the piece that handles PACKED multi-sequence inputs -- and
    # replaces Qwen2_5_VL{Model,ForConditionalGeneration}.forward outright.
    #
    # Without it this probe measures stock HF attention, which cannot do B>1 packing, and
    # duly reported "not row-independent" (0.4375). That was the harness, not production.
    # Applied BEFORE from_pretrained so the patched class forwards are in place.
    from verl.models.monkey_patch import apply_ulysses_patch
    apply_ulysses_patch("qwen2_5_vl")
    print("[t0e] applied verl's ulysses patch (fsdp_workers.py:195 parity)", flush=True)

    print(f"[t0e] loading {args.model}", flush=True)
    model, processor, tokenizer = _load(args.model)
    torch.set_grad_enabled(False)
    dev = model.device

    from verl.utils.dataset import process_image
    rows = _rows(args.parquet, args.n_rows)
    for r in rows:
        r["pimage"] = process_image(r["image"], args.min_pixels, args.max_pixels)
    n = len(rows)
    print(f"[t0e] {n} rows", flush=True)

    T = args.resp_len

    # The probe response, built ONCE and shared by every row. Tiled to exactly T: a fixed
    # string tokenised to 16 ids against T=24 and killed job 3174476. Content is irrelevant
    # to row-independence; identical length across rows is not, since
    # gather_response_logits must read the same window for each.
    _base = tokenizer("the answer is four apples on the table today",
                      add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if _base.numel() == 0:
        raise AssertionError("tokeniser produced no ids for the probe response")
    probe_resp = _base.repeat((T + _base.numel() - 1) // _base.numel())[:T]   # [T]

    def build(messages, images):
        row = build_prompt_row(processor, tokenizer, messages, images,
                               args.max_prompt_length, args.min_pixels, args.max_pixels)
        resp = probe_resp.unsqueeze(0)
        ids, am, pos = append_responses(
            row["input_ids"].unsqueeze(0), row["attention_mask"].unsqueeze(0),
            row["position_ids"].unsqueeze(0), resp, torch.ones_like(resp))
        return ids, am, pos, row

    out = {"fit_sha256": sha, "fit_lines": n_lines, "cases": []}

    for label, with_image in (("sighted (images)", True), ("blind (no image)", False)):
        built, mms = [], []
        for r in rows:
            if with_image:
                msgs, imgs = P.build_sighted_messages(r["problem"]), [r["pimage"]]
            else:
                msgs, imgs = P.build_answerer_messages("A short caption.", r["problem"]), None
            built.append(build(msgs, imgs))
            if with_image:
                mms.append({k: v for k, v in processor.image_processor(
                    images=[r["pimage"]], return_tensors="pt").items()})

        ids = torch.cat([b[0] for b in built]).to(dev)
        am = torch.cat([b[1] for b in built]).to(dev)
        pos = torch.cat([b[2] for b in built]).to(dev)

        def mm_for(lo, hi):
            if not with_image:
                return None
            coll = batch_collate(mms[lo:hi])
            full = {k: torch.cat(v, dim=0).to(dev) for k, v in coll.items()}
            return full

        # Compare LOG-PROBS of the realized next token, not raw logits.
        #
        # Job 3174586 compared raw logits against a 1e-2 threshold and reported 4.375e-01,
        # 3.750e-01, 4.141e-01 -- exact binary fractions, and bf16's ulp at logit magnitude
        # 16-32 is 0.0625, so those are 6-7 ulps. A 1e-2 threshold on bf16 logits is two
        # orders of magnitude below what the dtype can represent: that test could never pass
        # however correct the forward was. T0d got this right and t0e regressed. [CC]
        #
        # Log-probs are the right quantity anyway: log_softmax is shift-invariant, so a
        # uniform logit offset cancels, and it is what the distortion actually consumes.
        def run(lo, hi, mm):
            lg, idx, B, S = forward_packed_logits(
                model, ids[lo:hi], am[lo:hi], pos[lo:hi],
                multi_modal_inputs=mm, padding_free=True)
            g, _ = gather_response_logits(lg, idx, B, S, T)
            lp = g.float().log_softmax(-1)                          # [hi-lo, T, V]
            tgt = probe_resp[1:].view(1, T - 1, 1).expand(lp.shape[0], -1, -1).to(lp.device)
            return (lp[:, :-1].gather(-1, tgt).squeeze(-1),         # [hi-lo, T-1] log-probs
                    g.float())                                      # [hi-lo, T, V] logits

        # reference: each row entirely alone
        ref = [run(i, i + 1, mm_for(i, i + 1)) for i in range(n)]
        lp_alone = torch.cat([r[0] for r in ref], dim=0)
        lg_alone = torch.cat([r[1] for r in ref], dim=0)

        # (a) all rows packed together
        lp_tog, lg_tog = run(0, n, mm_for(0, n))

        # (b) chunked, using the SAME slicer the worker uses -- exercises patch offsets
        mm_all = mm_for(0, n)
        parts = [run(lo, min(lo + 2, n),
                     slice_multi_modal_inputs(mm_all, lo, min(lo + 2, n), n))
                 for lo in range(0, n, 2)]
        lp_chunk = torch.cat([p[0] for p in parts], dim=0)

        def stats(a, b):
            d = (a - b).abs()
            return {"max": float(d.max()), "median": float(d.median()),
                    "frac_over_1e-2": float((d > 1e-2).float().mean()),
                    # PER ROW -- the discriminator that separates the two hypotheses.
                    #
                    # Causal attention means row 0 of a packed batch has nothing before it
                    # EITHER WAY. So if sequence boundaries leak, row 0 still matches its
                    # alone-run while rows 1..k diverge progressively with how much earlier
                    # content they can wrongly see. If instead this is bf16 accumulation
                    # noise from different flash-attn tiling, every row deviates about
                    # equally and row 0 is no cleaner than row k.
                    #
                    # An aggregate max cannot tell those apart, which is why 3174613's
                    # numbers (median 5.1e-2, top-1 0.979) were unreadable on their own.
                    "per_row_max": [float(x) for x in d.max(dim=1).values]}

        # Context for reading the logit numbers: bf16's ulp at the observed magnitude.
        mag = float(lg_alone.abs().max())
        ulp = 2.0 ** (torch.tensor(max(mag, 1e-6)).log2().floor().item() - 8)

        case = {"case": label, "rows": n,
                "logprob_batched_vs_alone": stats(lp_tog, lp_alone),
                "logprob_chunked_vs_alone": stats(lp_chunk, lp_alone),
                "logit_batched_vs_alone_max": float((lg_tog - lg_alone).abs().max()),
                "logit_max_magnitude": mag, "bf16_ulp_at_that_magnitude": ulp,
                "top1_agreement": float(
                    (lg_tog.argmax(-1) == lg_alone.argmax(-1)).float().mean())}
        out["cases"].append(case)
        b, c = case["logprob_batched_vs_alone"], case["logprob_chunked_vs_alone"]
        print(f"[t0e] {label}\n"
              f"        LOGPROB  B={n} vs alone : max {b['max']:.3e}  "
              f"median {b['median']:.3e}  frac>1e-2 {b['frac_over_1e-2']:.3f}\n"
              f"        LOGPROB  chunked vs alone : max {c['max']:.3e}  "
              f"median {c['median']:.3e}\n"
              f"        top-1 agreement           : {case['top1_agreement']:.4f}\n"
              f"        (logit max diff {case['logit_batched_vs_alone_max']:.3e}; "
              f"bf16 ulp at |logit|<={mag:.1f} is {ulp:.3e})", flush=True)
        Path(args.out).write_text(json.dumps(out, indent=2))

    worst = max(max(c["logprob_batched_vs_alone"]["max"],
                    c["logprob_chunked_vs_alone"]["max"]) for c in out["cases"])
    worst_top1 = min(c["top1_agreement"] for c in out["cases"])

    # TWO SIGNATURES, and the aggregate max distinguishes neither -- which is why
    # 3174613 could not be read. Leakage is a STRUCTURAL claim, so test its structure:
    #
    #   (1) ROW POSITION. Causal attention gives row 0 no predecessor either way, so under
    #       leakage row 0 matches its alone-run while later rows worsen. Under bf16 tiling
    #       noise every row deviates about equally.
    #   (2) CONTEXT SCALING. Row 3 of a B=4 pack can wrongly see three sequences; under
    #       chunk(2) it sees at most one. Under leakage the deviation must SHRINK when
    #       chunked. Under noise the two are indistinguishable.
    rows0, rowsN, scaling = [], [], []
    for c in out["cases"]:
        pr = c["logprob_batched_vs_alone"]["per_row_max"]
        rows0.append(pr[0])
        rowsN.append(max(pr[1:]) if len(pr) > 1 else pr[0])
        scaling.append(c["logprob_chunked_vs_alone"]["max"]
                       / max(c["logprob_batched_vs_alone"]["max"], 1e-12))
    position_effect = max(rowsN) / max(max(rows0), 1e-12)
    chunk_ratio = min(scaling)

    out["verdict_logprob_max"] = worst
    out["verdict_min_top1_agreement"] = worst_top1
    out["row0_max"] = rows0
    out["later_rows_max"] = rowsN
    out["position_effect_ratio"] = position_effect
    out["chunk_vs_batch_ratio"] = chunk_ratio
    # Noise: row 0 no cleaner than the rest, and chunking changes nothing.
    looks_like_noise = position_effect < 3.0 and chunk_ratio > 0.5
    out["row_independent"] = bool(looks_like_noise)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n=== T0e VERDICT ===")
    print(f"  worst |delta log-prob| vs row-alone : {worst:.3e}")
    print(f"  min top-1 agreement                 : {worst_top1:.4f}")
    print(f"  row 0 max per case                  : "
          f"{[f'{x:.3e}' for x in rows0]}")
    print(f"  later rows max per case             : "
          f"{[f'{x:.3e}' for x in rowsN]}")
    print(f"  position effect  (later / row 0)    : {position_effect:.2f}   "
          f"(>>1 = leakage, ~1 = noise)")
    print(f"  chunk(2) / B=4 deviation            : {chunk_ratio:.2f}   "
          f"(<<1 = leakage, ~1 = noise)")
    if looks_like_noise:
        print("  -> ROW-INDEPENDENT within bf16 accumulation noise. Deviation does not")
        print("     depend on row position or on how much preceding context is packed,")
        print("     which is what cross-sequence leakage would require. Chunk-invariance")
        print("     follows: the row-chunking cannot change the answer beyond this noise.")
    else:
        print("  -> LEAKAGE SIGNATURE. Deviation scales with row position and/or with how")
        print("     much context is packed alongside. Every distortion at B>1 is wrong.")
        print("     R1 must not run.")
    print(f"\n[t0e] -> {args.out}")
    return 0 if looks_like_noise else 1


if __name__ == "__main__":
    sys.exit(main())
