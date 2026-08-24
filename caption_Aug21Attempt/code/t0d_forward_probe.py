"""T0d -- WHICH forward is wrong: the packed path, or the padded reference?

T0b's Phase 0 failed: packed and padded log-probs disagree by 0.115-0.478 nats on 11/16
items and agree BIT-EXACTLY on the other 5, with fp32 giving no improvement over bf16. Not
precision. `ca21_contexts.py:106` left-pads every prompt to max_prompt_length, and the split
is exactly padded-vs-unpadded, so it is a padding-handling discrepancy.

WHAT T0b COULD NOT TELL US. It compared our packed forward against a padded reference that
I ALSO wrote. Disagreement localises nothing: either implementation could be the wrong one,
and the consequence differs completely.

  packed wrong  -> ca21_worker.py uses it, so PRODUCTION is broken and everything voids.
  padded wrong  -> only the test scaffold is miscalibrated; production is fine.

THE DECOMPOSITION. Take one row and build it twice with IDENTICAL content: once left-padded
to max_prompt_length (P), once with those pads sliced off (U). Padding carries no
information, so a correct implementation is INVARIANT to it. Run both paths on both rows:

    packed(U) vs padded(U)   -- control. T0b's five zero-items say this is ~0.
    packed(P) vs packed(U)   -- is the PACKED path padding-invariant?
    padded(P) vs padded(U)   -- is the PADDED path padding-invariant?

Exactly one of the last two should be non-zero, and that one names the culprit. If BOTH are
non-zero the fault is shared and neither path can be trusted. If NEITHER is, the discrepancy
is not padding after all and this whole diagnosis is wrong -- which is also worth knowing.

This is decisive without needing a verl worker: the invariance property is checkable against
the model alone, so there is no third implementation to argue with.

Read-only. Generates nothing, trains nothing, writes one JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ca21_contexts import build_prompt_row          # noqa: E402
from ca21_worker import forward_packed_logits       # noqa: E402
import ca21_prompts as P                            # noqa: E402
from t0b_diagnostic import _load, _rows             # noqa: E402


def _next_token_logprobs(logits, ids, lo):
    """log p(token t+1 | <=t) at every real position, as a 1-D tensor.

    Compared position-by-position on the REALIZED next token, mirroring T0b's C1 so the
    numbers here are directly comparable to the ones that failed the gate.
    """
    import torch

    lp = logits.float().log_softmax(-1)             # [n_real, V]
    tgt = ids[0, lo + 1:].unsqueeze(-1)             # next token at each position
    return lp[:-1].gather(-1, tgt).squeeze(-1)


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-items", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=2048)
    ap.add_argument("--min-pixels", type=int, default=262144)
    ap.add_argument("--max-pixels", type=int, default=1048576)
    args = ap.parse_args()

    print(f"[t0d] loading {args.model}", flush=True)
    model, processor, tokenizer = _load(args.model)
    torch.set_grad_enabled(False)                   # measurement only (T0b's OOM lesson)
    dev = model.device

    rows = _rows(args.parquet, args.n_items)

    # process_image FIRST, and use the result everywhere. build_prompt_row sizes its image
    # placeholder tokens for the RESIZED image, so handing image_processor the raw one
    # yields a pixel_values whose patch count disagrees with the token count -- wrong
    # logits, not an error. t0b does this at :148 and uses `pimage` at both :158 and :209.
    from verl.utils.dataset import process_image
    for r in rows:
        r["pimage"] = process_image(r["image"], args.min_pixels, args.max_pixels)

    print(f"[t0d] {len(rows)} items", flush=True)

    out = []
    for r in rows:
        row = build_prompt_row(processor, tokenizer,
                               P.build_sighted_messages(r["problem"]), [r["pimage"]],
                               args.max_prompt_length, args.min_pixels, args.max_pixels)
        ids_p = row["input_ids"].unsqueeze(0).to(dev)
        am_p = row["attention_mask"].unsqueeze(0).to(dev)
        pos_p = row["position_ids"].unsqueeze(0).to(dev)
        mm = {k: v.to(dev) for k, v in processor.image_processor(
            images=[r["pimage"]], return_tensors="pt").items()}

        # Left padding => the pads are a prefix. Slice them off to build U.
        n_pad = int((am_p[0] == 0).sum())
        lo = n_pad
        ids_u, am_u, pos_u = ids_p[:, lo:], am_p[:, lo:], pos_p[..., lo:]
        n_real = int(ids_u.shape[1])
        if n_real < 8 or (am_p[0, lo:] == 0).any():
            print(f"[t0d] {r['problem_id']}: SKIP (n_real={n_real}, "
                  f"padding is not a clean prefix)", flush=True)
            continue

        def packed(i, a, p):
            lg, *_ = forward_packed_logits(model, i, a, p, multi_modal_inputs=mm,
                                           padding_free=True)
            return lg                                # [n_real, V]

        def padded(i, a, p):
            lg = model(input_ids=i, attention_mask=a, position_ids=p.transpose(0, 1),
                       **mm, use_cache=False).logits
            return lg[0, -n_real:]                   # real tokens are the suffix

        lp = {
            "packed_P": _next_token_logprobs(packed(ids_p, am_p, pos_p), ids_u, 0),
            "packed_U": _next_token_logprobs(packed(ids_u, am_u, pos_u), ids_u, 0),
            "padded_P": _next_token_logprobs(padded(ids_p, am_p, pos_p), ids_u, 0),
            "padded_U": _next_token_logprobs(padded(ids_u, am_u, pos_u), ids_u, 0),
        }

        def d(a, b):
            return float((lp[a] - lp[b]).abs().max())

        rec = {
            "problem_id": r["problem_id"], "n_pad": n_pad, "n_real": n_real,
            "control_packedU_vs_paddedU": d("packed_U", "padded_U"),
            "reproduce_packedP_vs_paddedP": d("packed_P", "padded_P"),
            "PACKED_padding_invariance": d("packed_P", "packed_U"),
            "PADDED_padding_invariance": d("padded_P", "padded_U"),
            "pos_ids_at_pad": pos_p[0, :, :min(4, n_pad)].tolist() if n_pad else [],
            "pos_ids_first_real": pos_p[0, :, lo:lo + 4].tolist(),
        }
        out.append(rec)
        print(f"[t0d] {r['problem_id']}: n_pad={n_pad} n_real={n_real}\n"
              f"        control  packed(U) vs padded(U) = {rec['control_packedU_vs_paddedU']:.3e}"
              f"   (expect ~0)\n"
              f"        repro    packed(P) vs padded(P) = {rec['reproduce_packedP_vs_paddedP']:.3e}"
              f"   (expect ~0.1-0.5)\n"
              f"        PACKED  invariance to padding   = {rec['PACKED_padding_invariance']:.3e}\n"
              f"        PADDED  invariance to padding   = {rec['PADDED_padding_invariance']:.3e}",
              flush=True)
        Path(args.out).write_text(json.dumps(out, indent=2))   # flush every item

    if not out:
        print("[t0d] no usable items", flush=True)
        return 1

    mx = lambda k: max(r[k] for r in out)   # noqa: E731
    packed_bad = mx("PACKED_padding_invariance") > 1e-2
    padded_bad = mx("PADDED_padding_invariance") > 1e-2

    print("\n=== T0d VERDICT ===")
    print(f"  control packed(U) vs padded(U) max : {mx('control_packedU_vs_paddedU'):.3e}")
    print(f"  repro   packed(P) vs padded(P) max : {mx('reproduce_packedP_vs_paddedP'):.3e}")
    print(f"  PACKED padding-invariance     max : {mx('PACKED_padding_invariance'):.3e}")
    print(f"  PADDED padding-invariance     max : {mx('PADDED_padding_invariance'):.3e}")
    if packed_bad and not padded_bad:
        print("  -> PACKED path is not padding-invariant. ca21_worker uses it, so")
        print("     PRODUCTION IS BROKEN and every distortion to date is void.")
    elif padded_bad and not packed_bad:
        print("  -> PADDED reference is not padding-invariant. The test scaffold was")
        print("     miscalibrated; the production packed path is vindicated.")
    elif packed_bad and padded_bad:
        print("  -> BOTH fail invariance. Neither path is trustworthy; fault is shared.")
    else:
        print("  -> NEITHER fails invariance, yet T0b's gap reproduces. The padding")
        print("     diagnosis is WRONG and the cause is something else.")
    print(f"\n[t0d] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
