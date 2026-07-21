#!/usr/bin/env python3
"""Verify the PAPO Conv3d->matmul patch-embed fix.

Proves, on the real model, before any training run:
  (A) ALGEBRAIC IDENTITY - in float64 the matmul reimpl equals the stock Conv3d
      patch embed to ~machine eps (maxdiff ~1e-12). This is the real proof: if the
      weight-reshape/ordering were wrong the error would be the SIZE OF THE OUTPUT
      (order 1-10), not machine eps.
  (B) PRECISION-ONLY at train precision - the fp32 / bf16 differences are then just
      floating-point rounding (different summation order), reported as RELATIVE error
      and compared against the dtype's ULP. Not a result change.
  (C) APPLIED - PAPO's apply_ulysses_patch() actually installs our forward.

Run inside the EasyR1/PAPO container. Auto-locates vendored PAPO at ../PAPO.

  python3 verify_conv_patch.py
"""
import argparse
import os
import sys

# rough relative machine epsilon per dtype (for context in the printout)
_EPS = {"torch.float64": 2.2e-16, "torch.float32": 1.2e-7, "torch.bfloat16": 3.9e-3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking")
    ap.add_argument("--papo-path", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PAPO"))
    ap.add_argument("--n", type=int, default=4096, help="number of random patches to test")
    args = ap.parse_args()

    papo = os.path.abspath(args.papo_path)
    sys.path.insert(0, papo)
    print(f"[info] PAPO path: {papo}")

    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed
    from verl.models.transformers.qwen3_vl import qwen3_vl_patch_embed_forward
    import verl
    print(f"[info] verl -> {verl.__file__}")

    # Capture the STOCK class forward BEFORE any patch is applied.
    stock_forward = Qwen3VLVisionPatchEmbed.forward

    print(f"[info] loading {args.model} (bf16, cpu) ...")
    try:
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
    except Exception as e:
        print(f"[warn] AutoModelForImageTextToText failed ({e!r}); falling back to AutoModel")
        from transformers import AutoModel
        m = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True)

    pe = next(mod for mod in m.modules() if isinstance(mod, Qwen3VLVisionPatchEmbed))
    in_dim = pe.proj.weight[0].numel()  # C * T * P * P = correct flattened patch width
    print(f"[info] patch-embed found | in_dim = {in_dim} | n = {args.n}")

    # ONE fixed fp64 input, cast down per dtype, so all dtypes see identical data.
    torch.manual_seed(0)
    x64 = torch.randn(args.n, in_dim, dtype=torch.float64)

    results = {}
    print("\n[diff] dtype            maxdiff      rel_err      mean_diff    |out|max     rel/ULP")
    for dt in (torch.float64, torch.float32, torch.bfloat16):
        ped = pe.to(dt)
        x = x64.to(dt)
        with torch.no_grad():
            a = stock_forward(ped, x)                    # stock Conv3d
            b = qwen3_vl_patch_embed_forward(ped, x)     # our matmul
        diff = (a.double() - b.double()).abs()
        md = diff.max().item()
        mean = diff.mean().item()
        scale = a.double().abs().max().item()
        rel = md / scale if scale > 0 else float("nan")
        results[dt] = (md, rel, mean, scale)
        eps = _EPS[str(dt)]
        print(f"       {str(dt):15s} {md:.3e}   {rel:.3e}   {mean:.3e}   {scale:.3e}   {rel/eps:6.2f}")

    # (C) applied-check: does PAPO's own patch path install our forward?
    from verl.models.monkey_patch import apply_ulysses_patch
    apply_ulysses_patch("qwen3_vl")
    applied = Qwen3VLVisionPatchEmbed.forward is qwen3_vl_patch_embed_forward

    md64 = results[torch.float64][0]
    rel32 = results[torch.float32][1]
    relbf = results[torch.bfloat16][1]

    identity_ok = md64 < 1e-9          # fp64: true algebraic identity
    fp32_ok = rel32 < 1e-4             # within fp32 accumulation noise
    bf16_ok = relbf < 5e-2             # within a few bf16 ULP (eps ~3.9e-3)
    ok = identity_ok and fp32_ok and bf16_ok and applied

    print("\n==== VERDICT ====")
    print(f"(A) fp64 algebraic identity (<1e-9): {'PASS' if identity_ok else 'FAIL'}  (maxdiff {md64:.3e})")
    print(f"(B) fp32 rel within noise   (<1e-4): {'PASS' if fp32_ok else 'FAIL'}  (rel {rel32:.3e})")
    print(f"(B) bf16 rel within few ULP (<5e-2): {'PASS' if bf16_ok else 'FAIL'}  (rel {relbf:.3e})")
    print(f"(C) patch installed                : {'PASS' if applied else 'FAIL'}")
    print("OVERALL:", "PASS" if ok else "FAIL")
    print("\nInterpretation: (A) PASS => the op is mathematically identical (a wrong reshape")
    print("would give rel_err ~O(1), not machine eps). (B) then shows the fp32/bf16 gaps are")
    print("only rounding = the same numerical noise bf16 training already has = result-neutral.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
