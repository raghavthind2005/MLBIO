#!/usr/bin/env python3
"""Verify the PAPO Conv3d->matmul patch-embed fix.

Proves two things on the real model, before any training run:
  (A) bit-identity  - the matmul reimpl equals the stock Conv3d patch embed
                      (fp32 => algebraic identity, maxdiff ~0; bf16 => rounding only)
  (B) applied       - PAPO's apply_ulysses_patch() actually installs our forward

Run inside the EasyR1/PAPO container. Auto-locates the vendored PAPO at ../PAPO
(sibling dir); override with --papo-path. No PYTHONPATH needed.

  python3 verify_conv_patch.py
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/capstor/store/cscs/swissai/a0174/models/Qwen3-VL-4B-Thinking")
    ap.add_argument("--papo-path", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PAPO"))
    ap.add_argument("--n", type=int, default=1024, help="number of random patches to test")
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
        m = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    except Exception as e:
        print(f"[warn] AutoModelForImageTextToText failed ({e!r}); falling back to AutoModel")
        from transformers import AutoModel
        m = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)

    pe = next(mod for mod in m.modules() if isinstance(mod, Qwen3VLVisionPatchEmbed))
    in_dim = pe.proj.weight[0].numel()  # C * T * P * P = correct flattened patch width
    print(f"[info] patch-embed found | in_dim = {in_dim}")

    results = {}
    for dt in (torch.float32, torch.bfloat16):
        ped = pe.to(dt)
        x = torch.randn(args.n, in_dim, dtype=dt)
        with torch.no_grad():
            a = stock_forward(ped, x)                    # stock Conv3d
            b = qwen3_vl_patch_embed_forward(ped, x)     # our matmul
        md = (a.float() - b.float()).abs().max().item()
        results[dt] = md
        print(f"[diff] {str(dt):16s} maxdiff = {md:.3e}   shapes {tuple(a.shape)} vs {tuple(b.shape)}")

    # (B) applied-check: does PAPO's own patch path install our forward?
    from verl.models.monkey_patch import apply_ulysses_patch
    apply_ulysses_patch("qwen3_vl")
    applied = Qwen3VLVisionPatchEmbed.forward is qwen3_vl_patch_embed_forward
    print(f"[applied] Qwen3VLVisionPatchEmbed.forward is patched: {applied}")

    fp32_ok = results[torch.float32] < 1e-5
    bf16_ok = results[torch.bfloat16] < 1e-2
    ok = fp32_ok and bf16_ok and applied
    print("\n==== VERDICT ====")
    print(f"fp32 algebraic identity (<1e-5): {'PASS' if fp32_ok else 'FAIL'}  ({results[torch.float32]:.3e})")
    print(f"bf16 rounding only      (<1e-2): {'PASS' if bf16_ok else 'FAIL'}  ({results[torch.bfloat16]:.3e})")
    print(f"patch installed                : {'PASS' if applied else 'FAIL'}")
    print("OVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
