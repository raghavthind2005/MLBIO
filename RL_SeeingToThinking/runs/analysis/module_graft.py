"""
module_graft.py — S3 causal test: which weights, when grafted onto base, recover perception?

The senior's key experiment. Build "Frankenstein" models:
    W_grafted[k] = ckpt[k]   if k is in the mask    (= base + delta on that subset)
                 = base[k]   otherwise
and run the MC perception probe on each. If MLP-graft recovers most of the trained model's
accuracy and attention-graft does not, the MLP change is *sufficient* for the perception fix
(causation, not just the correlation weight_delta showed).

Graft modes:
  base       — untouched base model (lower-bound sanity)
  full       — all ckpt weights (= trained model; upper-bound sanity)
  llm_all    — all LLM weights from ckpt (mlp+attn+norm)
  mlp        — LLM MLP weights only            <- the S3 hypothesis
  attn       — LLM attention weights only      <- the S3 control
  late_mlp   — LLM MLP weights, late third only (layer >= 2/3 depth)
  early_mlp  — LLM MLP weights, early third only

Reuses: qwen3vl_param_map.classify (SAME mlp/attn boundary as weight_delta),
        ckpt_model.reconstruct_full_state_dict, mc_eval.run_mc_probe, babyvision_data.

Requires container + GPU. Run AFTER mc_base validates the probe path.

Usage:
  python module_graft.py --base <model> --ckpt <global_step_96/actor> \
      --data-dir <babyvision_data> --modes base full mlp attn late_mlp early_mlp \
      --out graft_results.csv
"""

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from qwen3vl_param_map import classify
from ckpt_model import load_model, reconstruct_full_state_dict, _install_patch_embed_fix
from mc_eval import run_mc_probe
from probe_loader import add_probe_args, load_probe


def in_mask(key: str, mode: str, n_layers: int) -> bool:
    """Does this parameter get the checkpoint value under this graft mode?"""
    comp, module, idx = classify(key)
    if mode == "full":
        return True
    if mode == "base":
        return False
    if mode == "llm_all":
        return comp == "llm" or comp == "embed" or comp == "head"
    if mode == "mlp":
        return comp == "llm" and module == "mlp"
    if mode == "attn":
        return comp == "llm" and module == "attn"
    if mode == "late_mlp":
        return comp == "llm" and module == "mlp" and idx is not None and idx >= (2 * n_layers // 3)
    if mode == "early_mlp":
        return comp == "llm" and module == "mlp" and idx is not None and idx < (n_layers // 3)
    raise ValueError(f"unknown mode {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True, help="global_step_N/actor (the trained checkpoint to graft)")
    ap.add_argument("--modes", nargs="+",
                    default=["base", "full", "mlp", "attn", "late_mlp", "early_mlp"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="graft_results.csv")
    add_probe_args(ap)
    args = ap.parse_args()

    items = load_probe(args)
    print(f"[graft] {len(items)} MC items ({args.dataset}); modes={args.modes}")

    # Load base model ONCE; we'll overwrite/restore subsets per mode.
    model, processor = load_model(args.base, device=args.device)

    # Reconstruct the trained checkpoint's full weights.
    print("[graft] reconstructing checkpoint weights …")
    ckpt_sd = reconstruct_full_state_dict(args.ckpt)

    # Detect n_layers (for late/early bands) from LLM keys.
    n_layers = 1 + max((classify(k)[2] for k in ckpt_sd if classify(k)[0] == "llm"
                        and classify(k)[2] is not None), default=0)
    print(f"[graft] detected {n_layers} LLM layers")

    # Snapshot base values for every key we might overwrite (all ckpt keys present in model).
    model_sd = model.state_dict()
    graftable = [k for k in ckpt_sd if k in model_sd]
    base_snapshot = {k: model_sd[k].detach().clone() for k in graftable}
    print(f"[graft] {len(graftable)} graftable keys snapshotted")

    rows = []
    for mode in args.modes:
        # Build the overwrite set for this mode.
        overwrite = {k: ckpt_sd[k].to(base_snapshot[k].dtype) for k in graftable
                     if in_mask(k, mode, n_layers)}
        # Apply: ckpt values on masked keys, base values everywhere else (restore first).
        model.load_state_dict(base_snapshot, strict=False)      # reset to base
        if overwrite:
            model.load_state_dict(overwrite, strict=False)      # graft subset
        model.eval()

        n_params = sum(base_snapshot[k].numel() for k in overwrite)
        print(f"\n[graft] mode={mode}: grafting {len(overwrite)} tensors ({n_params/1e9:.3f}B params)")
        acc, results = run_mc_probe(model, processor, items, device=args.device, verbose=False)
        n_correct = sum(r["correct"] for r in results)
        print(f"[graft] mode={mode}: MC accuracy = {acc:.4f} ({n_correct}/{len(results)})")
        rows.append(dict(mode=mode, n_graft_tensors=len(overwrite),
                         n_graft_params=n_params, accuracy=acc,
                         n_correct=n_correct, n_total=len(results)))

    # Restore base (cleanliness).
    model.load_state_dict(base_snapshot, strict=False)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[graft] wrote {args.out}")

    # Verdict
    by = {r["mode"]: r["accuracy"] for r in rows}
    if {"base", "full"} <= by.keys():
        gain = by["full"] - by["base"]
        print(f"\n=== S3 VERDICT (gain base->full = {gain:+.4f}) ===")
        for m in ("mlp", "attn", "late_mlp", "early_mlp"):
            if m in by:
                rec = (by[m] - by["base"]) / gain if gain else float("nan")
                print(f"  {m:10s}: acc={by[m]:.4f}  recovers {rec*100:5.1f}% of the full gain")


if __name__ == "__main__":
    main()