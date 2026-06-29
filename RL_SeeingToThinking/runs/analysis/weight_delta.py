"""
weight_delta.py — S2 localization test + freeze sanity check

Tests senior's claim S2: "weight change is tiny, concentrated in late-layer MLPs, not attention."
Also serves as a gold-standard freeze verification: vit_only must show LLM-delta ≈ 0;
llm_only must show vision-delta ≈ 0.

Algorithm (CPU-only, no process group needed):
  For each param key:
    1. Load base weight from safetensors (lazy, per shard).
    2. Reconstruct full ckpt tensor from 4 FSDP rank shards (mirror model_merger.py logic).
    3. Compute: rel_fro = ‖Δ‖_F / ‖W_base‖_F  (primary); abs_fro; mean_abs; cos_sim.
    4. Classify(key) -> (component, module, layer_idx).
    5. Append row to CSV (append mode → accumulate across steps/conditions).

Usage:
  # Single checkpoint:
  python weight_delta.py --base <model_dir> --ckpt <global_step_N/actor> \
                         --condition full --step 96 --out deltas.csv

  # Probe names only (no computation — verify classifier before first real run):
  python weight_delta.py --base <model_dir> --ckpt <global_step_6/actor> --probe-names

  # Loop over all steps for one condition (use run_weight_delta.sh wrapper):
  for STEP in 6 12 18 24 30 36 42 48 54 60 66 72 78 84 90 96; do
    python weight_delta.py --base $BASE --ckpt $CKPT_DIR/global_step_$STEP/actor \
                           --condition full --step $STEP --out $OUT
  done

Output CSV columns:
  condition, step, key, component, module, layer_idx, n_params,
  base_fro, abs_fro, rel_fro, mean_abs, cos_sim
"""

import argparse
import csv
import json
import os
import re
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

# Import the shared classifier (same directory).
sys.path.insert(0, os.path.dirname(__file__))
from qwen3vl_param_map import classify, print_summary


# ── Base model loading (multi-shard safetensors) ─────────────────────────────

def build_base_index(base_dir: str) -> tuple[dict, dict]:
    """
    Returns:
      handles    : {shard_filename -> safe_open handle}
      weight_map : {param_key -> shard_filename}
    """
    index_path = os.path.join(base_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    else:
        single = os.path.join(base_dir, "model.safetensors")
        assert os.path.exists(single), f"No safetensors found in {base_dir}"
        handle = safe_open(single, framework="pt", device="cpu")
        return {single: handle}, {k: single for k in handle.keys()}

    handles = {}
    for shard in set(weight_map.values()):
        path = os.path.join(base_dir, shard)
        handles[shard] = safe_open(path, framework="pt", device="cpu")
    return handles, weight_map


def load_base_param(handles: dict, weight_map: dict, key: str) -> torch.Tensor:
    shard = weight_map[key]
    return handles[shard].get_tensor(key).to(torch.float32)


# ── FSDP checkpoint reconstruction ───────────────────────────────────────────

def detect_world_size(actor_dir: str) -> int:
    """Read world_size from the filename pattern model_world_size_N_rank_0.pt."""
    for fname in os.listdir(actor_dir):
        m = re.match(r"model_world_size_(\d+)_rank_0\.pt", fname)
        if m:
            return int(m.group(1))
    raise FileNotFoundError(f"No model_world_size_*_rank_0.pt found in {actor_dir}")


def load_ckpt_shards(actor_dir: str, world_size: int) -> list[dict]:
    shards = []
    for rank in range(world_size):
        path = os.path.join(actor_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        shards.append(torch.load(path, map_location="cpu", weights_only=False))
    return shards


def reconstruct_param(shards: list[dict], key: str) -> torch.Tensor:
    """
    Reconstruct the full parameter tensor from FSDP rank shards.
    Mirrors model_merger.py:36-44,160-202 exactly.
    DTensor -> cat along shard dim (or take rank-0 if replicated).
    Plain tensor -> cat along dim 0 (FSDP flat-param sharding fallback).
    """
    tensors = [s[key] for s in shards]
    t0 = tensors[0]

    if hasattr(t0, "_local_tensor"):  # DTensor
        local = [t._local_tensor for t in tensors]
        placement = t0.placements[0]
        if placement.is_replicate():
            return local[0].to(torch.float32)
        elif placement.is_shard():
            return torch.cat(local, dim=placement.dim).contiguous().to(torch.float32)
        else:
            raise RuntimeError(f"Unexpected placement {placement} for key {key}")
    else:
        # Non-DTensor: if all shards are the same shape, they're replicated; else cat.
        if all(t.shape == t0.shape for t in tensors[1:]):
            return t0.to(torch.float32)
        return torch.cat(tensors, dim=0).to(torch.float32)


# ── Metrics ───────────────────────────────────────────────────────────────────

_EPS = 1e-12

def compute_metrics(w_base: torch.Tensor, w_ckpt: torch.Tensor) -> dict:
    delta = w_ckpt - w_base
    base_fro = w_base.norm().item()
    abs_fro  = delta.norm().item()
    rel_fro  = abs_fro / max(base_fro, _EPS)
    mean_abs = delta.abs().mean().item()
    cos_sim  = F.cosine_similarity(
        w_base.reshape(1, -1), w_ckpt.reshape(1, -1)
    ).item()
    return dict(
        base_fro=base_fro,
        abs_fro=abs_fro,
        rel_fro=rel_fro,
        mean_abs=mean_abs,
        cos_sim=cos_sim,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "condition", "step", "key", "component", "module", "layer_idx",
    "n_params", "base_fro", "abs_fro", "rel_fro", "mean_abs", "cos_sim",
]


def main():
    parser = argparse.ArgumentParser(description="Compute per-param weight deltas.")
    parser.add_argument("--base", required=True,
                        help="Path to the base model directory (HF safetensors).")
    parser.add_argument("--ckpt", required=True,
                        help="Path to global_step_N/actor/ directory.")
    parser.add_argument("--condition", default="unknown",
                        help="Condition name: full | llm_only | vit_only.")
    parser.add_argument("--step", type=int, default=-1,
                        help="Training step number (for CSV labeling).")
    parser.add_argument("--out", default="deltas.csv",
                        help="Output CSV path (append mode).")
    parser.add_argument("--probe-names", action="store_true",
                        help="Print param classifications only; no delta computation.")
    args = parser.parse_args()

    # ── Load base index ───────────────────────────────────────────────────────
    print(f"[weight_delta] base     : {args.base}")
    print(f"[weight_delta] ckpt     : {args.ckpt}")
    print(f"[weight_delta] condition: {args.condition}  step: {args.step}")
    handles, weight_map = build_base_index(args.base)
    base_keys = sorted(weight_map.keys())
    print(f"[weight_delta] base params: {len(base_keys)}")

    # ── Always print classification summary (Step 0 verification) ────────────
    print_summary(base_keys, label=f"base model ({len(base_keys)} keys)")

    if args.probe_names:
        print("\n[probe-names] done — no delta computation.")
        return

    # ── Load FSDP shards ─────────────────────────────────────────────────────
    world_size = detect_world_size(args.ckpt)
    print(f"[weight_delta] ckpt world_size: {world_size}")
    print(f"[weight_delta] loading {world_size} rank shards …")
    shards = load_ckpt_shards(args.ckpt, world_size)

    ckpt_keys = set(shards[0].keys())
    missing = set(base_keys) - ckpt_keys
    extra   = ckpt_keys - set(base_keys)
    if missing:
        print(f"  WARNING: {len(missing)} base keys missing in ckpt: {sorted(missing)[:5]}…")
    if extra:
        print(f"  WARNING: {len(extra)} extra ckpt keys not in base (ignored): {sorted(extra)[:5]}…")

    # ── CSV setup (append mode) ───────────────────────────────────────────────
    write_header = not os.path.exists(args.out)
    out_f = open(args.out, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    # ── Main loop (one param at a time to bound memory) ───────────────────────
    n_done = 0
    for key in base_keys:
        if key not in ckpt_keys:
            continue
        w_base = load_base_param(handles, weight_map, key)
        w_ckpt = reconstruct_param(shards, key)

        if w_base.shape != w_ckpt.shape:
            print(f"  SHAPE MISMATCH {key}: base {w_base.shape} vs ckpt {w_ckpt.shape} — skipping")
            del w_base, w_ckpt
            continue

        comp, mod, idx = classify(key)
        n_params = w_ckpt.numel()
        metrics = compute_metrics(w_base, w_ckpt)
        del w_base, w_ckpt

        writer.writerow(dict(
            condition=args.condition,
            step=args.step,
            key=key,
            component=comp,
            module=mod,
            layer_idx="" if idx is None else idx,
            n_params=n_params,
            **metrics,
        ))

        n_done += 1
        if n_done % 50 == 0:
            print(f"  … {n_done}/{len(base_keys)} params processed")

    out_f.flush()
    out_f.close()
    print(f"[weight_delta] done. {n_done} params written to {args.out}")


if __name__ == "__main__":
    main()
