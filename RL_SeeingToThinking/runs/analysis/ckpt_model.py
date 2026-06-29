"""
ckpt_model.py — load a full Qwen3-VL model from base + (optional) FSDP checkpoint.

Reuses weight_delta's shard-reconstruction so we DON'T need to merge checkpoints to disk:
we rebuild the full state dict in memory and load_state_dict into a model instantiated from
the base config. Applies the same conv->matmul patch_embed fix used in training so the
vision forward is fast on aarch64.

Requires the container (torch + transformers 4.57.3). GPU recommended for inference.

  load_model(base_dir)                         -> base model (step 0)
  load_model(base_dir, ckpt_actor_dir=...)     -> trained model at that checkpoint
  load_model(base_dir, graft_state_dict=...)   -> model with custom (grafted) weights
"""

import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from weight_delta import detect_world_size, load_ckpt_shards, reconstruct_param  # reuse


# ── conv->matmul patch (identical math to training; maxdiff=0) ────────────────
def _install_patch_embed_fix():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed

    def patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.proj.weight.reshape(self.embed_dim, -1)
        hidden_states = hidden_states.to(dtype=weight.dtype).reshape(-1, weight.shape[1])
        return torch.nn.functional.linear(hidden_states, weight, self.proj.bias)

    Qwen3VLVisionPatchEmbed.forward = patched


def reconstruct_full_state_dict(actor_dir: str) -> dict:
    """Full {key: tensor} from the 4 FSDP rank shards (bf16)."""
    world_size = detect_world_size(actor_dir)
    shards = load_ckpt_shards(actor_dir, world_size)
    keys = list(shards[0].keys())
    out = {}
    for k in keys:
        out[k] = reconstruct_param(shards, k).to(torch.bfloat16)  # reconstruct_param returns fp32
    return out


def load_model(base_dir: str,
               ckpt_actor_dir: str = None,
               graft_state_dict: dict = None,
               device: str = "cuda",
               dtype=torch.bfloat16):
    """
    Instantiate Qwen3-VL from base_dir; optionally overwrite weights with a checkpoint
    (ckpt_actor_dir) or a custom grafted state dict (graft_state_dict). Returns (model, processor).
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    _install_patch_embed_fix()

    print(f"[ckpt_model] instantiating from base: {base_dir}")
    model = AutoModelForImageTextToText.from_pretrained(
        base_dir, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(base_dir)

    sd = None
    if graft_state_dict is not None:
        sd = graft_state_dict
        print("[ckpt_model] loading GRAFTED state dict")
    elif ckpt_actor_dir is not None:
        print(f"[ckpt_model] reconstructing weights from ckpt: {ckpt_actor_dir}")
        sd = reconstruct_full_state_dict(ckpt_actor_dir)

    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # lm_head is tied -> base may not list it separately; that's fine.
        miss_real = [m for m in missing if "lm_head" not in m]
        if miss_real:
            print(f"  WARNING missing keys ({len(miss_real)}): {miss_real[:5]} …")
        if unexpected:
            unexp_real = [u for u in unexpected if "lm_head" not in u]
            if unexp_real:
                print(f"  WARNING unexpected keys ({len(unexp_real)}): {unexp_real[:5]} …")

    model.to(device).eval()
    print(f"[ckpt_model] ready on {device}, dtype={dtype}")
    return model, processor


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Sanity-load a model and report param count.")
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", default=None, help="global_step_N/actor dir (optional)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    model, processor = load_model(args.base, ckpt_actor_dir=args.ckpt, device=args.device)
    n = sum(p.numel() for p in model.parameters())
    print(f"total params: {n/1e9:.3f}B")
    print("OK")
