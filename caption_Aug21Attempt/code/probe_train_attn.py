"""H3: does the HF training path load and step Qwen2.5-VL with a usable ViT kernel?

WHY THIS IS A SEPARATE, TINY JOB. vLLM's rollout path needed a container patch (S10)
because Qwen2.5-VL's vision tower has head_dim = 1280/16 = 80 and the bundled flash_attn
build rejects anything not a multiple of 32. verl trains through **HF, not vLLM**, and
loads with a global `attn_implementation="flash_attention_2"` (`fsdp_workers.py:215`)
that reaches the vision tower.

Read from the container source, `Qwen2_5_VLVisionAttention` computes head_dim = 80
(`modeling_qwen2_5_vl.py:187`) and dispatches on `config._attn_implementation` with an
explicit `flash_attention_2` branch (`:216-219`) into the same package. So it *should*
fail identically. But "should" is exactly the word that produced two dead jobs already:
the S10 fix was asserted twice on source reading that was correct as far as it went and
still wrong about the outcome. So this measures rather than infers.

THE CANDIDATE FIX is better than blanket-SDPA. transformers 4.57 accepts a per-subconfig
dict (`modeling_utils.py:2802-2838`):

    attn_implementation={"": "flash_attention_2", "vision_config": "sdpa"}

keeping FA2 on the language model (head_dim 128, works and is faster) and dropping to
SDPA only in the ViT, which is the one component that cannot run it.

Both arms are attempted and reported. A pass on the dict form with a fail on the global
form is the outcome that confirms both the diagnosis and the fix; anything else means the
model is wrong about itself and the plan needs revisiting before a smoke, not after.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def try_arm(model_path: str, attn_impl, label: str, dtype_str: str = "bfloat16") -> dict:
    """Load, forward, backward. Returns a verdict dict; never raises."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"\n{'=' * 68}\n[{label}] attn_implementation = {attn_impl!r}\n{'=' * 68}",
          flush=True)
    rec: dict = {"label": label, "attn_implementation": attn_impl}
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=getattr(torch, dtype_str),
            attn_implementation=attn_impl,
            device_map="cuda:0",
        )
        model.gradient_checkpointing_enable()
        model.train()

        # What the model believes it is doing, per component. Recorded because the
        # whole S10 lesson is that a setting can be accepted and then ignored.
        cfg = model.config
        rec["resolved"] = {
            "top": getattr(cfg, "_attn_implementation", None),
            "vision": getattr(getattr(cfg, "vision_config", None),
                              "_attn_implementation", None),
        }
        print(f"  resolved: {rec['resolved']}", flush=True)

        processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=3136, max_pixels=4_194_304)

        # A real image at the largest resolution the dev pool actually contains
        # (§4.5 measured grid_thw [1,116,180] -> 5,220 visual tokens), so the ViT
        # runs at the scale training will hit, not a 64x64 toy.
        from PIL import Image
        import numpy as np
        rng = np.random.default_rng(0)
        img = Image.fromarray(
            rng.integers(0, 255, (700, 1371, 3), dtype=np.uint8), mode="RGB")

        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": "What is shown? Answer briefly."}]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        batch = processor(text=[text], images=[img], return_tensors="pt").to("cuda:0")

        labels = batch["input_ids"].clone()
        out = model(**batch, labels=labels)
        loss = out.loss
        print(f"  forward OK, loss = {loss.item():.4f}", flush=True)

        loss.backward()
        gnorm = sum(p.grad.float().pow(2).sum().item()
                    for p in model.parameters() if p.grad is not None) ** 0.5
        print(f"  backward OK, grad-norm = {gnorm:.4f}", flush=True)

        if not (gnorm > 0):
            raise AssertionError(
                f"grad norm is {gnorm}; a step that produces no gradient is not a pass")

        rec.update(ok=True, loss=loss.item(), grad_norm=gnorm,
                   visual_tokens=int(batch["image_grid_thw"].prod(dim=-1).sum()) // 4)
        print(f"  [{label}] PASS", flush=True)

    except Exception as e:  # noqa: BLE001 -- the failure IS the measurement
        rec.update(ok=False, error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-2500:])
        print(f"  [{label}] FAIL -- {type(e).__name__}: {e}", flush=True)
    finally:
        try:
            del model
        except Exception:
            pass
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms = [
        # verl's current behaviour. Expected to FAIL on head_dim=80.
        ("global_fa2", "flash_attention_2"),
        # The proposed fix: FA2 everywhere except the vision tower.
        ("per_subconfig", {"": "flash_attention_2", "vision_config": "sdpa"}),
        # Fallback, and a control: if even this fails the problem is not the ViT.
        ("global_sdpa", "sdpa"),
    ]
    results = [try_arm(args.model, impl, label) for label, impl in arms]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 68}\n=== H3 VERDICT ===\n{'=' * 68}")
    for r in results:
        mark = "PASS" if r.get("ok") else "FAIL"
        extra = (f"loss {r['loss']:.4f}  gnorm {r['grad_norm']:.3f}"
                 if r.get("ok") else r.get("error", "")[:90])
        print(f"  {mark}  {r['label']:<14} {extra}")
        if r.get("ok"):
            print(f"        resolved: {r['resolved']}")

    by = {r["label"]: bool(r.get("ok")) for r in results}
    print()
    if by.get("per_subconfig"):
        if not by.get("global_fa2"):
            print("  CONFIRMED: verl's global flash_attention_2 fails on this ViT, and")
            print("  the per-subconfig dict fixes it. Apply it at the verl load site.")
        else:
            print("  NOTE: global FA2 also passed, so this container's flash_attn does")
            print("  NOT restrict head_dim on the HF path the way vLLM's bundled build")
            print("  does. The dict is then unnecessary -- prefer the unmodified path.")
    else:
        print("  per_subconfig FAILED. The plan for the training path is wrong; do not")
        print("  proceed to a smoke until this is understood.")

    print(f"\n[done] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
