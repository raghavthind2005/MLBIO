# Patch: vLLM 0.11.2 ignores `mm_encoder_attn_backend` on CUDA

**Status:** active, required for any Qwen2.5-VL run in `easyr1_vllm0112.sqsh`.
**Scope:** one logical line in `vllm/attention/layer.py`.

## The failure

Qwen2.5-VL's vision tower has `head_dim = hidden_size / num_heads = 1280 / 16 = 80`. The
`flash_attn` build in this container was compiled with a reduced head-dim set and refuses
anything that is not a multiple of 32:

```
RuntimeError: This flash attention build does not support headdim not being a multiple of 32.
  vllm/model_executor/models/qwen2_5_vl.py:400 -> vit_flash_attn_wrapper -> flash_attn_varlen_func
```

Observed in jobs **3167519** and **3167568**. The container was built for the Qwen3-VL/EasyR1
line, whose ViT has different head geometry, which is why this never surfaced before.

## Why the documented fix does not work

Passing `mm_encoder_attn_backend="TORCH_SDPA"` is accepted by `EngineArgs`, echoed back in
vLLM's own non-default-args log line, and **plumbed correctly** as far as
`get_vit_attn_backend`, which honours it (`models/vision.py:91`):

```python
if attn_backend_override is not None:
    return attn_backend_override          # -> TORCH_SDPA
```

The very next call throws it away. `maybe_get_vit_flash_attn_backend`, CUDA branch
(`attention/layer.py:116`, unpatched):

```python
elif current_platform.is_cuda():
    if (attn_backend != AttentionBackendEnum.FLASH_ATTN
        and check_upstream_fa_availability(torch.get_default_dtype())):
        attn_backend = AttentionBackendEnum.FLASH_ATTN   # <- reverts the override
        use_upstream_fa = True                           # <- `from flash_attn import ...`
```

`attn_backend_override` is never consulted. So the override is honoured once and then
unconditionally reverted, and `use_upstream_fa=True` selects the standalone `flash_attn`
build — exactly the build that cannot serve `head_dim=80`.

This is why job 3167568 failed **identically** to 3167519 despite the config change: the
setting was applied, logged, and ignored. A silently-ignored config is indistinguishable
from a working one in the logs, which is the whole reason gate **G-VITATTN** exists below.

## The patch

The guard already exists — in the ROCm branch of the same function, ten lines above
(`layer.py:106-110`, `and attn_backend_override is None`). It was omitted on CUDA. The patch
copies it across:

```diff
         if (
             attn_backend != AttentionBackendEnum.FLASH_ATTN
             and check_upstream_fa_availability(torch.get_default_dtype())
+            and attn_backend_override is None
         ):
```

We are resolving an internal inconsistency in one function, not imposing a preference. This
is a bugfix, not a reimplementation, and not a behavioural change for anyone who does not
pass an explicit override: with `attn_backend_override is None` the condition is unchanged.

## Verified consequence, traced end to end

With the guard in place and `attn_backend_override=TORCH_SDPA`:

1. `get_vit_attn_backend` returns `TORCH_SDPA` (`vision.py:91`).
2. The CUDA branch condition is now `False`, so no revert.
3. Control falls to `layer.py:131`; `TORCH_SDPA` is not in `{FLASH_ATTN, ROCM_AITER_FA}`,
   so `flash_attn_varlen_func = None` and the function returns `(TORCH_SDPA, None)`.
4. `Qwen2_5_VisionTransformer` propagates `attn_backend=self.attn_backend` to every block
   (`qwen2_5_vl.py:708`).
5. `Qwen2_5_VisionAttention.__init__` recomputes the same result, so
   `is_flash_attn_backend = False` (`qwen2_5_vl.py:357`).
6. `forward` takes the `elif self.attn_backend == TORCH_SDPA` branch (`qwen2_5_vl.py:410`).

`flash_attn_varlen_func` is never reached.

## Is this scientifically neutral?

Yes, on the two grounds that matter here.

**Exactness.** SDPA and FlashAttention both compute exact softmax attention. They differ in
tiling and summation order, so outputs differ only at bf16 rounding level. Neither is an
approximation of the other. `TORCH_SDPA` is also vLLM's own class default for
`Qwen2_5_VisionAttention` (`qwen2_5_vl.py:301`), so this selects the library's declared
default rather than an exotic path.

**Comparability.** Vision-SR1 ran standard NVIDIA builds where `head_dim=80` is supported, so
their ViT used FA2 and ours uses SDPA. This does not threaten our design, because we never
test our result against their 47.1 directly — we run our own control arm (S1/O8: accuracy-only
GRPO vs accuracy + caption-KL, identical data, steps and seed, same container). What the
contrast requires is that the kernel be identical *across our arms*, not identical to theirs.
It cancels. 47.1 is a landmark, not a bitwise target, and could not be reproduced bitwise on
different hardware regardless.

There is in any case no more faithful option available: the FA2 build in this container
cannot execute this ViT at all. The choice is SDPA or nothing.

## Upstream status

No upstream fix exists to adopt, checked 2026-08-23:

- [vllm#27821](https://github.com/vllm-project/vllm/issues/27821) — RFC reorganising ViT
  attention selection; lists *honouring* `--mm-encoder-attn-backend` as a future goal, which
  is itself an admission it is not reliably honoured today.
- [vllm#38411](https://github.com/vllm-project/vllm/issues/38411) — structurally identical
  crash (FA2 build incompatible with the hardware, "no equivalent override" for the vision
  encoder). **Closed as not planned.**

## Reproducibility controls

A bind-mount silently overrides whatever the image ships. If the image is ever rebuilt, a
stale patched file would mask the new one invisibly. Three controls prevent that:

1. **This file is version-controlled** — the diff is in git, not a mystery mount.
2. **`ORIGINAL.sha256` pins the file this was patched against.** `verify_container_original`
   in `code/container_gate.py` extracts that single file from the `.sqsh` and refuses to run
   if it no longer matches. Costs seconds.
   Run it as (the login node's default `python3` is 3.6.15 and cannot parse
   `from __future__ import annotations`; verified against the live image 2026-08-23):

   ```
   ssh clariden
   cd $CA21/code && python3.11 -c "import container_gate as G; print(G.verify_container_original())"
   ```

3. **Gate G-VITATTN asserts the outcome rather than trusting it** — see
   `assert_vit_attn_patch`. It checks that the imported `layer.py` is the patched one *and*
   that the function actually returns `TORCH_SDPA` under an override. It runs before the
   engine is built, so a mount failure costs one second instead of ninety.

## Provenance

| | |
|---|---|
| image | `/capstor/store/cscs/swissai/a0174/ce-images/easyr1_vllm0112.sqsh` |
| file | `usr/local/lib/python3.12/dist-packages/vllm/attention/layer.py` (1052 lines) |
| original sha256 | `11d6e56009e8dcb84ce0ac11393a45bc70cc30abf259667db80a5752e36e1ad8` |
| patched sha256 | see `PATCHED.sha256` |
| authored | 2026-08-23, after jobs 3167519 and 3167568 |

## Not yet resolved: the training path

verl loads the policy through HF, not vLLM, with `attn_implementation="flash_attention_2"`
applied globally (`fsdp_workers.py:215`), which reaches the vision tower. vLLM's
`use_upstream_fa` path imports from *the same* `flash_attn` package that just rejected
`head_dim=80`, so training is expected to hit this identically.

**Expected, NOT verified.** If it holds, the fix there is a config change (`sdpa`), not a
patch — this patch is specific to vLLM's rollout path.
