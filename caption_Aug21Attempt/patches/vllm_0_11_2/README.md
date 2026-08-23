# Patch: vLLM 0.11.2 ignores `mm_encoder_attn_backend` on CUDA

**Status:** active, required for any Qwen2.5-VL run in `easyr1_vllm0112.sqsh`.
**Scope:** one logical line in `vllm/attention/layer.py`.

## The failure

Qwen2.5-VL's vision tower has `head_dim = hidden_size / num_heads = 1280 / 16 = 80`, and the
FlashAttention kernel vLLM reaches for the ViT refuses anything not a multiple of 32:

```
RuntimeError: This flash attention build does not support headdim not being a multiple of 32.
  vllm/model_executor/models/qwen2_5_vl.py:400 -> vit_flash_attn_wrapper -> flash_attn_varlen_func
```

Observed in jobs **3167519** and **3167568**. The container was built for the Qwen3-VL/EasyR1
line, whose ViT has different head geometry, which is why this never surfaced before.

### ⚠️ Corrected 2026-08-24: which kernel is actually at fault

This section originally read *"the `flash_attn` build in this container was compiled with a
reduced head-dim set."* **That attribution was wrong**, disproved by job 3168489 (H3):

- HF's `Qwen2_5_VLVisionAttention` with `attn_implementation="flash_attention_2"` dispatches
  into the **standalone `flash_attn` package** at the same `head_dim=80` — and it **passes**,
  forward and backward, at 5,220 visual tokens.
- So the standalone package handles `head_dim=80` perfectly well.

vLLM has **two** entry points (`attention/layer.py:131-142`): `use_upstream_fa=True` imports
`from flash_attn import flash_attn_varlen_func` (the standalone package), and `False` imports
`from vllm.attention.utils.fa_utils import flash_attn_varlen_func` — **vLLM's own bundled
build**. Since the standalone package is demonstrably fine, the restriction lives in the
bundled one.

Consistent with the traces. In job 3167519 no override was passed, so
`get_vit_attn_backend` returned `FLASH_ATTN`, the CUDA branch condition
(`attn_backend != FLASH_ATTN`) was already false, `use_upstream_fa` stayed `False`, and the
**bundled** kernel was imported. In 3167568 the transformer flipped to upstream — but
`maybe_get_vit_flash_attn_backend` **does not return** the updated `use_upstream_fa`, so
`Qwen2_5_VisionTransformer`'s local stayed `False` and was passed down to every block
(`qwen2_5_vl.py:709`), which re-derived the bundled kernel again. A second propagation bug in
the same function family, and the reason the second job failed identically to the first.

*The patch is unaffected* — it selects SDPA and never reaches either kernel. Only the
explanation needed fixing, and a patch note that misidentifies its own cause is how someone
later repairs the wrong thing.

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

## The training path — RESOLVED, and not the way I predicted **[V] job 3168489**

This section previously predicted that verl's HF training path would "hit this identically."
**It does not.** All three probed arms load, forward, backward and produce non-zero gradients
at 5,220 visual tokens:

| arm | LM kernel | ViT kernel | loss | grad-norm |
|---|---|---|---|---|
| `global_fa2` (verl's default) | FA2 | **FA2** | 18.1687 | 295.35 |
| `per_subconfig` | FA2 | SDPA | 18.1538 | 472.47 |
| `global_sdpa` | SDPA | SDPA | 18.1523 | 525.15 |

`global_fa2` passing **is** the disproof above: HF at `head_dim=80` through the standalone
package is fine.

**We nevertheless use `per_subconfig`**, for a reason that has nothing to do with crashes:

```python
attn_implementation={"": "flash_attention_2", "vision_config": "sdpa"}
```

vLLM generates rollouts with **FA2 on the language model** (logged: `Using FLASH_ATTN
backend`, `head_dim=128`) and **SDPA on the ViT** (forced by this patch). In GRPO the rollout
log-probs and the training-time recomputed log-probs are meant to describe the *same policy*,
and verl already contends with a known rollout/training numerical gap. `per_subconfig` is the
**only** arm that matches rollout on both components — `global_fa2` mismatches the ViT,
`global_sdpa` mismatches the language model.

*Honest limit:* this minimises the mismatch rather than eliminating it. vLLM's `FLASH_ATTN`
is its own bundled `_vllm_fa2_C`; HF's is the upstream package. Same algorithm family,
different builds. Exact agreement is not available.

## ⬜ Open: the grad-norm spread is not explained

Identical weights, identical seeded input, **only the kernels differ** — yet grad norms span
**295 → 525 (+78%)** while losses agree to four significant figures. Both the ViT kernel
(295 → 472) and the language-model kernel (472 → 525) move it.

The likely explanation is bf16 accumulation on a **random-noise image**, which drives ViT
attention toward uniform and is close to a worst case for cancellation — a stress input, not
a representative one. But this document asserts kernel-neutrality above, and a 78% spread
should not sit unexplained beneath that claim. **To check on a real pool image**, folded into
the next job rather than given a dedicated one.
