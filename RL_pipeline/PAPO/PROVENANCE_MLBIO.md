# PAPO — vendored provenance (MLBIO)

This directory is a **read-only vendored copy** of the PAPO codebase, cloned for the
perception-degradation RL pipeline work. Do not edit files here as if they were ours;
treat upstream as the source of truth and record any local patch explicitly (mirroring
how `RL_SeeingToThinking/EasyR1` is handled).

## Upstream source
- Repo: https://github.com/MikeWangWZHL/PAPO.git
- Branch: `main_qwen3` (Qwen3-VL support branch)
- Commit: `1263a29cae716a3ed7a94df3dc2ca7b83ca85e14` ("Merge pull request #25 from SStoica12/main_qwen3")
- Vendored: 2026-07-21
- Paper: PAPO — Perception-Aware Policy Optimization for Multimodal Reasoning, arXiv:2507.06448

## What was removed vs. the upstream clone (packaging only — no code/logic changed)
- `.git/` (21M history) — so files are tracked directly by the MLBIO repo
- `static/`, `index.html` — project webpage assets (not part of the pipeline)
- `PAPO.egg-info/`, `.DS_Store` — build/OS cruft
- `PAPO-Eval/` submodule stub + `.gitmodules` — evaluation harness is a separate repo
  (https://github.com/xhguo7/PAPO-Eval); vendor it separately if/when we run PAPO's evals.

Everything under `verl/`, `examples/`, `scripts/`, `data/` (pointers only), `pyproject.toml`,
`requirements.txt`, `environment.yaml`, `setup.py`, `README.md` is upstream-unmodified,
EXCEPT the local patches documented below.

## Local patches (MLBIO)
- **Conv3d→matmul patch-embed fix (2026-07-21), RESULT-NEUTRAL (bit-identical, maxdiff=0.0).**
  - `verl/models/transformers/qwen3_vl.py`: added `qwen3_vl_patch_embed_forward` (matmul reimpl of the kernel==stride Conv3d patch embed).
  - `verl/models/monkey_patch.py`: in the `qwen3_vl` branch, import `Qwen3VLVisionPatchEmbed` +
    the new fn, and install `Qwen3VLVisionPatchEmbed.forward = qwen3_vl_patch_embed_forward`.
  - Why: on GH200/aarch64+cuDNN the stock Conv3d patch-embed is ~3.4e5x slower than the
    equivalent GEMM (conv3d 56,708 ms vs matmul 0.17 ms @N=10k) — the entire training-speed
    wall. PAPO pays it twice/step (real + masked image). Mirrors the same fix in
    `RL_SeeingToThinking/EasyR1`. Rides PAPO's existing `apply_ulysses_patch()` (fsdp_workers.py:187).
  - Verify on-cluster before any run: bit-identity assert (stock vs patched, maxdiff==0) +
    applied-check (`Qwen3VLVisionPatchEmbed.forward is qwen3_vl_patch_embed_forward`).

## Upstream dependency pins (from pyproject.toml — x86_64/cu124 reference stack)
- torch==2.6.0, vllm==0.8.4, transformers==4.51.3, flash-attn==2.7.4.post1, python>=3.10
- NOTE: this differs from our cluster EasyR1 image `easyr1_vllm0112.sqsh`
  (torch 2.9/cu129, vllm 0.11.2, transformers 4.57.3, flash-attn 2.8.3, aarch64).
  Compatibility must be smoke-tested before trusting the existing image (see session notes).

## Data / models (NOT vendored — pull on cluster scratch via HuggingFace)
- Train: `PAPOGalaxy/PAPO_ViRL39K_train`
- Val:   `PAPOGalaxy/PAPO_MMK12_test`
- Backbone (example): `Qwen/Qwen3-VL-2B-Thinking`; our locked backbone is Qwen3-VL-4B-Instruct
  (config adaptation + reward/format compatibility to be decided before any run).
