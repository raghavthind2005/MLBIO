"""Gate G-VITATTN: prove the vLLM ViT-attention patch is actually in force.

WHY THIS EXISTS, precisely. Job 3167568 passed `mm_encoder_attn_backend="TORCH_SDPA"`.
vLLM accepted it, echoed it back in its own non-default-args log line, and then reverted it
internally -- and the run failed *identically* to the unpatched job 3167519. Every log line
looked correct. The setting was applied and ignored.

That is the exact failure class G-PIXELS was built for, so it gets the same treatment: the
outcome is measured, not assumed. See ``patches/vllm_0_11_2/README.md``.

Two independent checks, both cheap, both BEFORE the engine is constructed so a mount failure
costs one second instead of the ninety it takes to load weights and reach the profile run:

  IDENTITY  the ``vllm.attention.layer`` module actually imported hashes to our patched file.
            Catches a mount that silently did not land.
  BEHAVIOUR the patched function really returns TORCH_SDPA under an override on this
            platform. Catches a file that is present but wrong, and is the property we
            actually depend on rather than a proxy for it.

Identity without behaviour would pass a patch that mounted correctly but did not work.
Behaviour without identity would pass if some future vLLM fixed this upstream -- which is
fine and worth knowing, hence the two are reported separately rather than merged.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

#: Path of the patched file INSIDE the container. Also the mount target.
CONTAINER_LAYER_PATH = "usr/local/lib/python3.12/dist-packages/vllm/attention/layer.py"

#: sha256 of the file as shipped in easyr1_vllm0112.sqsh, i.e. what we patched AGAINST.
#: If the image is rebuilt this stops matching, and the patch must be re-derived rather
#: than silently masking a newer file.
ORIGINAL_SHA256 = "11d6e56009e8dcb84ce0ac11393a45bc70cc30abf259667db80a5752e36e1ad8"

IMAGE = "/capstor/store/cscs/swissai/a0174/ce-images/easyr1_vllm0112.sqsh"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_container_original(image: str = IMAGE, workdir: str | None = None) -> str:
    """Assert the image still ships the file this patch was derived from.

    A bind-mount overrides whatever the image ships. If the image is ever rebuilt, a stale
    patched file would mask the new one INVISIBLY -- the run would succeed while executing
    code that no longer corresponds to the container. Extracting the single file costs a
    couple of seconds and turns that into a loud failure.

    Run from a login node (needs ``unsquashfs``), not from inside the container.
    """
    import tempfile

    tmp = workdir or tempfile.mkdtemp(prefix="ca21_sqsh_")
    subprocess.run(
        ["unsquashfs", "-q", "-f", "-d", tmp, image, CONTAINER_LAYER_PATH],
        check=True, capture_output=True,
    )
    got = sha256_file(Path(tmp) / CONTAINER_LAYER_PATH)
    if got != ORIGINAL_SHA256:
        raise AssertionError(
            f"container image no longer ships the layer.py this patch was built against.\n"
            f"  expected {ORIGINAL_SHA256}\n"
            f"  found    {got}\n"
            f"The patch must be re-derived against the new file. Mounting the old one would "
            f"silently revert whatever changed in the rebuild."
        )
    return got


def assert_vit_attn_patch(expect_sha256: str | None = None, verbose: bool = True) -> dict:
    """G-VITATTN. Raises AssertionError unless the patch is present AND effective.

    ``expect_sha256`` is the patched file's hash; when given, the identity check is enforced
    rather than merely reported.
    """
    from vllm.attention.backends.registry import AttentionBackendEnum
    from vllm.attention.layer import maybe_get_vit_flash_attn_backend
    import vllm.attention.layer as _layer

    module_path = _layer.__file__
    module_sha = sha256_file(module_path)

    # BEHAVIOUR. This is the property we depend on; the identity check is corroboration.
    # Ask for SDPA the same way the model does and confirm it survives the round trip.
    backend, fn = maybe_get_vit_flash_attn_backend(
        AttentionBackendEnum.TORCH_SDPA,
        False,
        attn_backend_override=AttentionBackendEnum.TORCH_SDPA,
    )

    if verbose:
        print(f"  [G-VITATTN] vllm.attention.layer -> {module_path}", flush=True)
        print(f"  [G-VITATTN] sha256 {module_sha}", flush=True)
        print(f"  [G-VITATTN] override TORCH_SDPA -> {backend}, "
              f"varlen_fn={'None' if fn is None else fn}", flush=True)

    if backend != AttentionBackendEnum.TORCH_SDPA:
        raise AssertionError(
            f"G-VITATTN FAILED: asked for TORCH_SDPA, got {backend}.\n"
            f"The patch is not in force -- vLLM is reverting the override, so the ViT will "
            f"call the flash_attn build that cannot serve Qwen2.5-VL's head_dim=80 and the "
            f"run would die ~90 s from now in the profile pass.\n"
            f"  module: {module_path}\n"
            f"  sha256: {module_sha}\n"
            f"Check the mount in the .toml actually landed on {CONTAINER_LAYER_PATH}."
        )

    # With SDPA selected, no flash-attn entry point should have been imported at all.
    if fn is not None:
        raise AssertionError(
            f"G-VITATTN FAILED: backend is TORCH_SDPA but a flash_attn varlen function was "
            f"still bound ({fn}). The SDPA path must not carry one."
        )

    # THE CASE THAT ACTUALLY RUNS, and which this gate did not previously cover.
    #
    # The check above asks for TORCH_SDPA explicitly. Nothing in EasyR1 can do that:
    # vllm_rollout_spmd.py:120-124 builds engine_kwargs with only
    # disable_mm_preprocessor_cache and limit_mm_per_prompt, and RolloutConfig has no
    # passthrough. So production calls this with attn_backend_override=None and
    # attn_backend=FLASH_ATTN -- which is how job 3177693 reached the ViT with a flash-attn
    # build that cannot serve head_dim=80, while this gate would have reported PASS.
    #
    # A gate that only exercises a path production never takes is not a gate.
    backend_default, fn_default = maybe_get_vit_flash_attn_backend(
        AttentionBackendEnum.FLASH_ATTN,
        False,
        attn_backend_override=None,
    )
    if verbose:
        print(f"  [G-VITATTN] no override (as production calls it) -> {backend_default}, "
              f"varlen_fn={'None' if fn_default is None else fn_default}", flush=True)

    if backend_default != AttentionBackendEnum.TORCH_SDPA or fn_default is not None:
        raise AssertionError(
            f"G-VITATTN FAILED: with NO override -- the way verl actually calls this -- the "
            f"ViT resolved to {backend_default} (varlen_fn="
            f"{'None' if fn_default is None else fn_default}), not TORCH_SDPA.\n"
            f"CA21 PATCH 2 is not in force, so the run will die in the ViT with "
            f"'headdim not being a multiple of 32' once vLLM initialises.\n"
            f"  module: {module_path}\n"
            f"  sha256: {module_sha}\n"
            f"Check the mount in the .toml landed on {CONTAINER_LAYER_PATH} and that the "
            f"mounted file is the CURRENT patch (PATCHED.sha256 changed when PATCH 2 was "
            f"added; a stale mount would pass the identity check against the old pin)."
        )

    patched = expect_sha256 is not None and module_sha == expect_sha256
    if expect_sha256 is not None and not patched:
        raise AssertionError(
            f"G-VITATTN FAILED (identity): behaviour is correct but the imported module is "
            f"not our patched file.\n"
            f"  expected {expect_sha256}\n"
            f"  found    {module_sha}  at {module_path}\n"
            f"Either the mount did not land and upstream has since fixed this, or a "
            f"different patch is in play. Both need a human before results are trusted."
        )

    if verbose:
        print("  [G-VITATTN] PASS -- override honoured, no flash_attn bound", flush=True)

    return {"module_path": module_path, "module_sha256": module_sha,
            "backend": str(backend), "identity_checked": bool(expect_sha256)}


if __name__ == "__main__":
    # CLI so a job script can run this BEFORE the expensive step.
    #
    # This gate existed and tested the right property, and job 3177693 still died on exactly
    # the failure it guards -- because nothing invoked it. It cost a four-GPU allocation
    # through a full model load to learn something one GPU could have reported in seconds.
    # Cheap gates have to run before expensive ones, and a gate with no entry point is not
    # a gate.
    #
    #   python3 container_gate.py [expected_sha256_of_patched_layer.py]
    #
    # Needs CUDA: the branch under test is current_platform.is_cuda().
    import sys

    expected = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        info = assert_vit_attn_patch(expected)
    except AssertionError as exc:
        print(f"\n{exc}", flush=True)
        sys.exit(1)
    print(f"[gate] G-VITATTN OK  backend={info['backend']}  "
          f"identity_checked={info['identity_checked']}", flush=True)
    sys.exit(0)
