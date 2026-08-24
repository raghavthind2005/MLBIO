"""Attach the caption method inside EVERY Ray worker process.

WHY A HOOK RATHER THAN JUST CALLING make_ca21_worker. main_ca21 calls it in the Runner
actor, which patches the FSDPWorker class object *in that process*. Ray does not ship the
patched class to the workers -- `ray.remote(FSDPWorker)` serialises a class REFERENCE, and
each WorkerDict process re-imports `verl.workers.fsdp_workers` from disk, getting the
pristine class. So dispatch found the method on WorkerDict (bound in the driver by
_bind_workers_method_to_parent) while the underlying instance did not have it:

    AttributeError: 'FSDPWorker' object has no attribute 'compute_caption_distortion'

which is what job 3178190 hit, one call after caption generation finally succeeded.

`worker_process_setup_hook` runs in each worker before any task, which is the only place
the attachment can happen for the instance that actually executes it.

Subclassing would avoid all of this, and cannot be used: create_colocated_worker_cls builds
`WorkerDict(cls.__base__)` and calls `super().__init__()` with no arguments, so a subclass
of FSDPWorker (whose __base__ is FSDPWorker, requiring config and role) cannot be
constructed. See ca21_worker._CA21Methods.
"""

from __future__ import annotations


def setup() -> None:
    """Idempotent: safe to run in every worker, and in a driver that already attached."""
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.fsdp_workers import FSDPWorker

    from ca21_worker import make_ca21_worker

    existing = getattr(FSDPWorker, "compute_caption_distortion", None)
    if existing is not None:
        # Ours already, from an earlier call in this same process -- nothing to do.
        # Anyone ELSE's would be masked by attaching, so make_ca21_worker raises on it.
        if getattr(existing, "__module__", None) == "ca21_worker":
            return

    make_ca21_worker(FSDPWorker, register, Dispatch.DP_COMPUTE_PROTO)
    print("[ca21] worker hook: compute_caption_distortion attached in "
          f"pid {__import__('os').getpid()}", flush=True)
