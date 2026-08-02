"""VRAM reclamation utilities.

Centralises the tear-down sequence that properly frees GPU memory
when unloading a diffusers pipeline.  The correct order is:

    1. Remove accelerate CPU-offload hooks (they pin GPU references)
    2. Null-out every component attribute on the pipeline object
       (breaks circular refs WITHOUT copying tensors to RAM)
    3. Delete the pipeline object
    4. Run two GC passes (circular-ref collector needs >1 sweep)
    5. Release the CUDA cache and IPC shared memory

Never call `.to("cpu")` — that *copies* every tensor into system
RAM before freeing VRAM, temporarily doubling total memory usage
and leaving orphan CPU copies that the GC may not collect promptly.
"""

from __future__ import annotations

import gc

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _get_vram_mb() -> tuple[int, int]:
    """Return (allocated_mb, reserved_mb).  (0, 0) if CUDA unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) // (1024 * 1024)
            reserved = torch.cuda.memory_reserved(0) // (1024 * 1024)
            return alloc, reserved
    except (ImportError, RuntimeError) as exc:
        log.debug("vram_query_failed", error=str(exc))
    return 0, 0


def release_pipeline(pipe, *, label: str = "pipeline") -> None:
    """Tear down a diffusers pipeline and reclaim all VRAM + system RAM.

    This runs synchronously and should be called from a worker thread
    (via ``asyncio.to_thread``) to avoid blocking the event loop.

    Args:
        pipe: A diffusers DiffusionPipeline (or compatible) instance.
        label: A human-readable tag for log messages.
    """
    import torch

    before_alloc, before_reserved = _get_vram_mb()

    # 1. Remove accelerate CPU-offload hooks — they hold references to
    #    GPU sub-modules and prevent the GC from freeing them.
    if hasattr(pipe, "maybe_free_model_hooks"):
        try:
            pipe.maybe_free_model_hooks()
        except Exception as exc:
            log.debug(
                "vram_free_hooks_failed",
                label=label,
                error=str(exc),
            )

    # 2. Null-out every component on the pipeline.
    #    This drops Python's reference to each nn.Module / tokenizer /
    #    scheduler without copying anything to CPU.
    component_names = list(getattr(pipe, "components", {}).keys())
    for name in component_names:
        try:
            setattr(pipe, name, None)
        except AttributeError:
            # Component is a property without a setter — skip and rely
            # on the `del pipe` below to drop the reference.
            pass

    # 3. Delete the pipeline object itself.
    del pipe

    # 4. Three GC passes — CPython's cyclic-ref collector sometimes needs
    #    multiple sweeps for nested ref-cycles (e.g. nn.Module ↔ Parameter ↔
    #    grad_fn chains). MUST precede empty_cache(): the CUDA pool only
    #    releases blocks whose owning tensors are already collected.
    gc.collect()
    gc.collect()
    gc.collect()

    # 5. Release the CUDA memory pool.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # ipc_collect() releases CUDA IPC shared-memory handles that
        # empty_cache() does not touch.
        torch.cuda.ipc_collect()

    # 6. Return system RAM pages to the OS.
    #    Python's allocator holds freed pages; malloc_trim releases them.
    #    Without this, unloaded model weight buffers linger as reserved
    #    system RAM even though Python considers them freed.
    #    This used to be inlined here — the ONLY unload path that did it, which
    #    is why every other teardown ratcheted (spec §5.5.1 H2). It now lives in
    #    resource/reclaim.py so every unload path can call the same thing.
    from augmentum.resource.reclaim import trim_allocator

    trim_allocator()

    after_alloc, after_reserved = _get_vram_mb()
    freed_alloc = before_alloc - after_alloc
    freed_reserved = before_reserved - after_reserved

    log.info(
        "vram_reclaimed",
        label=label,
        freed_allocated_mb=freed_alloc,
        freed_reserved_mb=freed_reserved,
        remaining_allocated_mb=after_alloc,
        remaining_reserved_mb=after_reserved,
    )


def flush_cuda_cache() -> None:
    """Lightweight cache flush without a full GC sweep.

    Use after smaller operations (LoRA unload, variant switch) where a
    full ``release_pipeline`` is overkill.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError) as exc:
        log.debug("flush_cuda_cache_failed", error=str(exc))
