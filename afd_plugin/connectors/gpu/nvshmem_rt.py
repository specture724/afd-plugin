# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Minimal NVSHMEM host-library binding for the async GPU connector.

``torch.distributed._symmetric_memory`` cannot serve AFD: its NVSHMEM backend
bootstraps on the *default* process group and carves teams out of
``NVSHMEM_TEAM_WORLD`` with ``nvshmem_team_split_strided``. Each AFD role runs
as its own ``vllm serve`` whose default group covers only that role's ranks, so
the cross-role AFD group is never a strided subset of it and team creation
fails.

Bootstrapping NVSHMEM ourselves from a unique id exchanged over the AFD group's
store makes the AFD world *be* ``NVSHMEM_TEAM_WORLD``, so no team split is
needed. The host library's ABI is pinned by static asserts in its own headers
(``uniqueid`` 128 B, ``init_attr`` 144 B, version = ``(1 << 16) + sizeof``),
which is why ctypes is enough and ``csrc/gpu/`` stays empty.
"""

from __future__ import annotations

import ctypes
import os
from typing import TYPE_CHECKING, Final

import torch

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup, Store

UNIQUEID_PADDING: Final[int] = 124
# 128 (init_args) - 4 (version) - 24 (uid_args) - 4 (trailing alignment)
INIT_ARGS_PADDING: Final[int] = 96
NVSHMEMX_INIT_WITH_UNIQUEID: Final[int] = 1 << 3
_UID_STORE_KEY: Final[str] = "afd_nvshmem_uid"
_LIB_RELATIVE: Final[str] = "nvidia/nvshmem/lib/libnvshmem_host.so.3"


class _UniqueId(ctypes.Structure):
    _fields_ = (
        ("version", ctypes.c_int),
        ("internal", ctypes.c_char * UNIQUEID_PADDING),
    )


class _UniqueIdArgs(ctypes.Structure):
    _fields_ = (
        ("version", ctypes.c_int),
        ("id", ctypes.POINTER(_UniqueId)),
        ("myrank", ctypes.c_int),
        ("nranks", ctypes.c_int),
    )


class _InitArgs(ctypes.Structure):
    _fields_ = (
        ("version", ctypes.c_int),
        ("uid_args", _UniqueIdArgs),
        ("content", ctypes.c_char * INIT_ARGS_PADDING),
    )


class _InitAttr(ctypes.Structure):
    _fields_ = (
        ("version", ctypes.c_int),
        ("mpi_comm", ctypes.c_void_p),
        ("args", _InitArgs),
    )


def _find_host_library() -> str:
    """Locate ``libnvshmem_host.so.3`` next to the installed nvshmem wheel."""
    import site
    import sysconfig

    roots = [sysconfig.get_paths()["purelib"], *site.getsitepackages()]
    for root in roots:
        candidate = os.path.join(root, _LIB_RELATIVE)
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        "AFD async GPU connector requires the NVSHMEM host library; "
        f"{_LIB_RELATIVE} was not found under {roots}. Install "
        "nvidia-nvshmem-cu13 matching the installed torch build.",
    )


def _load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(_find_host_library(), mode=ctypes.RTLD_GLOBAL)
    lib.nvshmemx_get_uniqueid.argtypes = [ctypes.POINTER(_UniqueId)]
    lib.nvshmemx_get_uniqueid.restype = ctypes.c_int
    lib.nvshmemx_set_attr_uniqueid_args.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(_UniqueId),
        ctypes.POINTER(_InitAttr),
    ]
    lib.nvshmemx_set_attr_uniqueid_args.restype = ctypes.c_int
    lib.nvshmemx_hostlib_init_attr.argtypes = [
        ctypes.c_uint,
        ctypes.POINTER(_InitAttr),
    ]
    lib.nvshmemx_hostlib_init_attr.restype = ctypes.c_int
    lib.nvshmem_malloc.argtypes = [ctypes.c_size_t]
    lib.nvshmem_malloc.restype = ctypes.c_void_p
    lib.nvshmem_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.nvshmem_ptr.restype = ctypes.c_void_p
    lib.nvshmem_my_pe.restype = ctypes.c_int
    lib.nvshmem_n_pes.restype = ctypes.c_int
    lib.nvshmemx_putmem_on_stream.argtypes = [
        ctypes.c_void_p,  # dest
        ctypes.c_void_p,  # source
        ctypes.c_size_t,  # bytes
        ctypes.c_int,  # pe
        ctypes.c_void_p,  # cudaStream_t
    ]
    lib.nvshmemx_putmem_on_stream.restype = None
    lib.nvshmemx_int32_p_on_stream.argtypes = [
        ctypes.c_void_p,  # dest (int32_t*)
        ctypes.c_int32,  # value, passed by value -- no source buffer needed
        ctypes.c_int,  # pe
        ctypes.c_void_p,  # cudaStream_t
    ]
    lib.nvshmemx_int32_p_on_stream.restype = None
    lib.nvshmemx_quiet_on_stream.argtypes = [ctypes.c_void_p]  # cudaStream_t
    lib.nvshmemx_quiet_on_stream.restype = None
    return lib


# NVSHMEM initialization is process-global: a process joins exactly one NVSHMEM
# world, so every connector in it shares this state.
_lib: ctypes.CDLL | None = None
_initialized_world: tuple[int, int] | None = None


def init(pg: ProcessGroup, rank: int, world_size: int) -> None:
    """Join the NVSHMEM world described by ``pg``, once per process.

    Rank 0 mints the unique id and publishes it on the group's store; every rank
    then initializes with the same id, so NVSHMEM's PE numbering equals the AFD
    world rank.
    """
    global _lib, _initialized_world

    if _initialized_world is not None:
        if _initialized_world != (rank, world_size):
            raise RuntimeError(
                "NVSHMEM is already initialized in this process as "
                f"rank {_initialized_world[0]} of {_initialized_world[1]}; "
                f"cannot re-initialize as rank {rank} of {world_size}",
            )
        return

    from torch.distributed.distributed_c10d import _get_process_group_store

    lib = _load_library()
    store: Store = _get_process_group_store(pg)

    unique_id = _UniqueId()
    unique_id.version = (1 << 16) + ctypes.sizeof(_UniqueId)
    if rank == 0:
        if lib.nvshmemx_get_uniqueid(ctypes.byref(unique_id)) != 0:
            raise RuntimeError("nvshmemx_get_uniqueid failed")
        store.set(_UID_STORE_KEY, bytes(memoryview(unique_id).cast("B")))
    else:
        raw = store.get(_UID_STORE_KEY)
        ctypes.memmove(ctypes.byref(unique_id), raw, ctypes.sizeof(_UniqueId))

    attr = _InitAttr()
    attr.version = (1 << 16) + ctypes.sizeof(_InitAttr)
    attr.args.version = (1 << 16) + ctypes.sizeof(_InitArgs)
    attr.args.uid_args.version = (1 << 16) + ctypes.sizeof(_UniqueIdArgs)
    if (
        lib.nvshmemx_set_attr_uniqueid_args(
            rank,
            world_size,
            ctypes.byref(unique_id),
            ctypes.byref(attr),
        )
        != 0
    ):
        raise RuntimeError("nvshmemx_set_attr_uniqueid_args failed")
    if (
        lib.nvshmemx_hostlib_init_attr(
            NVSHMEMX_INIT_WITH_UNIQUEID,
            ctypes.byref(attr),
        )
        != 0
    ):
        raise RuntimeError("nvshmemx_hostlib_init_attr failed")

    actual_pe, actual_world = lib.nvshmem_my_pe(), lib.nvshmem_n_pes()
    if (actual_pe, actual_world) != (rank, world_size):
        raise RuntimeError(
            f"NVSHMEM PE numbering does not match the AFD world: got PE "
            f"{actual_pe} of {actual_world}, expected {rank} of {world_size}",
        )
    _lib = lib
    _initialized_world = (rank, world_size)


def is_initialized() -> bool:
    return _initialized_world is not None


def _require_lib() -> ctypes.CDLL:
    if _lib is None:
        raise RuntimeError("NVSHMEM is not initialized; call init() first")
    return _lib


def malloc(nbytes: int) -> int:
    """Allocate a symmetric buffer. Collective: every PE must call it alike."""
    pointer = _require_lib().nvshmem_malloc(nbytes)
    if not pointer:
        raise RuntimeError(
            f"nvshmem_malloc({nbytes}) returned NULL; raise "
            "NVSHMEM_SYMMETRIC_SIZE or lower the window capacity",
        )
    return int(pointer)


def peer_ptr(local_ptr: int, pe: int) -> int:
    """Map a peer's copy of a symmetric allocation into this process.

    Unused by the write path: ``put_on_stream`` addresses a destination PE
    without a local mapping of its memory, which is what makes it work across
    nodes. Kept as a primitive for a possible future fast path that writes
    directly through a mapping when ``pe`` happens to be P2P-reachable.
    """
    pointer = _require_lib().nvshmem_ptr(ctypes.c_void_p(local_ptr), pe)
    if not pointer:
        raise RuntimeError(
            f"nvshmem_ptr returned NULL for PE {pe}: no direct peer access.",
        )
    return int(pointer)


def put_on_stream(dest_ptr: int, source: torch.Tensor, *, pe: int, stream: int) -> None:
    """One-sided write of ``source`` into PE ``pe``'s window at ``dest_ptr``.

    ``dest_ptr`` is this rank's own address for the destination symmetric
    object -- the same value every PE computes for its own copy of the
    window, the way ``malloc`` hands back one consistent address that every
    PE plugs into the same formula. NVSHMEM resolves it to the physical
    location on ``pe`` internally, over NVLink/P2P or over the network
    depending on reachability, so the caller never needs to know which.

    ``source`` must be a contiguous, device-resident tensor; NVSHMEM's
    on-stream put reads it directly, with no implicit dtype cast and no
    implicit gather -- callers must already have done both.
    """
    if not source.is_contiguous():
        raise ValueError("put_on_stream requires a contiguous source tensor")
    nbytes = source.numel() * source.element_size()
    if nbytes == 0:
        return
    _require_lib().nvshmemx_putmem_on_stream(
        ctypes.c_void_p(dest_ptr),
        ctypes.c_void_p(source.data_ptr()),
        ctypes.c_size_t(nbytes),
        ctypes.c_int(pe),
        ctypes.c_void_p(stream),
    )


def put_scalar_i32_on_stream(
    dest_ptr: int, value: int, *, pe: int, stream: int
) -> None:
    """One-sided write of a single int32 into PE ``pe``'s window.

    ``value`` travels as an immediate, not through a source buffer -- this is
    what the flag write uses in place of a local ``.fill_()`` into a mapped
    peer view.
    """
    _require_lib().nvshmemx_int32_p_on_stream(
        ctypes.c_void_p(dest_ptr),
        ctypes.c_int32(value),
        ctypes.c_int(pe),
        ctypes.c_void_p(stream),
    )


def fence_on_stream(stream: int) -> None:
    """Order this PE's prior puts before whatever is enqueued after.

    Same-stream issue order was enough to make "the flag is visible" imply
    "the payload is visible" when every write was a same-engine
    device-to-device copy (see ``symm_window.py``'s module docstring); once a
    write can be a put over the network, two independent puts to the same PE
    carry no such guarantee and need an explicit order point between them.

    NVSHMEM's host API exposes no stream-enqueued ``fence`` (only the
    device-side one), so this reaches for the next cheapest thing it does
    expose, ``quiet`` -- which waits for local completion of every
    outstanding put rather than merely ordering them. Correct, and probably
    stronger than what dispatch actually needs; revisit if it shows up as a
    stall once there is a cross-node deployment to profile.
    """
    _require_lib().nvshmemx_quiet_on_stream(ctypes.c_void_p(stream))


class _DeviceBuffer:
    """Hand a raw device pointer to torch via ``__cuda_array_interface__``."""

    def __init__(self, pointer: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {
            "data": (pointer, False),
            "shape": (nbytes,),
            "typestr": "|u1",
            "version": 3,
            "strides": None,
        }


def tensor_from_ptr(
    base_ptr: int,
    *,
    byte_offset: int,
    sizes: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """View symmetric memory as a tensor without copying.

    The buffer is exposed as bytes and then reinterpreted, because
    ``__cuda_array_interface__`` has no type string for dtypes like bfloat16.
    """
    itemsize = torch.empty(0, dtype=dtype).element_size()
    numel = 1
    for size in sizes:
        numel *= size
    if numel == 0:
        # A zero-length __cuda_array_interface__ buffer is rejected by the CUDA
        # runtime (cudaErrorInvalidValue); an empty slot is legitimate whenever
        # routing sends a peer nothing, so hand back a plain empty tensor.
        return torch.empty(sizes, dtype=dtype, device=device)
    nbytes = numel * itemsize
    if byte_offset % itemsize:
        raise ValueError(
            f"byte offset {byte_offset} is not aligned to {itemsize}-byte {dtype}",
        )
    raw = torch.as_tensor(
        _DeviceBuffer(base_ptr + byte_offset, nbytes),
        device=device,
    )
    return raw.view(dtype).reshape(sizes)


__all__ = [
    "fence_on_stream",
    "init",
    "is_initialized",
    "malloc",
    "peer_ptr",
    "put_on_stream",
    "put_scalar_i32_on_stream",
    "tensor_from_ptr",
]
