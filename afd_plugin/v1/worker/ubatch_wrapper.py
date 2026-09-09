# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD-owned vLLM ubatch wrapper.

This runtime module depends on vLLM's native ubatching stack.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
from vllm.compilation import monitor
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (
    ForwardContext,
    create_forward_context,
    get_forward_context,
)
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.v1.worker.gpu_ubatch_wrapper import UbatchMetadata, UBatchWrapper
from vllm.v1.worker.ubatching import make_ubatch_contexts

from afd_plugin.config import is_afd_active
from afd_plugin.connectors import AFDForwardContextMetadata
from afd_plugin.v1.worker.attention_metadata import build_ubatch_dp_metadata_list

# Bring-up instrument for the cooperative capture path: which key each call
# computes, and which are already captured. Set AFD_UBATCH_GRAPH_DEBUG=1.
_UBATCH_GRAPH_DEBUG = bool(os.environ.get("AFD_UBATCH_GRAPH_DEBUG"))

# How often to report the replay share. Frequent enough to see it in a short
# benchmark, rare enough not to write a line per step.
_REPLAY_LOG_EVERY = 200

logger = init_logger(f"vllm.{__name__}")


class AFDUBatchWrapper(UBatchWrapper):
    # Class-level so a wrapper built without touching __init__ still counts.
    _afd_replays = 0
    _afd_eager_ubatches = 0
    _afd_unsplit_steps = 0

    """Thin AFD-aware subclass of vLLM's native ``UBatchWrapper``."""

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        device: torch.cuda.device,
    ):
        super().__init__(runnable, vllm_config, runtime_mode, device)
        self._afd_metadata_installer: Callable[[ForwardContext], None] | None = None

    def configure_afd_context_provider(
        self,
        installer: Callable[[ForwardContext], None],
    ) -> None:
        """Save the same typed installer used by the forward-context provider."""

        self._afd_metadata_installer = installer

    # Patch reason: native SM partitioning conflicts with AFD connector work.
    # Patch functionality: disable native SM partitioning only for active AFD.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/v1/worker/gpu_ubatch_wrapper.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    @staticmethod
    def _create_sm_control_context(vllm_config: VllmConfig):
        # ### PATCH START: leave all SMs visible to AFD compute and communication.
        if is_afd_active(vllm_config):
            return nullcontext()
        # ### PATCH END: leave all SMs visible to AFD compute and communication.
        return UBatchWrapper._create_sm_control_context(vllm_config)

    # Patch reason: native ubatch contexts do not carry AFD transfer metadata.
    # Patch functionality: install per-ubatch AFD context and control-plane
    # metadata while preserving native capture, replay, and execution behavior.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/v1/worker/gpu_ubatch_wrapper.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    # Patch reason: when a FULL dispatch has no captured graph and no
    # cooperative capture is possible, upstream falls into an assert on
    # ``cudagraph_wrapper`` -- which the AFD wrapper never constructs. The
    # decode bucket below the DBO split threshold (e.g. batch size 1, which
    # the empty-first-ubatch guard refuses to split) hits this during graph
    # capture of its key.
    # Patch functionality: a dispatched-but-uncaptured FULL step runs eagerly
    # instead of asserting. Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/v1/worker/gpu_ubatch_wrapper.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        ubatch_slices = forward_context.ubatch_slices
        if ubatch_slices is None:
            # Counted because "DBO is on" and "this step actually split" are
            # different things: the splitter can decline every real step while
            # the run still pays DBO's setup cost.
            self._afd_unsplit_steps += 1
            self._maybe_log_replay_share()
            # ### PATCH START: uncaptured FULL without a cudagraph wrapper
            # runs eagerly. The wrapper owns only cooperative capture; a
            # whole-batch FULL dispatch whose key never captured has nowhere
            # to replay into.
            if (
                forward_context.cudagraph_runtime_mode is CUDAGraphMode.FULL
                and self.cudagraph_wrapper is None
                and forward_context.batch_descriptor is not None
                and forward_context.batch_descriptor.num_tokens not in self.cudagraphs
            ):
                return self.runnable(*args, **kwargs)
            # ### PATCH END: uncaptured FULL without a cudagraph wrapper.
            return super().__call__(*args, **kwargs)

        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
        # ### PATCH START: install AFD metadata before splitting ubatches.
        parent_additional_kwargs = dict(forward_context.additional_kwargs)
        if "afd_metadata" not in parent_additional_kwargs:
            self._install_missing_afd_metadata(forward_context)
            parent_additional_kwargs = dict(forward_context.additional_kwargs)

        num_tokens = sum(int(ubatch_slice.num_tokens) for ubatch_slice in ubatch_slices)
        dp_metadata = build_ubatch_dp_metadata_list(
            self.vllm_config,
            ubatch_slices,
        )
        # ### PATCH END: install AFD metadata before splitting ubatches.

        if _UBATCH_GRAPH_DEBUG:
            desc = forward_context.batch_descriptor
            logger.info(
                "AFD ubatch wrapper: num_tokens=%d mode=%s desc=%s captured=%s",
                num_tokens,
                getattr(cudagraph_runtime_mode, "name", cudagraph_runtime_mode),
                desc,
                sorted(self.cudagraphs),
            )
        if (
            num_tokens not in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=forward_context.attn_metadata,
                slot_mapping=forward_context.slot_mapping,
                input_ids=kwargs["input_ids"],
                positions=kwargs["positions"],
                inputs_embeds=kwargs["inputs_embeds"],
                intermediate_tensors=kwargs["intermediate_tensors"],
                compute_stream=torch.cuda.current_stream(),
                dp_metadata=dp_metadata,
                batch_descriptor=forward_context.batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            with self.sm_control:
                return self._capture_ubatches(ubatch_metadata, self.runnable)

        if (
            num_tokens in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
            # ### PATCH START: never replay while capture is still running.
            # A descriptor can be visited twice during the capture phase --
            # the prefill bucket key is registered by the plugin and derived
            # again from the vLLM capture-size list -- and the second visit
            # arrives with the key already captured. Replaying there runs a
            # graph against the capture pass's freshly built dummy tensors and
            # faults with an illegal memory access. Outside the capture phase
            # this flag is False, so real steps replay as before.
            and not monitor.cudagraph_capturing_enabled
            # ### PATCH END: never replay while capture is still running.
            # ### PATCH START: refresh the captured graph's metadata first.
            # A replay is only safe when this step's attention metadata has
            # the shape the graph was captured with; the refresh doubles as
            # that check and falls back to the eager path on mismatch.
            and refresh_captured_attention_metadata(
                self.cudagraphs[num_tokens],
                forward_context,
            )
            # ### PATCH END: refresh the captured graph's metadata first.
        ):
            get_offloader().sync_prev_onload()
            self._afd_replays += 1
            self._maybe_log_replay_share()
            cudagraph_metadata = self.cudagraphs[num_tokens]
            cudagraph_metadata.cudagraph.replay()
            return cudagraph_metadata.outputs

        self._afd_eager_ubatches += 1
        self._maybe_log_replay_share()
        ubatch_metadata = self._make_ubatch_metadata(
            ubatch_slices=ubatch_slices,
            attn_metadata=forward_context.attn_metadata,
            slot_mapping=forward_context.slot_mapping,
            input_ids=kwargs["input_ids"],
            positions=kwargs["positions"],
            inputs_embeds=kwargs["inputs_embeds"],
            intermediate_tensors=kwargs["intermediate_tensors"],
            compute_stream=torch.cuda.current_stream(),
            dp_metadata=dp_metadata,
            batch_descriptor=forward_context.batch_descriptor,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )
        with self.sm_control:
            return self._run_ubatches(ubatch_metadata, self.runnable)

    def _maybe_log_replay_share(self) -> None:
        """Report how often the ubatch path actually replays a captured graph.

        Capturing a graph proves nothing about whether steps use it: the replay
        preconditions can refuse every real step and leave the run eager while
        still looking like graphs are on.
        """
        total = self._afd_replays + self._afd_eager_ubatches + self._afd_unsplit_steps
        if total % _REPLAY_LOG_EVERY:
            return
        split = self._afd_replays + self._afd_eager_ubatches
        logger.info(
            "AFD ubatch path: replays=%d eager_ubatch=%d unsplit=%d "
            "split_share=%.0f%% replay_share_of_split=%.0f%%",
            self._afd_replays,
            self._afd_eager_ubatches,
            self._afd_unsplit_steps,
            100.0 * split / max(total, 1),
            100.0 * self._afd_replays / max(split, 1),
        )

    def _install_missing_afd_metadata(
        self,
        forward_context: ForwardContext,
    ) -> None:
        installer = self._afd_metadata_installer
        if installer is None:
            self._afd_use_native_ubatch_metadata = True
            return

        installer(forward_context)

    # Patch reason: native per-ubatch contexts omit AFD transfer metadata.
    # Patch functionality: clone the parent AFD context into each native ubatch.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.26.0, vllm/v1/worker/gpu_ubatch_wrapper.py
    # Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
    def _make_ubatch_metadata(
        self,
        ubatch_slices,
        attn_metadata,
        slot_mapping,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
        compute_stream,
        dp_metadata,
        batch_descriptor,
        cudagraph_runtime_mode,
    ) -> list[UbatchMetadata]:
        # ### PATCH START: resolve and validate the parent AFD context.
        parent_forward_context = get_forward_context()
        parent_additional_kwargs = dict(parent_forward_context.additional_kwargs)
        afd_metadata = parent_additional_kwargs.get("afd_metadata")
        if afd_metadata is None:
            if getattr(self, "_afd_use_native_ubatch_metadata", False):
                try:
                    return UBatchWrapper._make_ubatch_metadata(
                        self,
                        ubatch_slices,
                        attn_metadata,
                        slot_mapping,
                        input_ids,
                        positions,
                        inputs_embeds,
                        intermediate_tensors,
                        compute_stream,
                        dp_metadata,
                        batch_descriptor,
                        cudagraph_runtime_mode,
                    )
                finally:
                    self._afd_use_native_ubatch_metadata = False
            raise RuntimeError(
                "AFDUBatchWrapper requires "
                "ForwardContext.additional_kwargs['afd_metadata']",
            )
        # ### PATCH END: resolve and validate the parent AFD context.

        forward_contexts = []
        has_slot_mapping = slot_mapping and isinstance(slot_mapping, list)
        for idx, _ubatch_slice in enumerate(ubatch_slices):
            # ### PATCH START: attach one AFD context to each native ubatch.
            ubatch_afd_metadata = build_ubatch_afd_metadata(
                afd_metadata,
                ubatch_slices,
                idx,
            )
            forward_contexts.append(
                create_forward_context(
                    attn_metadata[idx] if attn_metadata is not None else None,
                    self.vllm_config,
                    dp_metadata=dp_metadata[idx],
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    slot_mapping=slot_mapping[idx] if has_slot_mapping else None,
                    additional_kwargs=build_ubatch_additional_kwargs(
                        parent_additional_kwargs,
                        ubatch_afd_metadata,
                    ),
                ),
            )
            # ### PATCH END: attach one AFD context to each native ubatch.

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        ubatch_metadata: list[UbatchMetadata] = []
        for idx, ubatch_slice in enumerate(ubatch_slices):
            (
                sliced_input_ids,
                sliced_positions,
                sliced_inputs_embeds,
                sliced_intermediate_tensors,
            ) = self._slice_model_inputs(
                ubatch_slice.token_slice,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
            )
            ubatch_metadata.append(
                UbatchMetadata(
                    context=ubatch_ctxs[idx],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.num_tokens,
                ),
            )

        return ubatch_metadata


def refresh_captured_attention_metadata(
    cudagraph_metadata: Any,
    forward_context: Any,
) -> bool:
    """Refresh the derived device tensors a captured graph reads, or refuse.

    The cooperative FULL graph bakes the attention kernels' *input pointers*:
    base buffers (query_start_loc, seq_lens, block table, slot mapping) are
    persistent and refreshed by the runner's per-step metadata build, but the
    prefill pathway derives its ``cu_seqlens`` from a subtraction that
    produces a fresh tensor every build. The graph only ever reads the tensor
    captured at capture time, so a replay step must copy the freshly derived
    values into it -- with the same shape, which only holds when this step's
    per-ubatch metadata took the same decode/prefill pathway as at capture.

    The copy is therefore also the safety check: any structural mismatch --
    a step whose first request classifies as decode, a request with chunked
    context, a different request count -- returns ``False`` and the caller
    runs the batch eagerly instead of replaying stale metadata.

    Returns:
        ``True`` when every captured ubatch's metadata is fed for this step
        and the graph may replay.
    """
    fresh_list = forward_context.attn_metadata
    ubatch_metadata_list = cudagraph_metadata.ubatch_metadata
    if not isinstance(fresh_list, list) or len(fresh_list) != len(ubatch_metadata_list):
        return False

    for ubatch_metadata, fresh_per_layer in zip(
        ubatch_metadata_list, fresh_list, strict=True
    ):
        captured_per_layer = ubatch_metadata.context.forward_context.attn_metadata
        if captured_per_layer is None:
            continue
        if not _refresh_metadata_mapping(captured_per_layer, fresh_per_layer):
            return False
    return True


def _refresh_metadata_mapping(
    captured_per_layer: Any,
    fresh_per_layer: Any,
) -> bool:
    """Pair each captured layer's metadata with this step's, then refresh."""
    if isinstance(captured_per_layer, dict):
        if not isinstance(fresh_per_layer, dict):
            return False
        pairs = [
            (metadata, fresh_per_layer.get(layer_name))
            for layer_name, metadata in captured_per_layer.items()
        ]
    elif isinstance(captured_per_layer, (list, tuple)):
        if not isinstance(fresh_per_layer, (list, tuple)) or len(
            fresh_per_layer
        ) != len(captured_per_layer):
            return False
        pairs = list(zip(captured_per_layer, fresh_per_layer, strict=True))
    else:
        return False

    refreshed: set[int] = set()
    for captured_metadata, fresh_metadata in pairs:
        if captured_metadata is None:
            continue
        if fresh_metadata is None:
            return False
        if id(captured_metadata) in refreshed:
            # Layers of a KV cache group share one metadata object; the
            # derived tensors only need one refresh.
            continue
        if not _refresh_metadata_pair(captured_metadata, fresh_metadata):
            return False
        refreshed.add(id(captured_metadata))
    return True


def _refresh_metadata_pair(
    captured_metadata: Any,
    fresh_metadata: Any,
) -> bool:
    """Refresh one captured metadata object from this step's rebuild.

    The decode pathway needs no refresh: every device tensor its captured
    kernels read is a persistent buffer the step's metadata build already
    wrote. Only the prefill pathway derives a private tensor, so only it is
    copied. ``prefill`` is probed rather than accessed because the metadata
    type is a per-backend union and backends without the decode/prefill
    split have nothing this hook can refresh.
    """
    # A replay reads the buffers baked in at capture, so this step must have
    # written those very buffers. That holds on the whole-batch path, where the
    # runner keeps persistent metadata. It does not hold under DBO: the
    # per-ubatch metadata comes from split_attn_metadata, which slices and even
    # clones per step. So identity, not shape, is the test, and a DBO prefill
    # step falls back to running its ubatches eagerly.
    #
    # Relaxing this to a shape check plus a copy does not make DBO prefill
    # replay, and that is measured rather than assumed. With the copy alone the
    # run is stable and correct and still never replays: the per-ubatch
    # slot_mapping shape refuses 94% of split steps, because a request-aligned
    # split lands wherever the requests happen to end. Forcing the split to half
    # the bucket makes the shapes match, the graph does replay, and the first
    # replay segfaults inside cuGraphLaunch. The open problem is therefore the
    # validity of the captured cooperative graph, not this precondition.
    for field in ("seq_lens", "block_table_tensor", "slot_mapping"):
        captured_field = getattr(captured_metadata, field, None)
        fresh_field = getattr(fresh_metadata, field, None)
        if captured_field is None and fresh_field is None:
            continue
        if captured_field is None or fresh_field is None:
            return False
        if captured_field.data_ptr() != fresh_field.data_ptr():
            return False

    captured_prefill = getattr(captured_metadata, "prefill", None)
    fresh_prefill = getattr(fresh_metadata, "prefill", None)
    if (captured_prefill is None) != (fresh_prefill is None):
        return False
    if captured_prefill is None:
        return True
    assert fresh_prefill is not None
    captured_cu = captured_prefill.query_start_loc
    fresh_cu = fresh_prefill.query_start_loc
    if captured_cu.shape != fresh_cu.shape:
        return False
    # The destination was allocated during capture, inside vLLM's inference
    # mode, so it is an inference tensor -- and writing one is only permitted
    # from inside inference mode. Refreshing it from a dummy run (which is not
    # in inference mode) otherwise raises "Inplace update to inference tensor
    # outside InferenceMode". Only this tensor's contents change; nothing here
    # is autograd-relevant.
    with torch.inference_mode():
        captured_cu.copy_(fresh_cu, non_blocking=True)
    return True


def build_ubatch_afd_metadata(
    afd_metadata: AFDForwardContextMetadata,
    ubatch_slices: Any,
    ubatch_idx: int,
) -> AFDForwardContextMetadata:
    """Clone parent AFD metadata for one vLLM ubatch."""

    if ubatch_idx < 0 or ubatch_idx >= len(ubatch_slices):
        raise IndexError(f"ubatch_idx {ubatch_idx} out of range")

    ubatch_slice = ubatch_slices[ubatch_idx]
    clone = afd_metadata.clone()
    clone.stage_idx = ubatch_idx
    clone.num_stages = len(ubatch_slices)
    clone.tokens_start_loc = [int(ubatch_slice.token_slice.start)]
    clone.requests_start_loc = [int(ubatch_slice.request_slice.start)]
    clone.tokens_lens = [int(ubatch_slice.num_tokens)]
    clone.tokens_unpadded_lens = [
        _resolve_ubatch_unpadded_tokens(afd_metadata, ubatch_slice, ubatch_idx),
    ]
    return clone


def build_ubatch_additional_kwargs(
    parent_additional_kwargs: dict[str, Any],
    afd_metadata: AFDForwardContextMetadata,
) -> dict[str, Any]:
    child_kwargs = dict(parent_additional_kwargs)
    child_kwargs["afd_metadata"] = afd_metadata
    return child_kwargs


def _resolve_ubatch_unpadded_tokens(
    afd_metadata: AFDForwardContextMetadata,
    ubatch_slice: Any,
    ubatch_idx: int,
) -> int:
    unpadded_lens = afd_metadata.tokens_unpadded_lens
    if ubatch_idx < len(unpadded_lens):
        return int(unpadded_lens[ubatch_idx])
    return int(ubatch_slice.num_tokens)


__all__ = [
    "AFDUBatchWrapper",
    "build_ubatch_additional_kwargs",
    "build_ubatch_afd_metadata",
    "build_ubatch_dp_metadata_list",
    "refresh_captured_attention_metadata",
]
