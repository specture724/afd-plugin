# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Allow MLA prefill metadata to be built for FULL cudagraph capture.

Upstream ``MLACommonMetadataBuilder.build_for_cudagraph_capture`` is
decode-only: it rejects any batch whose max query length exceeds the
builder's ``reorder_batch_threshold``, because upstream only ever captures
FULL graphs for uniform decode batches. The AFD prefill bucket scheme
(docs/design/prefill_bucketed_full_capture.md) deliberately captures the
varlen prefill pathway at a token bucket's upper bound and replays it for
ragged steps, so the guard has to yield for those capture batches.

The patch stays inert unless ``PREFILL_BUCKETS`` is set, i.e. only a
deployment that opted into the AFD prefill bucket scheme ever captures
prefill metadata; every other run keeps the upstream failure mode.
"""

from __future__ import annotations

from afd_plugin.v1.worker.prefill_buckets import resolve_prefill_buckets


# Patch reason: upstream build_for_cudagraph_capture rejects prefill-shaped
# batches ("MLA only supports decode-only full CUDAGraph capture"), which
# blocks the AFD prefill bucket capture.
# Patch functionality: under PREFILL_BUCKETS, allow capture batches whose max
# query length exceeds the decode threshold while keeping the request-count
# guard; without PREFILL_BUCKETS the upstream asserts are kept verbatim.
# Signature: matches upstream; no added parameters.
# Upstream: vLLM v0.26.0, vllm/model_executor/layers/attention/mla_attention.py
# Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
def build_for_cudagraph_capture(self, common_attn_metadata):
    """
    This method builds the metadata for full cudagraph capture.
    Currently, only decode is supported for full cudagraphs with MLA.
    """
    m = common_attn_metadata
    assert m.num_reqs <= (m.num_actual_tokens * self.reorder_batch_threshold), (
        "MLA only supports decode-only full CUDAGraph capture. "
        "Make sure all cudagraph capture sizes <= max_num_seq."
    )

    # ### PATCH START: AFD bucketed prefill capture
    if resolve_prefill_buckets() and m.max_query_len > self.reorder_batch_threshold:
        # A prefill-shaped capture batch: the varlen prefill pathway is what
        # gets baked into the graph, so the decode-only query-length guard
        # does not apply. Replay safety is enforced per step by the ubatch
        # wrapper's metadata check, not here.
        return self.build(0, m)
    # ### PATCH END: AFD bucketed prefill capture

    assert m.max_query_len <= self.reorder_batch_threshold  # decode only

    return self.build(0, m)


def apply_mla_prefill_cudagraph_capture() -> None:
    """Install the patched capture builder onto the MLA metadata builder."""
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonMetadataBuilder,
    )

    if getattr(build_for_cudagraph_capture, "_afd_prefill_capture", False):
        return
    build_for_cudagraph_capture._afd_prefill_capture = True  # type: ignore[attr-defined]
    MLACommonMetadataBuilder.build_for_cudagraph_capture = build_for_cudagraph_capture


apply_mla_prefill_cudagraph_capture()


__all__ = [
    "apply_mla_prefill_cudagraph_capture",
    "build_for_cudagraph_capture",
]
