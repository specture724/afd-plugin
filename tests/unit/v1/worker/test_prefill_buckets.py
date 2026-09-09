# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Unit tests for the AFD prefill bucket capture scheme.

Covers the bucket configuration, the policy wiring, the runner-side dispatch
decision, the prefill FULL key registration, and the ubatch wrapper's
replay-time metadata refresh. Everything runs on CPU against fakes.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.v1.worker.attention_model_runner import AFDAttentionModelRunner
from afd_plugin.v1.worker.cuda_graph import (
    FULL_DECODE_ONLY,
    AFDCUDAGraphPolicy,
    validate_cuda_graph_mode,
)
from afd_plugin.v1.worker.prefill_buckets import (
    MLA_REORDER_BATCH_THRESHOLD,
    PREFILL_BUCKETS_ENV,
    prefill_bucket_capture_reqs,
    resolve_prefill_bucket,
    resolve_prefill_buckets,
    synthetic_prefill_bucket_batch,
)
from afd_plugin.v1.worker.ubatch_wrapper import (
    refresh_captured_attention_metadata,
)

_DISABLED_POLICY = AFDCUDAGraphPolicy(
    enabled=False,
    mode_name=None,
    allow_attention_full_decode_only=False,
    enable_ffn_graph_cache=False,
)


# ---------------------------------------------------------------------------
# Bucket configuration


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ()),
        ("   ", ()),
        ("1024", (1024,)),
        ("1024,2048", (1024, 2048)),
        ("1024 2048 4096", (1024, 2048, 4096)),
        ("4096, 1024", (1024, 4096)),  # sorted into ascending order
    ],
)
def test_resolve_prefill_buckets_parses(raw, expected):
    assert resolve_prefill_buckets({PREFILL_BUCKETS_ENV: raw}) == expected


@pytest.mark.parametrize(
    "raw",
    ["0", "-8", "1024,1024", "abc"],
)
def test_resolve_prefill_buckets_rejects(raw):
    with pytest.raises(ValueError):
        resolve_prefill_buckets({PREFILL_BUCKETS_ENV: raw})


def test_resolve_prefill_buckets_defaults_to_off():
    assert resolve_prefill_buckets({}) == ()


def test_resolve_prefill_bucket_pads_up_not_down():
    buckets = (1024, 2048, 4096)
    assert resolve_prefill_bucket(buckets, 1) == 1024
    assert resolve_prefill_bucket(buckets, 1024) == 1024
    assert resolve_prefill_bucket(buckets, 1025) == 2048
    # Above the largest bucket there is nothing to pad onto.
    assert resolve_prefill_bucket(buckets, 4097) is None


def test_prefill_bucket_capture_reqs_matches_dispatcher_formula():
    assert prefill_bucket_capture_reqs(1024, 64) == 64
    assert prefill_bucket_capture_reqs(1024, 8) == 8


@pytest.mark.parametrize("bucket", [1024, 2048, 4096])
def test_synthetic_prefill_bucket_batch_shape(bucket):
    num_scheduled, max_query_len = synthetic_prefill_bucket_batch(bucket, 8)
    assert sum(num_scheduled) == bucket
    assert len(num_scheduled) == 8
    assert set(num_scheduled[2:]) == {0}
    # The first request must classify as prefill in the metadata builder.
    assert max_query_len > MLA_REORDER_BATCH_THRESHOLD
    # Unequal so the request-aligned DBO splitter has an interior boundary.
    assert num_scheduled[0] != num_scheduled[1]
    assert max_query_len == num_scheduled[0]


def test_synthetic_prefill_bucket_batch_needs_two_requests():
    with pytest.raises(ValueError):
        synthetic_prefill_bucket_batch(1024, 1)


# ---------------------------------------------------------------------------
# Policy wiring


def _config(*, enforce_eager=False, cudagraph_mode=FULL_DECODE_ONLY, role_config=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
        parallel_config=SimpleNamespace(
            use_ubatching=False,
            num_ubatches=1,
        ),
    )


def test_policy_carries_prefill_buckets(monkeypatch):
    monkeypatch.setenv(PREFILL_BUCKETS_ENV, "1024,2048")
    policy = validate_cuda_graph_mode(_config(), role="attention")
    assert policy.prefill_buckets == (1024, 2048)


def test_policy_defaults_to_no_buckets(monkeypatch):
    monkeypatch.delenv(PREFILL_BUCKETS_ENV, raising=False)
    policy = validate_cuda_graph_mode(_config(), role="attention")
    assert policy.prefill_buckets == ()


def test_policy_rejects_buckets_on_ffn_role(monkeypatch):
    monkeypatch.setenv(PREFILL_BUCKETS_ENV, "1024")
    with pytest.raises(RuntimeError, match="Attention-side"):
        validate_cuda_graph_mode(_config(), role="ffn")


def test_policy_reports_buckets_even_when_eager(monkeypatch):
    monkeypatch.setenv(PREFILL_BUCKETS_ENV, "1024")
    policy = validate_cuda_graph_mode(_config(enforce_eager=True), role="attention")
    assert policy.enabled is False
    assert policy.prefill_buckets == (1024,)


# ---------------------------------------------------------------------------
# Runner dispatch decision


def _bucket_runner(
    buckets,
    *,
    num_computed=None,
    is_warmup=False,
    is_graph_capturing=False,
    uniform_decode=False,
):
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_cudagraph_policy = AFDCUDAGraphPolicy(
        enabled=True,
        mode_name=FULL_DECODE_ONLY,
        allow_attention_full_decode_only=True,
        enable_ffn_graph_cache=False,
        prefill_buckets=buckets,
    )
    runner.uniform_decode_query_len = 1
    runner._is_warmup = is_warmup
    runner._afd_is_graph_capturing = is_graph_capturing
    runner._is_uniform_decode = lambda **_kwargs: uniform_decode
    runner.parallel_config = SimpleNamespace(dbo_prefill_token_threshold=12)
    num_reqs = 0 if num_computed is None else len(num_computed)
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=(
            np.asarray(num_computed) if num_computed is not None else np.zeros(0)
        ),
        num_reqs=num_reqs,
    )
    return runner


def test_dispatch_decision_disabled_without_buckets():
    runner = _bucket_runner(())
    assert runner._resolve_prefill_bucket_decision(
        num_tokens=2048,
        num_reqs=2,
        num_scheduled_tokens_np=np.array([1024, 1024], dtype=np.int32),
        max_num_scheduled_tokens=1024,
        force_uniform_decode=False,
    ) == (None, False)


def test_dispatch_decision_skips_decode_steps():
    runner = _bucket_runner((1024,), uniform_decode=True)
    assert runner._resolve_prefill_bucket_decision(
        num_tokens=8,
        num_reqs=8,
        num_scheduled_tokens_np=np.ones(8, dtype=np.int32),
        max_num_scheduled_tokens=1,
        force_uniform_decode=None,
    ) == (None, False)


def test_dispatch_decision_pads_eligible_prefill():
    runner = _bucket_runner((1024, 2048), num_computed=[0, 0, 0])
    assert runner._resolve_prefill_bucket_decision(
        num_tokens=1500,
        num_reqs=3,
        num_scheduled_tokens_np=np.array([700, 500, 300], dtype=np.int32),
        max_num_scheduled_tokens=700,
        force_uniform_decode=None,
    ) == (2048, False)


def test_dispatch_decision_forces_eager_when_first_request_is_decode_shaped():
    runner = _bucket_runner((1024,), num_computed=[0, 0])
    # First request at or below the MLA threshold: the metadata builder would
    # classify the step through the decode pathway.
    decision = runner._resolve_prefill_bucket_decision(
        num_tokens=1200,
        num_reqs=2,
        num_scheduled_tokens_np=np.array([400, 800], dtype=np.int32),
        max_num_scheduled_tokens=800,
        force_uniform_decode=None,
    )
    assert decision == (None, True)


def test_dispatch_decision_forces_eager_with_prior_context():
    runner = _bucket_runner((1024,), num_computed=[0, 64])
    decision = runner._resolve_prefill_bucket_decision(
        num_tokens=1200,
        num_reqs=2,
        num_scheduled_tokens_np=np.array([600, 600], dtype=np.int32),
        max_num_scheduled_tokens=600,
        force_uniform_decode=None,
    )
    assert decision == (None, True)


def test_dispatch_decision_ignores_guard_during_capture():
    # The synthetic capture batch manages its own shape; the guard must not
    # read live request state that does not exist yet. Padding itself is not
    # needed on this path: the capture batch already runs at the bucket size
    # and dispatch maps a bucket onto its own key.
    runner = _bucket_runner(
        (1024,),
        num_computed=[64],  # stale live state that would otherwise reject
        is_warmup=True,
    )
    assert runner._resolve_prefill_bucket_decision(
        num_tokens=1024,
        num_reqs=1,
        num_scheduled_tokens_np=np.array([1024], dtype=np.int32),
        max_num_scheduled_tokens=1024,
        force_uniform_decode=None,
    ) == (None, False)


def test_dispatch_decision_leaves_below_threshold_steps_alone():
    runner = _bucket_runner((1024,), num_computed=[0])
    assert runner._resolve_prefill_bucket_decision(
        num_tokens=6,
        num_reqs=1,
        num_scheduled_tokens_np=np.array([6], dtype=np.int32),
        max_num_scheduled_tokens=6,
        force_uniform_decode=None,
    ) == (None, False)


# ---------------------------------------------------------------------------
# Prefill FULL key registration


def test_register_prefill_bucket_keys():
    registered = []

    class _Dispatcher:
        def add_cudagraph_key(self, mode, descriptor):
            registered.append((mode, descriptor))

    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_cudagraph_policy = AFDCUDAGraphPolicy(
        enabled=True,
        mode_name=FULL_DECODE_ONLY,
        allow_attention_full_decode_only=True,
        enable_ffn_graph_cache=False,
        prefill_buckets=(1024, 2048),
    )
    runner.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[1, 2, 8, 1024, 2048],
    )
    runner.max_num_tokens = 2048
    runner.scheduler_config = SimpleNamespace(max_num_seqs=64)
    runner.cudagraph_dispatcher = _Dispatcher()

    runner._register_prefill_bucket_keys()

    assert [(mode, d.num_tokens, d.num_reqs, d.uniform) for mode, d in registered] == [
        (
            _registered_mode(registered, 0),
            1024,
            prefill_bucket_capture_reqs(1024, 64),
            False,
        ),
        (
            _registered_mode(registered, 1),
            2048,
            prefill_bucket_capture_reqs(2048, 64),
            False,
        ),
    ]


def _registered_mode(registered, index):
    return registered[index][0]


def test_register_prefill_bucket_keys_rejects_missing_capture_sizes():
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_cudagraph_policy = AFDCUDAGraphPolicy(
        enabled=True,
        mode_name=FULL_DECODE_ONLY,
        allow_attention_full_decode_only=True,
        enable_ffn_graph_cache=False,
        prefill_buckets=(4096,),
    )
    runner.compilation_config = SimpleNamespace(cudagraph_capture_sizes=[1024])
    runner.max_num_tokens = 4096
    runner.scheduler_config = SimpleNamespace(max_num_seqs=8)
    runner.cudagraph_dispatcher = SimpleNamespace(
        add_cudagraph_key=lambda *_: None,
    )

    with pytest.raises(RuntimeError, match="capture sizes"):
        runner._register_prefill_bucket_keys()


def test_register_prefill_bucket_keys_noop_without_buckets(monkeypatch):
    monkeypatch.delenv(PREFILL_BUCKETS_ENV, raising=False)
    runner = object.__new__(AFDAttentionModelRunner)
    runner.afd_cudagraph_policy = _DISABLED_POLICY
    runner.cudagraph_dispatcher = SimpleNamespace(
        add_cudagraph_key=lambda *_: pytest.fail("must not register keys"),
    )

    runner._register_prefill_bucket_keys()


# ---------------------------------------------------------------------------
# Replay-time metadata refresh


def _captured_graph(ubatch_forward_contexts):
    ubatch_metadata = [
        SimpleNamespace(context=SimpleNamespace(forward_context=context))
        for context in ubatch_forward_contexts
    ]
    return SimpleNamespace(ubatch_metadata=ubatch_metadata)


def test_refresh_copies_derived_prefill_cu_seqlens():
    captured_meta = SimpleNamespace(
        prefill=SimpleNamespace(
            query_start_loc=torch.full((3,), -1, dtype=torch.int32),
        ),
    )
    fresh_cu = torch.tensor([0, 5, 9], dtype=torch.int32)
    fresh_meta = SimpleNamespace(prefill=SimpleNamespace(query_start_loc=fresh_cu))

    graph = _captured_graph(
        [SimpleNamespace(attn_metadata={"layer": captured_meta})],
    )
    context = SimpleNamespace(attn_metadata=[{"layer": fresh_meta}])

    assert refresh_captured_attention_metadata(graph, context) is True
    assert torch.equal(captured_meta.prefill.query_start_loc, fresh_cu)


def test_refresh_accepts_decode_pathway_without_copy():
    # Decode-pathway ubatches read only persistent buffers: nothing to copy,
    # and a missing prefill on both sides is a match.
    captured_meta = SimpleNamespace(prefill=None)
    fresh_meta = SimpleNamespace(prefill=None)
    graph = _captured_graph(
        [SimpleNamespace(attn_metadata={"layer": captured_meta})],
    )
    context = SimpleNamespace(attn_metadata=[{"layer": fresh_meta}])

    assert refresh_captured_attention_metadata(graph, context) is True


def test_refresh_rejects_pathway_mismatch():
    captured_meta = SimpleNamespace(prefill=SimpleNamespace())
    fresh_meta = SimpleNamespace(prefill=None)
    graph = _captured_graph(
        [SimpleNamespace(attn_metadata={"layer": captured_meta})],
    )
    context = SimpleNamespace(attn_metadata=[{"layer": fresh_meta}])

    assert refresh_captured_attention_metadata(graph, context) is False


def test_refresh_rejects_shape_mismatch():
    captured_meta = SimpleNamespace(
        prefill=SimpleNamespace(
            query_start_loc=torch.full((3,), -1, dtype=torch.int32),
        ),
    )
    fresh_meta = SimpleNamespace(
        prefill=SimpleNamespace(
            query_start_loc=torch.full((5,), -1, dtype=torch.int32),
        ),
    )
    graph = _captured_graph(
        [SimpleNamespace(attn_metadata={"layer": captured_meta})],
    )
    context = SimpleNamespace(attn_metadata=[{"layer": fresh_meta}])

    assert refresh_captured_attention_metadata(graph, context) is False
    # Nothing was written on the refusal path.
    assert torch.equal(
        captured_meta.prefill.query_start_loc,
        torch.full((3,), -1, dtype=torch.int32),
    )


def test_refresh_rejects_ubatch_count_mismatch():
    graph = _captured_graph(
        [SimpleNamespace(attn_metadata=None)] * 2,
    )
    context = SimpleNamespace(
        attn_metadata=[{"layer": SimpleNamespace(prefill=None)}],
    )

    assert refresh_captured_attention_metadata(graph, context) is False


def test_refresh_skips_metadata_free_contexts():
    graph = _captured_graph(
        [SimpleNamespace(attn_metadata=None)],
    )
    context = SimpleNamespace(
        attn_metadata=[{"layer": SimpleNamespace(prefill=None)}],
    )

    assert refresh_captured_attention_metadata(graph, context) is True
