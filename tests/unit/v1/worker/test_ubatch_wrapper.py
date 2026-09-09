# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Replay safety for the AFD cooperative ubatch wrapper."""

from types import SimpleNamespace

import torch

from afd_plugin.v1.worker.ubatch_wrapper import _refresh_metadata_pair


def _metadata(seq_lens, block_table, slot_mapping, query_start_loc=None):
    prefill = (
        SimpleNamespace(query_start_loc=query_start_loc)
        if query_start_loc is not None
        else None
    )
    return SimpleNamespace(
        seq_lens=seq_lens,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        prefill=prefill,
    )


def test_replay_allowed_when_the_step_writes_the_captured_buffers():
    # The persistent case: the runner refreshed the very tensors the graph
    # baked in, so only the derived cu_seqlens still needs copying.
    seq_lens = torch.zeros(4, dtype=torch.int32)
    block_table = torch.zeros((4, 2), dtype=torch.int32)
    slot_mapping = torch.zeros(8, dtype=torch.int64)
    captured_cu = torch.zeros(5, dtype=torch.int32)
    fresh_cu = torch.arange(5, dtype=torch.int32)

    captured = _metadata(seq_lens, block_table, slot_mapping, captured_cu)
    fresh = _metadata(seq_lens, block_table, slot_mapping, fresh_cu)

    assert _refresh_metadata_pair(captured, fresh) is True
    # The refresh is what makes the replay correct, so it must have happened.
    assert torch.equal(captured_cu, fresh_cu)


def test_replay_refused_when_the_step_rebuilt_a_buffer():
    # What DBO does: split_attn_metadata slices and clones per step, so the
    # graph would read capture-time contents this step never wrote. The caller
    # runs the ubatches eagerly. Copying the values across instead is stable
    # but does not buy a replay either -- see the note on _refresh_metadata_pair.
    block_table = torch.zeros((4, 2), dtype=torch.int32)
    slot_mapping = torch.zeros(8, dtype=torch.int64)
    captured_cu = torch.zeros(5, dtype=torch.int32)

    captured = _metadata(
        torch.zeros(4, dtype=torch.int32), block_table, slot_mapping, captured_cu
    )
    # Same shape and values, different storage -- the clone DBO produces.
    fresh = _metadata(
        torch.zeros(4, dtype=torch.int32),
        block_table,
        slot_mapping,
        torch.arange(5, dtype=torch.int32),
    )

    assert _refresh_metadata_pair(captured, fresh) is False
    # And it must not have written into the captured tensor on the way out.
    assert int(captured_cu.sum()) == 0


def test_replay_copies_the_values_when_the_experiment_flag_is_on(monkeypatch):
    # AFD_UBATCH_REPLAY_COPY trades the identity precondition for a shape
    # check plus a copy, which is what a DBO step needs -- its per-ubatch
    # metadata is cloned per step, so identity can never hold.
    from afd_plugin.v1.worker import ubatch_wrapper

    monkeypatch.setattr(ubatch_wrapper, "_UBATCH_REPLAY_COPY", True)

    captured_seq = torch.zeros(4, dtype=torch.int32)
    fresh_seq = torch.arange(4, dtype=torch.int32)
    block_table = torch.zeros((4, 2), dtype=torch.int32)
    slot_mapping = torch.zeros(8, dtype=torch.int64)
    captured_cu = torch.zeros(5, dtype=torch.int32)

    captured = _metadata(captured_seq, block_table, slot_mapping, captured_cu)
    fresh = _metadata(fresh_seq, block_table.clone(), slot_mapping, torch.arange(5))

    assert _refresh_metadata_pair(captured, fresh) is True
    assert torch.equal(captured_seq, fresh_seq)


def test_replay_still_refuses_a_shape_mismatch_under_the_flag(monkeypatch):
    from afd_plugin.v1.worker import ubatch_wrapper

    monkeypatch.setattr(ubatch_wrapper, "_UBATCH_REPLAY_COPY", True)

    captured = _metadata(
        torch.zeros(4, dtype=torch.int32),
        torch.zeros((4, 2), dtype=torch.int32),
        torch.zeros(8, dtype=torch.int64),
        torch.zeros(5, dtype=torch.int32),
    )
    fresh = _metadata(
        torch.zeros(6, dtype=torch.int32),
        torch.zeros((6, 2), dtype=torch.int32),
        torch.zeros(12, dtype=torch.int64),
        torch.zeros(7, dtype=torch.int32),
    )

    assert _refresh_metadata_pair(captured, fresh) is False
