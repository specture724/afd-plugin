# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Token buckets for the AFD prefill FULL capture scheme.

The scheme captures one cooperative FULL CUDA graph per prefill token bucket
(docs/design/prefill_bucketed_full_capture.md) and pads every prefill step up
to the nearest bucket at runtime. The bucket list is a deployment knob: it
has to sit inside the vLLM cudagraph capture sizes for the dispatch padding
to land on a captured graph, and at or below ``max_num_batched_tokens``
because no step can exceed it.
"""

from __future__ import annotations

import os

PREFILL_BUCKETS_ENV = "PREFILL_BUCKETS"

# FLASH_ATTN_MLA's metadata builder routes every request whose query length
# is at or below this threshold through the decode pathway
# (``MLACommonMetadataBuilder.reorder_batch_threshold``). A step whose first
# request is at or below it builds decode-only attention metadata, which a
# prefill-shaped graph cannot replay -- such steps stay eager.
MLA_REORDER_BATCH_THRESHOLD = 512

# Split of the synthetic capture batch's tokens between its two real
# requests. The split has to be unequal so the request-aligned DBO splitter
# has a boundary to split on, and lopsided enough that both ubatches' token
# counts (70% and 30% of the bucket) match the shapes real steps produce.
CAPTURE_BATCH_PRIMARY_FRACTION_NUMERATOR = 7
CAPTURE_BATCH_FRACTION_DENOMINATOR = 10


def resolve_prefill_buckets(
    environ: dict[str, str] | os._Environ | None = None,
) -> tuple[int, ...]:
    """Parse ``PREFILL_BUCKETS`` into an ascending bucket tuple.

    Empty (the default) means the prefill bucket scheme is off and the
    runner behaves exactly as before. Invalid values raise ``ValueError``:
    a typo here would silently disable graph capture for most steps.
    """
    source = os.environ if environ is None else environ
    raw = source.get(PREFILL_BUCKETS_ENV, "").strip()
    if not raw:
        return ()

    buckets: list[int] = []
    for part in raw.replace(",", " ").split():
        value = int(part)
        if value <= 0:
            raise ValueError(
                f"{PREFILL_BUCKETS_ENV} entries must be positive; got {part!r}"
            )
        buckets.append(value)
    if len(set(buckets)) != len(buckets):
        raise ValueError(f"{PREFILL_BUCKETS_ENV} has duplicate entries: {raw!r}")
    return tuple(sorted(buckets))


def resolve_prefill_bucket(
    buckets: tuple[int, ...],
    num_tokens: int,
) -> int | None:
    """Nearest bucket at or above ``num_tokens``, or ``None`` below the first.

    Steps below the smallest bucket stay eager: their absolute host cost is
    small, and padding them to the first bucket would spend more real GPU
    work on padding than the saved launches are worth.
    """
    for bucket in buckets:
        if num_tokens <= bucket:
            return bucket
    return None


def fixed_bucket_split_token(bucket: int) -> int:
    """Where a bucket-padded step splits its two ubatches.

    A cooperative graph bakes each ubatch's metadata shapes, so a replay is
    only possible when every step of a bucket splits at the same token. That
    rules out the request-aligned split, whose boundary moves with the batch --
    which is why a DBO prefill step could never replay and always fell back to
    eager. Half the bucket, rounded down to the alignment the per-token views
    need, is the same point the capture uses.
    """
    half = bucket // 2
    return max(
        SPLIT_TOKEN_ALIGNMENT,
        (half // SPLIT_TOKEN_ALIGNMENT) * SPLIT_TOKEN_ALIGNMENT,
    )


# A ubatch's per-token views start at the split point, so it carries the same
# alignment requirement as the request-aligned splitter's boundaries.
SPLIT_TOKEN_ALIGNMENT = 16


def prefill_bucket_capture_reqs(bucket: int, max_num_seqs: int) -> int:
    """Requests the dispatch descriptor pins for one bucket.

    Must match ``CudagraphDispatcher._create_padded_batch_descriptor``'s
    non-uniform branch exactly, because FULL dispatch requires an exact key
    match: ``num_reqs = min(num_tokens_padded, max_num_seqs)``.
    """
    return min(bucket, max_num_seqs)


def synthetic_prefill_bucket_batch(
    bucket: int,
    num_reqs: int,
) -> tuple[list[int], int]:
    """Scheduled-token split of the capture batch, and its max query length.

    Two real requests carry ``numerator``/``1 - numerator`` of the bucket and
    the remaining pinned requests are empty (zero tokens), which is exactly
    the empty-request padding a runtime step produces: the padded rows repeat
    the last ``cu_seqlens`` offset and are skipped by the varlen kernel. The
    requests' sequence lengths equal their query lengths, so the capture
    builds zero-context prefill metadata -- the only shape a replay step is
    allowed to have (prior context would need chunked-context kernels the
    graph does not contain).
    """
    if num_reqs < 2:
        raise ValueError(
            "the synthetic prefill bucket batch needs two real requests; "
            f"got num_reqs={num_reqs}"
        )
    primary = bucket * CAPTURE_BATCH_PRIMARY_FRACTION_NUMERATOR
    primary //= CAPTURE_BATCH_FRACTION_DENOMINATOR
    num_scheduled_tokens = [primary, bucket - primary] + [0] * (num_reqs - 2)
    return num_scheduled_tokens, primary


__all__ = [
    "SPLIT_TOKEN_ALIGNMENT",
    "fixed_bucket_split_token",
    "CAPTURE_BATCH_FRACTION_DENOMINATOR",
    "CAPTURE_BATCH_PRIMARY_FRACTION_NUMERATOR",
    "MLA_REORDER_BATCH_THRESHOLD",
    "PREFILL_BUCKETS_ENV",
    "prefill_bucket_capture_reqs",
    "resolve_prefill_bucket",
    "resolve_prefill_buckets",
    "synthetic_prefill_bucket_batch",
]
