# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Prefill-bucket synthetic batch for cudagraph capture dummy runs.

Upstream ``GPUModelRunner._dummy_run`` can synthesize a uniform decode batch
or a mixed decode+prefill batch, but nothing shaped like the AFD prefill
bucket capture batch: ``num_reqs`` pinned to the dispatch descriptor, two
real requests carrying an unequal 7:3 split of the bucket (so the
request-aligned DBO splitter has an interior boundary), the remaining
pinned requests empty (zero tokens -- the same empty-request padding a
runtime step produces), and per-request ``seq_lens == query_len`` so the
captured attention metadata has no chunked context. A capture batch with a
prior context would bake chunked-context kernels that no replay step can
feed.

The patch adds one opt-in parameter, ``create_prefill_bucket_batch``; every
other path is the upstream function verbatim. Upstream: vLLM v0.26.0,
``vllm/v1/worker/gpu_model_runner.py``, commit
568afb3a13806beb53bb2e6bd518269357b237c0. On a vLLM upgrade, re-copy the
upstream function and re-apply the two marked AFD regions.
"""

from __future__ import annotations

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer
from vllm.v1.worker.cp_utils import (
    get_dcp_dummy_context_len,
    prepare_dcp_dummy_context_metadata,
)
from vllm.v1.worker.gpu_model_runner import PerLayerAttnMetadata

from afd_plugin.v1.worker.prefill_buckets import synthetic_prefill_bucket_batch

logger = init_logger(__name__)


def _resolve_maybe_create_ubatch_slices():
    """The splitter as installed in the runner's module namespace.

    The AFD request-aligned split patch replaces the attribute on
    ``vllm.v1.worker.gpu_model_runner``, and upstream ``_dummy_run`` resolves
    it through that namespace, so the copy must do the same to keep both
    call sites on one implementation.
    """
    from vllm.v1.worker import gpu_model_runner as gpu_model_runner_module

    return gpu_model_runner_module.maybe_create_ubatch_slices


# Patch reason: upstream _dummy_run cannot synthesize the AFD prefill bucket
# capture batch (pinned num_reqs, 7:3 two-request token split, empty padding
# requests, zero-context per-request seq_lens), so capturing a prefill bucket
# would bake the wrong batch shape -- an even split whose query lengths fall
# at or below MLA's decode threshold, plus chunked-context kernels no replay
# step can feed.
# Patch functionality: add one opt-in parameter, create_prefill_bucket_batch,
# which switches the synthetic batch to the AFD bucket shape; every other
# branch is the upstream function copied verbatim.
# Signature: matches upstream plus the documented create_prefill_bucket_batch
# parameter, defaulting to False so every existing caller is unaffected.
# Upstream: vLLM v0.26.0, vllm/v1/worker/gpu_model_runner.py
# Commit: 568afb3a13806beb53bb2e6bd518269357b237c0
def _dummy_run(
    self,
    num_tokens: int,
    cudagraph_runtime_mode: CUDAGraphMode | None = None,
    force_attention: bool = False,
    uniform_decode: bool = False,
    allow_microbatching: bool = True,
    skip_eplb: bool = False,
    is_profile: bool = False,
    create_mixed_batch: bool = False,
    remove_lora: bool = True,
    is_graph_capturing: bool = False,
    num_active_loras: int = 0,
    profile_seq_lens: int | None = None,
    create_prefill_bucket_batch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a dummy forward pass to warm up/profile run or capture the
    CUDA graph for the model.

    Args:
        num_tokens: Number of tokens to run the dummy forward pass.
        cudagraph_runtime_mode: used to control the behavior.
            - if not set will determine the cudagraph mode based on using
                the self.cudagraph_dispatcher.
            - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
            - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
            - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                needed.
        force_attention: If True, always create attention metadata. Used to
            warm up attention backend when mode is NONE.
        uniform_decode: If True, the batch is a uniform decode batch.
        skip_eplb: If True, skip EPLB state update.
        is_profile: If True, this is a profile run.
        create_mixed_batch: If True, create a mixed batch with both decode
            (1 token) and prefill (multiple tokens) requests.
        remove_lora: If False, dummy LoRAs are not destroyed after the run
        num_active_loras: Number of distinct active LoRAs to capture for.
            LoRA is activated when num_active_loras > 0.
        profile_seq_lens: If provided, use this value for seq_lens instead
            of max_query_len. Used to profile attention workspace that
            scales with context length.
        create_prefill_bucket_batch: AFD addition. If True, synthesize the
            AFD prefill bucket capture batch instead of the even split:
            ``num_reqs`` requests, the first two carrying an unequal 7:3
            split of ``num_tokens`` (the bucket) and the rest empty, with
            per-request ``seq_lens`` equal to the query lengths so the
            captured attention metadata has zero chunked context.
    """
    mm_config = self.vllm_config.model_config.multimodal_config
    if mm_config and mm_config.mm_encoder_only:
        # The current dummy run only covers LM execution, so we can skip it.
        # mm encoder dummy run may need to add in the future.
        return torch.tensor([]), torch.tensor([])

    assert (
        cudagraph_runtime_mode is None or cudagraph_runtime_mode.is_valid_runtime_mode()
    )

    # If cudagraph_mode.decode_mode() == FULL and
    # cudagraph_mode.separate_routine(). This means that we are using
    # different graphs and/or modes for mixed prefill-decode batches vs.
    # uniform decode batches. A uniform decode batch means that all
    # requests have identical query length, except a potential virtual
    # request (shorter) in the batch account for padding.
    # Uniform decode batch could either be common pure decode, where
    # max_query_len == 1, or speculative decode, where
    # max_query_len == 1 + num_spec_decode_tokens.

    # When setting max_query_len = 1, we switch to and capture the optimized
    # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
    # for GQA/MQA.
    max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

    # Set num_scheduled_tokens based on num_tokens and max_num_seqs
    # for dummy run with LoRA so that the num_reqs collectively
    # has num_tokens in total.
    assert num_tokens <= self.max_num_tokens
    max_num_reqs = self.scheduler_config.max_num_seqs
    if create_mixed_batch:
        assert not uniform_decode
        # Create mixed batch:
        # first half decode tokens, second half one prefill
        num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
        num_prefill_tokens = num_tokens - num_decode_tokens
        num_reqs = num_decode_tokens + 1

        # Create decode requests (1 token each) followed by prefill request
        num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
        # Note: Overriding max_query_len to be the prefill tokens
        max_query_len = num_prefill_tokens
    # ### PATCH START: AFD prefill bucket synthetic batch
    elif create_prefill_bucket_batch:
        assert not uniform_decode
        num_reqs = min(num_tokens, max_num_reqs)
        num_scheduled_tokens_list, max_query_len = synthetic_prefill_bucket_batch(
            num_tokens,
            num_reqs,
        )
    # ### PATCH END: AFD prefill bucket synthetic batch
    elif uniform_decode:
        assert not create_mixed_batch
        num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
        num_scheduled_tokens_list = [max_query_len] * num_reqs
        if num_tokens % max_query_len != 0:
            num_scheduled_tokens_list[-1] = num_tokens % max_query_len
    else:
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs

    assert sum(num_scheduled_tokens_list) == num_tokens
    assert len(num_scheduled_tokens_list) == num_reqs
    num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
    num_tokens_unpadded = int(num_scheduled_tokens.sum())

    num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

    _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
        self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens,
            max_num_scheduled_tokens=max_query_len,
            use_cascade_attn=False,
            allow_microbatching=allow_microbatching,
            force_eager=is_profile or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
            # `force_uniform_decode` is used for cudagraph capture; because for
            # capturing mixed prefill-decode batches, we sometimes use
            # num_tokens == num_reqs which looks like a uniform decode batch to the
            # dispatcher; but we actually want to capture a piecewise cudagraph
            force_uniform_decode=uniform_decode,
            # `force_has_lora` is used for cudagraph capture; because LoRA is
            # activated later in the context manager, but we need to know the
            # LoRA state when determining the batch descriptor for capture
            force_has_lora=num_active_loras > 0,
            # `force_num_active_loras` is used for cudagraph capture; because we
            # need to capture graphs for specific num_active_loras counts
            force_num_active_loras=num_active_loras,
        )
    )

    if cudagraph_runtime_mode is None:
        cudagraph_runtime_mode = _cudagraph_mode
    else:
        assert cudagraph_runtime_mode == _cudagraph_mode, (
            f"Cudagraph runtime mode mismatch in dummy_run. "
            f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
        )

    num_tokens_padded = batch_desc.num_tokens
    num_reqs_padded = (
        batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
    )
    dcp_dummy_context_len = get_dcp_dummy_context_len(
        self.dcp_world_size,
        self.parallel_config.cp_kv_cache_interleave_size,
        hasattr(self, "kv_cache_config"),
        create_mixed_batch,
        is_graph_capturing,
        uniform_decode,
    )
    maybe_create_ubatch_slices = _resolve_maybe_create_ubatch_slices()
    ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
        should_ubatch,
        num_scheduled_tokens,
        num_tokens_padded,
        num_reqs_padded,
        self.vllm_config.parallel_config.num_ubatches,
    )
    logger.debug(
        "ubatch_slices: %s, ubatch_slices_padded: %s",
        ubatch_slices,
        ubatch_slices_padded,
    )

    attn_metadata: PerLayerAttnMetadata | None = None

    slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
        num_tokens_padded=num_tokens_padded,
        num_reqs_padded=num_reqs_padded,
        num_tokens_unpadded=num_tokens_unpadded,
        ubatch_slices=ubatch_slices_padded,
    )

    # Dummy runs have no real slot assignments — fill with -1 so
    # concat_and_cache kernels skip the KV write.
    if slot_mappings_by_group is not None:
        for sm in slot_mappings_by_group.values():
            sm.fill_(-1)

    # _dummy_run shares pinned CPU buffers (seq_lens, query_start_loc,
    # etc.) with execute_model.  It must participate in the same event
    # protocol so that back-to-back dummy/real steps don't overwrite
    # pinned memory while a prior non_blocking H2D DMA is still reading.
    with self.synchronize_input_prep():
        # If force_attention is True, we always capture attention.
        # Otherwise, it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            if profile_seq_lens is not None:
                seq_lens = profile_seq_lens  # type: ignore[assignment]
            elif create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                if dcp_dummy_context_len > 0:
                    seq_lens = torch.tensor(  # type: ignore[assignment]
                        [1 + dcp_dummy_context_len] * num_decode_tokens
                        + [num_prefill_tokens + dcp_dummy_context_len],
                        dtype=torch.int,
                    )
                else:
                    seq_lens = torch.tensor(  # type: ignore[assignment]
                        [1] * num_decode_tokens + [num_prefill_tokens + 1],
                        dtype=torch.int,
                    )
            # ### PATCH START: AFD prefill bucket per-request seq_lens
            elif create_prefill_bucket_batch:
                # seq_len == query_len for every request: the capture batch
                # has no prior context, matching the replay steps the
                # bucketed prefill graphs are dispatched for.
                seq_lens = torch.tensor(  # type: ignore[assignment]
                    num_scheduled_tokens_list,
                    dtype=torch.int,
                )
            # ### PATCH END: AFD prefill bucket per-request seq_lens
            elif dcp_dummy_context_len > 0:
                seq_lens = max_query_len + dcp_dummy_context_len  # type: ignore[assignment]
            else:
                seq_lens = max_query_len  # type: ignore[assignment]
            self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
            self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
            self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)

            cum_num_tokens = self._get_cumsum_and_arange(
                num_scheduled_tokens, self.query_pos.np
            )
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1].fill(
                cum_num_tokens[-1]
            )
            self.query_start_loc.copy_to_gpu()

            prepare_dcp_dummy_context_metadata(
                input_batch=self.input_batch,
                kv_cache_config=getattr(self, "kv_cache_config", None),
                query_pos=self.query_pos,
                positions=self.positions,
                query_start_loc=self.query_start_loc,
                num_reqs=num_reqs,
                num_tokens_unpadded=num_tokens_unpadded,
                dcp_dummy_context_len=dcp_dummy_context_len,
            )

            # Sync block table CPU->GPU so cleared rows from
            # remove_request() are visible to the attention metadata
            # builder. Without this, stale block IDs from finished
            # requests can corrupt Mamba state.
            self.input_batch.block_table.commit_block_table(num_reqs_padded)

            pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded if pad_attn else None,
                num_reqs=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=(ubatch_slices_padded if pad_attn else ubatch_slices),
                for_cudagraph_capture=is_graph_capturing,
                slot_mappings=slot_mappings_by_group,
                use_spec_decode=self.speculative_config is not None,
            )

    with self.maybe_dummy_run_with_lora(
        self.lora_config,
        num_scheduled_tokens,
        num_sampled_tokens,
        remove_lora,
        num_active_loras,
    ):
        # Make sure padding doesn't exceed max_num_tokens
        assert num_tokens_padded <= self.max_num_tokens
        model_kwargs = self._init_model_kwargs()
        if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
            input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

            model_kwargs = {
                **model_kwargs,
                **self._dummy_mm_kwargs(num_reqs),
            }
        elif self.enable_prompt_embeds:
            input_ids = None
            inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            model_kwargs = self._init_model_kwargs()
        else:
            input_ids = self.input_ids.gpu[:num_tokens_padded]
            inputs_embeds = None

        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_tokens_padded]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
        else:
            positions = self.positions[:num_tokens_padded]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            if self.intermediate_tensors is None:
                self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                    batch_size=self.max_num_tokens,
                    dtype=self.model_config.dtype,
                    device=self.device,
                )

            intermediate_tensors = self.sync_and_gather_intermediate_tensors(
                num_tokens_padded, None, False
            )

        if ubatch_slices_padded is not None:
            # Adjust values to reflect a single ubatch.
            # TODO(sage,lucas): this is cruft that should be addressed in
            #  the padding refactor.
            num_tokens_padded = ubatch_slices_padded[0].num_tokens
            if num_tokens_across_dp is not None:
                num_tokens_across_dp[:] = num_tokens_padded

        with (
            self.maybe_randomize_inputs(input_ids, inputs_embeds),
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
                slot_mapping=slot_mappings,
            ),
        ):
            outputs = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        if self.use_aux_hidden_state_outputs:
            hidden_states, _ = outputs
        else:
            hidden_states = outputs

        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
            or self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(
                self.drafter,
                EagleProposer
                | DFlashProposer
                | DraftModelProposer
                | ExtractHiddenStatesProposer
                | Gemma4Proposer,
            )
            assert self.speculative_config is not None
            # Eagle currently only supports PIECEWISE cudagraphs.
            # Therefore only use cudagraphs if the main model uses PIECEWISE
            # NOTE(lucas): this is a hack, need to clean up.
            use_cudagraphs = (
                (
                    is_graph_capturing
                    and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                )
                or (
                    not is_graph_capturing
                    and cudagraph_runtime_mode != CUDAGraphMode.NONE
                )
            ) and not self.speculative_config.enforce_eager

            # Note(gnovack) - We need to disable cudagraphs for one of the two
            # lora cases when cudagraph_specialize_lora is enabled. This is a
            # short term mitigation for issue mentioned in
            # https://github.com/vllm-project/vllm/issues/28334
            if (
                self.compilation_config.cudagraph_specialize_lora
                and num_active_loras > 0
            ):
                use_cudagraphs = False

            self.drafter.dummy_run(
                num_tokens,
                use_cudagraphs=use_cudagraphs,
                is_graph_capturing=is_graph_capturing,
                slot_mappings=slot_mappings,
            )

    # We register layerwise NVTX hooks here after the first dynamo tracing is
    # done to avoid nvtx operations in hook functions being traced by
    # torch dynamo and causing graph breaks.
    # Note that for DYNAMO_ONCE and VLLM_COMPILE mode,
    # compiled model's dynamo tracing is only done once and the compiled model's
    # __call__ function is replaced by calling the compiled function.
    # So it's safe to register hooks here. Hooks will be registered to
    # both compiled and uncompiled models but they will never
    # be called on the compiled model execution path.
    self._register_layerwise_nvtx_hooks()

    # This is necessary to avoid blocking DP.
    # For dummy runs, we typically skip EPLB since we don't have any real
    # requests to process.
    # However, in DP settings, there may be cases when some DP ranks do
    # not have any requests to process, so they're executing dummy batches.
    # In such cases, we still have to trigger EPLB to make sure
    # ranks execute the rearrangement in synchronization.
    if not skip_eplb:
        self.eplb_step(is_dummy=True, is_profile=is_profile)

    logit_indices = np.cumsum(num_scheduled_tokens) - 1
    logit_indices_device = torch.from_numpy(logit_indices).to(
        self.device, non_blocking=True
    )
    return hidden_states, hidden_states[logit_indices_device]


def apply_prefill_bucket_dummy_run() -> None:
    """Install the bucket-aware dummy run onto vLLM's GPU model runner.

    Both vLLM's own code and the AFD runner resolve ``_dummy_run`` off the
    class, so replacing the class attribute covers every caller. The AFD
    runner's override delegates through ``super()``, so it lands here too.
    Idempotent via a marker attribute on the installed function.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(_dummy_run, "_afd_prefill_bucket", False):
        return
    _dummy_run._afd_prefill_bucket = True  # type: ignore[attr-defined]
    GPUModelRunner._dummy_run = _dummy_run


apply_prefill_bucket_dummy_run()

__all__ = [
    "apply_prefill_bucket_dummy_run",
    "_dummy_run",
]
