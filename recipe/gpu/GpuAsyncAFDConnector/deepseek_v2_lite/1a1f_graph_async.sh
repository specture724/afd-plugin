#!/usr/bin/env bash
# 1A1F async GPU connector with the Attention side on CUDA graphs.
#
# Launch under a GPU reservation, which sets CUDA_VISIBLE_DEVICES:
#   gpu run --gpu-ids 3,7 -- bash recipe/gpu/GpuAsyncAFDConnector/deepseek_v2_lite/1a1f_graph_async.sh
#
# Only the FFN server runs on CUDA graphs.
#
# The Attention side stays eager, and not by preference: enabling graphs there
# puts vLLM's AOT compilation in front of the model, and Dynamo traces through
# the MoE proxy into the connector, which builds tensors from raw NVSHMEM
# pointers and calls the CUDA driver through ctypes. Compilation aborts. The
# fix is to make the dispatch opaque to Dynamo -- a custom op it splits at --
# which does not exist yet.
#
# The FFN side has no such problem: its work never goes through a compiled
# forward. It has no control plane either, so it never learns the next work
# item's shape ahead of time -- it captures the experts once per MoE layer at
# the largest batch the sender can produce and pads every item up to that. The
# grouping rides in a device-side count vector, so a replay re-reads it. One
# shape covers prefill and decode alike.
#
# The two roles are separate vllm serve processes: the AFD process group hosts
# its own TCPStore, which cannot be created under a single torchrun/torchelastic
# launcher.
set -u

MODEL_PATH=${MODEL_PATH:-/path/model_weights/DeepSeek-V2-Lite}
# How to invoke vLLM. `uv run vllm` is right from a synced checkout, but it
# falls through to whatever `vllm` is on PATH when the venv lacks vLLM -- and a
# vLLM without this plugin installed serves the request correctly as a plain
# model, with no AFD in the picture at all. Override this to point at the
# interpreter that actually has both. Word-split on purpose.
VLLM_CMD=${VLLM_CMD:-uv run vllm}
# Set to 1 to run the FFN experts eagerly instead of from captured graphs. The
# A/B control: everything else about the two runs stays identical.
FFN_EAGER=${FFN_EAGER:-0}
# Set to 1 to run the Attention side with vLLM's dual batch overlap. The yield
# it hands off at already sits between send_attn_output and recv_ffn_output --
# exactly while the A->F->A round trip is in flight -- so one half's attention
# should run while the other waits on its experts.
# The flags go to both servers: each side sizes its window's ring depth from
# the stage count, and a stage count that only one role sees leaves the FFN
# with one ring slot while the Attention dispatches into two -- the second
# lands in a slot the FFN never polls, and the reply the first forward waits
# for never comes.
# Set to 1 to keep the Attention side eager (the long-standing default). Set
# to 0 to run it on FULL_DECODE_ONLY CUDA graphs: the MoE round trip is the
# opaque afd_async_moe_roundtrip custom op now, so AOT compilation splits at
# it and the DBO ubatches capture cooperatively -- replays keep the interleaved
# kernel order without any host ping-pong.
ATTN_EAGER=${ATTN_EAGER:-1}
ENABLE_DBO=${ENABLE_DBO:-0}
DBO_ARGS=""
if [ "$ENABLE_DBO" = 1 ]; then
    DBO_ARGS="--enable-dbo --dbo-decode-token-threshold ${DBO_DECODE_THRESHOLD:-2} --dbo-prefill-token-threshold ${DBO_PREFILL_THRESHOLD:-12}"
fi
FFN_EAGER_ARG=""
[ "$FFN_EAGER" = 1 ] && FFN_EAGER_ARG="--enforce-eager"
LOG_DIR=${LOG_DIR:-.}
mkdir -p "$LOG_DIR"
export VLLM_USE_V2_MODEL_RUNNER=0
# Single node over NVLink: skip the IB transport probe.
export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-none}
# Two servers on one box spawn a lot of threads; the HF tokenizer's rayon pool
# is the first thing to fail when thread creation gets refused.
# Line-buffer the servers' output. Both are torn down with a signal, and
# block-buffered stdout loses whatever had not reached 4KiB -- which is how a
# short run ends with two empty log files.
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
if [ "${#DEVICES[@]}" -lt 2 ]; then
    echo "need 2 visible GPUs, got ${#DEVICES[@]}: ${CUDA_VISIBLE_DEVICES:-unset}" >&2
    exit 1
fi
ATTN_DEVICES="${DEVICES[0]}"
FFN_DEVICES="${DEVICES[1]}"
echo "attention on ${ATTN_DEVICES}, ffn on ${FFN_DEVICES}"

# Lower this when sharing a box: vLLM refuses to start if the desired
# fraction exceeds what is actually free.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
# Prefill batch size drives whether each MoE call clears the compute-bound
# inflection point, so it is the knob to raise when benchmarking.
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-512}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
# Optional online quantization (e.g. "fp8") to fit larger experts on one FFN
# GPU. Empty keeps the checkpoint's native dtype.
QUANT=${QUANT:-}
QUANT_ARGS=""
[ -n "$QUANT" ] && QUANT_ARGS="--quantization $QUANT"
# Model-family knobs (DeepSeek-V4 needs its own tokenizer mode and fp8 KV).
TOKENIZER_MODE=${TOKENIZER_MODE:-}
TOKENIZER_MODE_ARGS=""
[ -n "$TOKENIZER_MODE" ] && TOKENIZER_MODE_ARGS="--tokenizer-mode $TOKENIZER_MODE"
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-}
KV_CACHE_DTYPE_ARGS=""
[ -n "$KV_CACHE_DTYPE" ] && KV_CACHE_DTYPE_ARGS="--kv-cache-dtype $KV_CACHE_DTYPE"
# One capture bucket per decode size 1..MAX_NUM_SEQS. A single bucket at the
# max would pad every decode batch up to it, and the empty-last-ubatch guard
# then refuses to split any batch below half the bucket -- DBO would silently
# never split under graphs. ATTN_EAGER=2 adds the prefill token buckets on top,
# as capture sizes so the dispatcher's padding lands on a captured graph. The
# plugin reads PREFILL_BUCKETS from the environment, and it is an
# Attention-side strategy only -- the FFN role refuses to start when it sees
# the variable -- so it goes on that one command, never on a shared export.
ATTN_GRAPH_ARGS="--enforce-eager"
ATTN_BUCKET_ENV=()
if [ "$ATTN_EAGER" != 1 ]; then
    BUCKET_LIST=""
    MAX_CAPTURE="$MAX_NUM_SEQS"
    if [ "$ATTN_EAGER" = 2 ]; then
        PREFILL_BUCKETS=${PREFILL_BUCKETS:-$MAX_NUM_BATCHED_TOKENS}
        ATTN_BUCKET_ENV=(env "PREFILL_BUCKETS=$PREFILL_BUCKETS")
        BUCKET_LIST=$(echo "$PREFILL_BUCKETS" | tr ',' ' ')
        for size in $BUCKET_LIST; do
            [ "$size" -gt "$MAX_CAPTURE" ] && MAX_CAPTURE=$size
        done
    fi
    CAPTURE_SIZES="$(seq -s' ' 1 "$MAX_NUM_SEQS") ${BUCKET_LIST}"
    ATTN_GRAPH_ARGS="--max-cudagraph-capture-size ${MAX_CAPTURE} --cudagraph-capture-sizes ${CAPTURE_SIZES} --compilation-config {\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}"
fi
# Free-form passthrough, e.g. EXTRA_ARGS="--no-enable-prefix-caching". Without
# it a benchmark that replays prompts measures the prefix cache instead of
# prefill, which reads as a 30x speedup.
EXTRA_ARGS=${EXTRA_ARGS:-}
AFD_PORT=${AFD_PORT:-6275}
API_PORT=${API_PORT:-18311}
# The FFN server never takes HTTP -- its EngineCore is a connector daemon --
# but it still starts an API server, and two of them racing for one port means
# whichever loses exits and takes its role down with it. Give it its own.
FFN_API_PORT=${FFN_API_PORT:-$((API_PORT + 1))}

CUDA_VISIBLE_DEVICES="$ATTN_DEVICES" "${ATTN_BUCKET_ENV[@]}" $VLLM_CMD serve "$MODEL_PATH" \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    $QUANT_ARGS \
    $TOKENIZER_MODE_ARGS \
    $KV_CACHE_DTYPE_ARGS \
    --additional-config '{
        "afd": {
            "role": "attention",
            "connector": "GpuAsyncAFDConnector",
            "async": true,
            "compute_gate_on_attention": true,
            "host": "127.0.0.1",
            "port": '"$AFD_PORT"',
            "num_attention_ranks": 1,
            "num_ffn_ranks": 1
        }
    }' \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --api-server-count 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    $ATTN_GRAPH_ARGS \
    $DBO_ARGS \
    $EXTRA_ARGS \
    --host 127.0.0.1 \
    --port "$API_PORT" \
    --trust-remote-code > "$LOG_DIR/attn.log" 2>&1 &
ATTN_PID=$!

CUDA_VISIBLE_DEVICES="$FFN_DEVICES" $VLLM_CMD serve "$MODEL_PATH" \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    $QUANT_ARGS \
    $TOKENIZER_MODE_ARGS \
    $KV_CACHE_DTYPE_ARGS \
    --additional-config '{
        "afd": {
            "role": "ffn",
            "connector": "GpuAsyncAFDConnector",
            "async": true,
            "compute_gate_on_attention": true,
            "host": "127.0.0.1",
            "port": '"$AFD_PORT"',
            "num_attention_ranks": 1,
            "num_ffn_ranks": 1
        }
    }' \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --api-server-count 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    $FFN_EAGER_ARG \
    $DBO_ARGS \
    $EXTRA_ARGS \
    --host 127.0.0.1 \
    --port "$FFN_API_PORT" \
    --trust-remote-code > "$LOG_DIR/ffn.log" 2>&1 &
FFN_PID=$!

cleanup() {
    kill "$ATTN_PID" "$FFN_PID" 2>/dev/null
    wait "$ATTN_PID" "$FFN_PID" 2>/dev/null
}
trap cleanup EXIT

for _ in $(seq 1 "${READY_TIMEOUT:-600}"); do
    if curl -sf "http://127.0.0.1:$API_PORT/health" > /dev/null 2>&1; then
        echo "server ready on http://127.0.0.1:$API_PORT"
        echo
        echo "curl -s http://127.0.0.1:$API_PORT/v1/completions \\"
        echo "  -H 'Content-Type: application/json' \\"
        echo "  -d '{\"model\":\"$MODEL_PATH\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}'"
        echo
        if [ -n "${SMOKE:-}" ]; then
            curl -s "http://127.0.0.1:$API_PORT/v1/completions" \
                -H 'Content-Type: application/json' \
                -d '{"model":"'"$MODEL_PATH"'","prompt":"The capital of France is",
                     "max_tokens":16,"temperature":0}'
            echo
            exit 0
        fi
        # Stay up so the servers can take requests; Ctrl-C tears both down.
        wait "$ATTN_PID" "$FFN_PID"
        exit 0
    fi
    if ! kill -0 "$ATTN_PID" 2>/dev/null || ! kill -0 "$FFN_PID" 2>/dev/null; then
        echo "a server exited early; see $LOG_DIR/attn.log and $LOG_DIR/ffn.log" >&2
        exit 1
    fi
    sleep 1
done
echo "timed out waiting for the server" >&2
exit 1
