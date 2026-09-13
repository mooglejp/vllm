#!/usr/bin/env bash
set -euo pipefail

profile_dir=${1:-/tmp/tq-radiance-delta-fork-profile}
server_log=${2:-/tmp/tq-radiance-delta-fork-server.log}

rm -rf "$profile_dir"
mkdir -p "$profile_dir"

export VLLM_TARGET_DEVICE=rocm
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_MOE=0
export VLLM_ROCM_USE_AITER_LINEAR=0
export VLLM_ROCM_USE_AITER_FP8BMM=0
export VLLM_ROCM_USE_AITER_FP4BMM=0
export VLLM_ROCM_USE_AITER_RMSNORM=0
export VLLM_TQ_GFX1201_K8V4=1
export VLLM_TQ_GFX1201_K8V4_PREFILL=0
export VLLM_TQ_GFX1201_K8V4_UNIFIED_CONTINUATION=0
export TQ_DISABLE_FLASH_PREFILL=1

exec /tmp/tq-venv/bin/python -m vllm.entrypoints.cli.main serve \
  /model \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen38-27b-tq-mtp \
  --trust-remote-code \
  --generation-config vllm \
  --load-format runai_streamer \
  --model-loader-extra-config '{"distributed":false,"memory_limit":3221225472}' \
  --attention-backend TURBOQUANT \
  --block-size 16 \
  --kv-cache-dtype turboquant_k8v4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 256 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"attention_backend":"TURBOQUANT","enable_adaptive_verification":false}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$profile_dir\",\"torch_profiler_with_stack\":false,\"torch_profiler_with_flops\":false,\"torch_profiler_use_gzip\":true,\"torch_profiler_dump_cuda_time_total\":true,\"torch_profiler_record_shapes\":false,\"torch_profiler_with_memory\":false,\"ignore_frontend\":true,\"delay_iterations\":0,\"max_iterations\":0,\"warmup_iterations\":0,\"active_iterations\":1,\"wait_iterations\":0}" \
  >"$server_log" 2>&1
