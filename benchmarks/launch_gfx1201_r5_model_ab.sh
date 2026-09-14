#!/usr/bin/env bash
set -euo pipefail

mode=${1:?baseline or candidate}
log_path=${2:?container log path}
pid_path=${3:?container pid path}
stats_path=${4:-/dev/shm/r5_hook_stats.json}

memory_limit=$(</sys/fs/cgroup/memory.max)
swap_limit=$(</sys/fs/cgroup/memory.swap.max)
if [[ "$memory_limit" == max || "$memory_limit" -gt 12884901888 || "$swap_limit" != 0 ]]; then
  echo "R5 requires an isolated cgroup with RAM <=12 GiB and swap disabled" >&2
  exit 2
fi

if [[ "$mode" != baseline && "$mode" != candidate ]]; then
  echo "mode must be baseline or candidate" >&2
  exit 2
fi

if [[ -e "$log_path" || -e "$pid_path" || -e "$stats_path" ]]; then
  echo "Use fresh output paths; existing artifacts are preserved" >&2
  exit 2
fi
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
export VLLM_ROCM_USE_GFX1201_MXFP4_GEMM=1
export VLLM_ROCM_USE_GFX1201_MXFP4_W4A8=0
export VLLM_ROCM_USE_GFX1201_MXFP4_W4A8_PREFILL=0
export VLLM_TQ_GFX1201_K8V4_PREFILL=0
export VLLM_TQ_GFX1201_K8V4_UNIFIED_CONTINUATION=0
export TMPDIR=/dev/shm/r5-tmp
export TEMP=/dev/shm/r5-tmp
export TMP=/dev/shm/r5-tmp
export XDG_CACHE_HOME=/dev/shm/r5-xdg
export VLLM_CACHE_ROOT=/dev/shm/r5-vllm-cache
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/r5-inductor
export TRITON_CACHE_DIR=/dev/shm/r5-triton
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export R5_HOOK_STATS="$stats_path"
export R5_MODE="$mode"
export R5_MODE_FILE=/dev/shm/r5_model_control.json
export VLLM_NO_USAGE_STATS=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTHONPATH=/dev/shm/r5_model_ab:/dev/shm/r5-flash-site:/dev/shm:/workspace/vllm

runner=(/tmp/tq-venv/bin/python -m vllm.entrypoints.cli.main)

nohup "${runner[@]}" serve /model \
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
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/dev/shm/r5-model-trace","torch_profiler_use_gzip":true,"torch_profiler_with_stack":false,"ignore_frontend":true,"max_iterations":4}' \
  >"$log_path" 2>&1 &
echo $! >"$pid_path"
