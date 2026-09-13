#!/usr/bin/env bash
set -euo pipefail

# Benchmark-only launcher for the pinned Radiance control image.  The caller
# supplies one label and one host cache directory; the model and base env can
# be overridden for a different machine without changing the control flags.

label=${1:?label is required}
cache_dir=${2:?cache directory is required}
profile_dir=${3:?profile directory is required}
server_log=${4:?server log path is required}
w4a8=${RADIANCE_DELTA_W4A8:-1}
r4d=${RADIANCE_DELTA_R4D:-1}

mm_profile_args=()
if [[ "$r4d" == "0" ]]; then
  # R4D-off falls back to Torch SDPA for the dummy vision profile, whose
  # maximum feature shape requests an infeasible dense score matrix.  The
  # control request is text-only, so skip that unrelated startup probe.
  mm_profile_args+=(--skip-mm-profiling)
fi

model_dir=${RADIANCE_MODEL_DIR:-/srv/ai/models/llm/hf/amd-Qwen3.8-27B-Quark-AWQ-MXFP4}
base_env=${RADIANCE_BASE_ENV:-/home/emmett/radiance-benchmark/results/mmtp-20260910T104329Z/base.env}
image=${RADIANCE_IMAGE:-magiccodingman/vllm-radiance@sha256:83a9dc02a8f8e75aabe81366d36ebaa2e35fcbe181cacf8e8e0a4cef4ebccbcc}

mkdir -p "$cache_dir" "$profile_dir"
if [[ -z "$profile_dir" || "$profile_dir" == "/" ]]; then
  echo "refusing unsafe profile directory: $profile_dir" >&2
  exit 2
fi
find "$profile_dir" -mindepth 1 -maxdepth 1 -type f -delete
mkdir -p "$(dirname "$server_log")"

docker run --rm --name "$label" \
  --env-file "$base_env" \
  --env RADIANCE_FAST_DRAFT=0 \
  --env RADIANCE_DYNAMIC_DRAFT=0 \
  --env RADIANCE_QUARK_BF16_MTP=0 \
  --env RADIANCE_RUN_BWTEST=0 \
  --env RADIANCE_SPECULATIVE_CONFIG= \
  --env RADIANCE_MXFP4_W4A8="$w4a8" \
  --env RADIANCE_USE_R4D="$r4d" \
  --env RADIANCE_USE_R4D_GDN="$r4d" \
  --env RADIANCE_R4D_REPORT=0 \
  --env RADIANCE_USE_R4D_AR="$r4d" \
  --env RADIANCE_USE_R4D_AR_QUANT="$r4d" \
  --env RADIANCE_GDN_META="$r4d" \
  --env RADIANCE_GDN_MERGE_INPROJ="$r4d" \
  --env RADIANCE_GDN_FUSED_UPDATE="$r4d" \
  --env RADIANCE_GDN_SHARED_BUILD="$r4d" \
  --env R4D_ATTN_FP8=0 \
  --env VLLM_CACHE_ROOT=/cache/vllm \
  --env TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  --env TRITON_CACHE_DIR=/cache/triton \
  --env AITER_ROOT_DIR=/cache/aiter \
  --volume "$model_dir:/model:ro" \
  --volume "$cache_dir:/cache" \
  --volume "$profile_dir:/cache/profiler" \
  --device /dev/kfd \
  --device /dev/dri \
  --ipc private \
  --shm-size 8g \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --publish 127.0.0.1:8000:8000 \
  "$image" \
  /model \
  --served-model-name qwen38-27b-quark-mxfp4 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --max-model-len 131072 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 256 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  "${mm_profile_args[@]}" \
  --load-format runai_streamer \
  --model-loader-extra-config '{"distributed":false,"memory_limit":3221225472}' \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/cache/profiler\",\"torch_profiler_with_stack\":false,\"torch_profiler_with_flops\":false,\"torch_profiler_use_gzip\":true,\"torch_profiler_dump_cuda_time_total\":true,\"torch_profiler_record_shapes\":false,\"torch_profiler_with_memory\":false,\"ignore_frontend\":true,\"delay_iterations\":0,\"max_iterations\":0,\"warmup_iterations\":0,\"active_iterations\":1,\"wait_iterations\":0}" \
  --host 0.0.0.0 \
  --port 8000 \
  >"$server_log" 2>&1
