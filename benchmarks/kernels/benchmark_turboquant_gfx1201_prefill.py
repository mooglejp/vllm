# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure P2.2 K8/V4 continuation-reader and streaming candidates.

This diagnostic deliberately does not change the production continuation
threshold or dispatch.  It keeps two references separate:

* ``quantized_reference`` dequantizes every visible position from the SoA
  cache into the same FP16 workspace used by the current large-continuation
  path, then runs SDPA.
* ``raw_current_reference`` dequantizes only the prefix and appends the raw
  current chunk, matching the current large-continuation numerical contract.

The generic reader is the existing SoA decode adapter with one request per
query row (the shape produced by the current <=128-token path).  The
specialized reader is the gfx1201 packed multi-token launcher.  The streaming
candidate reuses its stage-1 tile shape but reads quantized prefix K/V and raw
current-chunk K/V in one online-softmax pass.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)
from vllm.triton_utils import triton
from vllm.v1.attention.ops.turboquant_soa import (
    gfx1201_prefill as streaming_prefill,
)
from vllm.v1.attention.ops.turboquant_soa import (
    triton_turboquant_decode_gfx1201_k8v4 as specialized,
)
from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_decode import (
    _tq_full_dequant_kv,
    _use_fp8_e4b15,
)
from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_store import (
    triton_turboquant_store,
)
from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_unified_attention import (
    triton_turboquant_decode_attention_soa,
)

H_Q = 24
H_K = 4
HEAD_DIM = 256
BLOCK_SIZE = 16


@dataclass(frozen=True)
class Case:
    cached_len: int
    q_len: int

    @property
    def seq_len(self) -> int:
        return self.cached_len + self.q_len


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = actual.float() - reference.float()
    ref_norm = reference.float().norm().item()
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "relative_l2": float(delta.norm().item() / max(ref_norm, 1e-12)),
        "equal_fraction": float((actual == reference).float().mean().item()),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def allocate_cache(
    total_len: int,
    config: TurboQuantConfig,
    device: torch.device,
    permute_blocks: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = math.ceil(total_len / BLOCK_SIZE)
    cache = torch.empty(
        num_blocks,
        BLOCK_SIZE,
        H_K,
        config.slot_size_aligned,
        dtype=torch.uint8,
        device=device,
    )
    physical_blocks = torch.arange(num_blocks, dtype=torch.int32, device=device)
    if permute_blocks and num_blocks > 1:
        physical_blocks = torch.roll(physical_blocks, shifts=1)
    block_table = physical_blocks.view(1, -1)
    return cache, block_table


def store_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    cache: torch.Tensor,
    config: TurboQuantConfig,
    pit: torch.Tensor,
    midpoints: torch.Tensor,
    centroids: torch.Tensor,
    block_table: torch.Tensor | None = None,
) -> None:
    total_len = key.shape[0]
    if block_table is None:
        slots = torch.arange(total_len, dtype=torch.int32, device=key.device)
    else:
        num_blocks = math.ceil(total_len / BLOCK_SIZE)
        logical_blocks = torch.arange(num_blocks, dtype=torch.int32, device=key.device)
        physical = block_table[0, logical_blocks]
        slots = physical.repeat_interleave(BLOCK_SIZE)[:total_len] * BLOCK_SIZE
        slots += (
            torch.arange(total_len, dtype=torch.int32, device=key.device) % BLOCK_SIZE
        )
    triton_turboquant_store(
        key=key,
        value=value,
        kv_cache=cache,
        slot_mapping=slots,
        PiT=pit,
        midpoints=midpoints,
        mse_bits=config.key_mse_bits,
        key_packed_size=config.key_packed_size,
        value_quant_bits=config.effective_value_quant_bits,
        key_fp8=config.key_fp8,
        centroids=centroids,
        norm_correction=config.norm_correction,
    )


def dequantize_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    length: int,
    config: TurboQuantConfig,
    centroids: torch.Tensor,
    output_k: torch.Tensor | None = None,
    output_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the same SoA full-dequant kernel as large continuation."""
    alloc_len = math.ceil(length / BLOCK_SIZE) * BLOCK_SIZE
    if output_k is None:
        output_k = torch.empty(
            1, H_K, alloc_len, HEAD_DIM, dtype=torch.float16, device=cache.device
        )
    if output_v is None:
        output_v = torch.empty_like(output_k)

    key_data_bytes = (
        HEAD_DIM if config.key_fp8 else math.ceil(HEAD_DIM * config.key_mse_bits / 8)
    )
    val_data_bytes = math.ceil(HEAD_DIM * config.effective_value_quant_bits / 8)
    data_bytes_per_slot = key_data_bytes + val_data_bytes
    meta_region_offset = BLOCK_SIZE * H_K * data_bytes_per_slot
    num_soa_fields = 2 if config.key_fp8 else 3
    soa_v_scale = 0 if config.key_fp8 else 1
    soa_v_zero = 1 if config.key_fp8 else 2
    block_d = triton.next_power_of_2(HEAD_DIM)
    _tq_full_dequant_kv[(alloc_len, H_K)](
        cache,
        cache.view(torch.uint16),
        block_table,
        centroids,
        output_k,
        output_v,
        output_k.stride(0),
        output_k.stride(1),
        output_k.stride(2),
        output_v.stride(0),
        output_v.stride(1),
        output_v.stride(2),
        cache.stride(0),
        block_table.stride(0),
        HEAD_DIM=HEAD_DIM,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_KV_HEADS=H_K,
        MSE_BYTES=math.ceil(HEAD_DIM * config.key_mse_bits / 8),
        VQB=config.effective_value_quant_bits,
        VAL_DATA_BYTES=val_data_bytes,
        MSE_BITS=config.key_mse_bits,
        KEY_FP8=1 if config.key_fp8 else 0,
        KEY_DATA_BYTES=key_data_bytes,
        META_REGION_OFFSET=meta_region_offset,
        NUM_SOA_FIELDS=num_soa_fields,
        SOA_K_NORM=0,
        SOA_V_SCALE=soa_v_scale,
        SOA_V_ZERO=soa_v_zero,
        BLOCK_D=block_d,
        NORM_CORRECTION=1 if config.norm_correction else 0,
        FP8_E4B15=_use_fp8_e4b15(cache.device.index or 0),
        num_warps=4,
    )
    return output_k, output_v


def sdpa_with_causal_continuation(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cached_len: int,
    scale: float,
    backend: str = "math",
) -> torch.Tensor:
    q_len = query.shape[0]
    seq_len = key.shape[0]
    q_t = query.transpose(0, 1).unsqueeze(0)
    k_t = key.transpose(0, 1).unsqueeze(0)
    v_t = value.transpose(0, 1).unsqueeze(0)
    q_pos = torch.arange(q_len, device=query.device).unsqueeze(1) + cached_len
    k_pos = torch.arange(seq_len, device=query.device).unsqueeze(0)
    mask = k_pos <= q_pos

    def call_sdpa() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=mask,
            scale=scale,
            enable_gqa=H_K < H_Q,
        )[0].transpose(0, 1)

    if backend == "math":
        with sdpa_kernel(SDPBackend.MATH):
            return call_sdpa()
    if backend == "auto":
        return call_sdpa()
    raise ValueError(f"unsupported SDPA backend mode: {backend}")


def build_references(
    query: torch.Tensor,
    key_chunk: torch.Tensor,
    value_chunk: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    case: Case,
    config: TurboQuantConfig,
    centroids: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_k, prefix_v = dequantize_cache(
        cache, block_table, case.cached_len, config, centroids
    )
    all_k, all_v = dequantize_cache(cache, block_table, case.seq_len, config, centroids)

    # The all-quantized reference deliberately uses the same FP16 dequant
    # workspace and query-dtype conversion as the current path.
    quant_k = all_k[0, :, : case.seq_len, :].transpose(0, 1).to(query.dtype)
    quant_v = all_v[0, :, : case.seq_len, :].transpose(0, 1).to(query.dtype)
    quantized_reference = sdpa_with_causal_continuation(
        query, quant_k, quant_v, case.cached_len, scale
    )

    # This is the current large-continuation contract: only the prefix comes
    # from TQ, while the current chunk remains raw in query dtype.
    raw_k = torch.empty(
        case.seq_len, H_K, HEAD_DIM, dtype=query.dtype, device=query.device
    )
    raw_v = torch.empty_like(raw_k)
    raw_k[: case.cached_len] = (
        prefix_k[0, :, : case.cached_len, :].transpose(0, 1).to(query.dtype)
    )
    raw_v[: case.cached_len] = (
        prefix_v[0, :, : case.cached_len, :].transpose(0, 1).to(query.dtype)
    )
    raw_k[case.cached_len :] = key_chunk
    raw_v[case.cached_len :] = value_chunk
    raw_current_reference = sdpa_with_causal_continuation(
        query, raw_k, raw_v, case.cached_len, scale
    )
    del prefix_k, prefix_v, all_k, all_v, quant_k, quant_v, raw_k, raw_v
    return quantized_reference, raw_current_reference


def workspace_bytes(
    case: Case,
    query_dtype: torch.dtype,
    config: TurboQuantConfig,
    splits: int,
) -> dict[str, int | float]:
    dtype_bytes = torch.empty((), dtype=query_dtype).element_size()
    fp16_bytes = torch.empty((), dtype=torch.float16).element_size()
    cache_bytes = (
        math.ceil(case.seq_len / BLOCK_SIZE)
        * BLOCK_SIZE
        * H_K
        * config.slot_size_aligned
    )
    old_prefix = 2 * H_K * case.cached_len * HEAD_DIM * fp16_bytes
    old_full = 2 * H_K * case.seq_len * HEAD_DIM * dtype_bytes
    old_mask = case.q_len * case.seq_len
    generic_partial = (
        case.q_len * H_Q * splits * HEAD_DIM * 4 + 2 * case.q_len * H_Q * splits * 4
    )
    specialized_mid = case.q_len * H_Q * splits * (HEAD_DIM + 1) * 4
    specialized_lse = case.q_len * H_Q * 4
    output = case.q_len * H_Q * HEAD_DIM * dtype_bytes
    return {
        "shared_cache_bytes": cache_bytes,
        "old_dequant_fp16_bytes": old_prefix,
        "old_full_kv_bytes": old_full,
        "old_causal_mask_bytes": old_mask,
        "old_additional_bytes": old_prefix + old_full + old_mask,
        "generic_split_scratch_bytes": generic_partial,
        "generic_additional_bytes": generic_partial + output,
        "specialized_mid_o_bytes": specialized_mid,
        "specialized_lse_bytes": specialized_lse,
        "specialized_additional_bytes": specialized_mid + specialized_lse + output,
        "specialized_query_block_size": 2 if case.q_len <= 2 else 4,
        "specialized_query_blocks": math.ceil(
            case.q_len / (2 if case.q_len <= 2 else 4)
        ),
        "mid_o_mib": specialized_mid / 2**20,
        "streaming_additional_bytes": output,
    }


def run_generic(
    query: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    case: Case,
    config: TurboQuantConfig,
    centroids: torch.Tensor,
    pit: torch.Tensor,
    output: torch.Tensor,
    splits: int,
    seq_lens: torch.Tensor | None = None,
    row_block_table: torch.Tensor | None = None,
) -> torch.Tensor:
    # This is the existing <=128 direct-reader shape: one synthetic request
    # per query row, with the row's causal visible length.
    if seq_lens is None:
        seq_lens = torch.arange(
            case.cached_len + 1,
            case.seq_len + 1,
            dtype=torch.int32,
            device=query.device,
        )
    if row_block_table is None:
        row_block_table = block_table.expand(case.q_len, -1)
    return triton_turboquant_decode_attention_soa(
        query=query,
        kv_cache=cache,
        block_table=row_block_table,
        seq_lens=seq_lens,
        Pi=pit,
        centroids=centroids,
        scale=HEAD_DIM**-0.5,
        mse_bits=config.key_mse_bits,
        key_packed_size=config.key_packed_size,
        value_quant_bits=config.effective_value_quant_bits,
        value_packed_size=config.value_packed_size,
        key_fp8=config.key_fp8,
        norm_correction=config.norm_correction,
        max_seq_len=case.seq_len,
        output_buf=output,
        max_num_kv_splits=splits,
    )


def run_specialized(
    query: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    case: Case,
    output: torch.Tensor,
    mid_o: torch.Tensor,
    lse: torch.Tensor,
    splits: int,
    qsl: torch.Tensor,
    qsl_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    return specialized.triton_turboquant_decode_gfx1201_k8v4_multi_token(
        query=query,
        kv_cache=cache,
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=qsl,
        query_start_loc_cpu=qsl_cpu,
        scale=HEAD_DIM**-0.5,
        output=output,
        mid_o_buf=mid_o,
        lse_buf=lse,
        max_num_kv_splits=splits,
        max_query_len=case.q_len,
    )


def run_raw_current(
    query: torch.Tensor,
    key_chunk: torch.Tensor,
    value_chunk: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    case: Case,
    config: TurboQuantConfig,
    centroids: torch.Tensor,
    dequant_k: torch.Tensor,
    dequant_v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    dequantize_cache(
        cache,
        block_table,
        case.cached_len,
        config,
        centroids,
        output_k=dequant_k,
        output_v=dequant_v,
    )
    full_k = torch.empty(
        case.seq_len, H_K, HEAD_DIM, dtype=query.dtype, device=query.device
    )
    full_v = torch.empty_like(full_k)
    full_k[: case.cached_len] = (
        dequant_k[0, :, : case.cached_len, :].transpose(0, 1).to(query.dtype)
    )
    full_v[: case.cached_len] = (
        dequant_v[0, :, : case.cached_len, :].transpose(0, 1).to(query.dtype)
    )
    full_k[case.cached_len :] = key_chunk
    full_v[case.cached_len :] = value_chunk
    return sdpa_with_causal_continuation(
        query, full_k, full_v, case.cached_len, scale, backend="auto"
    )


def run_streaming_raw_current(
    query: torch.Tensor,
    key_chunk: torch.Tensor,
    value_chunk: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    case: Case,
    scale: float,
    output: torch.Tensor,
) -> torch.Tensor:
    return streaming_prefill.launch_gfx1201_tq_continuation_prefill(
        query=query,
        key_chunk=key_chunk,
        value_chunk=value_chunk,
        kv_cache=cache,
        block_table=block_table,
        cached_len=case.cached_len,
        seq_len=case.seq_len,
        scale=scale,
        output=output,
    )


def timed_samples(
    run,
    warmups: int,
    samples: int,
    flush: torch.Tensor,
) -> tuple[torch.Tensor, list[float]]:
    for _ in range(warmups):
        output = run()
    torch.accelerator.synchronize()
    starts = [torch.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.Event(enable_timing=True) for _ in range(samples)]
    times_us: list[float] = []
    output = run()
    torch.accelerator.synchronize()
    for start, end in zip(starts, ends):
        flush.zero_()
        start.record()
        output = run()
        end.record()
        torch.accelerator.synchronize()
        times_us.append(start.elapsed_time(end) * 1000)
    return output, times_us


def memory_baseline(device: torch.device) -> int | None:
    """Reset allocator peak tracking and return current allocated bytes."""
    try:
        torch.accelerator.reset_peak_memory_stats(device)
        return int(torch.accelerator.memory_allocated(device))
    except (AttributeError, RuntimeError):
        return None


def measured_peak_bytes(device: torch.device, baseline: int | None) -> int | None:
    """Return allocator peak and the increase over the candidate baseline."""
    if baseline is None:
        return None
    try:
        return int(torch.accelerator.max_memory_allocated(device)) - baseline
    except (AttributeError, RuntimeError):
        return None


def benchmark_case(
    case: Case,
    config: TurboQuantConfig,
    device: torch.device,
    dtype: torch.dtype,
    splits: list[int],
    warmups: int,
    samples: int,
    flush: torch.Tensor,
    seed: int,
    do_correctness: bool,
    permute_blocks: bool = False,
) -> list[dict]:
    generator = torch.Generator(device=device).manual_seed(seed)
    total_len = case.seq_len
    key = torch.randn(
        total_len, H_K, HEAD_DIM, dtype=dtype, device=device, generator=generator
    )
    value = torch.randn_like(key, generator=generator)
    query = key[case.cached_len :].new_empty(case.q_len, H_Q, HEAD_DIM)
    query.normal_(generator=generator)
    key_chunk = key[case.cached_len :]
    value_chunk = value[case.cached_len :]

    cache, block_table = allocate_cache(total_len, config, device, permute_blocks)
    pit = torch.eye(HEAD_DIM, dtype=torch.float32, device=device)
    midpoints = torch.zeros(config.n_centroids - 1, dtype=torch.float32, device=device)
    centroids = torch.ones(config.n_centroids, dtype=torch.float32, device=device)
    store_cache(key, value, cache, config, pit, midpoints, centroids, block_table)
    torch.accelerator.synchronize()

    scale = HEAD_DIM**-0.5
    quantized_reference: torch.Tensor | None = None
    raw_reference: torch.Tensor | None = None
    reference_status = "skipped"
    reference_error: str | None = None
    if do_correctness:
        try:
            quantized_reference, raw_reference = build_references(
                query,
                key_chunk,
                value_chunk,
                cache,
                block_table,
                case,
                config,
                centroids,
                scale,
            )
            reference_status = "complete"
        except torch.OutOfMemoryError as exc:
            reference_status = "reference_oom"
            reference_error = str(exc)
        except RuntimeError as exc:
            reference_status = "reference_error"
            reference_error = str(exc)
        finally:
            torch.accelerator.synchronize()
            if reference_status != "complete":
                torch.accelerator.empty_cache()

    # Keep metadata fixed for candidate-internal measurements. The generic
    # whole-path record below deliberately omits these arguments so its setup
    # cost remains visible.
    seq_lens = torch.tensor([case.seq_len], dtype=torch.int32, device=device)
    qsl = torch.tensor([0, case.q_len], dtype=torch.int32, device=device)
    qsl_cpu = torch.tensor([0, case.q_len], dtype=torch.int32)
    row_seq_lens = torch.arange(
        case.cached_len + 1,
        case.seq_len + 1,
        dtype=torch.int32,
        device=device,
    )
    row_block_table = block_table.expand(case.q_len, -1)
    workspaces = {
        split: workspace_bytes(case, dtype, config, split) for split in splits
    }
    if 1 not in workspaces:
        workspaces[1] = workspace_bytes(case, dtype, config, 1)
    results: list[dict] = []
    raw_methods = {"current_large_continuation", "streaming_raw_current_prefill"}

    def append_result(
        method: str,
        split: int | None,
        status: str,
        times_us: list[float] | None = None,
        output: torch.Tensor | None = None,
        error: str | None = None,
        measurement_scope: str = "whole_path",
        measured_peak: int | None = None,
        calculated_workspace: int | None = None,
    ) -> None:
        result: dict = {
            "record_type": "p2.2",
            "case": {"cached_len": case.cached_len, "q_len": case.q_len},
            "shape": {"Hq": H_Q, "Hk": H_K, "D": HEAD_DIM},
            "block_size": BLOCK_SIZE,
            "dtype": str(dtype),
            "method": method,
            "split": split,
            "status": status,
            "measurement_scope": measurement_scope,
            "current_chunk_source": ("raw" if method in raw_methods else "quantized"),
            "dequant_workspace_dtype": "torch.float16",
            "query_attention_dtype": str(dtype),
            "attention_backend": (
                "runtime_auto_sdpa"
                if method == "current_large_continuation"
                else "fused_online_softmax"
                if method == "streaming_raw_current_prefill"
                else "quantized_reader"
            ),
            "reference_status": reference_status,
            "reference_backend": "SDPBackend.MATH",
            "reference_contract": {
                "quantized": (
                    "all visible K/V dequantized to FP16, then cast to query dtype"
                ),
                "raw_current": (
                    "prefix dequantized to FP16, current chunk remains raw query dtype"
                ),
                "attention": (
                    "SDPA math fallback with causal bound "
                    "kv_pos <= cached_len + query_row"
                ),
            },
        }
        if reference_error is not None:
            result["reference_error"] = reference_error
        if split is not None:
            result["workspace"] = workspaces[split]
        if calculated_workspace is not None:
            result["calculated_workspace_bytes"] = calculated_workspace
        if measured_peak is not None:
            result["measured_peak_delta_bytes"] = measured_peak
        if times_us:
            median = statistics.median(times_us)
            result.update(
                {
                    "samples_us": times_us,
                    "median_us": median,
                    "p95_us": percentile(times_us, 0.95),
                    "min_us": min(times_us),
                }
            )
        if output is not None and quantized_reference is not None:
            output_cpu = output.detach().float().cpu()
            result["vs_quantized_reference"] = error_metrics(
                output_cpu, quantized_reference.float().cpu()
            )
            result["vs_raw_current_reference"] = error_metrics(
                output_cpu, raw_reference.float().cpu()
            )
        if error is not None:
            result["error"] = error
        results.append(result)

    prefix_alloc = math.ceil(case.cached_len / BLOCK_SIZE) * BLOCK_SIZE
    try:
        baseline = memory_baseline(device)
        dequant_k = torch.empty(
            1, H_K, prefix_alloc, HEAD_DIM, dtype=torch.float16, device=device
        )
        dequant_v = torch.empty_like(dequant_k)
        output, times = timed_samples(
            lambda: run_raw_current(
                query,
                key_chunk,
                value_chunk,
                cache,
                block_table,
                case,
                config,
                centroids,
                dequant_k,
                dequant_v,
                scale,
            ),
            warmups,
            samples,
            flush,
        )
        append_result(
            "current_large_continuation",
            None,
            "complete",
            times,
            output,
            measurement_scope="whole_path",
            measured_peak=measured_peak_bytes(device, baseline),
            calculated_workspace=workspaces[splits[0]]["old_additional_bytes"]
            if splits
            else None,
        )
    except torch.OutOfMemoryError as exc:
        append_result("current_large_continuation", None, "oom", error=str(exc))
    except RuntimeError as exc:
        append_result("current_large_continuation", None, "error", error=str(exc))
    finally:
        if "dequant_k" in locals():
            del dequant_k, dequant_v

    if case.q_len > 128:
        baseline = memory_baseline(device)
        streaming_output = None
        try:
            streaming_output = torch.empty_like(query)
            output, times = timed_samples(
                lambda output=streaming_output: run_streaming_raw_current(
                    query,
                    key_chunk,
                    value_chunk,
                    cache,
                    block_table,
                    case,
                    scale,
                    output,
                ),
                warmups,
                samples,
                flush,
            )
            append_result(
                "streaming_raw_current_prefill",
                1,
                "complete",
                times,
                output,
                measurement_scope="fixed_metadata",
                measured_peak=measured_peak_bytes(device, baseline),
                calculated_workspace=workspace_bytes(case, dtype, config, 1)[
                    "streaming_additional_bytes"
                ],
            )
        except torch.OutOfMemoryError as exc:
            append_result("streaming_raw_current_prefill", 1, "oom", error=str(exc))
        except RuntimeError as exc:
            append_result("streaming_raw_current_prefill", 1, "error", error=str(exc))
        finally:
            del streaming_output
    else:
        append_result(
            "streaming_raw_current_prefill",
            1,
            "not_eligible",
            measurement_scope="fixed_metadata",
        )

    for split in splits:
        methods = [
            "generic_soa_direct_rowwise",
            "specialized_multi_token",
            "generic_soa_direct_rowwise_fixed",
        ]
        if (seed + split) % 2:
            methods.reverse()
        for method in methods:
            baseline = memory_baseline(device)
            output_buf = mid_o = lse = None
            try:
                if method.startswith("generic"):
                    output_buf = torch.empty_like(query)
                    fixed = method.endswith("_fixed")
                    output, times = timed_samples(
                        lambda fixed=fixed, output_buf=output_buf, split=split: (
                            run_generic(  # noqa: E501
                                query,
                                cache,
                                block_table,
                                case,
                                config,
                                centroids,
                                pit,
                                output_buf,
                                split,
                                seq_lens=row_seq_lens if fixed else None,
                                row_block_table=row_block_table if fixed else None,
                            )
                        ),
                        warmups,
                        samples,
                        flush,
                    )
                    append_result(
                        method,
                        split,
                        "complete",
                        times,
                        output,
                        measurement_scope=("fixed_metadata" if fixed else "whole_path"),
                        measured_peak=measured_peak_bytes(device, baseline),
                        calculated_workspace=workspaces[split][
                            "generic_additional_bytes"
                        ],
                    )
                else:
                    output_buf = torch.empty_like(query)
                    mid_o = torch.empty(
                        case.q_len,
                        H_Q,
                        split,
                        HEAD_DIM + 1,
                        dtype=torch.float32,
                        device=device,
                    )
                    lse = torch.empty(
                        case.q_len, H_Q, dtype=torch.float32, device=device
                    )

                    def run_specialized_candidate(
                        output_buf=output_buf,
                        mid_o=mid_o,
                        lse=lse,
                        split=split,
                    ):
                        return run_specialized(
                            query,
                            cache,
                            block_table,
                            case,
                            output_buf,
                            mid_o,
                            lse,
                            split,
                            qsl,
                            qsl_cpu,
                            seq_lens,
                        )

                    output, times = timed_samples(
                        run_specialized_candidate,
                        warmups,
                        samples,
                        flush,
                    )
                    append_result(
                        method,
                        split,
                        "complete",
                        times,
                        output,
                        measurement_scope="fixed_metadata",
                        measured_peak=measured_peak_bytes(device, baseline),
                        calculated_workspace=workspaces[split][
                            "specialized_additional_bytes"
                        ],
                    )
            except torch.OutOfMemoryError as exc:
                append_result(method, split, "oom", error=str(exc))
            except RuntimeError as exc:
                append_result(method, split, "error", error=str(exc))
            finally:
                del output_buf, mid_o, lse

    return results


def parse_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cached-lens", type=int, nargs="+", default=[4096, 8192, 16384, 32768, 65536]
    )
    parser.add_argument(
        "--q-lens", type=int, nargs="+", default=[64, 128, 256, 512, 1024]
    )
    parser.add_argument("--splits", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--flush-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--permute-blocks", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    config = TurboQuantConfig.from_cache_dtype("turboquant_k8v4", HEAD_DIM)
    flush = torch.empty(
        args.flush_mib * 1024 * 1024 // 4, dtype=torch.float32, device=device
    )
    cases = [
        Case(cached_len, q_len)
        for cached_len in args.cached_lens
        for q_len in args.q_lens
    ]
    random.Random(args.seed).shuffle(cases)

    with args.output.open("x") as output_file:
        for index, case in enumerate(cases):
            print(
                f"[{index + 1}/{len(cases)}] cached={case.cached_len} q={case.q_len}",
                flush=True,
            )
            for result in benchmark_case(
                case,
                config,
                device,
                dtype,
                args.splits,
                args.warmups,
                args.samples,
                flush,
                args.seed + index,
                not args.skip_correctness,
                args.permute_blocks,
            ):
                encoded = json.dumps(result, sort_keys=True)
                output_file.write(encoded + "\n")
                output_file.flush()
                print(encoded, flush=True)


if __name__ == "__main__":
    main()
