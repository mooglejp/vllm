# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare K/V cache formats under one explicit Math-SDPA contract.

This is a benchmark-only control for the post-P4 Radiance-delta plan.  It
does not change a vLLM cache dtype, attention backend, or model dispatch.  The
three paths use the same BF16 Q/K/V tensors and causal continuation mask:

* ``bf16_math`` uses the logical BF16 K/V directly;
* ``fp8_prefix_math`` stores the prefix as FP8 E4M3, decodes it to BF16, and
  then uses Math SDPA;
* ``k8v4_prefix_math`` stores the prefix with the existing TurboQuant K8/V4
  writer, decodes it with the existing reader, and then uses Math SDPA.

The full-path timings include prefix decode and dense K/V assembly.  Separate
attention-only timings use already materialized BF16 K/V, so a large common
attention cost is not mistaken for a cache-format cost.  This control is not
an implementation or adoption benchmark for an FP8-KV production path.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)
from vllm.triton_utils import triton
from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_decode import (
    _tq_full_dequant_kv,
    _use_fp8_e4b15,
)
from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_store import (
    triton_turboquant_store,
)

H_Q = 24
H_K = 4
HEAD_DIM = 256
BLOCK_SIZE = 16
FP8_DTYPE = torch.float8_e4m3fn


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
        "rmse": float(delta.square().mean().sqrt().item()),
        "relative_l2": float(delta.norm().item() / max(ref_norm, 1e-12)),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def synchronize() -> None:
    torch.accelerator.synchronize()


def allocate_block_table(
    total_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = math.ceil(total_len / BLOCK_SIZE)
    physical_blocks = torch.arange(num_blocks, dtype=torch.int32, device=device)
    if num_blocks > 1:
        physical_blocks = torch.roll(physical_blocks, shifts=1)
    return physical_blocks.view(1, -1), physical_blocks


def cache_slot_indices(
    length: int, physical_blocks: torch.Tensor, device: torch.device
) -> torch.Tensor:
    logical_blocks = torch.arange(
        math.ceil(length / BLOCK_SIZE), dtype=torch.int64, device=device
    )
    physical = physical_blocks[logical_blocks]
    return physical.repeat_interleave(BLOCK_SIZE)[:length] * BLOCK_SIZE + (
        torch.arange(length, dtype=torch.int64, device=device) % BLOCK_SIZE
    )


def store_k8v4_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    config: TurboQuantConfig,
    pit: torch.Tensor,
    midpoints: torch.Tensor,
    centroids: torch.Tensor,
) -> None:
    total_len = key.shape[0]
    num_blocks = math.ceil(total_len / BLOCK_SIZE)
    logical_blocks = torch.arange(num_blocks, dtype=torch.int32, device=key.device)
    physical = block_table[0, logical_blocks]
    slots = physical.repeat_interleave(BLOCK_SIZE)[:total_len] * BLOCK_SIZE
    slots += torch.arange(total_len, dtype=torch.int32, device=key.device) % BLOCK_SIZE
    triton_turboquant_store(
        key=key,
        value=value,
        kv_cache=cache,
        slot_mapping=slots,
        PiT=pit,
        centroids=centroids,
        midpoints=midpoints,
        mse_bits=config.key_mse_bits,
        key_packed_size=config.key_packed_size,
        value_quant_bits=config.effective_value_quant_bits,
        key_fp8=config.key_fp8,
        norm_correction=config.norm_correction,
    )


def allocate_k8v4_cache(
    total_len: int,
    config: TurboQuantConfig,
    device: torch.device,
) -> torch.Tensor:
    num_blocks = math.ceil(total_len / BLOCK_SIZE)
    return torch.empty(
        num_blocks,
        BLOCK_SIZE,
        H_K,
        config.slot_size_aligned,
        dtype=torch.uint8,
        device=device,
    )


def dequantize_k8v4_prefix(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    length: int,
    config: TurboQuantConfig,
    centroids: torch.Tensor,
    output_k: torch.Tensor,
    output_v: torch.Tensor,
) -> None:
    alloc_len = math.ceil(length / BLOCK_SIZE) * BLOCK_SIZE
    key_data_bytes = (
        HEAD_DIM if config.key_fp8 else math.ceil(HEAD_DIM * config.key_mse_bits / 8)
    )
    val_data_bytes = math.ceil(HEAD_DIM * config.effective_value_quant_bits / 8)
    data_bytes_per_slot = key_data_bytes + val_data_bytes
    meta_region_offset = BLOCK_SIZE * H_K * data_bytes_per_slot
    num_soa_fields = 2 if config.key_fp8 else 3
    soa_v_scale = 0 if config.key_fp8 else 1
    soa_v_zero = 1 if config.key_fp8 else 2
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
        BLOCK_D=triton.next_power_of_2(HEAD_DIM),
        NORM_CORRECTION=1 if config.norm_correction else 0,
        FP8_E4B15=_use_fp8_e4b15(cache.device.index or 0),
        num_warps=4,
    )


def build_fp8_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    physical_blocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = physical_blocks.numel()
    key_cache = torch.empty(
        num_blocks,
        BLOCK_SIZE,
        H_K,
        HEAD_DIM,
        dtype=FP8_DTYPE,
        device=key.device,
    )
    value_cache = torch.empty_like(key_cache)
    slots = cache_slot_indices(key.shape[0], physical_blocks, key.device)
    # Float8 does not implement ``index_copy_cuda`` on this ROCm build, while
    # indexed assignment is supported and preserves the paged mapping.
    key_cache.view(-1, H_K, HEAD_DIM)[slots] = key.to(FP8_DTYPE)
    value_cache.view(-1, H_K, HEAD_DIM)[slots] = value.to(FP8_DTYPE)
    return key_cache, value_cache


def run_math_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    q_t = query.transpose(0, 1).unsqueeze(0)
    k_t = key.transpose(0, 1).unsqueeze(0)
    v_t = value.transpose(0, 1).unsqueeze(0)
    with sdpa_kernel(SDPBackend.MATH):
        result = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=mask,
            scale=scale,
            enable_gqa=True,
        )[0].transpose(0, 1)
    output.copy_(result)
    return output


def time_samples(
    run: Callable[[], torch.Tensor],
    warmups: int,
    samples: int,
    flush: torch.Tensor,
) -> tuple[torch.Tensor, list[float]]:
    for _ in range(warmups):
        run()
    synchronize()
    times_us: list[float] = []
    output = run()
    synchronize()
    for index in range(samples):
        flush.zero_()
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        output = run()
        end.record()
        synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)
    return output, times_us


def oracle_rows(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    case: Case,
    scale: float,
) -> dict[tuple[int, int], torch.Tensor]:
    rows = sorted({0, case.q_len // 2, case.q_len - 1})
    group = H_Q // H_K
    result: dict[tuple[int, int], torch.Tensor] = {}
    for row in rows:
        visible = case.cached_len + row + 1
        for head in (0, H_Q - 1):
            kv_head = head // group
            q_row = query[row, head].double()
            k_rows = key[:visible, kv_head].double()
            v_rows = value[:visible, kv_head].double()
            scores = torch.matmul(k_rows, q_row) * scale
            weights = torch.softmax(scores, dim=0)
            result[(row, head)] = torch.matmul(weights, v_rows)
    synchronize()
    return result


def row_metrics(
    output: torch.Tensor,
    oracle: dict[tuple[int, int], torch.Tensor],
) -> dict[str, float]:
    actual = torch.stack([output[row, head].double() for row, head in oracle], dim=0)
    reference = torch.stack(list(oracle.values()), dim=0)
    return error_metrics(actual, reference)


def time_record(times_us: list[float]) -> dict[str, object]:
    return {
        "samples_us": times_us,
        "median_us": statistics.median(times_us),
        "p95_us": percentile(times_us, 0.95),
        "min_us": min(times_us),
        "finite": True,
    }


def benchmark_case(
    case: Case,
    device: torch.device,
    warmups: int,
    samples: int,
    flush: torch.Tensor,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(seed)
    dtype = torch.bfloat16
    scale = HEAD_DIM**-0.5
    key = torch.randn(
        case.seq_len,
        H_K,
        HEAD_DIM,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    value = torch.randn_like(key, generator=generator)
    query = torch.randn(
        case.q_len,
        H_Q,
        HEAD_DIM,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    key_chunk = key[case.cached_len :]
    value_chunk = value[case.cached_len :]
    q_pos = torch.arange(case.q_len, device=device).unsqueeze(1) + case.cached_len
    k_pos = torch.arange(case.seq_len, device=device).unsqueeze(0)
    mask = k_pos <= q_pos
    output = torch.empty_like(query)
    output_aux = torch.empty_like(query)
    fp8_full_k = torch.empty_like(key)
    fp8_full_v = torch.empty_like(value)
    k8v4_full_k = torch.empty_like(key)
    k8v4_full_v = torch.empty_like(value)
    decode_k = torch.empty(
        1,
        H_K,
        math.ceil(case.cached_len / BLOCK_SIZE) * BLOCK_SIZE,
        HEAD_DIM,
        dtype=torch.float16,
        device=device,
    )
    decode_v = torch.empty_like(decode_k)
    oracle = oracle_rows(query, key, value, case, scale)

    block_table, physical_blocks = allocate_block_table(case.seq_len, device)
    slots = cache_slot_indices(case.cached_len, physical_blocks, device)
    fp8_key_cache, fp8_value_cache = build_fp8_cache(key, value, physical_blocks)
    fp8_key_flat = fp8_key_cache.view(-1, H_K, HEAD_DIM)
    fp8_value_flat = fp8_value_cache.view(-1, H_K, HEAD_DIM)
    config = TurboQuantConfig.from_cache_dtype("turboquant_k8v4", HEAD_DIM)
    tq_cache = allocate_k8v4_cache(case.seq_len, config, device)
    pit = torch.eye(HEAD_DIM, dtype=torch.float32, device=device)
    midpoints = torch.zeros(config.n_centroids - 1, dtype=torch.float32, device=device)
    centroids = torch.ones(config.n_centroids, dtype=torch.float32, device=device)
    store_k8v4_cache(
        key,
        value,
        tq_cache,
        block_table,
        config,
        pit,
        midpoints,
        centroids,
    )
    synchronize()

    def run_bf16_attention() -> torch.Tensor:
        return run_math_sdpa(query, key, value, mask, output, scale)

    def run_fp8_attention() -> torch.Tensor:
        return run_math_sdpa(query, fp8_full_k, fp8_full_v, mask, output, scale)

    def run_k8v4_attention() -> torch.Tensor:
        return run_math_sdpa(query, k8v4_full_k, k8v4_full_v, mask, output, scale)

    def decode_fp8() -> None:
        fp8_full_k[: case.cached_len].copy_(fp8_key_flat[slots].to(dtype))
        fp8_full_v[: case.cached_len].copy_(fp8_value_flat[slots].to(dtype))
        fp8_full_k[case.cached_len :].copy_(key_chunk)
        fp8_full_v[case.cached_len :].copy_(value_chunk)

    def run_fp8_full() -> torch.Tensor:
        decode_fp8()
        return run_fp8_attention()

    def decode_k8v4() -> None:
        dequantize_k8v4_prefix(
            tq_cache,
            block_table,
            case.cached_len,
            config,
            centroids,
            decode_k,
            decode_v,
        )
        k8v4_full_k[: case.cached_len].copy_(
            decode_k[0, :, : case.cached_len].transpose(0, 1).to(dtype)
        )
        k8v4_full_v[: case.cached_len].copy_(
            decode_v[0, :, : case.cached_len].transpose(0, 1).to(dtype)
        )
        k8v4_full_k[case.cached_len :].copy_(key_chunk)
        k8v4_full_v[case.cached_len :].copy_(value_chunk)

    def run_k8v4_full() -> torch.Tensor:
        decode_k8v4()
        return run_k8v4_attention()

    def run_k8v4_decode() -> torch.Tensor:
        decode_k8v4()
        return output_aux

    def run_fp8_decode() -> torch.Tensor:
        decode_fp8()
        return output_aux

    # Materialize the attention-only inputs once before rotating the timing
    # order.  Full-path methods use separate buffers and cannot invalidate
    # these inputs between samples.
    decode_fp8()
    decode_k8v4()
    synchronize()

    methods: dict[str, Callable[[], torch.Tensor]] = {
        "bf16_math_attention_only": run_bf16_attention,
        "fp8_prefix_math_attention_only": run_fp8_attention,
        "k8v4_prefix_math_attention_only": run_k8v4_attention,
        "fp8_prefix_math_full": run_fp8_full,
        "k8v4_prefix_math_full": run_k8v4_full,
        "fp8_prefix_decode_only": run_fp8_decode,
        "k8v4_prefix_decode_only": run_k8v4_decode,
    }
    results: dict[str, object] = {}
    names = list(methods)
    for index in range(warmups + samples):
        order = names if (index + seed) % 2 == 0 else list(reversed(names))
        if index < warmups:
            for name in order:
                methods[name]()
            continue
        sample_index = index - warmups
        for name in order:
            flush.zero_()
            start = torch.Event(enable_timing=True)
            end = torch.Event(enable_timing=True)
            start.record()
            output_value = methods[name]()
            end.record()
            synchronize()
            elapsed = start.elapsed_time(end) * 1000.0
            row = results.setdefault(name, {"samples_us": []})
            assert isinstance(row, dict)
            row["samples_us"].append(elapsed)
            # Preserve each method's final output for post-timing correctness;
            # the timed methods intentionally reuse output buffers.
            row["last_output"] = output_value.detach().clone()
            row["sample_index"] = sample_index

    output_results: dict[str, object] = {}
    for name, raw in results.items():
        assert isinstance(raw, dict)
        times = raw["samples_us"]
        assert isinstance(times, list)
        record = time_record(times)
        last_output = raw["last_output"]
        assert isinstance(last_output, torch.Tensor)
        record["finite_full_output"] = bool(torch.isfinite(last_output).all().item())
        if "decode_only" not in name:
            record["oracle_rows"] = row_metrics(last_output, oracle)
        output_results[name] = record

    baseline_us = output_results["bf16_math_attention_only"]["median_us"]
    for record in output_results.values():
        assert isinstance(record, dict)
        record["relative_to_bf16_math"] = float(baseline_us) / float(
            record["median_us"]
        )
    return {
        "case": {"cached_len": case.cached_len, "q_len": case.q_len},
        "shape": {"Hq": H_Q, "Hk": H_K, "D": HEAD_DIM},
        "dtype": str(dtype),
        "cache_block_size": BLOCK_SIZE,
        "attention_backend": "torch SDPBackend.MATH",
        "current_chunk_contract": "raw BF16 K/V; only prefix cache format varies",
        "seed": seed,
        "results": output_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cached-lens", type=int, nargs="+", default=[4096, 32768])
    parser.add_argument("--q-lens", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--flush-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires a ROCm GPU")
    device = torch.device("cuda")
    flush = torch.empty(args.flush_mib * 2**20, dtype=torch.uint8, device=device)
    cases = [
        Case(cached_len, q_len)
        for cached_len in args.cached_lens
        for q_len in args.q_lens
    ]
    report: dict[str, object] = {
        "schema_version": 1,
        "hardware": str(torch.cuda.get_device_properties(device)),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "contract": {
            "Hq": H_Q,
            "Hk": H_K,
            "D": HEAD_DIM,
            "dtype": "torch.bfloat16",
            "causal": True,
            "current_chunk": "raw BF16 K/V",
            "backend": "SDPBackend.MATH",
            "fp8_dtype": str(FP8_DTYPE),
            "cache_format": "K8/V4 prefix vs FP8 E4M3 prefix vs BF16 logical prefix",
        },
        "warmups": args.warmups,
        "samples": args.samples,
        "flush_mib": args.flush_mib,
        "cases": [],
    }
    for index, case in enumerate(cases):
        print(f"[{index + 1}/{len(cases)}] cached={case.cached_len} q={case.q_len}")
        try:
            result = benchmark_case(
                case,
                device,
                args.warmups,
                args.samples,
                flush,
                args.seed + index,
            )
        except torch.OutOfMemoryError as exc:
            result = {
                "case": {"cached_len": case.cached_len, "q_len": case.q_len},
                "status": "oom",
                "error": str(exc),
            }
            torch.accelerator.empty_cache()
        except RuntimeError as exc:
            result = {
                "case": {"cached_len": case.cached_len, "q_len": case.q_len},
                "status": "error",
                "error": str(exc),
            }
            torch.accelerator.empty_cache()
        else:
            result["status"] = "complete"
        report["cases"].append(result)
        print(json.dumps(result, indent=2, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
