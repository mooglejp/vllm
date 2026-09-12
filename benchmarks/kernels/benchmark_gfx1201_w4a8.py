# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the opt-in gfx1201 MXFP4 W4A8 decode prototype.

The timed operations use preallocated outputs and precomputed activation
quantization.  The wrapper timings are intentionally not part of the adoption
gate because they include allocation and Python-side quantization overhead.
"""

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from vllm.model_executor.kernels.linear.mxfp4.gfx1201_w4a8 import (
    _gfx1201_w4a8_decode_kernel,
    gfx1201_w4a8_linear_reference,
    quantize_activation_fp8_reference,
)
from vllm.model_executor.kernels.linear.mxfp4.triton_gfx1201 import (
    _mxfp4_small_m_kernel,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    quant_dequant_mxfp4,
)
from vllm.triton_utils import triton

MODEL_SHAPES = (
    (5120, 6144),
    (34816, 5120),
    (5120, 17408),
    (16384, 5120),
    (96, 5120),
    (14336, 5120),
)


@dataclass(frozen=True)
class KernelConfig:
    block_m: int = 16
    block_n: int = 64
    block_k: int = 128
    num_warps: int = 4
    num_stages: int = 1


def _launch_w4a8(
    x: torch.Tensor,
    row_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    config: KernelConfig,
):
    m, k = x.shape
    n = weight.shape[0]
    return _gfx1201_w4a8_decode_kernel[
        (triton.cdiv(m, config.block_m), triton.cdiv(n, config.block_n))
    ](
        x,
        row_scale,
        weight,
        weight_scale,
        output,
        m,
        n,
        k,
        *x.stride(),
        *weight.stride(),
        *weight_scale.stride(),
        *output.stride(),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )


def _launch_mxfp4(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    config: KernelConfig,
):
    m, k = x.shape
    n = weight.shape[0]
    return _mxfp4_small_m_kernel[
        (triton.cdiv(m, config.block_m), triton.cdiv(n, config.block_n))
    ](
        x,
        weight,
        weight_scale,
        output,
        m,
        n,
        k,
        *x.stride(),
        *weight.stride(),
        *weight_scale.stride(),
        *output.stride(),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )


def _measure_round_robin(
    operations: dict[str, Callable[[], object]],
    flush: torch.Tensor,
    warmups: int,
    samples: int,
) -> dict[str, list[float]]:
    for _ in range(warmups):
        for operation in operations.values():
            operation()
    torch.accelerator.synchronize()
    timings = {name: [] for name in operations}
    names = list(operations)
    for sample in range(samples):
        offset = sample % len(names)
        for name in names[offset:] + names[:offset]:
            flush.zero_()
            start = torch.Event(enable_timing=True)
            end = torch.Event(enable_timing=True)
            start.record()
            operations[name]()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end) * 1000)
    return timings


def _summary(samples: list[float]) -> dict[str, object]:
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def _reference_check() -> None:
    torch.manual_seed(1201)
    for m in (1, 2, 3, 4):
        for n, k in ((5, 64), (67, 128), (65, 160)):
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            weight = torch.randint(
                0, 256, (n, k // 2), device="cuda", dtype=torch.uint8
            )
            scale = torch.randint(
                120, 132, (n, k // 32), device="cuda", dtype=torch.uint8
            )
            x_fp8, row_scale = quantize_activation_fp8_reference(x)
            candidate = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            config = KernelConfig()
            _launch_w4a8(x_fp8, row_scale, weight, scale, candidate, config)
            reference = gfx1201_w4a8_linear_reference(x, weight, scale)
            torch.accelerator.synchronize()
            if not torch.equal(candidate, reference):
                delta = (candidate.float() - reference.float()).abs()
                raise AssertionError(
                    f"W4A8 reference mismatch for M={m}, N={n}, K={k}: "
                    f"max_abs={delta.max().item()}"
                )
            if not torch.isfinite(candidate).all():
                raise AssertionError(f"non-finite W4A8 output for M={m}, N={n}, K={k}")


def _benchmark_shape(
    m: int,
    n: int,
    k: int,
    flush: torch.Tensor,
    warmups: int,
    samples: int,
    seed: int,
    config: KernelConfig,
) -> dict[str, object]:
    torch.manual_seed(seed)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    scale = torch.randint(120, 132, (n, k // 32), device="cuda", dtype=torch.uint8)
    x_fp8, row_scale = quantize_activation_fp8_reference(x)
    old_x = quant_dequant_mxfp4(x)
    new_output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    old_output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    def new_kernel():
        return _launch_w4a8(x_fp8, row_scale, weight, scale, new_output, config)

    def old_kernel():
        return _launch_mxfp4(old_x, weight, scale, old_output, config)

    new_kernel()
    old_kernel()
    torch.accelerator.synchronize()
    raw = _measure_round_robin(
        {"w4a8": new_kernel, "mxfp4": old_kernel},
        flush,
        warmups,
        samples,
    )
    w4a8_median = statistics.median(raw["w4a8"])
    mxfp4_median = statistics.median(raw["mxfp4"])
    timing = {name: _summary(values) for name, values in raw.items()}
    return {
        "m": m,
        "n": n,
        "k": k,
        "config": asdict(config),
        "direct_output_preallocated": True,
        "timing": timing,
        "speedup_mxfp4_over_w4a8": (mxfp4_median / w4a8_median),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--shape-index", type=int, nargs="+")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--flush-mib", type=int, default=64)
    args = parser.parse_args()

    if not torch.accelerator.is_available() or torch.version.hip is None:
        raise RuntimeError("This benchmark requires a ROCm GPU")
    config = KernelConfig()
    torch.manual_seed(1201)
    flush = torch.empty(args.flush_mib * 1024 * 1024, device="cuda", dtype=torch.uint8)
    _reference_check()
    selected = (
        range(len(MODEL_SHAPES)) if args.shape_index is None else args.shape_index
    )
    metadata = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": str(torch.accelerator.current_accelerator()),
        "config": asdict(config),
        "rows": args.rows,
        "warmups": args.warmups,
        "samples": args.samples,
        "flush_mib": args.flush_mib,
        "reference_check": "M=1..4, (N,K)=(5,64),(67,128),(65,160), bitwise exact",
    }
    with args.output.open("w") as output:
        output.write(json.dumps({"metadata": metadata}) + "\n")
        for shape_index in selected:
            n, k = MODEL_SHAPES[shape_index]
            for m in args.rows:
                result = _benchmark_shape(
                    m,
                    n,
                    k,
                    flush,
                    args.warmups,
                    args.samples,
                    1201 + 10 * shape_index + m,
                    config,
                )
                output.write(json.dumps(result) + "\n")
                output.flush()
                print(json.dumps(result), flush=True)
        print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
