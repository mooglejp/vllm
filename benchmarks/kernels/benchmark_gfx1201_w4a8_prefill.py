# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the opt-in native gfx1201 FP8-WMMA W4A8 large-M candidate.

The candidate exposes quantization and GEMM as separate custom ops so the
report can distinguish quantization, GEMM, and combined time.  The production
linear dispatch is not used by this benchmark.
"""

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from vllm.model_executor.kernels.linear.mxfp4.gfx1201_w4a8 import (
    dequantize_mxfp4_weight_reference,
    gfx1201_w4a8_linear_reference,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    quant_dequant_mxfp4,
)

MODEL_SHAPES = (
    (5120, 6144),
    (34816, 5120),
    (5120, 17408),
    (16384, 5120),
    (96, 5120),
    (14336, 5120),
)
QUERY_ROWS = (128, 256, 512, 1024, 2048, 3072, 4096)


@dataclass(frozen=True)
class TimingConfig:
    warmups: int = 5
    samples: int = 20


def _ops():
    if not hasattr(torch.ops, "_rocm_C"):
        raise RuntimeError("_rocm_C is not loaded")
    namespace = torch.ops._rocm_C
    if not hasattr(namespace, "gfx1201_w4a8_quantize") or not hasattr(
        namespace, "gfx1201_w4a8_gemm"
    ):
        raise RuntimeError("_rocm_C lacks the native gfx1201 W4A8 prefill ops")
    return namespace


def _summary(samples: list[float]) -> dict[str, object]:
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def _measure_round_robin(
    operations: dict[str, Callable[[], object]],
    flush: torch.Tensor,
    config: TimingConfig,
) -> dict[str, list[float]]:
    for _ in range(config.warmups):
        for operation in operations.values():
            operation()
    torch.accelerator.synchronize()
    timings = {name: [] for name in operations}
    names = list(operations)
    for sample in range(config.samples):
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


def _native_reference(
    quantized: torch.Tensor,
    row_scale: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    # Use FP64 for the oracle accumulation so WMMA versus Torch FP32
    # reduction-order differences are not mistaken for a quantization error.
    activation = quantized.view(torch.float8_e4m3fn).double()
    activation = activation * row_scale.double()[:, None]
    weight = dequantize_mxfp4_weight_reference(
        packed_weight, weight_scale, dtype=torch.float32
    ).double()
    return torch.mm(activation, weight.t()).to(torch.bfloat16)


def _correctness_check() -> dict[str, object]:
    ops = _ops()
    m, n, k = 128, 17, 64
    torch.manual_seed(1201)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    packed = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    scales = torch.randint(120, 132, (n, k // 32), device="cuda", dtype=torch.uint8)
    quantized = torch.empty((m, k), device="cuda", dtype=torch.uint8)
    row_scale = torch.empty((m,), device="cuda", dtype=torch.float32)
    output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    ops.gfx1201_w4a8_quantize(x, quantized, row_scale)
    ops.gfx1201_w4a8_gemm(quantized, row_scale, packed, scales, output)
    reference = _native_reference(quantized, row_scale, packed, scales)
    delta = output.float() - reference.float()
    torch_reference = gfx1201_w4a8_linear_reference(x, packed, scales)
    torch_delta = output.float() - torch_reference.float()
    return {
        "shape": [m, n, k],
        "native_fp8_byte_reference": True,
        "native_reference_accumulation": "float64",
        "max_abs": delta.abs().max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "bitwise_equal": bool(torch.equal(output, reference)),
        "torch_fp8_conversion_reference": True,
        "torch_reference_max_abs": torch_delta.abs().max().item(),
        "torch_reference_rmse": torch_delta.square().mean().sqrt().item(),
        "torch_reference_bitwise_equal": bool(torch.equal(output, torch_reference)),
    }


def _benchmark_case(
    m: int,
    n: int,
    k: int,
    flush: torch.Tensor,
    config: TimingConfig,
    seed: int,
) -> dict[str, object]:
    ops = _ops()
    torch.manual_seed(seed)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    packed = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    scales = torch.randint(120, 132, (n, k // 32), device="cuda", dtype=torch.uint8)

    quantized = torch.empty((m, k), device="cuda", dtype=torch.uint8)
    row_scale = torch.empty((m,), device="cuda", dtype=torch.float32)
    output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    old_output = torch.empty_like(output)
    old_x = quant_dequant_mxfp4(x)
    old_weight = dequantize_mxfp4_weight_reference(packed, scales, dtype=torch.bfloat16)

    def quantize():
        ops.gfx1201_w4a8_quantize(x, quantized, row_scale)

    def gemm():
        ops.gfx1201_w4a8_gemm(quantized, row_scale, packed, scales, output)

    def combined():
        quantize()
        gemm()

    def old_emulation():
        torch.mm(old_x, old_weight.t(), out=old_output)

    # Populate the candidate workspace before timing GEMM-only.
    quantize()
    torch.accelerator.synchronize()
    raw = _measure_round_robin(
        {"quantization": quantize, "gemm": gemm, "combined": combined},
        flush,
        config,
    )
    old_raw = _measure_round_robin({"old_emulation_mm": old_emulation}, flush, config)
    timing = {name: _summary(values) for name, values in raw.items()}
    old_timing = {name: _summary(values) for name, values in old_raw.items()}
    return {
        "m": m,
        "n": n,
        "k": k,
        "timing": timing,
        "old_emulation_mm_timing": old_timing,
        "old_emulation_mm_over_candidate_combined": (
            old_timing["old_emulation_mm"]["median_us"]
            / timing["combined"]["median_us"]
        ),
        "workspace_bytes": {
            "quantized_activation": quantized.numel() * quantized.element_size(),
            "row_scale": row_scale.numel() * row_scale.element_size(),
            "output": output.numel() * output.element_size(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape-index", type=int, nargs="+")
    parser.add_argument("--rows", type=int, nargs="+", default=list(QUERY_ROWS))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--flush-mib", type=int, default=64)
    args = parser.parse_args()

    if not torch.accelerator.is_available() or torch.version.hip is None:
        raise RuntimeError("This benchmark requires a ROCm GPU")
    config = TimingConfig(warmups=args.warmups, samples=args.samples)
    flush = torch.empty(args.flush_mib * 1024 * 1024, device="cuda", dtype=torch.uint8)
    correctness = _correctness_check()
    selected = (
        range(len(MODEL_SHAPES)) if args.shape_index is None else args.shape_index
    )
    metadata = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": str(torch.accelerator.current_accelerator()),
        "timing_config": asdict(config),
        "rows": args.rows,
        "flush_mib": args.flush_mib,
        "correctness": correctness,
        "baseline": "pre-dequantized BF16 weight + activation MXFP4 QDQ + torch.mm",
        "candidate": (
            "native FP8-WMMA custom ops; quantization and GEMM separately timed"
        ),
    }
    with args.output.open("w") as output:
        output.write(json.dumps({"metadata": metadata}) + "\n")
        for shape_index in selected:
            n, k = MODEL_SHAPES[shape_index]
            for m in args.rows:
                result = _benchmark_case(
                    m,
                    n,
                    k,
                    flush,
                    config,
                    1201 + 10 * shape_index + m,
                )
                output.write(json.dumps(result) + "\n")
                output.flush()
                print(json.dumps(result), flush=True)
        print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
