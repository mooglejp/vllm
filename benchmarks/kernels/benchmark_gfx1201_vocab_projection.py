# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the ROCm BF16 full-vocabulary projection used at small M."""

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.utils.platform_utils import num_compute_units


def _measure_round_robin(
    operations: dict[str, Callable[[], torch.Tensor]],
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
            operation = operations[name]
            operation()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end) * 1000)
    return timings


def _timing_summary(values: list[float], estimated_bytes: int) -> dict:
    median_us = statistics.median(values)
    deciles = statistics.quantiles(values, n=10)
    return {
        "median_us": median_us,
        "min_us": min(values),
        "max_us": max(values),
        "p10_us": deciles[0],
        "p90_us": deciles[-1],
        "estimated_gbps": estimated_bytes / median_us / 1e3,
        "samples_us": values,
    }


def _difference(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    delta = candidate.float() - reference.float()
    return {
        "bitwise_exact": torch.equal(candidate, reference),
        "max_abs": delta.abs().max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "finite": torch.isfinite(candidate).all().item(),
    }


def _benchmark_rows(
    rows: int,
    n: int,
    k: int,
    weight: torch.Tensor,
    cu_counts: list[int],
    flush: torch.Tensor,
    warmups: int,
    samples: int,
) -> dict:
    x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
    default_cu_count = num_compute_units()
    reference = ops.wvSplitK(weight, x, default_cu_count)
    candidates = {
        str(cu_count): ops.wvSplitK(weight, x, cu_count) for cu_count in cu_counts
    }
    torch_linear = F.linear(x, weight)
    torch.accelerator.synchronize()

    correctness = {
        f"wvsplitk_cu{cu_count}": _difference(candidates[str(cu_count)], reference)
        for cu_count in cu_counts
    }
    correctness["torch_linear"] = _difference(torch_linear, reference)
    for name, result in correctness.items():
        if name.startswith("wvsplitk") and not result["bitwise_exact"]:
            raise AssertionError(f"{name} differs from the default: {result}")

    operations = {
        f"wvsplitk_cu{cu_count}": (
            lambda cu_count=cu_count: ops.wvSplitK(weight, x, cu_count)
        )
        for cu_count in cu_counts
    }
    operations["torch_linear"] = lambda: F.linear(x, weight)
    raw_timings = _measure_round_robin(operations, flush, warmups, samples)
    estimated_bytes = (
        weight.numel() * weight.element_size()
        + x.numel() * x.element_size()
        + rows * n * x.element_size()
    )
    return {
        "shape": {"m": rows, "n": n, "k": k},
        "estimated_bytes_per_call": estimated_bytes,
        "correctness_vs_default_wvsplitk": correctness,
        "timing": {
            name: _timing_summary(values, estimated_bytes)
            for name, values in raw_timings.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--n", type=int, default=248320)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument(
        "--cu-counts", nargs="+", type=int, default=[32, 40, 48, 56, 60, 64, 72]
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if any(row < 1 or row > 5 for row in args.rows):
        parser.error("--rows must be between 1 and 5 for wvSplitK")
    if args.n <= 0 or args.k <= 0 or args.k % 8 != 0:
        parser.error("--n must be positive and --k must be positive and divisible by 8")
    if any(cu_count <= 0 for cu_count in args.cu_counts):
        parser.error("--cu-counts values must be positive")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("This benchmark requires a ROCm GPU")
    torch.manual_seed(args.seed)
    properties = torch.cuda.get_device_properties(0)
    weight = torch.randn(
        (args.n, args.k), device="cuda", dtype=torch.bfloat16
    ).contiguous()
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    result = {
        "device": properties.name or getattr(properties, "gcnArchName", "unknown"),
        "runtime_multiprocessor_count": properties.multi_processor_count,
        "torch_version": torch.__version__,
        "rocm_version": torch.version.hip,
        "dtype": "torch.bfloat16",
        "default_cu_count": num_compute_units(),
        "seed": args.seed,
        "warmups": args.warmups,
        "samples": args.samples,
        "results": [
            _benchmark_rows(
                rows,
                args.n,
                args.k,
                weight,
                args.cu_counts,
                flush,
                args.warmups,
                args.samples,
            )
            for rows in args.rows
        ],
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
