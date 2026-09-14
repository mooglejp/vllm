# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the benchmark-only P3-v2a pure-FP8 mapping candidates.

Inputs are pre-expanded FP8 E4M3 byte matrices.  There is no MXFP4 decode,
E8M0 scale, activation quantization, production registration, or vLLM
dispatch.  The old one-wave 16x16 kernel is measured beside the proposed
2/4-wave large-tile candidates.  Correctness and compilation happen before
the timed round-robin samples.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from benchmarks.kernels.benchmark_gfx1201_fp8_wmma import (
        FP8_DTYPE,
        MODEL_SHAPES,
        _error,
        _fp8_bytes,
        _git_revision,
        _load_extension,
        _oracle_slice,
    )
except ModuleNotFoundError:
    from benchmark_gfx1201_fp8_wmma import (
        FP8_DTYPE,
        MODEL_SHAPES,
        _error,
        _fp8_bytes,
        _git_revision,
        _load_extension,
        _oracle_slice,
    )


@dataclass(frozen=True)
class TimingConfig:
    warmups: int = 5
    samples: int = 20


CANDIDATES: tuple[tuple[str, str, int, int, int], ...] = (
    ("a0_1wave_16x16", "fp8_wmma_gemm", 1, 16, 16),
    ("a1_2wave_64x64", "fp8_wmma_gemm_2wave_64x64", 2, 64, 64),
    ("a2_4wave_64x64", "fp8_wmma_gemm_4wave_64x64", 4, 64, 64),
    ("a3_4wave_64x128", "fp8_wmma_gemm_4wave_64x128", 4, 64, 128),
    ("a4_4wave_128x64", "fp8_wmma_gemm_4wave_128x64", 4, 128, 64),
)
CANDIDATE_BY_NAME = {candidate[0]: candidate for candidate in CANDIDATES}
DEFAULT_ROWS = (64, 256)
MAPPING_SPEEDUP_GATE = 5.0


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


def _invoke(
    extension: Any, method: str, a: torch.Tensor, b: torch.Tensor, output: torch.Tensor
) -> None:
    getattr(extension, method)(a, b, output)


def _candidate_specs(names: list[str] | None) -> list[tuple[str, str, int, int, int]]:
    if names is None:
        return list(CANDIDATES)
    unknown = sorted(set(names) - set(CANDIDATE_BY_NAME))
    if unknown:
        raise ValueError(f"unknown candidate(s): {', '.join(unknown)}")
    return [CANDIDATE_BY_NAME[name] for name in names]


def _correctness_case(
    extension: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    specs: list[tuple[str, str, int, int, int]],
) -> dict[str, object]:
    if outputs[specs[0][0]].numel() <= 1_000_000:
        rows, cols = a.size(0), b.size(0)
        scope = "full output"
    else:
        rows = min(a.size(0), 16)
        cols = min(b.size(0), 32)
        scope = "first output slice"
    for name, method, _, _, _ in specs:
        _invoke(extension, method, a, b, outputs[name])
    torch.accelerator.synchronize()
    oracle = _oracle_slice(a, b, rows, cols)
    torch_fp32 = torch.mm(
        a[:rows].view(FP8_DTYPE).float(),
        b[:cols].view(FP8_DTYPE).float().t(),
    )
    baseline_name = specs[0][0]
    baseline = outputs[baseline_name][:rows, :cols]
    baseline_error = _error(baseline, oracle)
    result: dict[str, object] = {
        "oracle": "FP64 decode of exact FP8 bytes with FP64 accumulation",
        "scope": scope,
        "slice": [rows, cols],
        "baseline": baseline_name,
        "baseline_fp64": baseline_error,
        "baseline_torch_fp32": _error(baseline, torch_fp32.double()),
        "candidates": {},
    }
    candidate_results = result["candidates"]
    assert isinstance(candidate_results, dict)
    for name, _, _, _, _ in specs:
        actual = outputs[name][:rows, :cols]
        finite = bool(torch.isfinite(actual).all())
        oracle_error = _error(actual, oracle)
        candidate_results[name] = {
            "finite": finite,
            "fp64": oracle_error,
            "vs_baseline": _error(actual, baseline),
            "torch_fp32": _error(actual, torch_fp32.double()),
            "sanity_passed": (
                finite
                and oracle_error["max_abs"] <= 0.25
                and oracle_error["rmse"] <= 0.01
            ),
        }
    result["finite"] = all(
        bool(candidate_results[name]["finite"]) for name, *_ in specs
    )
    result["sanity_tolerance"] = {
        "max_abs": 0.25,
        "purpose": "pure-FP8 correctness smoke check; not a W4A8 adoption gate",
    }
    result["passed"] = bool(
        result["finite"]
        and all(bool(candidate_results[name]["sanity_passed"]) for name, *_ in specs)
    )
    return result


def _tail_correctness_probe(
    extension: Any, specs: list[tuple[str, str, int, int, int]]
) -> dict[str, object]:
    m, n, k = 129, 17, 64
    a = _fp8_bytes((m, k), 1201001)
    b = _fp8_bytes((n, k), 1201002)
    outputs = {
        name: torch.empty((m, n), device="cuda", dtype=torch.float32)
        for name, *_ in specs
    }
    result = _correctness_case(extension, a, b, outputs, specs)
    result["shape"] = [m, n, k]
    result["purpose"] = "M/N tail correctness before timing"
    return result


def _benchmark_case(
    extension: Any,
    m: int,
    n: int,
    k: int,
    flush: torch.Tensor,
    config: TimingConfig,
    seed: int,
    specs: list[tuple[str, str, int, int, int]],
    skip_correctness: bool,
) -> dict[str, object]:
    a = _fp8_bytes((m, k), seed)
    b = _fp8_bytes((n, k), seed + 1)
    outputs = {
        name: torch.empty((m, n), device="cuda", dtype=torch.float32)
        for name, *_ in specs
    }
    correctness = (
        {"skipped": True}
        if skip_correctness
        else _correctness_case(extension, a, b, outputs, specs)
    )
    if not skip_correctness and not correctness["passed"]:
        raise RuntimeError(f"pure-FP8 correctness failed for M={m}, N={n}, K={k}")

    operations = {
        name: lambda method=method, output=outputs[name]: _invoke(
            extension, method, a, b, output
        )
        for name, method, *_ in specs
    }
    raw = _measure_round_robin(operations, flush, config)
    timing = {name: _summary(values) for name, values in raw.items()}
    flop = 2.0 * m * n * k
    effective_tflops = {
        name: flop / (values["median_us"] * 1.0e6) for name, values in timing.items()
    }
    baseline_name = specs[0][0]
    baseline_us = timing[baseline_name]["median_us"]
    speedup_over_baseline = {
        name: baseline_us / values["median_us"] for name, values in timing.items()
    }
    return {
        "m": m,
        "n": n,
        "k": k,
        "seed": seed,
        "timing": timing,
        "effective_tflops": effective_tflops,
        "speedup_over_a0": speedup_over_baseline,
        "correctness": correctness,
        "inputs": {
            "a": "raw FP8 E4M3 bytes, prequantized",
            "b": "raw FP8 E4M3 bytes, preexpanded",
            "scales": "none",
            "mxfp4_decode": False,
        },
        "timed_region": (
            "one fixed-buffer GEMM call; allocation, conversion, compilation, "
            "oracle, and input generation are excluded"
        ),
    }


def _aggregate_gate(
    records: list[dict[str, object]],
    specs: list[tuple[str, str, int, int, int]],
) -> dict[str, object]:
    minima: dict[str, float] = {name: float("inf") for name, *_ in specs}
    for record in records:
        speedups = record["speedup_over_a0"]
        assert isinstance(speedups, dict)
        for name in minima:
            minima[name] = min(minima[name], float(speedups[name]))
    return {
        "baseline": specs[0][0],
        "threshold": MAPPING_SPEEDUP_GATE,
        "min_speedup_over_a0": minima,
        "pass": {
            name: value >= MAPPING_SPEEDUP_GATE
            for name, value in minima.items()
            if name != specs[0][0]
        },
        "decision": (
            "continue to P3-v2b only if every selected large-tile candidate "
            "under test meets the 5x mapping gate; otherwise stop"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape-index", type=int, nargs="+", default=None)
    parser.add_argument("--rows", type=int, nargs="+", default=list(DEFAULT_ROWS))
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--flush-mib", type=int, default=64)
    parser.add_argument(
        "--build-directory",
        type=Path,
        default=Path("/tmp/tq-gfx1201-fp8-wmma-v2a-build"),
    )
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    if not torch.accelerator.is_available() or torch.version.hip is None:
        raise RuntimeError("This benchmark requires a ROCm GPU")
    if any(row < 1 for row in args.rows):
        raise ValueError("rows must be positive")
    if args.warmups < 0 or args.samples < 1:
        raise ValueError("warmups must be nonnegative and samples must be positive")
    specs = _candidate_specs(args.candidates)
    if specs[0][0] != CANDIDATES[0][0]:
        raise ValueError("A0 must remain selected as the speed baseline")

    extension = _load_extension(args.build_directory, args.verbose_build)
    config = TimingConfig(args.warmups, args.samples)
    flush = torch.empty(args.flush_mib * 1024 * 1024, device="cuda", dtype=torch.uint8)
    if not args.skip_correctness:
        tail_correctness = _tail_correctness_probe(extension, specs)
        if not tail_correctness["passed"]:
            raise RuntimeError(f"tail correctness failed: {tail_correctness}")
    else:
        tail_correctness = {"skipped": True}

    selected = (
        range(len(MODEL_SHAPES)) if args.shape_index is None else args.shape_index
    )
    for index in selected:
        if index < 0 or index >= len(MODEL_SHAPES):
            raise ValueError(f"shape index {index} is out of range")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "revision": _git_revision(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": str(torch.accelerator.current_accelerator()),
        "timing_config": asdict(config),
        "rows": args.rows,
        "shape_indices": list(selected),
        "flush_mib": args.flush_mib,
        "build_directory": str(args.build_directory),
        "source": "benchmarks/kernels/gfx1201_fp8_wmma_microbenchmark.cu",
        "phase": "P3-v2a pure FP8 mapping only",
        "candidates": [
            {
                "name": name,
                "method": method,
                "waves": waves,
                "tile_m": tile_m,
                "tile_n": tile_n,
            }
            for name, method, waves, tile_m, tile_n in specs
        ],
        "mapping_gate": {
            "baseline": CANDIDATES[0][0],
            "threshold": MAPPING_SPEEDUP_GATE,
            "scope": "same-shape preexpanded FP8 GEMM median latency",
        },
        "scope": (
            "diagnostic only; no MXFP4 decode, E8M0 scale, production "
            "registration, dispatch, threshold, or model integration"
        ),
        "tail_correctness": tail_correctness,
    }
    records: list[dict[str, object]] = []
    with args.output.open("w") as output_file:
        output_file.write(json.dumps({"metadata": metadata}) + "\n")
        output_file.flush()
        print(json.dumps({"metadata": metadata}), flush=True)
        for shape_index in selected:
            n, k = MODEL_SHAPES[shape_index]
            for m in args.rows:
                record = _benchmark_case(
                    extension,
                    m,
                    n,
                    k,
                    flush,
                    config,
                    1201 + shape_index * 10000 + m,
                    specs,
                    args.skip_correctness,
                )
                records.append(record)
                output_file.write(json.dumps(record) + "\n")
                output_file.flush()
                print(json.dumps(record), flush=True)
        gate = _aggregate_gate(records, specs)
        output_file.write(json.dumps({"mapping_gate": gate}) + "\n")
        output_file.flush()
        print(json.dumps({"mapping_gate": gate}), flush=True)
        print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
