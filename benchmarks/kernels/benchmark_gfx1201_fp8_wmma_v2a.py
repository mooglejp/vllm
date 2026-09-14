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
import math
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
) -> list[dict[str, object]]:
    """Check both the existing tail and a full multi-wave column tail."""

    def column_distinct_inputs(
        m: int, n: int, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a_values = torch.ones((m, k), device="cuda", dtype=torch.float32)
        column_values = torch.linspace(-4.0, 4.0, n, device="cuda")
        b_values = column_values[:, None].expand(n, k)
        return (
            a_values.to(FP8_DTYPE).view(torch.uint8),
            b_values.to(FP8_DTYPE).view(torch.uint8),
        )

    cases: list[tuple[str, int, int, int, torch.Tensor, torch.Tensor]] = []
    m, n, k = 129, 17, 64
    cases.append(
        (
            "random_tail",
            m,
            n,
            k,
            _fp8_bytes((m, k), 1201001),
            _fp8_bytes((n, k), 1201002),
        )
    )
    m, n, k = 129, 129, 64
    distinct_a, distinct_b = column_distinct_inputs(m, n, k)
    cases.append(("column_distinct_full", m, n, k, distinct_a, distinct_b))

    results: list[dict[str, object]] = []
    for name, m, n, k, a, b in cases:
        outputs = {
            candidate_name: torch.empty((m, n), device="cuda", dtype=torch.float32)
            for candidate_name, *_ in specs
        }
        result = _correctness_case(extension, a, b, outputs, specs)
        result["case"] = name
        result["shape"] = [m, n, k]
        result["purpose"] = "M/N tail correctness before timing"
        results.append(result)
    return results


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
    a_fp32 = a.view(FP8_DTYPE).float()
    b_fp32 = b.view(FP8_DTYPE).float()
    outputs = {
        name: torch.empty((m, n), device="cuda", dtype=torch.float32)
        for name, *_ in specs
    }
    fp32_output = torch.empty((m, n), device="cuda", dtype=torch.float32)
    correctness = (
        {"skipped": True}
        if skip_correctness
        else _correctness_case(extension, a, b, outputs, specs)
    )

    operations = {
        name: lambda method=method, output=outputs[name]: _invoke(
            extension, method, a, b, output
        )
        for name, method, *_ in specs
    }
    operations["torch_fp32_mm"] = lambda: torch.mm(a_fp32, b_fp32.t(), out=fp32_output)
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
        "fp32_baseline": "pre-expanded FP8 bytes converted before timing",
        "timed_region": (
            "one fixed-buffer GEMM call; allocation, conversion, compilation, "
            "oracle, and input generation are excluded"
        ),
    }


def _candidate_correctness_passed(
    record: dict[str, object], name: str, requested: bool
) -> bool:
    if not requested:
        return False
    correctness = record.get("correctness")
    if not isinstance(correctness, dict) or correctness.get("skipped"):
        return False
    candidate_results = correctness.get("candidates")
    if not isinstance(candidate_results, dict):
        return False
    result = candidate_results.get(name)
    return isinstance(result, dict) and bool(result.get("sanity_passed", False))


def _aggregate_gate(
    records: list[dict[str, object]],
    specs: list[tuple[str, str, int, int, int]],
    expected_cases: list[tuple[int, int]],
    correctness_requested: bool,
    tail_correctness: list[dict[str, object]],
) -> dict[str, object]:
    expected_records = {
        (int(record["shape_index"]), int(record["m"])): record
        for record in records
        if "shape_index" in record
    }
    completed_cases = [case for case in expected_cases if case in expected_records]
    missing_cases = [case for case in expected_cases if case not in expected_records]
    candidate_gates: dict[str, dict[str, object]] = {}
    for name, *_ in specs:
        case_records = [expected_records.get(case) for case in expected_cases]
        all_cases_completed = not missing_cases
        speedups: list[float] = []
        correctness_passed = correctness_requested and bool(expected_cases)
        for record in case_records:
            if record is None:
                correctness_passed = False
                continue
            speedup_data = record.get("speedup_over_a0")
            speedup = speedup_data.get(name) if isinstance(speedup_data, dict) else None
            if isinstance(speedup, (int, float)) and math.isfinite(float(speedup)):
                speedups.append(float(speedup))
            else:
                all_cases_completed = False
            if not _candidate_correctness_passed(record, name, correctness_requested):
                correctness_passed = False
        tail_passed = (
            correctness_requested
            and bool(tail_correctness)
            and all(
                _candidate_correctness_passed(
                    {"correctness": case}, name, correctness_requested
                )
                for case in tail_correctness
            )
        )
        correctness_passed = correctness_passed and tail_passed
        speedup_passed = (
            all_cases_completed
            and len(speedups) == len(expected_cases)
            and all(value >= MAPPING_SPEEDUP_GATE for value in speedups)
        )
        eligible = (
            bool(expected_cases)
            and all_cases_completed
            and correctness_passed
            and speedup_passed
        )
        candidate_gates[name] = {
            "min_speedup_over_a0": min(speedups) if speedups else None,
            "all_cases_completed": all_cases_completed,
            "correctness_passed": correctness_passed,
            "speedup_passed": speedup_passed,
            "eligible_for_v2b": eligible,
        }
    large_tile_names = [name for name, *_ in specs[1:]]
    eligible_candidates = [
        name
        for name in large_tile_names
        if bool(candidate_gates[name]["eligible_for_v2b"])
    ]
    if eligible_candidates:
        decision = "continue to P3-v2b with eligible candidates: " + ", ".join(
            eligible_candidates
        )
    else:
        decision = (
            "stop P3-v2a; no large-tile candidate completed every required "
            "case with correctness and the 5x mapping gate"
        )
    return {
        "baseline": specs[0][0],
        "threshold": MAPPING_SPEEDUP_GATE,
        "expected_cases": [list(case) for case in expected_cases],
        "completed_cases": [list(case) for case in completed_cases],
        "missing_cases": [list(case) for case in missing_cases],
        "correctness_required": True,
        "correctness_run": correctness_requested,
        "tail_correctness_required": correctness_requested,
        "candidates": candidate_gates,
        "min_speedup_over_a0": {
            name: candidate["min_speedup_over_a0"]
            for name, candidate in candidate_gates.items()
        },
        "pass": {
            name: bool(candidate_gates[name]["eligible_for_v2b"])
            for name in large_tile_names
        },
        "eligible_candidates": eligible_candidates,
        "decision": decision,
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
    else:
        tail_correctness = {"skipped": True}

    selected = list(
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
            "correctness_required": not args.skip_correctness,
            "eligibility": (
                "tail and selected cases must be complete, correct, and at least 5x A0"
            ),
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
                record["shape_index"] = shape_index
                records.append(record)
                output_file.write(json.dumps(record) + "\n")
                output_file.flush()
                print(json.dumps(record), flush=True)
        expected_cases = [
            (shape_index, m) for shape_index in selected for m in args.rows
        ]
        tail_cases = tail_correctness if isinstance(tail_correctness, list) else []
        gate = _aggregate_gate(
            records, specs, expected_cases, not args.skip_correctness, tail_cases
        )
        output_file.write(json.dumps({"mapping_gate": gate}) + "\n")
        output_file.flush()
        print(json.dumps({"mapping_gate": gate}), flush=True)
        print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
